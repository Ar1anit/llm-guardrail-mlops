"""ONNX inference for the exported classifier, and its score on the golden evaluation set.

This is the single inference path: serving reuses ``GoldenEvaluator.evaluate``, so the golden
numbers describe exactly what production runs. Build it with ``GoldenEvaluator.from_export``,
which verifies the export directory against its ``manifest.json`` and takes the tokenizer,
sequence length and decision rule from it. The module imports neither torch nor transformers;
only the default tokenizer loader imports transformers, when it is called.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, Self, TypedDict

import numpy as np
import numpy.typing as npt
import onnxruntime as ort  # type: ignore[import-untyped]  # onnxruntime ships no py.typed marker

from src.evaluation.metrics import (
    ID_TO_LABEL,
    MALICIOUS_ID,
    attack_type_metrics,
    binary_metrics,
)

# Mirror the ONNX interface written by src/training/export_onnx.py, which imports torch.
INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAME = "logits"
# Mirror the export layout written by src/training/distilbert_pipeline.py.
MODEL_FILE_NAME = "model.onnx"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1
# Flags a prompt when the malicious probability is strictly above the threshold, so 0.5
# reproduces the argmax decision used to select checkpoints during training.
DEFAULT_THRESHOLD = 0.5
# The manifest decision rules this evaluator can reproduce, as malicious-probability thresholds.
DECISION_RULE_THRESHOLDS: Mapping[str, float] = MappingProxyType({"argmax": DEFAULT_THRESHOLD})
DEFAULT_BATCH_SIZE = 32

Scores = npt.NDArray[np.float64]


class Tokenizer(Protocol):
    """The callable part of a Hugging Face tokenizer that inference needs."""

    def __call__(self, text: list[str], **kwargs: Any) -> Mapping[str, Any]: ...


class Verdict(TypedDict):
    label: str
    score: float


class GoldenRecord(TypedDict):
    text: str
    label: int
    attack_type: str


class ArtifactVerificationError(ValueError):
    """An export directory does not match its manifest or the inference contract."""


@dataclass(frozen=True, slots=True)
class ExportProvenance:
    """What ``from_export`` verified: the manifest's digest and the sections a report cites."""

    export_directory: Path
    manifest_sha256: str
    lineage: Mapping[str, object]
    inference_contract: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class GoldenResult:
    """Golden-set metrics tied to the exact model and golden file bytes that produced them."""

    model_sha256: str
    golden_sha256: str
    metrics: dict[str, float]


def load_exported_tokenizer(directory: Path) -> Tokenizer:
    """Load an exported Hugging Face tokenizer from disk, never from the network."""
    from transformers import AutoTokenizer  # Imported here so serving can avoid transformers.

    tokenizer: Tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    return tokenizer


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _section(manifest: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        raise ArtifactVerificationError(f"manifest.json has no '{key}' object.")
    return value


def _read_manifest(export_directory: Path) -> tuple[dict[str, object], str]:
    path = export_directory / MANIFEST_FILE_NAME
    try:
        data = path.read_bytes()
    except FileNotFoundError as error:
        raise ArtifactVerificationError(f"{path} does not exist.") from error
    try:
        manifest = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ArtifactVerificationError(f"{path} is not valid JSON.") from error
    if not isinstance(manifest, dict):
        raise ArtifactVerificationError(f"{path} is not a JSON object.")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ArtifactVerificationError(
            f"Unsupported manifest schema_version {manifest.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}.",
        )
    return manifest, _sha256(data)


def _verify_files(export_directory: Path, manifest: Mapping[str, object]) -> dict[str, str]:
    """Require the directory to hold exactly the manifest's files, each with its SHA-256."""
    expected = _section(manifest, "files")
    if not all(isinstance(digest, str) for digest in expected.values()):
        raise ArtifactVerificationError("manifest.json 'files' must map paths to SHA-256 strings.")
    present: dict[str, Path] = {}
    for path in sorted(export_directory.rglob("*")):
        relative = path.relative_to(export_directory).as_posix()
        if path.is_symlink():
            raise ArtifactVerificationError(f"Export contains a symlink: {relative}.")
        if path.is_file() and relative != MANIFEST_FILE_NAME:
            present[relative] = path
    missing = sorted(set(expected) - set(present))
    unexpected = sorted(set(present) - set(expected))
    if missing or unexpected:
        raise ArtifactVerificationError(
            f"Export files differ from manifest.json: missing {missing}, unexpected {unexpected}.",
        )
    mismatched = [
        name for name, path in present.items() if _sha256(path.read_bytes()) != expected[name]
    ]
    if mismatched:
        raise ArtifactVerificationError(f"SHA-256 mismatch for {mismatched}.")
    if MODEL_FILE_NAME not in expected:
        raise ArtifactVerificationError(f"manifest.json lists no {MODEL_FILE_NAME}.")
    return {name: str(digest) for name, digest in expected.items()}


@dataclass(frozen=True, slots=True)
class _Contract:
    max_sequence_length: int
    threshold: float
    tokenizer_directory: str


def _verify_contract(manifest: Mapping[str, object], files: Mapping[str, str]) -> _Contract:
    contract = _section(manifest, "inference_contract")
    problems: list[str] = []
    if contract.get("input_names") != list(INPUT_NAMES):
        problems.append(f"input_names must be {list(INPUT_NAMES)}")
    if contract.get("output_name") != OUTPUT_NAME:
        problems.append(f"output_name must be '{OUTPUT_NAME}'")
    expected_labels = {str(label_id): name for label_id, name in ID_TO_LABEL.items()}
    if contract.get("labels") != expected_labels:
        problems.append(f"labels must be {expected_labels}")
    rule = contract.get("decision_rule")
    if not isinstance(rule, str) or rule not in DECISION_RULE_THRESHOLDS:
        problems.append(f"decision_rule must be one of {sorted(DECISION_RULE_THRESHOLDS)}")
    length = contract.get("max_sequence_length")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        problems.append("max_sequence_length must be a positive integer")
    directory = contract.get("tokenizer_directory")
    if not isinstance(directory, str) or not any(
        name.startswith(f"{directory}/") for name in files
    ):
        problems.append("tokenizer_directory must name a directory of checksummed files")
    if problems:
        raise ArtifactVerificationError(
            "manifest.json breaks the inference contract: " + "; ".join(problems) + ".",
        )
    assert isinstance(rule, str) and isinstance(length, int) and isinstance(directory, str)
    return _Contract(length, DECISION_RULE_THRESHOLDS[rule], directory)


def _parse_golden_records(data: bytes) -> list[GoldenRecord]:
    records: list[GoldenRecord] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            text, label, attack_type = row["text"], row["label"], row["attack_type"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(
                f"Golden file line {line_number} is not a JSON object with 'text', "
                "'label' and 'attack_type' fields.",
            ) from error
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Golden file line {line_number} has an empty 'text'.")
        if isinstance(label, bool) or label not in ID_TO_LABEL:
            raise ValueError(
                f"Golden file line {line_number} has label {label!r}; "
                f"expected one of {sorted(ID_TO_LABEL)}.",
            )
        if not isinstance(attack_type, str) or not attack_type:
            raise ValueError(f"Golden file line {line_number} has an empty 'attack_type'.")
        records.append({"text": text, "label": label, "attack_type": attack_type})
    if not records:
        raise ValueError("The golden file has no records.")
    return records


def _malicious_probabilities(logits: npt.NDArray[np.float32]) -> Scores:
    if logits.ndim != 2 or logits.shape[1] != len(ID_TO_LABEL):
        raise ValueError(
            f"Expected logits of shape (rows, {len(ID_TO_LABEL)}), got {logits.shape}.",
        )
    # A NaN score compares False against the threshold and would silently fail open.
    if not np.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits; refusing to derive a verdict.")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    # Annotated because numpy < 2.5 (resolved on Python 3.11) types this index as Any.
    malicious: Scores = probabilities[:, MALICIOUS_ID]
    return malicious


def read_golden_records(path: Path) -> list[GoldenRecord]:
    """Read and validate the golden set; errors carry line numbers, never content."""
    return _parse_golden_records(path.read_bytes())


class GoldenEvaluator:
    """Classify prompts with an exported ONNX model and score it on the golden set."""

    def __init__(
        self,
        model_path: Path,
        tokenizer: Tokenizer,
        *,
        max_sequence_length: int,
        threshold: float = DEFAULT_THRESHOLD,
        expected_sha256: str | None = None,
    ) -> None:
        """Load ``model_path``; prefer ``from_export``, which also verifies the manifest.

        The model is read once, hashed, and loaded from the same bytes, so ``model_sha256``
        names exactly the graph that runs. A mismatch with ``expected_sha256`` raises.
        """
        if model_path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx model file, got '{model_path.name}'.")
        if max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be at least 1, got {max_sequence_length}.",
            )
        if not (math.isfinite(threshold) and 0.0 < threshold < 1.0):
            raise ValueError(f"threshold must lie strictly between 0 and 1, got {threshold}.")
        model_bytes = model_path.read_bytes()
        self.model_sha256 = _sha256(model_bytes)
        if expected_sha256 is not None and self.model_sha256 != expected_sha256:
            raise ArtifactVerificationError(f"SHA-256 mismatch for {model_path.name}.")
        self._tokenizer = tokenizer
        self._max_sequence_length = max_sequence_length
        self._threshold = threshold
        # Prompts are sensitive; no usage telemetry leaves the process that scores them.
        ort.disable_telemetry_events()
        self._session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        input_names = tuple(node.name for node in self._session.get_inputs())
        output_names = [node.name for node in self._session.get_outputs()]
        if input_names != INPUT_NAMES or output_names != [OUTPUT_NAME]:
            raise ArtifactVerificationError(
                f"Unexpected ONNX interface: inputs {input_names}, outputs {output_names}.",
            )
        self.provenance: ExportProvenance | None = None

    @classmethod
    def from_export(
        cls,
        export_directory: Path,
        *,
        tokenizer_loader: Callable[[Path], Tokenizer] | None = None,
    ) -> Self:
        """Build the evaluator an export directory describes, or raise.

        Fails closed with ``ArtifactVerificationError`` unless the directory holds exactly the
        files its ``manifest.json`` lists, each with the recorded SHA-256, and the manifest's
        ``inference_contract`` matches what this evaluator implements. The tokenizer, sequence
        length and decision threshold all come from the manifest. The checksums detect
        accidental change, not tampering by someone who can rewrite the whole directory.
        """
        manifest, manifest_sha256 = _read_manifest(export_directory)
        files = _verify_files(export_directory, manifest)
        contract = _verify_contract(manifest, files)
        load_tokenizer = tokenizer_loader or load_exported_tokenizer
        evaluator = cls(
            export_directory / MODEL_FILE_NAME,
            load_tokenizer(export_directory / contract.tokenizer_directory),
            max_sequence_length=contract.max_sequence_length,
            threshold=contract.threshold,
            expected_sha256=files[MODEL_FILE_NAME],
        )
        evaluator.provenance = ExportProvenance(
            export_directory=export_directory,
            manifest_sha256=manifest_sha256,
            lineage=_section(manifest, "lineage"),
            inference_contract=_section(manifest, "inference_contract"),
        )
        return evaluator

    @property
    def threshold(self) -> float:
        return self._threshold

    def _scores(self, texts: Sequence[str]) -> Scores:
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_sequence_length,
            return_tensors="np",
            return_token_type_ids=False,
        )
        inputs = {name: np.asarray(encoded[name], dtype=np.int64) for name in INPUT_NAMES}
        (logits,) = self._session.run([OUTPUT_NAME], inputs)
        return _malicious_probabilities(np.asarray(logits))

    def _labels(self, scores: Scores) -> npt.NDArray[np.int64]:
        return (scores > self._threshold).astype(np.int64)

    def evaluate(self, text: str) -> Verdict:
        """Classify one prompt; ``score`` is the probability that it is malicious."""
        if not text.strip():
            raise ValueError("Cannot classify an empty prompt.")
        scores = self._scores([text])
        return {"label": ID_TO_LABEL[int(self._labels(scores)[0])], "score": float(scores[0])}

    def evaluate_file(
        self,
        path: Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> GoldenResult:
        """Malicious-class metrics and per-``attack_type`` slices on a golden JSONL file.

        The file is read once and hashed, so ``golden_sha256`` names exactly the rows scored.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        data = path.read_bytes()
        records = _parse_golden_records(data)
        texts = [record["text"] for record in records]
        scores = np.concatenate(
            [
                self._scores(texts[start : start + batch_size])
                for start in range(0, len(texts), batch_size)
            ],
        )
        labels = np.array([record["label"] for record in records], dtype=np.int64)
        predictions = self._labels(scores)
        return GoldenResult(
            model_sha256=self.model_sha256,
            golden_sha256=_sha256(data),
            metrics={
                "rows": float(len(records)),
                "threshold": self._threshold,
                **binary_metrics(labels, predictions),
                **attack_type_metrics([record["attack_type"] for record in records], predictions),
            },
        )
