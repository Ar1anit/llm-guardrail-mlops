"""DistilBERT sequence-classification implementation of the training pipeline."""

import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import datasets  # type: ignore[import-untyped]  # datasets ships no py.typed marker
import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorWithPadding,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.evaluation.metrics import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    Labels,
    attack_type_metrics,
    binary_metrics,
)
from src.training.export_onnx import (
    INPUT_NAMES,
    OUTPUT_NAME,
    export_sequence_classifier,
    verify_onnx_parity,
)
from src.training.pipeline import (
    ExportedArtifact,
    TrainingPipeline,
    TrainingPipelineConfig,
    TrainingPipelineResult,
)

logger = logging.getLogger(__name__)

Logits = npt.NDArray[np.float32]


TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
EXPECTED_SPLITS = frozenset({TRAIN_SPLIT, VALIDATION_SPLIT})
REQUIRED_COLUMNS = frozenset({"text", "label", "attack_type"})
MAX_REPORTED_ROWS = 5
# Mirrors the PromptRecord text limit in src/data/schema.py.
MAX_TEXT_CHARACTERS = 10_000
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")

# Checkpoints are selected on the validation split by this metric. Malicious-class F1 stops a
# model that flags everything from winning on recall alone; the choice shifts the
# false-positive/false-negative balance, so it needs approval (docs/model-governance.md).
SELECTION_METRIC = "malicious_f1"
DECISION_RULE = "argmax"

EXPORT_DIRECTORY_NAME = "export"
MODEL_FILE_NAME = "model.onnx"
TOKENIZER_DIRECTORY_NAME = "tokenizer"
MANIFEST_FILE_NAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class DegenerateModelError(RuntimeError):
    """The model predicts a single class on the validation split and cannot discriminate."""


class MissingEvaluationEvidenceError(RuntimeError):
    """``export`` was called on weights that ``evaluate`` has not scored on this pipeline."""


@dataclass(frozen=True, slots=True)
class DistilBertTrainingConfig:
    """DistilBERT settings; field names match the keys in ``configs/`` where one exists.

    ``base_model_revision`` must be an immutable commit SHA, because a branch or tag can move
    between runs. ``golden_file`` is the protected golden evaluation set: it is read only to
    reject any overlap with the training and validation data, never to train or select on.
    """

    base_model_name: str
    base_model_revision: str
    max_sequence_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    use_cpu: bool
    train_file: Path
    validation_file: Path
    golden_file: Path

    def __post_init__(self) -> None:
        if not isinstance(self.base_model_revision, str) or not COMMIT_SHA.fullmatch(
            self.base_model_revision,
        ):
            raise ValueError(
                "base_model_revision must be a 40-character lowercase commit SHA; "
                "a branch, tag, or missing revision can change between runs.",
            )
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}.")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError(
                f"learning_rate must be positive and finite, got {self.learning_rate}."
            )


def _format_row_indices(indices: Sequence[int]) -> str:
    shown = ", ".join(str(index) for index in indices[:MAX_REPORTED_ROWS])
    return f"{shown}, ..." if len(indices) > MAX_REPORTED_ROWS else shown


def _is_valid_label(label: object) -> bool:
    # bool is an int subclass; reject it so True/False cannot pass as class IDs.
    return isinstance(label, int) and not isinstance(label, bool) and label in ID_TO_LABEL


def _blank_row_indices(values: Sequence[object]) -> list[int]:
    return [
        index
        for index, value in enumerate(values)
        if not isinstance(value, str) or not value.strip()
    ]


def _validate_split(name: str, split: datasets.Dataset) -> None:
    """Raise ``ValueError`` unless ``split`` satisfies the training data contract.

    Messages contain split names, counts, and row indices only, never prompt text.
    """
    missing_columns = sorted(REQUIRED_COLUMNS - set(split.column_names))
    if missing_columns:
        raise ValueError(f"Split '{name}' is missing required column(s): {missing_columns}.")
    if split.num_rows == 0:
        raise ValueError(f"Split '{name}' is empty.")

    for column in ("text", "attack_type"):
        blank_rows = _blank_row_indices(list(split[column]))
        if blank_rows:
            raise ValueError(
                f"Split '{name}' contains {len(blank_rows)} row(s) with a missing or blank "
                f"'{column}' (first indices: {_format_row_indices(blank_rows)}).",
            )

    too_long_rows = [
        index for index, text in enumerate(split["text"]) if len(text) > MAX_TEXT_CHARACTERS
    ]
    if too_long_rows:
        raise ValueError(
            f"Split '{name}' contains {len(too_long_rows)} row(s) with more than "
            f"{MAX_TEXT_CHARACTERS} characters (first indices: "
            f"{_format_row_indices(too_long_rows)}).",
        )

    labels: list[object] = list(split["label"])
    invalid_rows = [index for index, label in enumerate(labels) if not _is_valid_label(label)]
    if invalid_rows:
        raise ValueError(
            f"Split '{name}' contains {len(invalid_rows)} row(s) with an invalid 'label' "
            f"(expected an integer in {sorted(ID_TO_LABEL)}; "
            f"first indices: {_format_row_indices(invalid_rows)}).",
        )

    present_labels = set(labels)
    for label_id, label_name in sorted(ID_TO_LABEL.items()):
        if label_id not in present_labels:
            raise ValueError(
                f"Split '{name}' contains no examples of class '{label_name}' (label {label_id}).",
            )


def _validate_no_shared_texts(train: datasets.Dataset, validation: datasets.Dataset) -> None:
    shared_texts = set(train["text"]) & set(validation["text"])
    if shared_texts:
        raise ValueError(
            f"{len(shared_texts)} distinct text(s) appear in both the '{TRAIN_SPLIT}' and "
            f"'{VALIDATION_SPLIT}' splits, which would inflate validation metrics.",
        )


def _token_keys(input_ids: Iterable[Sequence[int]]) -> set[tuple[int, ...]]:
    """Model inputs as hashable keys: texts that tokenize alike are the same example."""
    return {tuple(ids) for ids in input_ids}


def _read_golden_texts(path: Path) -> list[str]:
    """Read each golden record's ``text``; errors carry line numbers, never content."""
    texts: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Golden file line {line_number} is not a JSON object with a 'text' field.",
                ) from error
            if not isinstance(text, str):
                raise ValueError(f"Golden file line {line_number} has a non-string 'text'.")
            texts.append(text)
    if not texts:
        raise ValueError("The golden file has no records, so overlap with it cannot be checked.")
    return texts


def _state_fingerprint(model: PreTrainedModel) -> str:
    """SHA-256 over every parameter and buffer, so evidence can be tied to exact weights."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items(), key=lambda item: item[0]):
        digest.update(name.encode("utf-8"))
        flat = tensor.detach().cpu().contiguous().reshape(-1)
        digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_binary_label_contract(model: PreTrainedModel) -> None:
    if model.config.id2label != dict(ID_TO_LABEL) or model.config.label2id != dict(LABEL_TO_ID):
        raise ValueError("The model's label mapping does not match the benign/malicious contract.")


def _refuse_constant_predictions(metrics: Mapping[str, float]) -> None:
    flagged = metrics["true_positives"] + metrics["false_positives"]
    rows = flagged + metrics["false_negatives"] + metrics["true_negatives"]
    if flagged == 0 or flagged == rows:
        verdict = "benign" if flagged == 0 else "malicious"
        raise DegenerateModelError(
            f"The model predicts '{verdict}' for every validation row, so it cannot "
            "discriminate; refusing to export it.",
        )


def _predicted_labels(logits: Logits) -> Labels:
    if logits.ndim != 2 or logits.shape[1] != len(ID_TO_LABEL):
        raise ValueError(
            f"Expected logits of shape (rows, {len(ID_TO_LABEL)}), got {logits.shape}.",
        )
    # argmax over NaN returns class 0 (benign), which would silently fail open.
    if not np.isfinite(logits).all():
        raise ValueError("Model produced non-finite logits; refusing to derive predictions.")
    # Annotated because numpy < 2.5 (resolved on Python 3.11) types this argmax as Any.
    predictions: Labels = np.argmax(logits, axis=1).astype(np.int64)
    return predictions


def _trainer_metrics(prediction: EvalPrediction) -> dict[str, float]:
    raw_logits = prediction.predictions
    logits = np.asarray(
        raw_logits[0] if isinstance(raw_logits, tuple) else raw_logits,
        dtype=np.float32,
    )
    labels = np.asarray(prediction.label_ids, dtype=np.int64)
    return binary_metrics(labels, _predicted_labels(logits))


def _predict_logits(
    model: PreTrainedModel,
    split: datasets.Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
) -> Logits:
    """Return the logits for every row of ``split`` in order, padding each batch dynamically."""
    loader = torch.utils.data.DataLoader(
        split.select_columns(["input_ids", "attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
    )
    device = next(model.parameters()).device
    model.eval()  # type: ignore[no-untyped-call]  # transformers leaves PreTrainedModel.eval unannotated
    batches: list[Logits] = []
    with torch.inference_mode():
        for batch in loader:
            output = model(**{name: tensor.to(device) for name, tensor in batch.items()})
            batches.append(output.logits.float().cpu().numpy())
    return np.concatenate(batches, axis=0)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _file_checksums(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _library_versions() -> dict[str, str]:
    packages = ("datasets", "onnx", "onnxruntime", "torch", "transformers")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


class DistilBertPipeline(
    TrainingPipeline[datasets.DatasetDict, datasets.DatasetDict, PreTrainedModel],
):
    """Fine-tune DistilBERT as a binary prompt-injection classifier."""

    def __init__(
        self,
        config: TrainingPipelineConfig,
        model_config: DistilBertTrainingConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> None:
        super().__init__(config)
        self.model_config = model_config
        self._tokenizer = tokenizer
        self._dataset_digests: dict[str, str] | None = None
        # (weights fingerprint, metrics): evidence is tied to exact weights, not object identity.
        self._evaluated: tuple[str, dict[str, float]] | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The tokenizer pinned by ``model_config``, loaded on first use unless injected."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.base_model_name,
                revision=self.model_config.base_model_revision,
            )
        return self._tokenizer

    def load_dataset(self) -> datasets.DatasetDict:
        """Load the train and validation splits, never the golden evaluation set.

        The files are hashed before and after parsing and a change in between raises. The
        digests are kept so ``export`` can show that its manifest describes the data the
        model was trained on.
        """
        files = self._dataset_files()
        before = self._dataset_digests_now()
        dataset = datasets.load_dataset(
            "json",
            data_files={split: str(path) for split, path in files.items()},
        )
        if self._dataset_digests_now() != before:
            raise RuntimeError("A dataset file changed while it was being loaded.")
        self._dataset_digests = before
        return dataset

    def _dataset_files(self) -> dict[str, Path]:
        return {
            TRAIN_SPLIT: self.model_config.train_file,
            VALIDATION_SPLIT: self.model_config.validation_file,
        }

    def _dataset_digests_now(self) -> dict[str, str]:
        return {split: _sha256_file(path) for split, path in self._dataset_files().items()}

    def prepare_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        """Validate the train and validation splits, then tokenize them.

        Invalid data raises ``ValueError`` instead of being repaired or dropped, and
        messages never include prompt text. Original columns are kept, since evaluation
        slices by ``attack_type``.

        Leakage is checked on what the model sees, not on raw text: inputs that are equal
        after tokenization (differing only in case, whitespace, or beyond the truncation
        limit) must not appear in both splits, and none may appear in the golden file.

        Tokenization truncates to ``max_sequence_length`` and does not pad; pad per batch
        with ``DataCollatorWithPadding`` in ``train``. Only ``input_ids`` and
        ``attention_mask`` are produced, the inputs DistilBERT and its ONNX export accept.
        Serving must use the same tokenizer revision and truncation length.
        """
        found_splits = set(dataset)
        if found_splits != EXPECTED_SPLITS:
            raise ValueError(
                f"Expected exactly the dataset splits {sorted(EXPECTED_SPLITS)} but found "
                f"{sorted(found_splits)}; held-out evaluation data must not enter training.",
            )
        for name in (TRAIN_SPLIT, VALIDATION_SPLIT):
            _validate_split(name, dataset[name])
        _validate_no_shared_texts(dataset[TRAIN_SPLIT], dataset[VALIDATION_SPLIT])
        golden_texts = _read_golden_texts(self.model_config.golden_file)

        tokenizer = self.tokenizer
        max_length = self.model_config.max_sequence_length
        if not 0 < max_length <= tokenizer.model_max_length:
            raise ValueError(
                f"max_sequence_length must be between 1 and {tokenizer.model_max_length}, "
                f"got {max_length}.",
            )

        def tokenize(batch: dict[str, list[str]]) -> BatchEncoding:
            encoded: BatchEncoding = tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )
            return encoded

        prepared = dataset.map(tokenize, batched=True)
        train_keys = _token_keys(prepared[TRAIN_SPLIT]["input_ids"])
        validation_keys = _token_keys(prepared[VALIDATION_SPLIT]["input_ids"])
        shared_keys = train_keys & validation_keys
        if shared_keys:
            raise ValueError(
                f"{len(shared_keys)} distinct tokenized input(s) appear in both the "
                f"'{TRAIN_SPLIT}' and '{VALIDATION_SPLIT}' splits, which would inflate "
                "validation metrics.",
            )
        golden_keys = _token_keys(tokenize({"text": golden_texts})["input_ids"])
        for name, keys in ((TRAIN_SPLIT, train_keys), (VALIDATION_SPLIT, validation_keys)):
            leaked_keys = keys & golden_keys
            if leaked_keys:
                raise ValueError(
                    f"{len(leaked_keys)} golden evaluation input(s) also appear in the "
                    f"'{name}' split; the golden set must never be used for training or "
                    "model selection.",
                )
        return prepared

    def run(self) -> TrainingPipelineResult:
        """Fail before any work if the export already exists, then run every stage."""
        self._ensure_export_target_is_free()
        return super().run()

    def train(self, dataset: datasets.DatasetDict) -> PreTrainedModel:
        """Fine-tune on the train split and return the checkpoint with the best validation F1.

        Everything random is seeded from ``config.random_seed``, including the new
        classification head. Checkpoints live in a temporary directory that is removed
        afterwards, and no experiment tracker is enabled. Results are reproducible on the
        same device and library versions, not across devices. Raises ``RuntimeError`` if
        the training loss is not finite.
        """
        seed = self.config.random_seed
        set_seed(seed)
        model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.base_model_name,
            revision=self.model_config.base_model_revision,
            num_labels=len(ID_TO_LABEL),
            id2label=dict(ID_TO_LABEL),
            label2id=dict(LABEL_TO_ID),
        )
        with tempfile.TemporaryDirectory(prefix="guardrail-checkpoints-") as checkpoint_directory:
            arguments = TrainingArguments(
                output_dir=checkpoint_directory,
                num_train_epochs=self.model_config.epochs,
                per_device_train_batch_size=self.model_config.batch_size,
                per_device_eval_batch_size=self.model_config.batch_size,
                learning_rate=self.model_config.learning_rate,
                use_cpu=self.model_config.use_cpu,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_only_model=True,
                load_best_model_at_end=True,
                metric_for_best_model=SELECTION_METRIC,
                greater_is_better=True,
                seed=seed,
                data_seed=seed,
                report_to="none",
            )
            trainer = Trainer(
                model=model,
                args=arguments,
                train_dataset=dataset[TRAIN_SPLIT],
                eval_dataset=dataset[VALIDATION_SPLIT],
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                processing_class=self.tokenizer,
                compute_metrics=_trainer_metrics,
            )
            result = trainer.train()
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Training diverged: the training loss is not finite.")
        return model

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: datasets.DatasetDict,
    ) -> Mapping[str, float]:
        """Return malicious-class metrics on the validation split, plus ``attack_type`` slices.

        Predictions are the argmax of the logits, equivalent to a 0.5 malicious-probability
        threshold; no threshold calibration happens here. For an ``attack_type`` slice,
        ``flag_rate`` is the share classified malicious: recall for attack slices and the
        false-positive rate for benign ones. The golden benchmark belongs to
        ``src/evaluation``.

        ``export`` refuses weights that this method has not scored.
        """
        split = dataset[VALIDATION_SPLIT]
        logits = _predict_logits(model, split, self.tokenizer, self.model_config.batch_size)
        predictions = _predicted_labels(logits)
        labels = np.asarray(list(split["label"]), dtype=np.int64)
        metrics = binary_metrics(labels, predictions)
        metrics.update(
            attack_type_metrics([str(value) for value in split["attack_type"]], predictions)
        )
        self._evaluated = (_state_fingerprint(model), dict(metrics))
        return metrics

    def export(self, model: PreTrainedModel) -> ExportedArtifact:
        """Publish the ONNX model, tokenizer, and manifest as one immutable directory.

        Requires evaluation evidence: ``evaluate`` must have scored these exact weights on this
        pipeline, and the model must not have changed since. Otherwise this raises
        ``MissingEvaluationEvidenceError`` before anything is written. It also refuses a model
        whose label mapping is not the benign/malicious contract, whose validation predictions
        are all one class, or whose dataset files changed after loading. Files are written to a
        staging directory, checked against PyTorch with ONNX Runtime, and only then renamed to
        ``<output_directory>/export``. This pipeline never overwrites an existing export
        (concurrent writers to one output directory are not supported) and a failed export
        leaves nothing behind. ``manifest.json`` records the lineage, the validation metrics
        for these exact weights, and a SHA-256 for every other file; the returned checksum is
        the model's. The checksums detect accidental change, not tampering by someone who can
        rewrite the whole directory.
        """
        self._ensure_export_target_is_free()
        _require_binary_label_contract(model)
        validation_metrics = self._validation_metrics_for(model)
        _refuse_constant_predictions(validation_metrics)
        dataset_digests = self._verified_dataset_digests()

        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".export-", dir=self.config.output_directory))
        published = False
        try:
            onnx_path = staging / MODEL_FILE_NAME
            max_length = self.model_config.max_sequence_length
            export_sequence_classifier(
                model, self.tokenizer, onnx_path, max_sequence_length=max_length
            )
            verify_onnx_parity(model, self.tokenizer, onnx_path, max_sequence_length=max_length)
            self.tokenizer.save_pretrained(staging / TOKENIZER_DIRECTORY_NAME)

            checksums = _file_checksums(staging)
            manifest = json.dumps(
                self._build_manifest(checksums, validation_metrics, dataset_digests),
                indent=2,
                sort_keys=True,
            )
            (staging / MANIFEST_FILE_NAME).write_text(manifest + "\n", encoding="utf-8")

            self._ensure_export_target_is_free()
            staging.rename(self._export_directory)
            published = True
        finally:
            if not published:
                shutil.rmtree(staging, ignore_errors=True)
                if staging.exists():
                    logger.warning("Could not remove staging directory %s; delete it.", staging)

        artifact = ExportedArtifact(
            path=self._export_directory / MODEL_FILE_NAME,
            sha256=checksums[MODEL_FILE_NAME],
        )
        logger.info("Exported model to %s (sha256 %s)", artifact.path, artifact.sha256)
        return artifact

    @property
    def _export_directory(self) -> Path:
        return self.config.output_directory / EXPORT_DIRECTORY_NAME

    def _ensure_export_target_is_free(self) -> None:
        target = self._export_directory
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                f"Export directory {target} already exists. Exports are immutable; "
                "choose a new output_directory.",
            )

    def _validation_metrics_for(self, model: PreTrainedModel) -> dict[str, float]:
        if self._evaluated is None or self._evaluated[0] != _state_fingerprint(model):
            raise MissingEvaluationEvidenceError(
                "Export requires validation metrics for these exact weights: call evaluate() "
                "on this model first and do not change it before export().",
            )
        return self._evaluated[1]

    def _verified_dataset_digests(self) -> dict[str, str]:
        current = self._dataset_digests_now()
        if self._dataset_digests is None:
            return current
        changed = sorted(
            split for split in current if current[split] != self._dataset_digests[split]
        )
        if changed:
            raise RuntimeError(
                f"Dataset file(s) for {changed} changed after loading; the export would not "
                "describe the data the model was trained on.",
            )
        return self._dataset_digests

    def _build_manifest(
        self,
        checksums: Mapping[str, str],
        validation_metrics: Mapping[str, float],
        dataset_digests: Mapping[str, str],
    ) -> dict[str, object]:
        model_config = self.model_config
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_name": self.config.experiment_name,
            "lineage": {
                "code_revision": self.config.code_revision,
                "dataset_revision": self.config.dataset_revision,
                "random_seed": self.config.random_seed,
                "dataset_files": {
                    split: {"name": path.name, "sha256": dataset_digests[split]}
                    for split, path in self._dataset_files().items()
                },
                "base_model": {
                    "name": model_config.base_model_name,
                    "revision": model_config.base_model_revision,
                },
            },
            "training": {
                "epochs": model_config.epochs,
                "batch_size": model_config.batch_size,
                "learning_rate": model_config.learning_rate,
                "use_cpu": model_config.use_cpu,
                "selection_metric": SELECTION_METRIC,
            },
            "inference_contract": {
                "max_sequence_length": model_config.max_sequence_length,
                "input_names": list(INPUT_NAMES),
                "output_name": OUTPUT_NAME,
                "labels": {str(label_id): name for label_id, name in sorted(ID_TO_LABEL.items())},
                "decision_rule": DECISION_RULE,
                "tokenizer_directory": TOKENIZER_DIRECTORY_NAME,
            },
            "environment": _library_versions(),
            "validation_metrics": dict(validation_metrics),
            "files": dict(checksums),
        }
