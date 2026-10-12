"""ONNX inference for the exported classifier, and its score on the golden evaluation set.

This is the single inference path: serving reuses ``GoldenEvaluator.evaluate``, so the golden
numbers describe exactly what production runs. It imports neither torch nor transformers.
"""

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, TypedDict

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
DEFAULT_MAX_SEQUENCE_LENGTH = 512
# Flags a prompt when the malicious probability is strictly above the threshold, so 0.5
# reproduces the argmax decision used to select checkpoints during training.
DEFAULT_THRESHOLD = 0.5
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
    records: list[GoldenRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
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


class GoldenEvaluator:
    """Classify prompts with an exported ONNX model and score it on the golden set."""

    def __init__(
        self,
        model_path: Path,
        tokenizer: Tokenizer,
        *,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if model_path.suffix != ".onnx":
            raise ValueError(f"Expected an .onnx model file, got '{model_path.name}'.")
        if max_sequence_length < 1:
            raise ValueError(
                f"max_sequence_length must be at least 1, got {max_sequence_length}.",
            )
        if not (math.isfinite(threshold) and 0.0 < threshold < 1.0):
            raise ValueError(f"threshold must lie strictly between 0 and 1, got {threshold}.")
        self._tokenizer = tokenizer
        self._max_sequence_length = max_sequence_length
        self._threshold = threshold
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

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
    ) -> dict[str, float]:
        """Malicious-class metrics and per-``attack_type`` flag rates on a golden JSONL file."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}.")
        records = read_golden_records(path)
        texts = [record["text"] for record in records]
        scores = np.concatenate(
            [
                self._scores(texts[start : start + batch_size])
                for start in range(0, len(texts), batch_size)
            ],
        )
        labels = np.array([record["label"] for record in records], dtype=np.int64)
        predictions = self._labels(scores)
        return {
            "rows": float(len(records)),
            "threshold": self._threshold,
            **binary_metrics(labels, predictions),
            **attack_type_metrics([record["attack_type"] for record in records], predictions),
        }
