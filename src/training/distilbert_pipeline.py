"""DistilBERT sequence-classification implementation of the training pipeline."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import datasets  # type: ignore[import-untyped]  # datasets ships no py.typed marker
from transformers import AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase

from src.training.pipeline import (
    ExportedArtifact,
    TrainingPipeline,
    TrainingPipelineConfig,
)

# Mirrors the PromptRecord label contract in src/data/ingestion.py.
ID_TO_LABEL: Mapping[int, str] = MappingProxyType({0: "benign", 1: "malicious"})
LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: label_id for label_id, label in ID_TO_LABEL.items()},
)

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"
EXPECTED_SPLITS = frozenset({TRAIN_SPLIT, VALIDATION_SPLIT})
REQUIRED_COLUMNS = frozenset({"text", "label"})
MAX_REPORTED_ROWS = 5


@dataclass(frozen=True, slots=True)
class DistilBertTrainingConfig:
    """DistilBERT settings; field names match the keys in ``configs/`` where one exists."""

    base_model_name: str
    base_model_revision: str
    max_sequence_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    train_file: Path
    validation_file: Path


def _format_row_indices(indices: Sequence[int]) -> str:
    shown = ", ".join(str(index) for index in indices[:MAX_REPORTED_ROWS])
    return f"{shown}, ..." if len(indices) > MAX_REPORTED_ROWS else shown


def _is_valid_label(label: object) -> bool:
    # bool is an int subclass; reject it so True/False cannot pass as class IDs.
    return isinstance(label, int) and not isinstance(label, bool) and label in ID_TO_LABEL


def _validate_split(name: str, split: datasets.Dataset) -> None:
    """Raise ``ValueError`` unless ``split`` satisfies the training data contract.

    Messages contain split names, counts, and row indices only, never prompt text.
    """
    missing_columns = sorted(REQUIRED_COLUMNS - set(split.column_names))
    if missing_columns:
        raise ValueError(f"Split '{name}' is missing required column(s): {missing_columns}.")
    if split.num_rows == 0:
        raise ValueError(f"Split '{name}' is empty.")

    texts: list[object] = list(split["text"])
    blank_rows = [
        index for index, text in enumerate(texts) if not isinstance(text, str) or not text.strip()
    ]
    if blank_rows:
        raise ValueError(
            f"Split '{name}' contains {len(blank_rows)} row(s) with a missing or blank 'text' "
            f"(first indices: {_format_row_indices(blank_rows)}).",
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
        """Load the train and validation splits, never the golden evaluation set."""
        return datasets.load_dataset(
            "json",
            data_files={
                TRAIN_SPLIT: str(self.model_config.train_file),
                VALIDATION_SPLIT: str(self.model_config.validation_file),
            },
        )

    def prepare_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        """Validate the train and validation splits, then tokenize them.

        Invalid data raises ``ValueError`` instead of being repaired or dropped, and
        messages never include prompt text. Original columns are kept, since evaluation
        slices by ``attack_type``.

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

        tokenizer = self.tokenizer
        max_length = self.model_config.max_sequence_length
        if not 0 < max_length <= tokenizer.model_max_length:
            raise ValueError(
                f"max_sequence_length must be between 1 and {tokenizer.model_max_length}, "
                f"got {max_length}.",
            )

        def tokenize(batch: dict[str, list[str]]) -> BatchEncoding:
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )

        return dataset.map(tokenize, batched=True)

    def train(self, dataset: datasets.DatasetDict) -> PreTrainedModel:
        """Fine-tune on the train split, seeding with ``config.random_seed``.

        Use ``ID_TO_LABEL`` and ``LABEL_TO_ID`` for the model's label mapping, and select
        checkpoints on the validation split only.
        """
        raise NotImplementedError

    def evaluate(
        self,
        model: PreTrainedModel,
        dataset: datasets.DatasetDict,
    ) -> Mapping[str, float]:
        """Return named metrics for the validation split.

        Include malicious-class recall and false-negative counts. The golden benchmark
        belongs to ``src/evaluation``.
        """
        raise NotImplementedError

    def export(self, model: PreTrainedModel) -> ExportedArtifact:
        """Write the model and tokenizer under ``config.output_directory`` and checksum them.

        Delegate ONNX conversion to ``src/training/export_onnx.py``.
        """
        raise NotImplementedError
