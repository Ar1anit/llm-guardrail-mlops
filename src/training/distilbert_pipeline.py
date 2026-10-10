"""DistilBERT sequence-classification implementation of the training pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import datasets  # type: ignore[import-untyped]  # datasets ships no py.typed marker
from transformers import PreTrainedModel, AutoTokenizer

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


class DistilBertPipeline(
    TrainingPipeline[datasets.DatasetDict, datasets.DatasetDict, PreTrainedModel],
):
    """Fine-tune DistilBERT as a binary prompt-injection classifier."""

    def __init__(
        self,
        config: TrainingPipelineConfig,
        model_config: DistilBertTrainingConfig,
    ) -> None:
        super().__init__(config)
        self.model_config = model_config

    def load_dataset(self) -> datasets.DatasetDict:
        """Load the train and validation splits, never the golden evaluation set."""
        return datasets.load_dataset(
            "json",
            data_files={
                "train": str(self.model_config.train_file),
                "validation": str(self.model_config.validation_file),
            },
        )

    def prepare_dataset(self, dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

        return tokenizer()

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
