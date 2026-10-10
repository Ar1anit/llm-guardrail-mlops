"""Interfaces and orchestration skeleton for reproducible model training."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

RawDatasetT = TypeVar("RawDatasetT")
PreparedDatasetT = TypeVar("PreparedDatasetT")
ModelT = TypeVar("ModelT")


@dataclass(frozen=True, slots=True)
class TrainingPipelineConfig:
    """Configuration and lineage required for a reproducible training run."""

    experiment_name: str
    output_directory: Path
    dataset_revision: str
    code_revision: str
    random_seed: int


@dataclass(frozen=True, slots=True)
class ExportedArtifact:
    """Reference to an immutable exported model artifact."""

    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class TrainingPipelineResult:
    """Outputs produced by a completed training pipeline."""

    artifact: ExportedArtifact
    metrics: Mapping[str, float]


class TrainingPipeline(ABC, Generic[RawDatasetT, PreparedDatasetT, ModelT]):
    """Base pipeline whose stage implementations are supplied by a concrete trainer."""

    def __init__(self, config: TrainingPipelineConfig) -> None:
        self.config = config

    def run(self) -> TrainingPipelineResult:
        """Execute each training stage in order."""
        raw_dataset = self.load_dataset()
        prepared_dataset = self.prepare_dataset(raw_dataset)
        model = self.train(prepared_dataset)
        metrics = self.evaluate(model, prepared_dataset)
        artifact = self.export(model)
        return TrainingPipelineResult(artifact=artifact, metrics=metrics)

    @abstractmethod
    def load_dataset(self) -> RawDatasetT:
        """Load a dataset pinned to ``config.dataset_revision``."""
        raise NotImplementedError

    @abstractmethod
    def prepare_dataset(self, dataset: RawDatasetT) -> PreparedDatasetT:
        """Validate, split, and preprocess the source dataset."""
        raise NotImplementedError

    @abstractmethod
    def train(self, dataset: PreparedDatasetT) -> ModelT:
        """Train a model deterministically using ``config.random_seed``."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        model: ModelT,
        dataset: PreparedDatasetT,
    ) -> Mapping[str, float]:
        """Evaluate the trained model and return named metrics."""
        raise NotImplementedError

    @abstractmethod
    def export(self, model: ModelT) -> ExportedArtifact:
        """Export the trained model and calculate its SHA-256 checksum."""
        raise NotImplementedError
