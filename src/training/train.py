"""Training entry point: compose the Hydra config, record provenance, and run the pipeline.

``uv run python -m src.training.train [overrides...]``, for example
``training.batch_size=8 training.use_cpu=true``. Each run writes to a new directory,
``<training.output_root>/<UTC time>-<commit>/export``, so exports are never overwritten.
Training refuses a dirty git tree, or data that differs from its DVC pointers, unless
``training.allow_dirty=true``. Run it from the repository root: config paths are relative.
"""

import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import MISSING, DictConfig, OmegaConf

from src.evaluation.metrics import ID_TO_LABEL
from src.training.distilbert_pipeline import DistilBertPipeline, DistilBertTrainingConfig
from src.training.pipeline import TrainingPipelineConfig
from src.training.provenance import code_revision, dataset_revision

logger = logging.getLogger(__name__)

CONFIG_DIRECTORY = Path(__file__).resolve().parents[2] / "configs"
CONFIG_NAME = "config"


@dataclass
class ModelSettings:
    """``configs/model``: the base checkpoint, pinned to a Hub commit, and its input limit."""

    name: str = MISSING
    revision: str = MISSING
    num_labels: int = MISSING
    max_sequence_length: int = MISSING


@dataclass
class TrainingSettings:
    """``configs/training``: hyperparameters, data files, and run policy."""

    experiment_name: str = MISSING
    output_root: Path = MISSING
    epochs: int = MISSING
    batch_size: int = MISSING
    learning_rate: float = MISSING
    use_cpu: bool = MISSING
    allow_dirty: bool = MISSING
    train_file: Path = MISSING
    validation_file: Path = MISSING
    golden_file: Path = MISSING
    dvc_files: list[Path] = field(default_factory=list)


def _typed(schema: type, section: DictConfig) -> object:
    """Validate ``section`` against ``schema``: unknown keys, wrong types and gaps raise."""
    return OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(schema), section))


def compose_config(overrides: Sequence[str]) -> DictConfig:
    with initialize_config_dir(config_dir=str(CONFIG_DIRECTORY), version_base="1.3"):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides))


def build_pipeline(config: DictConfig, *, repository: Path, now: datetime) -> DistilBertPipeline:
    """Validate the config, establish provenance, and return a pipeline for a fresh run dir."""
    model = _typed(ModelSettings, config.model)
    training = _typed(TrainingSettings, config.training)
    assert isinstance(model, ModelSettings) and isinstance(training, TrainingSettings)
    if model.num_labels != len(ID_TO_LABEL):
        raise ValueError(
            f"model.num_labels must be {len(ID_TO_LABEL)} for the benign/malicious contract.",
        )
    seed = config.project.random_seed
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("project.random_seed must be an integer.")

    revision = code_revision(repository, allow_dirty=training.allow_dirty)
    data_revision = dataset_revision(
        training.dvc_files,
        [training.train_file, training.validation_file, training.golden_file],
        allow_dirty=training.allow_dirty,
    )
    run_name = f"{now.astimezone(UTC):%Y%m%dT%H%M%SZ}-{revision[:12]}"
    return DistilBertPipeline(
        TrainingPipelineConfig(
            experiment_name=training.experiment_name,
            output_directory=training.output_root / run_name,
            dataset_revision=data_revision,
            code_revision=revision,
            random_seed=seed,
        ),
        DistilBertTrainingConfig(
            base_model_name=model.name,
            base_model_revision=model.revision,
            max_sequence_length=model.max_sequence_length,
            epochs=training.epochs,
            batch_size=training.batch_size,
            learning_rate=training.learning_rate,
            use_cpu=training.use_cpu,
            train_file=training.train_file,
            validation_file=training.validation_file,
            golden_file=training.golden_file,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    overrides = list(sys.argv[1:] if argv is None else argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = build_pipeline(
        compose_config(overrides),
        repository=Path.cwd(),
        now=datetime.now(UTC),
    )
    logger.info(
        "Run %s: code %s, data %s.",
        pipeline.config.output_directory,
        pipeline.config.code_revision,
        pipeline.config.dataset_revision,
    )
    result = pipeline.run()
    logger.info("Exported %s (sha256 %s).", result.artifact.path, result.artifact.sha256)
    for name, value in sorted(result.metrics.items()):
        logger.info("validation %s = %.4f", name, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
