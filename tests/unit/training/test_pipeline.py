import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from src.training.pipeline import (
    ExportedArtifact,
    TrainingPipeline,
    TrainingPipelineConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPECTED_ARTIFACT = ExportedArtifact(path=Path("model.onnx"), sha256="0" * 64)

IMPORT_PROBE = """
import sys

import src.training.pipeline

loaded = sorted({"datasets", "torch", "transformers"} & set(sys.modules))
if loaded:
    raise SystemExit(f"pipeline.py imported ML dependencies: {loaded}")
"""


class RecordingPipeline(TrainingPipeline[list[str], list[str], str]):
    def __init__(self, config: TrainingPipelineConfig, fail_at: str | None = None) -> None:
        super().__init__(config)
        self.fail_at = fail_at
        self.calls: list[str] = []

    def _enter(self, stage: str) -> None:
        self.calls.append(stage)
        if stage == self.fail_at:
            raise RuntimeError(stage)

    def load_dataset(self) -> list[str]:
        self._enter("load_dataset")
        return ["raw"]

    def prepare_dataset(self, dataset: list[str]) -> list[str]:
        self._enter("prepare_dataset")
        return [*dataset, "prepared"]

    def train(self, dataset: list[str]) -> str:
        self._enter("train")
        return "+".join(dataset)

    def evaluate(self, model: str, dataset: list[str]) -> Mapping[str, float]:
        self._enter("evaluate")
        return {"model_length": float(len(model)), "dataset_size": float(len(dataset))}

    def export(self, model: str) -> ExportedArtifact:
        self._enter("export")
        return EXPECTED_ARTIFACT


@pytest.fixture
def config(tmp_path: Path) -> TrainingPipelineConfig:
    return TrainingPipelineConfig(
        experiment_name="test-experiment",
        output_directory=tmp_path,
        dataset_revision="dataset-revision",
        code_revision="code-revision",
        random_seed=0,
    )


def test_run_executes_each_stage_once_in_order(config: TrainingPipelineConfig) -> None:
    pipeline = RecordingPipeline(config)

    pipeline.run()

    assert pipeline.calls == [
        "load_dataset",
        "prepare_dataset",
        "train",
        "evaluate",
        "export",
    ]


def test_run_feeds_each_stage_the_previous_stage_output(config: TrainingPipelineConfig) -> None:
    result = RecordingPipeline(config).run()

    assert result.artifact == EXPECTED_ARTIFACT
    assert result.metrics == {
        "model_length": float(len("raw+prepared")),
        "dataset_size": 2.0,
    }


@pytest.mark.parametrize("failing_stage", ["load_dataset", "prepare_dataset", "train", "evaluate"])
def test_run_does_not_export_when_an_earlier_stage_fails(
    config: TrainingPipelineConfig,
    failing_stage: str,
) -> None:
    pipeline = RecordingPipeline(config, fail_at=failing_stage)

    with pytest.raises(RuntimeError, match=failing_stage):
        pipeline.run()

    assert "export" not in pipeline.calls


def test_pipeline_module_does_not_import_ml_dependencies() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_PROBE],
        capture_output=True,
        check=False,
        cwd=PROJECT_ROOT,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
