"""The Hydra entry point: config validation, provenance, and an end-to-end run."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError, ValidationError
from src.training.distilbert_pipeline import COMMIT_SHA
from src.training.provenance import DirtyWorkingTreeError, dvc_md5
from src.training.train import build_pipeline, compose_config, main

from tests.unit.training.support import (
    git,
    init_repository,
    write_dvc_pointer,
    write_golden_file,
    write_training_files,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning",
    "ignore:The feature will be removed:DeprecationWarning",
    "ignore:Exporting aten.{2}index operator:UserWarning",
)

NOW = datetime(2026, 10, 17, 9, 30, tzinfo=UTC)


def _data_repository(root: Path) -> tuple[str, list[str]]:
    """A committed git repo with train/val/golden files under DVC pointers; returns overrides."""
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    train_file, validation_file = write_training_files(processed)
    golden_file = write_golden_file(root / "data" / "golden.jsonl")
    processed_dvc = write_dvc_pointer(
        root / "data" / "processed.dvc",
        "processed",
        dvc_md5(processed),
    )
    golden_dvc = write_dvc_pointer(
        root / "data" / "golden.jsonl.dvc",
        "golden.jsonl",
        dvc_md5(golden_file),
    )
    sha = init_repository(root)
    return sha, [
        f"training.train_file={train_file}",
        f"training.validation_file={validation_file}",
        f"training.golden_file={golden_file}",
        f"training.dvc_files=[{processed_dvc},{golden_dvc}]",
    ]


def test_shipped_config_pins_the_base_model_to_a_commit() -> None:
    config = compose_config([])

    assert config.model.name == "distilbert/distilbert-base-uncased"
    assert COMMIT_SHA.fullmatch(config.model.revision)
    assert config.training.allow_dirty is False


def test_build_pipeline_records_provenance_and_a_fresh_run_directory(tmp_path: Path) -> None:
    sha, overrides = _data_repository(tmp_path / "repo")
    config = compose_config([*overrides, f"training.output_root={tmp_path / 'runs'}"])

    pipeline = build_pipeline(config, repository=tmp_path / "repo", now=NOW)

    assert pipeline.config.code_revision == sha
    assert pipeline.config.output_directory == tmp_path / "runs" / f"20261017T093000Z-{sha[:12]}"
    assert "processed.dvc=" in pipeline.config.dataset_revision
    assert "golden.jsonl.dvc=" in pipeline.config.dataset_revision
    assert pipeline.config.random_seed == 42


def test_build_pipeline_refuses_a_dirty_tree(tmp_path: Path) -> None:
    _, overrides = _data_repository(tmp_path / "repo")
    (tmp_path / "repo" / "README.md").write_text("changed\n", encoding="utf-8")

    with pytest.raises(DirtyWorkingTreeError):
        build_pipeline(compose_config(overrides), repository=tmp_path / "repo", now=NOW)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("model.revision=null", ValidationError),
        ("training.epochs=three", ValidationError),
        ("+training.unknown=1", ConfigKeyError),
        ("model.num_labels=3", ValueError),
    ],
    ids=["null-revision", "wrong-type", "unknown-key", "label-count"],
)
def test_build_pipeline_rejects_an_invalid_config(
    tmp_path: Path,
    override: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_pipeline(compose_config([override]), repository=tmp_path, now=NOW)


def test_main_trains_and_exports_with_recorded_provenance(
    tiny_model_directory: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    sha, overrides = _data_repository(repository)
    monkeypatch.chdir(repository)

    exit_code = main(
        [
            *overrides,
            f"training.output_root={tmp_path / 'runs'}",
            f"model.name={tiny_model_directory}",
            "model.max_sequence_length=16",
            "training.epochs=6",
            "training.batch_size=4",
            "training.learning_rate=0.03",
            "training.use_cpu=true",
            "project.random_seed=0",
        ],
    )

    assert exit_code == 0
    (export,) = (tmp_path / "runs").glob("*/export")
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["lineage"]["code_revision"] == sha
    assert manifest["lineage"]["dataset_revision"].count(".dvc=") == 2
    assert git(repository, "status", "--porcelain") == ""
