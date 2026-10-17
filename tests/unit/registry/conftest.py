"""Fixtures for the registry tests: release inputs and a real local MLflow registry."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from tests.unit.evaluation.support import write_export
from tests.unit.registry.support import write_report


@pytest.fixture
def release_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A fake export, a passing benchmark report for it, and a model card."""
    export = write_export(tmp_path / "export")
    card = tmp_path / "card.md"
    card.write_text("# Model card\n", encoding="utf-8")
    return export, write_report(export, tmp_path / "report.json"), card


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """A local SQLite MLflow server; artifacts land under ``tmp_path/mlruns``."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield MlflowClient()
    mlflow.set_tracking_uri("")
