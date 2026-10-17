"""Release verification and the registry flow, against a real local MLflow (SQLite) registry."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import Mock, patch

import mlflow
import pytest
from mlflow import MlflowClient
from src.evaluation.golden_evaluator import GoldenEvaluator
from src.registry.release import (
    DEFAULT_CONFIG_PATH,
    RegistrySettings,
    ReleaseCandidate,
    ReleaseError,
    load_settings,
    main,
    promote,
    publish,
    verify_candidate,
)

from tests.unit.evaluation.support import INFERENCE_SESSION, fake_session, sha256, write_export

SETTINGS = RegistrySettings(model_name="guardrail-test", experiment_name="releases-test")


def _load_evaluator(directory: Path) -> GoldenEvaluator:
    with patch(INFERENCE_SESSION, return_value=fake_session()):
        return GoldenEvaluator.from_export(directory, tokenizer_loader=Mock())


def _write_report(export: Path, path: Path, **overrides: object) -> Path:
    report: dict[str, object] = {
        "schema_version": 1,
        "passed": True,
        "model": {
            "model_sha256": sha256(export / "model.onnx"),
            "manifest_sha256": sha256(export / "manifest.json"),
        },
        "golden_set": {"path": "data/golden_eval.jsonl", "sha256": "f" * 64},
        "metrics": {"malicious_recall": 0.9, "false_positive_rate": 0.0},
        **overrides,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _set_lineage(export: Path, **lineage: str) -> None:
    manifest_path = export / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lineage"].update(lineage)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def release_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    export = write_export(tmp_path / "export")
    card = tmp_path / "card.md"
    card.write_text("# Model card\n", encoding="utf-8")
    return export, _write_report(export, tmp_path / "report.json"), card


def _candidate(files: tuple[Path, Path, Path], version: str = "0.1.0") -> ReleaseCandidate:
    export, report, card = files
    return verify_candidate(
        version=version,
        export_directory=export,
        report_path=report,
        model_card_path=card,
        load_evaluator=_load_evaluator,
    )


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MlflowClient]:
    """A local SQLite MLflow server; artifacts land under ``tmp_path/mlruns``."""
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield MlflowClient()
    mlflow.set_tracking_uri("")


def test_verify_candidate_collects_the_release_facts(
    release_files: tuple[Path, Path, Path],
) -> None:
    candidate = _candidate(release_files)

    assert candidate.model_sha256 == sha256(release_files[0] / "model.onnx")
    assert candidate.code_revision == "abc123"
    assert candidate.metrics["malicious_recall"] == 0.9


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"passed": False}, "failed release gates"),
        ({"schema_version": 2}, "schema_version 1"),
        ({"model": {"model_sha256": "0" * 64, "manifest_sha256": ""}}, "different model.onnx"),
        ({"golden_set": {}}, "golden-set digest"),
    ],
    ids=["gates-failed", "schema", "other-model", "no-golden-digest"],
)
def test_verify_candidate_rejects_a_report_that_does_not_vouch_for_the_export(
    release_files: tuple[Path, Path, Path],
    change: dict[str, object],
    message: str,
) -> None:
    export, report, _ = release_files
    _write_report(export, report, **change)

    with pytest.raises(ReleaseError, match=message):
        _candidate(release_files)


def test_verify_candidate_rejects_a_report_for_another_manifest(
    release_files: tuple[Path, Path, Path],
) -> None:
    export = release_files[0]
    _set_lineage(export, experiment="changed")  # Changes manifest.json after benchmarking.

    with pytest.raises(ReleaseError, match="different manifest"):
        _candidate(release_files)


@pytest.mark.parametrize("key", ["code_revision", "dataset_revision"])
def test_verify_candidate_refuses_a_dirty_lineage(
    release_files: tuple[Path, Path, Path],
    key: str,
) -> None:
    export, report, _ = release_files
    _set_lineage(export, **{key: "abc123-dirty"})
    _write_report(export, report)

    with pytest.raises(ReleaseError, match="dirty"):
        _candidate(release_files)


@pytest.mark.parametrize("version", ["v0.1.0", "0.1", "01.0.0", "0.1.0-rc1"])
def test_verify_candidate_requires_a_plain_semantic_version(
    release_files: tuple[Path, Path, Path],
    version: str,
) -> None:
    with pytest.raises(ReleaseError, match=r"MAJOR\.MINOR\.PATCH"):
        _candidate(release_files, version)


def test_verify_candidate_requires_a_model_card(release_files: tuple[Path, Path, Path]) -> None:
    release_files[2].unlink()

    with pytest.raises(ReleaseError, match="Model card"):
        _candidate(release_files)


def test_publish_registers_a_tagged_version_with_its_artifacts(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    candidate = _candidate(release_files)

    version = publish(registry, SETTINGS, candidate, deployment="shadow")

    model_version = registry.get_model_version(SETTINGS.model_name, version)
    assert model_version.tags["release"] == "0.1.0"
    assert model_version.tags["model_sha256"] == candidate.model_sha256
    assert model_version.tags["code_revision"] == "abc123"
    tags = registry.get_registered_model(SETTINGS.model_name).tags
    assert tags["deployment.shadow"] == version
    downloaded = Path(
        mlflow.artifacts.download_artifacts(
            f"models:/{SETTINGS.model_name}/{version}",
            dst_path=str(tmp_path / "download"),
        ),
    )
    assert sha256(downloaded / "model.onnx") == candidate.model_sha256
    run_artifacts = {item.path for item in registry.list_artifacts(model_version.run_id or "")}
    assert {"export", "evaluation", "docs"} <= run_artifacts


def test_publish_refuses_a_repeated_release_or_model(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
) -> None:
    publish(registry, SETTINGS, _candidate(release_files))

    with pytest.raises(ReleaseError, match="already exists"):
        publish(registry, SETTINGS, _candidate(release_files))
    with pytest.raises(ReleaseError, match="already registered as version 1"):
        publish(registry, SETTINGS, _candidate(release_files, "0.2.0"))


def test_promote_moves_the_deployment_pointer(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    publish(registry, SETTINGS, _candidate(release_files), deployment="shadow")
    other = write_export(tmp_path / "other")
    (other / "model.onnx").write_bytes(b"another graph")
    _set_lineage(other)  # Keep the manifest; recompute checksums below.
    manifest = json.loads((other / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["model.onnx"] = sha256(other / "model.onnx")
    (other / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    files = (other, _write_report(other, tmp_path / "other.json"), release_files[2])
    second = publish(registry, SETTINGS, _candidate(files, "0.2.0"))

    assert promote(registry, SETTINGS, "0.2.0", "shadow") == second
    tags = registry.get_registered_model(SETTINGS.model_name).tags
    assert tags["deployment.shadow"] == second


@pytest.mark.parametrize(
    ("release", "deployment", "message"),
    [("9.9.9", "shadow", "does not exist"), ("0.1.0", "Prod!", "Deployment name")],
    ids=["unknown-release", "bad-name"],
)
def test_promote_rejects_unknown_releases_and_bad_names(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    release: str,
    deployment: str,
    message: str,
) -> None:
    publish(registry, SETTINGS, _candidate(release_files))

    with pytest.raises(ReleaseError, match=message):
        promote(registry, SETTINGS, release, deployment)


def test_cli_refuses_to_run_without_a_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    release_files: tuple[Path, Path, Path],
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    export, report, card = release_files

    exit_code = main(
        [
            "publish",
            "--export-dir",
            str(export),
            "--report",
            str(report),
            "--model-card",
            str(card),
            "--version",
            "0.1.0",
        ],
    )

    assert exit_code == 2


def test_shipped_registry_settings_name_the_model() -> None:
    settings = load_settings(DEFAULT_CONFIG_PATH)

    assert settings.model_name == "guardrail-classifier"
    assert settings.experiment_name == "guardrail-releases"
