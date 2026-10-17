"""Fetching a deployment's model from a real local MLflow registry."""

import json
from pathlib import Path

import pytest
from mlflow import MlflowClient
from src.registry.fetch import RELEASE_FILE_NAME, fetch, main
from src.registry.release import ReleaseError, publish

from tests.unit.evaluation.support import sha256
from tests.unit.registry.support import SETTINGS, load_evaluator, make_candidate


def _fetch(
    client: MlflowClient,
    output: Path,
    *,
    release: str | None = None,
) -> dict[str, str]:
    """Fetch by ``release`` when given, otherwise through the ``shadow`` deployment."""
    fetch(
        client,
        SETTINGS,
        output,
        deployment=None if release else "shadow",
        release=release,
        load_evaluator=load_evaluator,
    )
    release = json.loads((output / RELEASE_FILE_NAME).read_text(encoding="utf-8"))
    assert isinstance(release, dict)
    return release


def test_fetch_writes_the_verified_export_and_its_release_record(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    version = publish(registry, SETTINGS, make_candidate(release_files), deployment="shadow")
    output = tmp_path / "models" / "shadow"

    release = _fetch(registry, output)

    assert sha256(output / "export" / "model.onnx") == sha256(release_files[0] / "model.onnx")
    assert release == {
        "model_name": SETTINGS.model_name,
        "selector": "deployment=shadow",
        "registry_version": version,
        "release": "0.1.0",
        "model_sha256": sha256(release_files[0] / "model.onnx"),
        "code_revision": "abc123",
        "dataset_revision": "data-v1",
    }
    assert sorted(path.name for path in output.iterdir()) == ["export", RELEASE_FILE_NAME]


def test_fetch_refuses_an_unset_deployment(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    publish(registry, SETTINGS, make_candidate(release_files))

    with pytest.raises(ReleaseError, match="points to no version"):
        _fetch(registry, tmp_path / "out")


def test_fetch_refuses_an_existing_output_directory(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    publish(registry, SETTINGS, make_candidate(release_files), deployment="shadow")
    (tmp_path / "out").mkdir()

    with pytest.raises(ReleaseError, match="already exists"):
        _fetch(registry, tmp_path / "out")


def test_fetch_refuses_a_model_that_contradicts_its_tag_and_leaves_nothing(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    version = publish(registry, SETTINGS, make_candidate(release_files), deployment="shadow")
    registry.set_model_version_tag(SETTINGS.model_name, version, "model_sha256", "0" * 64)
    parent = tmp_path / "models"

    with pytest.raises(ReleaseError, match="does not match its model_sha256 tag"):
        _fetch(registry, parent / "shadow")

    assert list(parent.iterdir()) == []


def test_cli_refuses_to_run_without_a_tracking_uri(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert main(["--release", "0.1.0", "--output", str(tmp_path / "out")]) == 2


def test_fetch_by_release_number_and_download_its_evidence(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    publish(registry, SETTINGS, make_candidate(release_files))
    evidence = tmp_path / "evidence"

    info = fetch(
        registry,
        SETTINGS,
        tmp_path / "serving",
        release="0.1.0",
        evidence=evidence,
        load_evaluator=load_evaluator,
    )

    assert info.selector == "release=0.1.0"
    assert (evidence / "evaluation" / "report.json").read_bytes() == release_files[1].read_bytes()
    assert (evidence / "docs" / "card.md").read_bytes() == release_files[2].read_bytes()


def test_fetch_refuses_an_unknown_release(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    publish(registry, SETTINGS, make_candidate(release_files))

    with pytest.raises(ReleaseError, match=r"Release 9\.9\.9"):
        _fetch(registry, tmp_path / "out", release="9.9.9")


def test_fetch_requires_exactly_one_selector(registry: MlflowClient, tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="exactly one"):
        fetch(registry, SETTINGS, tmp_path / "out", deployment="shadow", release="0.1.0")
    with pytest.raises(ReleaseError, match="exactly one"):
        fetch(registry, SETTINGS, tmp_path / "out")


def test_failed_fetch_also_removes_the_evidence_directory(
    registry: MlflowClient,
    release_files: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    version = publish(registry, SETTINGS, make_candidate(release_files))
    registry.set_model_version_tag(SETTINGS.model_name, version, "model_sha256", "0" * 64)

    with pytest.raises(ReleaseError, match="model_sha256"):
        fetch(
            registry,
            SETTINGS,
            tmp_path / "out" / "serving",
            release="0.1.0",
            evidence=tmp_path / "out" / "evidence",
            load_evaluator=load_evaluator,
        )

    assert list((tmp_path / "out").iterdir()) == []
