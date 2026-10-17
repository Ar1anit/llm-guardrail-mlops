"""Download one released model version, verify it, and record what it is.

``python -m src.registry.fetch (--release 0.1.0 | --deployment shadow) --output models/serving``
selects a model version, either by its release number or by the registered-model tag
``deployment.<name>``. It downloads that version's export to ``<output>/export``, verifies it with
``GoldenEvaluator.from_export`` and against the version's ``model_sha256`` tag, and writes
``<output>/release.json``. The serving image is built from that directory, so one image holds
exactly one verified model version. ``--evidence <dir>`` also downloads the release run's
benchmark report and model card, for publishing next to the release.

Output directories must not exist: a fetch never mixes files from two versions, and a failed
fetch leaves nothing behind.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from src.evaluation.golden_evaluator import GoldenEvaluator
from src.registry.release import (
    DEFAULT_CONFIG_PATH,
    DEPLOYMENT_TAG_PREFIX,
    EXPORT_ARTIFACT_PATH,
    RegistrySettings,
    ReleaseError,
    find_version,
    load_settings,
    tracking_uri,
)

logger = logging.getLogger(__name__)

RELEASE_FILE_NAME = "release.json"
EVIDENCE_ARTIFACT_PATHS = ("evaluation", "docs")
REQUIRED_TAGS = ("release", "model_sha256", "code_revision", "dataset_revision")


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """What the serving image holds; written to ``release.json`` beside the export.

    ``selector`` records how the version was chosen: ``release=<number>`` or
    ``deployment=<name>``.
    """

    model_name: str
    selector: str
    registry_version: str
    release: str
    model_sha256: str
    code_revision: str
    dataset_revision: str


def _select_version(
    client: MlflowClient,
    name: str,
    *,
    deployment: str | None,
    release: str | None,
) -> tuple[str, str]:
    """Return (registry version, selector) for exactly one of ``deployment`` or ``release``."""
    if (deployment is None) == (release is None):
        raise ReleaseError("Select the model with exactly one of a deployment or a release.")
    if deployment is not None:
        version = client.get_registered_model(name).tags.get(f"{DEPLOYMENT_TAG_PREFIX}{deployment}")
        if version is None:
            raise ReleaseError(f"Deployment {deployment!r} of {name} points to no version.")
        return version, f"deployment={deployment}"
    assert release is not None
    found = find_version(client, name, "release", release)
    if found is None:
        raise ReleaseError(f"Release {release} of {name} does not exist.")
    return found, f"release={release}"


def _new_staging_directory(target: Path) -> Path:
    if target.exists():
        raise ReleaseError(f"{target} already exists; fetch into a new directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=".fetch-", dir=target.parent))


def fetch(
    client: MlflowClient,
    settings: RegistrySettings,
    output: Path,
    *,
    deployment: str | None = None,
    release: str | None = None,
    evidence: Path | None = None,
    load_evaluator: Callable[[Path], GoldenEvaluator] = GoldenEvaluator.from_export,
) -> ReleaseInfo:
    """Fetch and verify one model version into ``output``, which must not exist yet."""
    name = settings.model_name
    version, selector = _select_version(client, name, deployment=deployment, release=release)
    model_version = client.get_model_version(name, version)
    tags = model_version.tags
    missing = [key for key in REQUIRED_TAGS if not tags.get(key)]
    if missing:
        raise ReleaseError(f"{name} version {version} lacks the release tags {missing}.")
    if release is not None and tags["release"] != release:
        raise ReleaseError(f"{name} version {version} is tagged as release {tags['release']}.")

    staging = _new_staging_directory(output)
    evidence_staging = _new_staging_directory(evidence) if evidence is not None else None
    try:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                f"models:/{name}/{version}",
                dst_path=str(staging / "download"),
            ),
        )
        downloaded.rename(staging / EXPORT_ARTIFACT_PATH)
        shutil.rmtree(staging / "download", ignore_errors=True)
        evaluator = load_evaluator(staging / EXPORT_ARTIFACT_PATH)
        if evaluator.model_sha256 != tags["model_sha256"]:
            raise ReleaseError(
                f"{name} version {version}: model.onnx does not match its model_sha256 tag.",
            )
        info = ReleaseInfo(
            model_name=name,
            selector=selector,
            registry_version=str(version),
            release=tags["release"],
            model_sha256=evaluator.model_sha256,
            code_revision=tags["code_revision"],
            dataset_revision=tags["dataset_revision"],
        )
        (staging / RELEASE_FILE_NAME).write_text(
            json.dumps(asdict(info), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if evidence is not None and evidence_staging is not None:
            if model_version.run_id is None:
                raise ReleaseError(f"{name} version {version} has no run to take evidence from.")
            for artifact_path in EVIDENCE_ARTIFACT_PATHS:
                mlflow.artifacts.download_artifacts(
                    run_id=model_version.run_id,
                    artifact_path=artifact_path,
                    dst_path=str(evidence_staging),
                )
            evidence_staging.rename(evidence)
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        if evidence_staging is not None:
            shutil.rmtree(evidence_staging, ignore_errors=True)
            if evidence is not None and evidence.exists() and not output.exists():
                shutil.rmtree(evidence, ignore_errors=True)
        raise
    logger.info(
        "Fetched %s release %s (version %s, %s, sha256 %s) into %s.",
        name,
        info.release,
        info.registry_version,
        selector,
        info.model_sha256,
        output,
    )
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and verify one released model version.")
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--release", help="Release number, e.g. 0.1.0")
    selector.add_argument("--deployment", help="Deployment name, e.g. shadow")
    parser.add_argument("--output", type=Path, required=True, help="New directory to create")
    parser.add_argument("--evidence", type=Path, help="New directory for report and model card")
    args = parser.parse_args(argv)
    # Third-party libraries log at WARNING: the Azure SDK logs request URLs and headers at INFO.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    for logger_name in ("src", __name__):
        logging.getLogger(logger_name).setLevel(logging.INFO)

    try:
        settings = load_settings(args.config)
        uri = tracking_uri(args.tracking_uri)
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        fetch(
            MlflowClient(),
            settings,
            args.output,
            deployment=args.deployment,
            release=args.release,
            evidence=args.evidence,
        )
    except ReleaseError as error:
        logger.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
