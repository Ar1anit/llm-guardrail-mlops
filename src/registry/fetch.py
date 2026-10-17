"""Download the model version a deployment points to, verify it, and record what it is.

``python -m src.registry.fetch --deployment shadow --output models/shadow`` resolves the
registered-model tag ``deployment.shadow`` to a version, downloads that version's export to
``<output>/export``, verifies it with ``GoldenEvaluator.from_export``, and writes
``<output>/release.json``. The serving image is built from that directory, so one image holds
exactly one verified model version. The output directory must not exist: a fetch never mixes
files from two versions.
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
    load_settings,
    tracking_uri,
)

logger = logging.getLogger(__name__)

RELEASE_FILE_NAME = "release.json"


@dataclass(frozen=True, slots=True)
class ReleaseInfo:
    """What the serving image holds; written to ``release.json`` beside the export."""

    model_name: str
    deployment: str
    registry_version: str
    release: str
    model_sha256: str
    code_revision: str
    dataset_revision: str


def fetch(
    client: MlflowClient,
    settings: RegistrySettings,
    deployment: str,
    output: Path,
    *,
    load_evaluator: Callable[[Path], GoldenEvaluator] = GoldenEvaluator.from_export,
) -> ReleaseInfo:
    """Fetch and verify the deployment's model into ``output``, which must not exist yet."""
    if output.exists():
        raise ReleaseError(f"{output} already exists; fetch into a new directory.")
    name = settings.model_name
    version = client.get_registered_model(name).tags.get(f"{DEPLOYMENT_TAG_PREFIX}{deployment}")
    if version is None:
        raise ReleaseError(f"Deployment {deployment!r} of {name} points to no version.")
    model_version = client.get_model_version(name, version)
    tags = model_version.tags
    required = ("release", "model_sha256", "code_revision", "dataset_revision")
    missing = [key for key in required if not tags.get(key)]
    if missing:
        raise ReleaseError(f"{name} version {version} lacks the release tags {missing}.")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=output.parent))
    try:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                f"models:/{name}/{version}",
                dst_path=str(staging / "download"),
            ),
        )
        export = staging / EXPORT_ARTIFACT_PATH
        downloaded.rename(export)
        evaluator = load_evaluator(export)
        if evaluator.model_sha256 != tags["model_sha256"]:
            raise ReleaseError(
                f"{name} version {version}: model.onnx does not match its model_sha256 tag.",
            )
        shutil.rmtree(staging / "download", ignore_errors=True)
        info = ReleaseInfo(
            model_name=name,
            deployment=deployment,
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
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    logger.info(
        "Fetched %s release %s (version %s, sha256 %s) into %s.",
        name,
        info.release,
        info.registry_version,
        info.model_sha256,
        output,
    )
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the model a deployment points to.")
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--deployment", required=True, help="Deployment name, e.g. shadow")
    parser.add_argument("--output", type=Path, required=True, help="New directory to create")
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
        fetch(MlflowClient(), settings, args.deployment, args.output)
    except ReleaseError as error:
        logger.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
