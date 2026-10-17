"""Publish a benchmarked export to the MLflow model registry, and move deployment pointers.

``python -m src.registry.release publish --export-dir <dir> --report <file> --model-card <md>
--version 0.1.0 [--deployment shadow]`` verifies the candidate, logs the export, report and model
card to an MLflow run, and registers that run's ``export`` folder as a new model version tagged
with its release number, checksums, and lineage. ``promote --version 0.1.0 --deployment shadow``
points a deployment at an existing release.

Publishing fails closed with ``ReleaseError`` unless: the export verifies against its manifest
(``GoldenEvaluator.from_export``); the benchmark report passed every gate and names the same
model and manifest SHA-256; neither revision is ``-dirty``; the version is new; and the model
has not been released under another version.

The registry is the Azure ML workspace's MLflow endpoint, which supports neither the MLflow 3
``register_model`` call nor aliases. Versions are therefore created with
``create_model_version``, and a deployment is a registered-model tag
``deployment.<name> = <version>``. The tracking URI must be given explicitly, through
``--tracking-uri`` or ``MLFLOW_TRACKING_URI``, so a missing setting never falls back to a local
``./mlruns`` store.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from omegaconf import OmegaConf

from src.evaluation.golden_evaluator import GoldenEvaluator

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/registry/default.yaml")
DIRTY_SUFFIX = "-dirty"
SEMANTIC_VERSION = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
DEPLOYMENT_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}")
DEPLOYMENT_TAG_PREFIX = "deployment."
EXPORT_ARTIFACT_PATH = "export"


class ReleaseError(ValueError):
    """The candidate or the registry state does not allow this release or promotion."""


@dataclass(frozen=True, slots=True)
class RegistrySettings:
    model_name: str
    experiment_name: str


@dataclass(frozen=True, slots=True)
class ReleaseCandidate:
    """A verified export with the facts its model version is tagged with."""

    version: str
    export_directory: Path
    report_path: Path
    model_card_path: Path
    model_sha256: str
    manifest_sha256: str
    golden_sha256: str
    code_revision: str
    dataset_revision: str
    metrics: Mapping[str, float]


def load_settings(path: Path) -> RegistrySettings:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise ReleaseError(f"{path} must be a mapping.")
    model_name, experiment_name = raw.get("model_name"), raw.get("experiment_name")
    if not isinstance(model_name, str) or not model_name:
        raise ReleaseError(f"{path}: 'model_name' must be a non-empty string.")
    if not isinstance(experiment_name, str) or not experiment_name:
        raise ReleaseError(f"{path}: 'experiment_name' must be a non-empty string.")
    return RegistrySettings(model_name, experiment_name)


def _section(document: Mapping[str, object], key: str, source: str) -> Mapping[str, object]:
    value = document.get(key)
    if not isinstance(value, dict):
        raise ReleaseError(f"{source} has no '{key}' object.")
    return value


def _clean_revision(lineage: Mapping[str, object], key: str) -> str:
    value = lineage.get(key)
    if not isinstance(value, str) or not value:
        raise ReleaseError(f"The manifest lineage has no '{key}'.")
    if DIRTY_SUFFIX in value:
        raise ReleaseError(f"'{key}' is dirty ({value}); a dirty model must not be released.")
    return value


def verify_candidate(
    *,
    version: str,
    export_directory: Path,
    report_path: Path,
    model_card_path: Path,
    load_evaluator: Callable[[Path], GoldenEvaluator] = GoldenEvaluator.from_export,
) -> ReleaseCandidate:
    """Check everything a release asserts, before anything is uploaded."""
    if not SEMANTIC_VERSION.fullmatch(version):
        raise ReleaseError(f"Version {version!r} is not MAJOR.MINOR.PATCH.")
    if not model_card_path.is_file():
        raise ReleaseError(f"Model card {model_card_path} does not exist.")
    evaluator = load_evaluator(export_directory)
    provenance = evaluator.provenance
    if provenance is None:
        raise ReleaseError("The evaluator was not built from an export directory.")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ReleaseError(f"{report_path} is not a schema_version 1 benchmark report.")
    if report.get("passed") is not True:
        raise ReleaseError(f"{report_path} records failed release gates.")
    model = _section(report, "model", str(report_path))
    if model.get("model_sha256") != evaluator.model_sha256:
        raise ReleaseError("The benchmark report describes a different model.onnx.")
    if model.get("manifest_sha256") != provenance.manifest_sha256:
        raise ReleaseError("The benchmark report describes a different manifest.json.")
    golden_sha256 = _section(report, "golden_set", str(report_path)).get("sha256")
    metrics = report.get("metrics")
    if not isinstance(golden_sha256, str) or not isinstance(metrics, dict):
        raise ReleaseError(f"{report_path} lacks the golden-set digest or the metrics.")

    return ReleaseCandidate(
        version=version,
        export_directory=export_directory,
        report_path=report_path,
        model_card_path=model_card_path,
        model_sha256=evaluator.model_sha256,
        manifest_sha256=provenance.manifest_sha256,
        golden_sha256=golden_sha256,
        code_revision=_clean_revision(provenance.lineage, "code_revision"),
        dataset_revision=_clean_revision(provenance.lineage, "dataset_revision"),
        metrics={str(name): float(value) for name, value in metrics.items()},
    )


def _quoted(value: str) -> str:
    if "'" in value:
        raise ReleaseError(f"Value {value!r} cannot be used in a registry search.")
    return f"'{value}'"


def _ensure_registered_model(client: MlflowClient, name: str) -> None:
    try:
        client.get_registered_model(name)
    except MlflowException as error:
        if error.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        client.create_registered_model(
            name,
            description="Binary prompt-injection and jailbreak classifier (ONNX).",
        )


def _find_version(client: MlflowClient, name: str, tag: str, value: str) -> str | None:
    found = client.search_model_versions(
        f"name={_quoted(name)} and tags.{tag}={_quoted(value)}",
    )
    return str(found[0].version) if found else None


def publish(
    client: MlflowClient,
    settings: RegistrySettings,
    candidate: ReleaseCandidate,
    *,
    deployment: str | None = None,
) -> str:
    """Register ``candidate`` as a new model version and return the registry's version number."""
    name = settings.model_name
    if deployment is not None:
        _check_deployment_name(deployment)
    if _find_version(client, name, "release", candidate.version) is not None:
        raise ReleaseError(f"Release {candidate.version} of {name} already exists.")
    existing = _find_version(client, name, "model_sha256", candidate.model_sha256)
    if existing is not None:
        raise ReleaseError(f"This model.onnx is already registered as version {existing}.")
    _ensure_registered_model(client, name)

    experiment = mlflow.set_experiment(settings.experiment_name)
    tags = {
        "release": candidate.version,
        "model_sha256": candidate.model_sha256,
        "manifest_sha256": candidate.manifest_sha256,
        "golden_sha256": candidate.golden_sha256,
        "code_revision": candidate.code_revision,
        "dataset_revision": candidate.dataset_revision,
    }
    with mlflow.start_run(
        experiment_id=experiment.experiment_id, run_name=f"release-{candidate.version}"
    ) as run:
        mlflow.set_tags(tags)
        mlflow.log_metrics(dict(candidate.metrics))
        mlflow.log_artifacts(str(candidate.export_directory), artifact_path=EXPORT_ARTIFACT_PATH)
        mlflow.log_artifact(str(candidate.report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(candidate.model_card_path), artifact_path="docs")
    if run.info.artifact_uri is None:
        raise ReleaseError("The MLflow run has no artifact location.")

    model_version = client.create_model_version(
        name,
        source=f"{run.info.artifact_uri}/{EXPORT_ARTIFACT_PATH}",
        run_id=run.info.run_id,
        tags=tags,
        description=f"Release {candidate.version}; see the run's docs/ for the model card.",
    )
    version = str(model_version.version)
    logger.info("Registered %s version %s as release %s.", name, version, candidate.version)
    if deployment is not None:
        promote(client, settings, candidate.version, deployment)
    return version


def _check_deployment_name(deployment: str) -> None:
    if not DEPLOYMENT_NAME.fullmatch(deployment):
        raise ReleaseError(f"Deployment name {deployment!r} must match {DEPLOYMENT_NAME.pattern}.")


def promote(
    client: MlflowClient,
    settings: RegistrySettings,
    release: str,
    deployment: str,
) -> str:
    """Point ``deployment`` at the model version of ``release`` and return that version."""
    _check_deployment_name(deployment)
    version = _find_version(client, settings.model_name, "release", release)
    if version is None:
        raise ReleaseError(f"Release {release} of {settings.model_name} does not exist.")
    tag = f"{DEPLOYMENT_TAG_PREFIX}{deployment}"
    previous = client.get_registered_model(settings.model_name).tags.get(tag)
    client.set_registered_model_tag(settings.model_name, tag, version)
    logger.info(
        "Deployment %s now points to version %s (release %s; previously %s).",
        deployment,
        version,
        release,
        previous or "unset",
    )
    return version


def _tracking_uri(explicit: str | None) -> str:
    uri = explicit or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise ReleaseError(
            "Set MLFLOW_TRACKING_URI or pass --tracking-uri; refusing to use a local default.",
        )
    return uri


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish and promote model releases.")
    parser.add_argument(
        "--tracking-uri", help="MLflow tracking URI (default: $MLFLOW_TRACKING_URI)"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    publish_parser = commands.add_parser("publish", help="Register a benchmarked export")
    publish_parser.add_argument("--export-dir", type=Path, required=True)
    publish_parser.add_argument("--report", type=Path, required=True)
    publish_parser.add_argument("--model-card", type=Path, required=True)
    publish_parser.add_argument("--version", required=True, help="Release number, e.g. 0.1.0")
    publish_parser.add_argument("--deployment", help="Also point this deployment at the release")
    promote_parser = commands.add_parser("promote", help="Point a deployment at a release")
    promote_parser.add_argument("--version", required=True)
    promote_parser.add_argument("--deployment", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        settings = load_settings(args.config)
        mlflow.set_tracking_uri(_tracking_uri(args.tracking_uri))
        mlflow.set_registry_uri(mlflow.get_tracking_uri())
        client = MlflowClient()
        if args.command == "publish":
            candidate = verify_candidate(
                version=args.version,
                export_directory=args.export_dir,
                report_path=args.report,
                model_card_path=args.model_card,
            )
            publish(client, settings, candidate, deployment=args.deployment)
        else:
            promote(client, settings, args.version, args.deployment)
    except ReleaseError as error:
        logger.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
