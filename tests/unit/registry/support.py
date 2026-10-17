"""Helpers shared by the registry tests: a fake verified export and its benchmark report."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.evaluation.golden_evaluator import GoldenEvaluator
from src.registry.release import RegistrySettings, ReleaseCandidate, verify_candidate

from tests.unit.evaluation.support import INFERENCE_SESSION, fake_session, sha256

SETTINGS = RegistrySettings(model_name="guardrail-test", experiment_name="releases-test")


def load_evaluator(directory: Path) -> GoldenEvaluator:
    with patch(INFERENCE_SESSION, return_value=fake_session()):
        return GoldenEvaluator.from_export(directory, tokenizer_loader=Mock())


def write_report(export: Path, path: Path, **overrides: object) -> Path:
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


def set_lineage(export: Path, **lineage: str) -> None:
    manifest_path = export / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["lineage"].update(lineage)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def make_candidate(files: tuple[Path, Path, Path], version: str = "0.1.0") -> ReleaseCandidate:
    export, report, card = files
    return verify_candidate(
        version=version,
        export_directory=export,
        report_path=report,
        model_card_path=card,
        load_evaluator=load_evaluator,
    )
