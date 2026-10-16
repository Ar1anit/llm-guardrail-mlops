"""Fakes shared by the evaluation unit tests: an ONNX session and a training-style export."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

MODEL_BYTES = b"fake onnx graph"
INFERENCE_SESSION = "src.evaluation.golden_evaluator.ort.InferenceSession"


def model_file(tmp_path: Path, name: str = "guardrail.onnx") -> Path:
    path = tmp_path / name
    path.write_bytes(MODEL_BYTES)
    return path


def fake_session(
    inputs: tuple[str, ...] = ("input_ids", "attention_mask"),
    outputs: tuple[str, ...] = ("logits",),
) -> Mock:
    session = Mock()
    session.get_inputs.return_value = [SimpleNamespace(name=name) for name in inputs]
    session.get_outputs.return_value = [SimpleNamespace(name=name) for name in outputs]
    return session


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contract(**overrides: object) -> dict[str, object]:
    return {
        "max_sequence_length": 16,
        "input_names": ["input_ids", "attention_mask"],
        "output_name": "logits",
        "labels": {"0": "benign", "1": "malicious"},
        "decision_rule": "argmax",
        "tokenizer_directory": "tokenizer",
        **overrides,
    }


def write_export(directory: Path, inference_contract: dict[str, object] | None = None) -> Path:
    """A directory shaped like a training export, with checksums that match its files."""
    (directory / "tokenizer").mkdir(parents=True)
    (directory / "model.onnx").write_bytes(MODEL_BYTES)
    (directory / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    files = {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "lineage": {"code_revision": "abc123", "dataset_revision": "data-v1"},
        "inference_contract": inference_contract or contract(),
        "files": files,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory
