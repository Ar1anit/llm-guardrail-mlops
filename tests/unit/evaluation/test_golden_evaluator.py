from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from src.evaluation.golden_evaluator import GoldenEvaluator


def _tokenizer_output(sequence_length: int) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((1, sequence_length), dtype=np.int64),
        "attention_mask": np.ones((1, sequence_length), dtype=np.int64),
    }


def test_loads_onnx_model_when_evaluator_is_created(tmp_path: Path) -> None:
    model_path = tmp_path / "guardrail.onnx"
    tokenizer = Mock()

    with patch("src.evaluation.golden_evaluator.ort.InferenceSession") as inference_session:
        GoldenEvaluator(model_path=model_path, tokenizer=tokenizer)

    inference_session.assert_called_once()
    assert inference_session.call_args.args[0] == str(model_path)


def test_rejects_model_with_invalid_file_extension(tmp_path: Path) -> None:
    model_path = tmp_path / "guardrail.pt"

    with (
        patch("src.evaluation.golden_evaluator.ort.InferenceSession") as inference_session,
        pytest.raises(ValueError, match=r"\.onnx"),
    ):
        GoldenEvaluator(model_path=model_path, tokenizer=Mock())

    inference_session.assert_not_called()


def test_rejects_empty_string_before_tokenization(tmp_path: Path) -> None:
    tokenizer = Mock()

    with patch("src.evaluation.golden_evaluator.ort.InferenceSession"):
        evaluator = GoldenEvaluator(
            model_path=tmp_path / "guardrail.onnx",
            tokenizer=tokenizer,
        )

    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate("")

    tokenizer.assert_not_called()


def test_truncates_sequences_longer_than_512_tokens(tmp_path: Path) -> None:
    tokenizer = Mock(return_value=_tokenizer_output(sequence_length=512))
    session = Mock()
    session.run.return_value = [np.array([[0.1, 0.9]], dtype=np.float32)]

    with patch(
        "src.evaluation.golden_evaluator.ort.InferenceSession",
        return_value=session,
    ):
        evaluator = GoldenEvaluator(
            model_path=tmp_path / "guardrail.onnx",
            tokenizer=tokenizer,
        )

    evaluator.evaluate("token " * 600)

    tokenizer.assert_called_once()
    assert tokenizer.call_args.kwargs["truncation"] is True
    assert tokenizer.call_args.kwargs["max_length"] == 512
    model_inputs = session.run.call_args.args[1]
    assert all(value.shape[1] <= 512 for value in model_inputs.values())


def test_evaluate_returns_only_label_and_score_keys(tmp_path: Path) -> None:
    tokenizer = Mock(return_value=_tokenizer_output(sequence_length=4))
    session = Mock()
    session.run.return_value = [np.array([[0.8, 0.2]], dtype=np.float32)]

    with patch(
        "src.evaluation.golden_evaluator.ort.InferenceSession",
        return_value=session,
    ):
        evaluator = GoldenEvaluator(
            model_path=tmp_path / "guardrail.onnx",
            tokenizer=tokenizer,
        )

    result = evaluator.evaluate("Explain secure parameterized SQL queries.")

    assert set(result) == {"label", "score"}
