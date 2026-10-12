import numpy as np
import pytest
from src.evaluation.metrics import binary_metrics
from src.training.distilbert_pipeline import (
    SELECTION_METRIC,
    _predicted_labels,
    _trainer_metrics,
)
from transformers import EvalPrediction

LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
PREDICTIONS = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)


def test_selection_metric_is_reported_so_checkpoint_selection_can_find_it() -> None:
    assert SELECTION_METRIC in binary_metrics(LABELS, PREDICTIONS)


def test_predicted_labels_take_the_argmax_of_the_logits() -> None:
    logits = np.array([[2.0, 1.0], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)

    assert _predicted_labels(logits).tolist() == [0, 1, 0]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_predicted_labels_refuse_non_finite_logits(bad_value: float) -> None:
    logits = np.array([[0.0, 1.0], [bad_value, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="non-finite"):
        _predicted_labels(logits)


@pytest.mark.parametrize(
    "logits",
    [np.zeros(4, dtype=np.float32), np.zeros((4, 3), dtype=np.float32)],
    ids=["one-dimensional", "three-classes"],
)
def test_predicted_labels_reject_logits_of_the_wrong_shape(logits: np.ndarray) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError, match="Expected logits of shape"):
        _predicted_labels(logits)


@pytest.mark.parametrize("wrap_in_tuple", [False, True], ids=["array", "tuple"])
def test_trainer_metrics_read_logits_from_an_eval_prediction(wrap_in_tuple: bool) -> None:
    logits = np.array([[0.0, 2.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.int64)
    predictions = (logits,) if wrap_in_tuple else logits

    metrics = _trainer_metrics(EvalPrediction(predictions=predictions, label_ids=labels))

    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_negatives"] == 1
