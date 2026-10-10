import numpy as np
import pytest
from src.training.distilbert_pipeline import (
    SELECTION_METRIC,
    _attack_type_metrics,
    _binary_metrics,
    _predicted_labels,
    _trainer_metrics,
)
from transformers import EvalPrediction

LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
PREDICTIONS = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)


def test_binary_metrics_report_counts_and_malicious_class_rates() -> None:
    metrics = _binary_metrics(LABELS, PREDICTIONS)

    assert metrics["true_positives"] == 2
    assert metrics["false_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 5
    assert metrics["malicious_precision"] == pytest.approx(2 / 3)
    assert metrics["malicious_recall"] == pytest.approx(1 / 2)
    assert metrics["malicious_f1"] == pytest.approx(4 / 7)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 6)
    assert metrics["false_negative_rate"] == pytest.approx(1 / 2)


def test_binary_metrics_are_zero_rather_than_undefined_when_nothing_is_flagged() -> None:
    metrics = _binary_metrics(
        np.array([1, 0], dtype=np.int64),
        np.array([0, 0], dtype=np.int64),
    )

    assert metrics["malicious_precision"] == 0.0
    assert metrics["malicious_recall"] == 0.0
    assert metrics["malicious_f1"] == 0.0
    assert metrics["false_negative_rate"] == 1.0


@pytest.mark.parametrize("label", [0, 1], ids=["only-benign", "only-malicious"])
def test_binary_metrics_reject_data_missing_a_class(label: int) -> None:
    labels = np.array([label, label], dtype=np.int64)

    with pytest.raises(ValueError, match="both classes"):
        _binary_metrics(labels, labels)


def test_binary_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="labels but"):
        _binary_metrics(LABELS, PREDICTIONS[:-1])


def test_selection_metric_is_reported_so_checkpoint_selection_can_find_it() -> None:
    assert SELECTION_METRIC in _binary_metrics(LABELS, PREDICTIONS)


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


def test_attack_type_metrics_report_support_and_flag_rate_per_slice() -> None:
    attack_types = ["benign", "benign", "hard_negative", "injection", "injection", "injection"]
    predictions = np.array([0, 1, 1, 1, 1, 0], dtype=np.int64)

    assert _attack_type_metrics(attack_types, predictions) == {
        "attack_type.benign.support": 2.0,
        "attack_type.benign.flag_rate": 0.5,
        "attack_type.hard_negative.support": 1.0,
        "attack_type.hard_negative.flag_rate": 1.0,
        "attack_type.injection.support": 3.0,
        "attack_type.injection.flag_rate": pytest.approx(2 / 3),
    }


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
