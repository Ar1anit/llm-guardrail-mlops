import numpy as np
import pytest
from src.evaluation.metrics import attack_type_metrics, binary_metrics

LABELS = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int64)
PREDICTIONS = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.int64)


def test_binary_metrics_report_counts_and_malicious_class_rates() -> None:
    metrics = binary_metrics(LABELS, PREDICTIONS)

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
    metrics = binary_metrics(
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
        binary_metrics(labels, labels)


def test_binary_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="labels but"):
        binary_metrics(LABELS, PREDICTIONS[:-1])


def test_attack_type_metrics_report_support_and_flag_rate_per_slice() -> None:
    attack_types = ["benign", "benign", "hard_negative", "injection", "injection", "injection"]
    predictions = np.array([0, 1, 1, 1, 1, 0], dtype=np.int64)

    assert attack_type_metrics(attack_types, predictions) == {
        "attack_type.benign.support": 2.0,
        "attack_type.benign.flag_rate": 0.5,
        "attack_type.hard_negative.support": 1.0,
        "attack_type.hard_negative.flag_rate": 1.0,
        "attack_type.injection.support": 3.0,
        "attack_type.injection.flag_rate": pytest.approx(2 / 3),
    }
