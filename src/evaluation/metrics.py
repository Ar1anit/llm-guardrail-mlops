"""Label contract and classification metrics shared by training, golden evaluation and serving.

Kept free of torch and transformers so that the ONNX inference path can import it.
"""

from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import numpy.typing as npt

Labels = npt.NDArray[np.int64]

# Mirrors the PromptRecord label contract in src/data/ingestion.py.
ID_TO_LABEL: Mapping[int, str] = MappingProxyType({0: "benign", 1: "malicious"})
LABEL_TO_ID: Mapping[str, int] = MappingProxyType(
    {label: label_id for label_id, label in ID_TO_LABEL.items()},
)
MALICIOUS_ID = LABEL_TO_ID["malicious"]


def binary_metrics(labels: Labels, predictions: Labels) -> dict[str, float]:
    """Malicious-class metrics; raises if a class is absent, since its rates would be vacuous."""
    if labels.shape != predictions.shape:
        raise ValueError(
            f"Got {labels.shape[0]} labels but {predictions.shape[0]} predictions.",
        )
    actual_malicious = labels == MALICIOUS_ID
    predicted_malicious = predictions == MALICIOUS_ID
    true_positives = int(np.sum(actual_malicious & predicted_malicious))
    false_positives = int(np.sum(~actual_malicious & predicted_malicious))
    false_negatives = int(np.sum(actual_malicious & ~predicted_malicious))
    true_negatives = int(np.sum(~actual_malicious & ~predicted_malicious))

    actual_positives = true_positives + false_negatives
    actual_negatives = true_negatives + false_positives
    if actual_positives == 0 or actual_negatives == 0:
        raise ValueError("Metrics are undefined unless both classes are present.")

    predicted_positives = true_positives + false_positives
    # With no positive predictions precision is undefined; 0.0 is the conservative value.
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_positives
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "true_negatives": float(true_negatives),
        "malicious_precision": precision,
        "malicious_recall": recall,
        "malicious_f1": f1,
        "false_positive_rate": false_positives / actual_negatives,
        "false_negative_rate": false_negatives / actual_positives,
    }


def attack_type_metrics(attack_types: Sequence[str], predictions: Labels) -> dict[str, float]:
    """Support and flag rate (the share classified malicious) for each ``attack_type``."""
    flagged = predictions == MALICIOUS_ID
    names = np.asarray(attack_types)
    metrics: dict[str, float] = {}
    for attack_type in sorted(set(attack_types)):
        in_slice = names == attack_type
        metrics[f"attack_type.{attack_type}.support"] = float(in_slice.sum())
        metrics[f"attack_type.{attack_type}.flag_rate"] = float(flagged[in_slice].mean())
    return metrics
