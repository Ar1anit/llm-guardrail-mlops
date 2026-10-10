from pathlib import Path

import datasets  # type: ignore[import-untyped]  # datasets ships no py.typed marker
import numpy as np
import pytest
import torch
from src.training.distilbert_pipeline import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    SELECTION_METRIC,
    DistilBertPipeline,
    _predict_logits,
)
from transformers import PreTrainedModel, TrainingArguments
from transformers.trainer_utils import TrainOutput

from tests.unit.training.support import build_tiny_pipeline

VALIDATION_ROWS = 12


def _install_fake_trainer(
    monkeypatch: pytest.MonkeyPatch,
    training_loss: float,
) -> dict[str, object]:
    """Replace ``Trainer`` so a test can inspect how ``train`` configures it."""
    captured: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def train(self) -> TrainOutput:
            return TrainOutput(global_step=1, training_loss=training_loss, metrics={})

    monkeypatch.setattr("src.training.distilbert_pipeline.Trainer", FakeTrainer)
    return captured


@pytest.fixture(scope="module")
def tiny_pipeline(
    tiny_model_directory: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> DistilBertPipeline:
    return build_tiny_pipeline(tmp_path_factory.mktemp("training"), tiny_model_directory)


@pytest.fixture(scope="module")
def prepared(tiny_pipeline: DistilBertPipeline) -> datasets.DatasetDict:
    return tiny_pipeline.prepare_dataset(tiny_pipeline.load_dataset())


@pytest.fixture(scope="module")
def trained_model(
    tiny_pipeline: DistilBertPipeline,
    prepared: datasets.DatasetDict,
) -> PreTrainedModel:
    return tiny_pipeline.train(prepared)


def _weights(model: PreTrainedModel) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_train_returns_a_binary_model_carrying_the_label_contract(
    trained_model: PreTrainedModel,
) -> None:
    assert trained_model.config.num_labels == len(ID_TO_LABEL)
    assert trained_model.config.id2label == dict(ID_TO_LABEL)
    assert trained_model.config.label2id == dict(LABEL_TO_ID)


def test_train_is_reproducible_for_a_seed_and_differs_across_seeds(
    tiny_model_directory: Path,
    tmp_path: Path,
    prepared: datasets.DatasetDict,
    trained_model: PreTrainedModel,
) -> None:
    same_seed = build_tiny_pipeline(tmp_path / "same", tiny_model_directory, seed=0).train(prepared)
    other_seed = build_tiny_pipeline(tmp_path / "other", tiny_model_directory, seed=1).train(
        prepared
    )

    reference = _weights(trained_model)
    assert all(torch.equal(reference[name], w) for name, w in _weights(same_seed).items())
    assert not all(torch.equal(reference[name], w) for name, w in _weights(other_seed).items())


def test_train_leaves_no_checkpoints_in_the_output_directory(
    tiny_pipeline: DistilBertPipeline,
    trained_model: PreTrainedModel,
) -> None:
    assert trained_model is not None
    assert not tiny_pipeline.config.output_directory.exists()


def test_train_uses_the_train_split_selects_on_validation_and_disables_tracking(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model_directory: Path,
    tmp_path: Path,
    prepared: datasets.DatasetDict,
) -> None:
    captured = _install_fake_trainer(monkeypatch, training_loss=0.5)

    model = build_tiny_pipeline(tmp_path, tiny_model_directory, seed=7).train(prepared)

    arguments = captured["args"]
    assert isinstance(arguments, TrainingArguments)
    assert (arguments.seed, arguments.data_seed) == (7, 7)
    assert arguments.num_train_epochs == 6
    assert arguments.per_device_train_batch_size == 4
    assert arguments.per_device_eval_batch_size == 4
    assert arguments.learning_rate == 3e-2
    assert arguments.use_cpu is True
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == SELECTION_METRIC
    assert arguments.greater_is_better is True
    assert arguments.report_to == []
    assert captured["train_dataset"] is prepared["train"]
    assert captured["eval_dataset"] is prepared["validation"]
    assert Path(arguments.output_dir) != tmp_path
    assert model.config.id2label == dict(ID_TO_LABEL)


@pytest.mark.parametrize("bad_loss", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_train_refuses_a_model_whose_training_loss_is_not_finite(
    monkeypatch: pytest.MonkeyPatch,
    tiny_model_directory: Path,
    tmp_path: Path,
    prepared: datasets.DatasetDict,
    bad_loss: float,
) -> None:
    _install_fake_trainer(monkeypatch, training_loss=bad_loss)

    with pytest.raises(RuntimeError, match="not finite"):
        build_tiny_pipeline(tmp_path, tiny_model_directory).train(prepared)


def test_evaluate_reports_consistent_metrics_for_the_validation_split_only(
    tiny_pipeline: DistilBertPipeline,
    trained_model: PreTrainedModel,
    prepared: datasets.DatasetDict,
) -> None:
    metrics = tiny_pipeline.evaluate(trained_model, prepared)

    counts = ("true_positives", "false_positives", "false_negatives", "true_negatives")
    assert sum(metrics[name] for name in counts) == VALIDATION_ROWS
    assert metrics["malicious_recall"] + metrics["false_negative_rate"] == pytest.approx(1.0)
    assert sum(v for k, v in metrics.items() if k.endswith(".support")) == VALIDATION_ROWS
    assert {k for k in metrics if k.endswith(".support")} == {
        "attack_type.benign.support",
        "attack_type.hard_negative.support",
        "attack_type.injection.support",
    }
    assert SELECTION_METRIC in metrics


def test_evaluate_computes_exact_metrics_and_slices_from_known_logits(
    monkeypatch: pytest.MonkeyPatch,
    tiny_pipeline: DistilBertPipeline,
    trained_model: PreTrainedModel,
    prepared: datasets.DatasetDict,
) -> None:
    # Validation labels are [0,0,0,1,1,1] twice, with attack types
    # [benign, benign, hard_negative, injection x3] twice. Flag rows 2-4 and 9-11.
    flagged = {2, 3, 4, 9, 10, 11}
    logits = np.array(
        [[0.0, 1.0] if row in flagged else [1.0, 0.0] for row in range(VALIDATION_ROWS)],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        "src.training.distilbert_pipeline._predict_logits", lambda *_a, **_k: logits
    )

    metrics = tiny_pipeline.evaluate(trained_model, prepared)

    assert (metrics["true_positives"], metrics["false_negatives"]) == (5, 1)
    assert (metrics["false_positives"], metrics["true_negatives"]) == (1, 5)
    assert metrics["malicious_precision"] == pytest.approx(5 / 6)
    assert metrics["malicious_recall"] == pytest.approx(5 / 6)
    assert metrics["false_positive_rate"] == pytest.approx(1 / 6)
    assert metrics["false_negative_rate"] == pytest.approx(1 / 6)
    assert metrics["attack_type.benign.support"] == 4
    assert metrics["attack_type.benign.flag_rate"] == 0.0
    assert metrics["attack_type.hard_negative.support"] == 2
    assert metrics["attack_type.hard_negative.flag_rate"] == 0.5
    assert metrics["attack_type.injection.support"] == 6
    assert metrics["attack_type.injection.flag_rate"] == pytest.approx(5 / 6)


def test_evaluate_refuses_non_finite_logits(
    monkeypatch: pytest.MonkeyPatch,
    tiny_pipeline: DistilBertPipeline,
    trained_model: PreTrainedModel,
    prepared: datasets.DatasetDict,
) -> None:
    broken = np.full((VALIDATION_ROWS, 2), np.nan, dtype=np.float32)
    monkeypatch.setattr(
        "src.training.distilbert_pipeline._predict_logits", lambda *_a, **_k: broken
    )

    with pytest.raises(ValueError, match="non-finite"):
        tiny_pipeline.evaluate(trained_model, prepared)


def test_predicted_logits_do_not_depend_on_the_batch_size(
    tiny_pipeline: DistilBertPipeline,
    trained_model: PreTrainedModel,
    prepared: datasets.DatasetDict,
) -> None:
    validation = prepared["validation"]
    tokenizer = tiny_pipeline.tokenizer
    reference = _predict_logits(trained_model, validation, tokenizer, batch_size=VALIDATION_ROWS)

    assert reference.shape == (VALIDATION_ROWS, len(ID_TO_LABEL))
    for batch_size in (1, 5):
        batched = _predict_logits(trained_model, validation, tokenizer, batch_size=batch_size)
        np.testing.assert_allclose(batched, reference, atol=1e-5)
