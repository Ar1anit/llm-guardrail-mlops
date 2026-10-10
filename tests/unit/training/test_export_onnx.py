from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]  # onnxruntime ships no py.typed marker
import pytest
import torch
from src.training.export_onnx import (
    INPUT_NAMES,
    OUTPUT_NAME,
    ExportVerificationError,
    export_sequence_classifier,
    verify_onnx_parity,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# The legacy exporter is used on purpose (see export_onnx.py); its notices are expected here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:You are using the legacy TorchScript-based ONNX export:DeprecationWarning",
    "ignore:The feature will be removed:DeprecationWarning",
    "ignore:Exporting aten.{2}index operator:UserWarning",
)

MAX_SEQUENCE_LENGTH = 16


def _classifier(base_directory: Path, seed: int) -> PreTrainedModel:
    torch.manual_seed(seed)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        base_directory,
        num_labels=2,
    )
    return model


@pytest.fixture(scope="module")
def classifier(tiny_model_directory: Path) -> PreTrainedModel:
    return _classifier(tiny_model_directory, seed=0)


@pytest.fixture(scope="module")
def local_tokenizer(tiny_model_directory: Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(tiny_model_directory)


@pytest.fixture(scope="module")
def onnx_path(
    classifier: PreTrainedModel,
    local_tokenizer: PreTrainedTokenizerBase,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_sequence_classifier(
        classifier,
        local_tokenizer,
        path,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    return path


def test_export_writes_a_single_self_contained_file(onnx_path: Path) -> None:
    assert [path.name for path in onnx_path.parent.iterdir()] == ["model.onnx"]
    assert onnx_path.stat().st_size > 0


def test_exported_graph_has_the_serving_interface_and_dynamic_axes(onnx_path: Path) -> None:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    assert tuple(node.name for node in session.get_inputs()) == INPUT_NAMES
    assert [node.name for node in session.get_outputs()] == [OUTPUT_NAME]
    # Shapes that differ from the export example prove batch and sequence are dynamic.
    for batch, sequence in ((3, 5), (1, 9)):
        tokens = np.full((batch, sequence), 7, dtype=np.int64)
        (logits,) = session.run(
            [OUTPUT_NAME],
            {"input_ids": tokens, "attention_mask": np.ones_like(tokens)},
        )
        assert logits.shape == (batch, 2)


def test_parity_check_accepts_the_exported_model(
    classifier: PreTrainedModel,
    local_tokenizer: PreTrainedTokenizerBase,
    onnx_path: Path,
) -> None:
    verify_onnx_parity(
        classifier,
        local_tokenizer,
        onnx_path,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )


def test_parity_check_rejects_a_model_with_different_weights(
    tiny_model_directory: Path,
    local_tokenizer: PreTrainedTokenizerBase,
    onnx_path: Path,
) -> None:
    different_model = _classifier(tiny_model_directory, seed=1)

    with pytest.raises(ExportVerificationError, match="differ"):
        verify_onnx_parity(
            different_model,
            local_tokenizer,
            onnx_path,
            max_sequence_length=MAX_SEQUENCE_LENGTH,
        )


def test_export_leaves_the_callers_model_untouched(
    tiny_model_directory: Path,
    local_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
) -> None:
    model = _classifier(tiny_model_directory, seed=2)
    model.train()
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    export_sequence_classifier(
        model,
        local_tokenizer,
        tmp_path / "model.onnx",
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

    assert model.training is True
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())
