"""Fixtures shared by the training unit tests."""

from pathlib import Path

import pytest
import torch
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerBase

from tests.unit.training.support import (
    TINY_MAX_POSITIONS,
    TOKENIZER_LIMIT,
    TOKENIZER_VOCAB,
    build_tokenizer,
)


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return build_tokenizer(TOKENIZER_LIMIT)


@pytest.fixture(scope="session")
def tiny_model_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A randomly initialised base DistilBERT with a local tokenizer, loadable offline.

    It has no classification head, like the Hub checkpoint, so training initialises one.
    """
    directory = tmp_path_factory.mktemp("tiny-distilbert")
    build_tokenizer(TINY_MAX_POSITIONS).save_pretrained(directory)
    torch.manual_seed(0)
    config = DistilBertConfig(
        vocab_size=len(TOKENIZER_VOCAB),
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=TINY_MAX_POSITIONS,
        pad_token_id=0,
    )
    DistilBertModel(config).save_pretrained(directory)
    return directory
