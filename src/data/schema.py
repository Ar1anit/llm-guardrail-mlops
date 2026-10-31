"""Data contract for prompt classification records, shared by ingestion and validation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

AttackType = Literal["benign", "hard_negative", "injection", "jailbreak"]
ATTACK_TYPES: tuple[AttackType, ...] = get_args(AttackType)
# Each attack type implies one label: 0 = benign/safe, 1 = malicious/attack.
ATTACK_TYPE_LABELS: Mapping[str, int] = MappingProxyType(
    {"benign": 0, "hard_negative": 0, "injection": 1, "jailbreak": 1},
)


class PromptRecord(BaseModel):
    """Data contract for individual prompt classification records."""

    text: str = Field(..., min_length=3, max_length=10000)
    label: Literal[0, 1]  # 0 = Benign/Safe, 1 = Malicious/Attack
    attack_type: AttackType
    source: str = Field(..., description="Origin dataset or generator")

    @field_validator("text")
    @classmethod
    def clean_whitespace(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Prompt text must not be empty or whitespace only.")
        return cleaned

    @model_validator(mode="after")
    def label_matches_attack_type(self) -> PromptRecord:
        if ATTACK_TYPE_LABELS[self.attack_type] != self.label:
            raise ValueError(f"attack_type '{self.attack_type}' requires a different label.")
        return self
