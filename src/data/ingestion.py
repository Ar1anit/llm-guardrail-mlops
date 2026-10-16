"""Data ingestion, schema validation, and dataset preparation pipeline.

Downloads pinned Hugging Face datasets, adds curated hard negatives, validates every row
against ``PromptRecord``, and writes deterministic splits stratified on ``attack_type``:

- data/raw/raw_combined.jsonl and data/raw/ingestion_manifest.json (source lineage)
- data/golden_eval.jsonl (held-out benchmark for release gates)
- data/processed/train.jsonl
- data/processed/val.jsonl

Rows that fail the text constraints are rejected and counted, never repaired. A missing or
unknown label, a source that rejects too many rows, a text with two labels, or a missing
attack type raises ``IngestionError``. Duplicates are found on the model input: the text as
the training tokenizer encodes and truncates it (``configs/model/classifier.yaml``), which is
how the training pipeline checks for leakage between splits. Errors and logs carry counts and
row indices, never prompt text.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, get_args

from datasets import load_dataset  # type: ignore[import-untyped]  # datasets ships no py.typed marker
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]  # no stubs
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_MODEL_CONFIG = Path("configs/model/classifier.yaml")
# A source that rejects more than this share of its rows has probably changed shape.
MAX_REJECT_FRACTION = 0.05


class IngestionError(ValueError):
    """A source or the combined dataset violates the data contract."""


class LabelConflictError(IngestionError):
    """The same prompt text arrived with both labels."""


# =====================================================================
# 1. DATA CONTRACT / SCHEMA
# =====================================================================
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


@dataclass(frozen=True, slots=True)
class HubSource:
    """A Hugging Face dataset pinned to an immutable commit, so reruns read the same rows."""

    name: str
    revision: str
    data_files: str | None = None
    split: str = "train"


DEEPSET_SOURCE = HubSource("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd")
DOLLY_SOURCE = HubSource(
    "databricks/databricks-dolly-15k",
    "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
)
# Apache-2.0. Columns: "prompt", and "type" set to "benign" or "jailbreak". The repository
# has no named configs, so the file is pinned: otherwise the loader picks files by pattern.
# The balanced file holds about as many jailbreaks as benign prompts.
JAILBREAK_SOURCE = HubSource(
    "jackhhao/jailbreak-classification",
    "2f2ceeb39658696fd3f462403562b6eea5306287",
    data_files="balanced/jailbreak_dataset_train_balanced.csv",
)
JAILBREAK_TYPES: Mapping[str, AttackType] = MappingProxyType(
    {"benign": "benign", "jailbreak": "jailbreak"},
)


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """The valid records from one source and the number of rows rejected on the way."""

    name: str
    revision: str | None
    data_files: str | None
    split: str | None
    records: list[PromptRecord]
    rows_read: int
    rejected: int


# =====================================================================
# 2. HARD NEGATIVES (Curated Educational & Benign Security Prompts)
# =====================================================================
HARD_NEGATIVES: list[dict[str, str | int]] = [
    {
        "text": (
            "Can you explain how a SQL injection "
            "attack works conceptually for my cyber security exam?"
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": "What is the definition of a prompt injection in the context of LLM applications?",
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": (
            "In Python, how do I safely sanitize user "
            "inputs to prevent XSS attacks in a FastAPI application?"
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": (
            "Explain the difference between direct prompt injection and indirect prompt injection."
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": (
            "Why do developers use delimiters like triple "
            "quotes or markdown tags in system prompts?"
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": "Show me a code snippet that demonstrates using parameterized queries in psycopg2.",
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": "I am reviewing our firewall configuration. What port does SSH typically run on?",
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": (
            "Write a unit test with pytest to verify that an API returns a 401 Unauthorized status."
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": (
            "Explain what a DAN (Do Anything Now) jailbreak "
            "prompt was and why early models were vulnerable."
        ),
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
    {
        "text": "Can you summarize the OWASP Top 10 for Large Language Model Applications?",
        "label": 0,
        "attack_type": "hard_negative",
        "source": "curated_security_education",
    },
]


# =====================================================================
# 3. DATA INGESTION FUNCTIONS
# =====================================================================
Row = Mapping[str, object]


def _load_rows(source: HubSource) -> Iterable[Row]:
    logger.info("Loading '%s' at revision %s...", source.name, source.revision)
    rows: Iterable[Row] = load_dataset(
        source.name,
        data_files=source.data_files,
        split=source.split,
        revision=source.revision,
    )
    return rows


def _field(row: Row, key: str, source: HubSource, index: int) -> object:
    if key not in row:
        raise IngestionError(f"{source.name} row {index} has no '{key}' field.")
    return row[key]


def _text_field(row: Row, key: str, source: HubSource, index: int) -> str:
    value = _field(row, key, source, index)
    if not isinstance(value, str):
        raise IngestionError(
            f"{source.name} row {index} has a {type(value).__name__} '{key}'; expected a string.",
        )
    return value


def _ingest(
    source: HubSource,
    limit: int,
    to_fields: Callable[[Row, int], dict[str, object]],
) -> SourceBatch:
    """Validate rows until ``limit`` records are kept; count rows that fail ``PromptRecord``.

    ``to_fields`` raises ``IngestionError`` for structural problems such as a missing or unknown
    label. Those are never counted as rejects, because they mean the source changed shape.
    """
    if limit < 1:
        raise ValueError(f"The sample limit must be at least 1, got {limit}.")
    records: list[PromptRecord] = []
    rows_read = 0
    rejected = 0
    for index, row in enumerate(_load_rows(source)):
        if len(records) >= limit:
            break
        rows_read += 1
        try:
            records.append(PromptRecord.model_validate(to_fields(row, index)))
        except ValidationError:
            rejected += 1
    batch = SourceBatch(
        source.name,
        source.revision,
        source.data_files,
        source.split,
        records,
        rows_read,
        rejected,
    )
    _check_rejects(batch)
    return batch


def _check_rejects(batch: SourceBatch) -> None:
    if batch.rows_read == 0:
        raise IngestionError(f"{batch.name} returned no rows.")
    if batch.rejected:
        logger.warning(
            "%s: rejected %d of %d rows that failed PromptRecord validation.",
            batch.name,
            batch.rejected,
            batch.rows_read,
        )
    if batch.rejected / batch.rows_read > MAX_REJECT_FRACTION:
        raise IngestionError(
            f"{batch.name} rejected {batch.rejected} of {batch.rows_read} rows, above the "
            f"{MAX_REJECT_FRACTION:.0%} limit; the source may have changed shape.",
        )


def fetch_deepset_injections(max_samples: int = 1500) -> SourceBatch:
    """Ingests labelled injection and benign samples from deepset/prompt-injections."""
    source = DEEPSET_SOURCE

    def to_fields(row: Row, index: int) -> dict[str, object]:
        label = _field(row, "label", source, index)
        if isinstance(label, bool) or label not in (0, 1):
            raise IngestionError(f"{source.name} row {index} has label {label!r}; expected 0 or 1.")
        return {
            "text": _text_field(row, "text", source, index),
            "label": label,
            "attack_type": "injection" if label == 1 else "benign",
            "source": source.name,
        }

    return _ingest(source, max_samples, to_fields)


def fetch_benign_dolly(target_count: int = 1500) -> SourceBatch:
    """Ingests general benign prompts from databricks/databricks-dolly-15k."""
    source = DOLLY_SOURCE

    def to_fields(row: Row, index: int) -> dict[str, object]:
        instruction = _text_field(row, "instruction", source, index).strip()
        context = _text_field(row, "context", source, index).strip()
        return {
            "text": f"{instruction} {context}".strip() if context else instruction,
            "label": 0,
            "attack_type": "benign",
            "source": source.name,
        }

    return _ingest(source, target_count, to_fields)


def fetch_jailbreaks(max_samples: int = 2000) -> SourceBatch:
    """Ingests jailbreak and benign prompts from jackhhao/jailbreak-classification."""
    source = JAILBREAK_SOURCE

    def to_fields(row: Row, index: int) -> dict[str, object]:
        kind = _field(row, "type", source, index)
        if not isinstance(kind, str) or kind not in JAILBREAK_TYPES:
            raise IngestionError(
                f"{source.name} row {index} has type {kind!r}; "
                f"expected one of {sorted(JAILBREAK_TYPES)}.",
            )
        attack_type = JAILBREAK_TYPES[kind]
        return {
            "text": _text_field(row, "prompt", source, index),
            "label": ATTACK_TYPE_LABELS[attack_type],
            "attack_type": attack_type,
            "source": source.name,
        }

    return _ingest(source, max_samples, to_fields)


def get_curated_hard_negatives() -> SourceBatch:
    """Returns validated hard negative records; a curated row that fails validation raises."""
    records = [PromptRecord.model_validate(item) for item in HARD_NEGATIVES]
    return SourceBatch("curated_security_education", None, None, None, records, len(records), 0)


# =====================================================================
# 4. DEDUPLICATE, SPLIT & SAVE
# =====================================================================
@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """The tokenizer and truncation length that training uses, from ``configs/model``."""

    name: str
    revision: str
    max_sequence_length: int


def read_tokenizer_spec(path: Path) -> TokenizerSpec:
    """Read the base model, its pinned revision, and the truncation length from a model config."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise IngestionError(f"{path} must be a mapping.")
    name, revision, length = raw.get("name"), raw.get("revision"), raw.get("max_sequence_length")
    if not isinstance(name, str) or not name:
        raise IngestionError(f"{path}: 'name' must name the base model.")
    if not isinstance(revision, str) or not revision:
        raise IngestionError(f"{path}: 'revision' must pin the base model to a commit.")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise IngestionError(f"{path}: 'max_sequence_length' must be a positive integer.")
    return TokenizerSpec(name, revision, length)


def model_input_key(spec: TokenizerSpec) -> Callable[[str], Hashable]:
    """Map a text to the token ids the model sees, encoded exactly as training encodes it."""
    tokenizer = AutoTokenizer.from_pretrained(spec.name, revision=spec.revision)

    def key(text: str) -> tuple[int, ...]:
        encoded = tokenizer(text, truncation=True, max_length=spec.max_sequence_length)
        return tuple(encoded["input_ids"])

    return key


def normalized_text(text: str) -> str:
    """Case- and whitespace-insensitive text; a tokenizer-free duplicate key."""
    return " ".join(text.casefold().split())


def deduplicate(
    records: Iterable[PromptRecord],
    key: Callable[[str], Hashable] = normalized_text,
) -> list[PromptRecord]:
    """Keep the first record per ``key(text)``; the pipeline passes ``model_input_key``.

    With the model-input key, texts that differ only beyond the truncation limit are one
    example, because the model cannot tell them apart. Raises ``LabelConflictError`` when a
    repeated input carries the other label, because no order of sources makes one of the two
    labels correct.
    """
    kept: dict[Hashable, PromptRecord] = {}
    conflicts = 0
    for record in records:
        first = kept.setdefault(key(record.text), record)
        if first.label != record.label:
            conflicts += 1
    if conflicts:
        raise LabelConflictError(
            f"{conflicts} record(s) repeat a model input with the other label; "
            "resolve the sources before splitting.",
        )
    return list(kept.values())


def _require_every_attack_type(records: Sequence[PromptRecord]) -> None:
    counts = Counter(record.attack_type for record in records)
    missing = [attack_type for attack_type in ATTACK_TYPES if not counts[attack_type]]
    if missing:
        raise IngestionError(
            f"No rows for attack type(s) {missing}; every type must be measurable as a slice.",
        )


def _stratified_split(
    records: Sequence[PromptRecord],
    test_size: float | int,
) -> tuple[list[PromptRecord], list[PromptRecord]]:
    kept, held_out = train_test_split(
        list(records),
        test_size=test_size,
        stratify=[record.attack_type for record in records],
        random_state=RANDOM_SEED,
    )
    return kept, held_out


def split_records(
    records: Sequence[PromptRecord],
    golden_size: int,
    val_ratio: float,
) -> tuple[list[PromptRecord], list[PromptRecord], list[PromptRecord]]:
    """Return (golden, train, val), each stratified on ``attack_type`` so every slice appears."""
    _require_every_attack_type(records)
    pool, golden = _stratified_split(records, golden_size)
    train, val = _stratified_split(pool, val_ratio)
    return golden, train, val


def write_jsonl(records: list[PromptRecord], file_path: Path) -> None:
    """Serializes a list of PromptRecords to JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    logger.info("Saved %d records to %s", len(records), file_path)


def _attack_type_counts(records: Sequence[PromptRecord]) -> dict[str, int]:
    counts = Counter(record.attack_type for record in records)
    return {attack_type: counts[attack_type] for attack_type in ATTACK_TYPES}


def run_pipeline(
    output_dir: Path,
    golden_size: int = 200,
    val_ratio: float = 0.15,
    *,
    tokenizer: TokenizerSpec,
) -> None:
    """Executes the full ingestion, validation, and splitting routine."""
    batches = [
        fetch_deepset_injections(max_samples=1500),
        fetch_benign_dolly(target_count=1500),
        fetch_jailbreaks(max_samples=2000),
        get_curated_hard_negatives(),
    ]
    all_records = [record for batch in batches for record in batch.records]
    logger.info("Total validated samples collected: %d", len(all_records))

    records = deduplicate(all_records, model_input_key(tokenizer))
    logger.info("Total records after deduplication: %d", len(records))
    golden, train, val = split_records(records, golden_size, val_ratio)

    raw_dir = output_dir / "raw"
    write_jsonl(records, raw_dir / "raw_combined.jsonl")
    write_jsonl(golden, output_dir / "golden_eval.jsonl")
    write_jsonl(train, output_dir / "processed" / "train.jsonl")
    write_jsonl(val, output_dir / "processed" / "val.jsonl")

    splits = {"golden": golden, "train": train, "val": val}
    lineage = {
        "random_seed": RANDOM_SEED,
        "sources": [
            {
                "name": batch.name,
                "revision": batch.revision,
                "data_files": batch.data_files,
                "split": batch.split,
                "rows_read": batch.rows_read,
                "rejected": batch.rejected,
                "kept": len(batch.records),
            }
            for batch in batches
        ],
        "deduplicated_rows": len(records),
        "deduplication_tokenizer": {
            "name": tokenizer.name,
            "revision": tokenizer.revision,
            "max_sequence_length": tokenizer.max_sequence_length,
        },
        "attack_type_counts": {name: _attack_type_counts(rows) for name, rows in splits.items()},
    }
    (raw_dir / "ingestion_manifest.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, rows in splits.items():
        logger.info("%s: %d rows | %s", name, len(rows), _attack_type_counts(rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest and prepare guardrail training data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for output data artifacts (default: data)",
    )
    parser.add_argument(
        "--golden-size",
        type=int,
        default=200,
        help="Number of samples to reserve for golden benchmark (default: 200)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Share of the remaining samples used for validation (default: 0.15)",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model config naming the tokenizer used to find duplicates "
        f"(default: {DEFAULT_MODEL_CONFIG})",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_pipeline(
        output_dir=args.output_dir,
        golden_size=args.golden_size,
        val_ratio=args.val_ratio,
        tokenizer=read_tokenizer_spec(args.model_config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
