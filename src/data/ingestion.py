"""Data ingestion, schema validation, and dataset preparation pipeline.

Downloads pinned Hugging Face datasets, adds curated hard negatives, validates every row
against ``PromptRecord``, and writes deterministic splits. The golden set takes a fixed quota
of each ``attack_type`` (``GOLDEN_QUOTAS``), so every slice is large enough to measure; train
and validation split the rest in proportion to ``attack_type``. Splits are drawn by
near-duplicate cluster (``src.data.near_duplicates``), so variants of one template never land
in two splits, and ``src.data.validation`` checks them before anything is written:

- data/raw/raw_combined.jsonl and data/raw/ingestion_manifest.json (source lineage)
- data/raw/validation_report.json (the checks of ``src.data.validation``)
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

import numpy as np
from datasets import load_dataset  # type: ignore[import-untyped]  # datasets ships no py.typed marker
from omegaconf import OmegaConf
from pydantic import ValidationError
from transformers import AutoTokenizer

from src.data.near_duplicates import (
    NEAR_DUPLICATE_THRESHOLD,
    SHINGLE_SIZE,
    near_duplicate_clusters,
)
from src.data.schema import ATTACK_TYPE_LABELS, ATTACK_TYPES, AttackType, PromptRecord
from src.data.validation import normalized_text, validate_splits, write_report

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_MODEL_CONFIG = Path("configs/model/classifier.yaml")
# A source that rejects more than this share of its rows has probably changed shape.
MAX_REJECT_FRACTION = 0.05
# Golden rows per attack type. A slice of 200 moves its rate by half a point per error. The
# quotas fix the class mix of the golden set, so its precision is not the precision at the
# traffic's real prevalence.
GOLDEN_QUOTAS: Mapping[str, int] = MappingProxyType(
    {"benign": 300, "hard_negative": 200, "injection": 200, "jailbreak": 200},
)


class IngestionError(ValueError):
    """A source or the combined dataset violates the data contract."""


class LabelConflictError(IngestionError):
    """The same prompt text arrived with both labels."""


@dataclass(frozen=True, slots=True)
class HubSource:
    """A Hugging Face dataset pinned to an immutable commit, so reruns read the same rows."""

    name: str
    revision: str
    license: str
    data_files: str | None = None
    split: str = "train"
    config: str | None = None


DEEPSET_SOURCE = HubSource(
    "deepset/prompt-injections",
    "4f61ecb038e9c3fb77e21034b22511b523772cdd",
    "apache-2.0",
)
DEEPSET_TEST_SOURCE = HubSource(
    "deepset/prompt-injections",
    "4f61ecb038e9c3fb77e21034b22511b523772cdd",
    "apache-2.0",
    split="test",
)
DOLLY_SOURCE = HubSource(
    "databricks/databricks-dolly-15k",
    "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
    "cc-by-sa-3.0",
)
# Apache-2.0. Columns: "prompt", and "type" set to "benign" or "jailbreak". The repository
# has no named configs, so the file is pinned: otherwise the loader picks files by pattern.
# The balanced file holds about as many jailbreaks as benign prompts.
JAILBREAK_SOURCE = HubSource(
    "jackhhao/jailbreak-classification",
    "2f2ceeb39658696fd3f462403562b6eea5306287",
    "apache-2.0",
    data_files="balanced/jailbreak_dataset_train_balanced.csv",
)
# MIT. Column "text": prompts players sent to make the Gandalf game reveal its password,
# selected for similarity to "ignore all previous instructions". Every row is an injection.
GANDALF_SOURCE = HubSource(
    "Lakera/gandalf_ignore_instructions",
    "04737b65e90a6794ec227012e4a255a7def6344b",
    "mit",
    split="train+validation+test",
)
# Apache-2.0. Column "prompt_injections": injections, prompt leaking, and mode switching in
# English, French, German, Spanish, Portuguese, Italian, and Romanian. Every row is an injection.
MULTILINGUAL_INJECTION_SOURCE = HubSource(
    "yanismiraoui/prompt_injections",
    "bd55359f2f332afc35f277ac3dd08f7111b024c9",
    "apache-2.0",
)
# MIT. Jailbreak prompts collected from Discord, Reddit, and prompt sites up to December 2023
# (Shen et al., "Do Anything Now"). Column "jailbreak" is true on every row of this config.
WILD_JAILBREAK_SOURCE = HubSource(
    "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "a10aab8eff1c73165a442d4464dce192bd28b9c5",
    "mit",
    config="jailbreak_2023_12_25",
)
# CC-BY-4.0. XSTest v2 (Röttger et al.): 250 safe prompts that resemble unsafe ones ("How do I
# kill a Python process?") and 200 unsafe "contrast_" prompts, which are excluded. The splits
# differ only in the model completion, so one is read.
XSTEST_SOURCE = HubSource(
    "natolambert/xstest-v2-copy",
    "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5",
    "cc-by-4.0",
    split="gpt4",
)
# CC-BY-4.0. OR-Bench-Hard-1K (Cui et al.): prompts that read as harmful, that most aligned
# models refuse, and that an LLM-judge ensemble rated safe to answer.
OR_BENCH_SOURCE = HubSource(
    "bench-llm/or-bench",
    "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272",
    "cc-by-4.0",
    config="or-bench-hard-1k",
)
JAILBREAK_TYPES: Mapping[str, AttackType] = MappingProxyType(
    {"benign": "benign", "jailbreak": "jailbreak"},
)


@dataclass(frozen=True, slots=True)
class SourceBatch:
    """The valid records from one source and the number of rows rejected on the way."""

    name: str
    revision: str | None
    license: str
    config: str | None
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
        source.config,
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
    to_fields: Callable[[Row, int], dict[str, object] | None],
) -> SourceBatch:
    """Validate rows until ``limit`` records are kept; count rows that fail ``PromptRecord``.

    ``to_fields`` returns ``None`` for a row the source marks as out of scope; it is read but
    neither kept nor rejected. It raises ``IngestionError`` for structural problems such as a
    missing or unknown label. Those are never counted as rejects, because they mean the source
    changed shape.
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
        fields = to_fields(row, index)
        if fields is None:
            continue
        try:
            records.append(PromptRecord.model_validate(fields))
        except ValidationError:
            rejected += 1
    batch = SourceBatch(
        source.name,
        source.revision,
        source.license,
        source.config,
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


def fetch_deepset_injections(
    max_samples: int = 1500,
    source: HubSource = DEEPSET_SOURCE,
) -> SourceBatch:
    """Ingests labelled injection and benign samples from deepset/prompt-injections."""

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


def _single_type(
    source: HubSource,
    limit: int,
    text_key: str,
    attack_type: AttackType,
) -> SourceBatch:
    """Ingest a source whose every row is one attack type, read from the ``text_key`` column."""

    def to_fields(row: Row, index: int) -> dict[str, object]:
        return {
            "text": _text_field(row, text_key, source, index),
            "label": ATTACK_TYPE_LABELS[attack_type],
            "attack_type": attack_type,
            "source": source.name,
        }

    return _ingest(source, limit, to_fields)


def fetch_gandalf_injections(max_samples: int = 1000) -> SourceBatch:
    """Ingests password-extraction injections from Lakera/gandalf_ignore_instructions."""
    return _single_type(GANDALF_SOURCE, max_samples, "text", "injection")


def fetch_multilingual_injections(max_samples: int = 1100) -> SourceBatch:
    """Ingests multilingual injections from yanismiraoui/prompt_injections."""
    return _single_type(
        MULTILINGUAL_INJECTION_SOURCE, max_samples, "prompt_injections", "injection"
    )


def fetch_wild_jailbreaks(max_samples: int = 1500) -> SourceBatch:
    """Ingests jailbreaks from TrustAIRLab/in-the-wild-jailbreak-prompts."""
    source = WILD_JAILBREAK_SOURCE

    def to_fields(row: Row, index: int) -> dict[str, object]:
        flag = _field(row, "jailbreak", source, index)
        if flag is not True:
            raise IngestionError(
                f"{source.name} row {index} has jailbreak {flag!r}; expected True.",
            )
        return {
            "text": _text_field(row, "prompt", source, index),
            "label": ATTACK_TYPE_LABELS["jailbreak"],
            "attack_type": "jailbreak",
            "source": source.name,
        }

    return _ingest(source, max_samples, to_fields)


def fetch_xstest_hard_negatives(max_samples: int = 450) -> SourceBatch:
    """Ingests the safe XSTest prompts as hard negatives; unsafe contrast prompts are skipped."""
    source = XSTEST_SOURCE

    def to_fields(row: Row, index: int) -> dict[str, object] | None:
        if _text_field(row, "type", source, index).startswith("contrast_"):
            return None
        return {
            "text": _text_field(row, "prompt", source, index),
            "label": ATTACK_TYPE_LABELS["hard_negative"],
            "attack_type": "hard_negative",
            "source": source.name,
        }

    return _ingest(source, max_samples, to_fields)


def fetch_or_bench_hard_negatives(max_samples: int = 1500) -> SourceBatch:
    """Ingests OR-Bench-Hard-1K prompts as hard negatives."""
    return _single_type(OR_BENCH_SOURCE, max_samples, "prompt", "hard_negative")


def get_curated_hard_negatives() -> SourceBatch:
    """Returns validated hard negative records; a curated row that fails validation raises."""
    records = [PromptRecord.model_validate(item) for item in HARD_NEGATIVES]
    return SourceBatch(
        "curated_security_education",
        None,
        "apache-2.0",
        None,
        None,
        None,
        records,
        len(records),
        0,
    )


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


Clustered = Sequence[tuple[PromptRecord, int]]


def _take_clusters(
    rows: Clustered,
    targets: Mapping[str, int],
    split_name: str,
) -> tuple[list[tuple[PromptRecord, int]], list[tuple[PromptRecord, int]]]:
    """Move whole near-duplicate clusters out of ``rows`` until each type meets its target.

    Clusters are visited in a seeded random order; one that would overshoot any target is
    skipped. Returns (kept, taken) in the input order.
    """
    members: dict[int, list[int]] = {}
    for index, (_, cluster) in enumerate(rows):
        members.setdefault(cluster, []).append(index)
    clusters = list(members.values())
    remaining = dict(targets)
    taken: set[int] = set()
    for position in np.random.default_rng(RANDOM_SEED).permutation(len(clusters)):
        if not any(remaining.values()):
            break
        indices = clusters[position]
        counts = Counter(rows[index][0].attack_type for index in indices)
        if all(count <= remaining.get(attack_type, 0) for attack_type, count in counts.items()):
            for attack_type, count in counts.items():
                remaining[attack_type] -= count
            taken.update(indices)
    short = {attack_type: count for attack_type, count in remaining.items() if count}
    if short:
        raise IngestionError(
            f"Whole near-duplicate clusters cannot fill the {split_name} split; short by {short}.",
        )
    kept = [row for index, row in enumerate(rows) if index not in taken]
    return kept, [row for index, row in enumerate(rows) if index in taken]


def _check_golden_quotas(records: Sequence[PromptRecord], quotas: Mapping[str, int]) -> None:
    """Each attack type needs a quota of at least 1 and at most half of its rows.

    The upper bound keeps at least as many rows of each type for training as golden measures.
    """
    unknown = sorted(set(quotas) - set(ATTACK_TYPES))
    missing = [attack_type for attack_type in ATTACK_TYPES if attack_type not in quotas]
    if unknown or missing:
        raise IngestionError(
            f"Golden quotas must name each attack type once; unknown {unknown}, missing {missing}.",
        )
    counts = Counter(record.attack_type for record in records)
    for attack_type in ATTACK_TYPES:
        quota = quotas[attack_type]
        if quota < 1 or 2 * quota > counts[attack_type]:
            raise IngestionError(
                f"Golden quota {quota} for '{attack_type}' must be at least 1 and at most half "
                f"of its {counts[attack_type]} rows.",
            )


def split_records(
    records: Sequence[PromptRecord],
    golden_quotas: Mapping[str, int],
    val_ratio: float,
) -> tuple[list[PromptRecord], list[PromptRecord], list[PromptRecord]]:
    """Return (golden, train, val), drawn by near-duplicate cluster.

    Golden takes each type's quota. Validation takes ``val_ratio`` of each type's remaining
    rows, rounded and at least one; train keeps the rest.
    """
    _require_every_attack_type(records)
    _check_golden_quotas(records, golden_quotas)
    clusters = near_duplicate_clusters([record.text for record in records])
    rows = list(zip(records, clusters, strict=True))
    pool, golden = _take_clusters(rows, golden_quotas, "golden")
    pool_counts = Counter(record.attack_type for record, _ in pool)
    val_targets = {t: max(1, round(val_ratio * count)) for t, count in pool_counts.items()}
    train, val = _take_clusters(pool, val_targets, "validation")
    return [r for r, _ in golden], [r for r, _ in train], [r for r, _ in val]


def write_jsonl(records: list[PromptRecord], file_path: Path) -> None:
    """Serializes a list of PromptRecords to JSONL."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    logger.info("Saved %d records to %s", len(records), file_path)


def run_pipeline(
    output_dir: Path,
    golden_quotas: Mapping[str, int] = GOLDEN_QUOTAS,
    val_ratio: float = 0.15,
    *,
    tokenizer: TokenizerSpec,
) -> None:
    """Executes the full ingestion, validation, and splitting routine."""
    batches = [
        fetch_deepset_injections(max_samples=1500),
        fetch_deepset_injections(max_samples=500, source=DEEPSET_TEST_SOURCE),
        fetch_gandalf_injections(),
        fetch_multilingual_injections(),
        fetch_benign_dolly(target_count=1500),
        fetch_jailbreaks(max_samples=2000),
        fetch_wild_jailbreaks(),
        fetch_xstest_hard_negatives(),
        fetch_or_bench_hard_negatives(),
        get_curated_hard_negatives(),
    ]
    all_records = [record for batch in batches for record in batch.records]
    logger.info("Total validated samples collected: %d", len(all_records))

    records = deduplicate(all_records, model_input_key(tokenizer))
    logger.info("Total records after deduplication: %d", len(records))
    golden, train, val = split_records(records, golden_quotas, val_ratio)
    splits = {"golden": golden, "train": train, "val": val}
    report = validate_splits(splits, golden_quotas)
    if not report.ok:
        details = "; ".join(f"{f.check}: {f.message}" for f in report.findings)
        raise IngestionError(f"The splits fail validation, so nothing was written: {details}")

    raw_dir = output_dir / "raw"
    write_jsonl(records, raw_dir / "raw_combined.jsonl")
    write_jsonl(golden, output_dir / "golden_eval.jsonl")
    write_jsonl(train, output_dir / "processed" / "train.jsonl")
    write_jsonl(val, output_dir / "processed" / "val.jsonl")
    write_report(report, raw_dir / "validation_report.json")

    lineage = {
        "random_seed": RANDOM_SEED,
        "golden_quotas": dict(golden_quotas),
        "sources": [
            {
                "name": batch.name,
                "revision": batch.revision,
                "license": batch.license,
                "config": batch.config,
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
        "near_duplicate_split": {
            "shingle_size": SHINGLE_SIZE,
            "threshold": NEAR_DUPLICATE_THRESHOLD,
        },
        "attack_type_counts": report.attack_type_counts,
        "golden_source_counts": dict(sorted(Counter(r.source for r in golden).items())),
    }
    (raw_dir / "ingestion_manifest.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, rows in splits.items():
        logger.info("%s: %d rows | %s", name, len(rows), report.attack_type_counts[name])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest and prepare guardrail training data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for output data artifacts (default: data)",
    )
    parser.add_argument(
        "--golden-per-type",
        type=int,
        default=None,
        help="Golden rows for every attack type, replacing GOLDEN_QUOTAS (for small test runs)",
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
    quotas = (
        GOLDEN_QUOTAS
        if args.golden_per_type is None
        else dict.fromkeys(ATTACK_TYPES, args.golden_per_type)
    )
    run_pipeline(
        output_dir=args.output_dir,
        golden_quotas=quotas,
        val_ratio=args.val_ratio,
        tokenizer=read_tokenizer_spec(args.model_config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
