"""Validation of the prepared datasets: schema, coverage, quotas, lineage, and leakage.

``uv run python -m src.data.validation [--data-dir data] [--report path]`` checks the files
that ingestion writes and exits non-zero on any finding:

- **schema**: every line is JSON that satisfies ``PromptRecord``.
- **coverage**: every split is non-empty and contains every attack type.
- **golden_quota**: the golden set holds exactly the manifest's quota of each attack type.
- **manifest**: the attack-type counts in ``ingestion_manifest.json`` match the files.
- **exact_duplicate**: no normalised text appears twice, within a split or across splits.
- **near_duplicate_leakage**: no pair of texts in different splits reaches the near-duplicate
  threshold (``src.data.near_duplicates``). Ingestion keeps each near-duplicate cluster in one
  split, so a finding here means the split was not produced by ingestion or was edited.

Ingestion runs the same checks on the splits before writing them. Findings and the report
carry split names, line numbers, and counts, never prompt text.
"""

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import ValidationError

from src.data.near_duplicates import NEAR_DUPLICATE_THRESHOLD, SHINGLE_SIZE, near_duplicate_pairs
from src.data.schema import ATTACK_TYPES, PromptRecord

logger = logging.getLogger(__name__)

SPLIT_FILES: Mapping[str, str] = {
    "golden": "golden_eval.jsonl",
    "train": "processed/train.jsonl",
    "val": "processed/val.jsonl",
}
MANIFEST_FILE = "raw/ingestion_manifest.json"
# Row references listed per finding; the counts are always complete.
MAX_EXAMPLES = 10


class DataValidationError(ValueError):
    """The prepared datasets violate the data contract."""


@dataclass(frozen=True, slots=True)
class Finding:
    """One failed check, described without prompt text."""

    check: str
    message: str


@dataclass(slots=True)
class ValidationReport:
    """Split sizes, attack-type counts, near-duplicate settings, and every finding."""

    split_rows: dict[str, int]
    attack_type_counts: dict[str, dict[str, int]]
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD
    shingle_size: int = SHINGLE_SIZE
    cross_split_near_duplicates: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ok": self.ok}


def normalized_text(text: str) -> str:
    """Case- and whitespace-insensitive text; a tokenizer-free duplicate key."""
    return " ".join(text.casefold().split())


def _attack_type_counts(records: Sequence[PromptRecord]) -> dict[str, int]:
    counts = Counter(record.attack_type for record in records)
    return {attack_type: counts[attack_type] for attack_type in ATTACK_TYPES}


def _references(rows: Sequence[tuple[str, int]]) -> str:
    shown = ", ".join(f"{split}:{index + 1}" for split, index in rows[:MAX_EXAMPLES])
    return shown + (f" and {len(rows) - MAX_EXAMPLES} more" if len(rows) > MAX_EXAMPLES else "")


def _check_coverage(splits: Mapping[str, Sequence[PromptRecord]]) -> list[Finding]:
    findings = []
    for name, records in splits.items():
        counts = _attack_type_counts(records)
        missing = [attack_type for attack_type, count in counts.items() if not count]
        if not records:
            findings.append(Finding("coverage", f"{name} is empty."))
        elif missing:
            findings.append(Finding("coverage", f"{name} has no rows of {missing}."))
    return findings


def _check_golden_quotas(
    golden: Sequence[PromptRecord],
    quotas: Mapping[str, int],
) -> list[Finding]:
    counts = _attack_type_counts(golden)
    wrong = {
        attack_type: (counts.get(attack_type, 0), quota)
        for attack_type, quota in quotas.items()
        if counts.get(attack_type, 0) != quota
    }
    return [
        Finding("golden_quota", f"golden '{attack_type}' has {actual} rows; quota is {quota}.")
        for attack_type, (actual, quota) in sorted(wrong.items())
    ]


def _check_exact_duplicates(rows: Sequence[tuple[str, int, PromptRecord]]) -> list[Finding]:
    seen: dict[str, tuple[str, int]] = {}
    repeats: list[tuple[str, int]] = []
    for split, index, record in rows:
        first = seen.setdefault(normalized_text(record.text), (split, index))
        if first != (split, index):
            repeats.append((split, index))
    if not repeats:
        return []
    return [
        Finding(
            "exact_duplicate",
            f"{len(repeats)} row(s) repeat an earlier text: {_references(repeats)}.",
        ),
    ]


def _check_near_duplicate_leakage(
    rows: Sequence[tuple[str, int, PromptRecord]],
    threshold: float,
) -> tuple[int, list[Finding]]:
    pairs = [
        pair
        for pair in near_duplicate_pairs([record.text for _, _, record in rows], threshold)
        if rows[pair.first][0] != rows[pair.second][0]
    ]
    if not pairs:
        return 0, []
    by_splits = Counter(
        tuple(sorted((rows[pair.first][0], rows[pair.second][0]))) for pair in pairs
    )
    summary = ", ".join(f"{a}/{b}: {count}" for (a, b), count in sorted(by_splits.items()))
    examples = [
        f"{rows[p.first][0]}:{rows[p.first][1] + 1}~{rows[p.second][0]}:{rows[p.second][1] + 1}"
        f" ({p.similarity:.2f})"
        for p in pairs[:MAX_EXAMPLES]
    ]
    message = (
        f"{len(pairs)} near-duplicate pair(s) span splits at similarity >= {threshold} "
        f"({summary}); for example {', '.join(examples)}."
    )
    return len(pairs), [Finding("near_duplicate_leakage", message)]


def validate_splits(
    splits: Mapping[str, Sequence[PromptRecord]],
    golden_quotas: Mapping[str, int] | None = None,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> ValidationReport:
    """Check coverage, golden quotas, and duplicates within and across ``splits``.

    Row references are 1-based positions within each split, which are line numbers in the
    written files.
    """
    report = ValidationReport(
        split_rows={name: len(records) for name, records in splits.items()},
        attack_type_counts={name: _attack_type_counts(records) for name, records in splits.items()},
        near_duplicate_threshold=threshold,
    )
    report.findings.extend(_check_coverage(splits))
    if golden_quotas is not None:
        report.findings.extend(_check_golden_quotas(splits.get("golden", []), golden_quotas))
    rows = [
        (name, index, record)
        for name, records in splits.items()
        for index, record in enumerate(records)
    ]
    report.findings.extend(_check_exact_duplicates(rows))
    count, findings = _check_near_duplicate_leakage(rows, threshold)
    report.cross_split_near_duplicates = count
    report.findings.extend(findings)
    return report


def read_split(path: Path, name: str) -> tuple[list[PromptRecord], list[Finding]]:
    """Parse a JSONL split; every unreadable or invalid line is a schema finding."""
    records: list[PromptRecord] = []
    invalid: list[tuple[str, int]] = []
    # Split on "\n" only: str.splitlines also splits at U+2028 and similar characters, which
    # JSON encoders leave unescaped inside a text.
    lines = path.read_text(encoding="utf-8").split("\n")
    for index, line in enumerate(lines[:-1] if lines and not lines[-1] else lines):
        try:
            records.append(PromptRecord.model_validate_json(line))
        except ValidationError:
            invalid.append((name, index))
    if not invalid:
        return records, []
    return records, [
        Finding("schema", f"{len(invalid)} line(s) fail PromptRecord: {_references(invalid)}."),
    ]


def _check_manifest(
    manifest: Mapping[str, object],
    splits: Mapping[str, Sequence[PromptRecord]],
) -> list[Finding]:
    recorded = manifest.get("attack_type_counts")
    if not isinstance(recorded, dict):
        return [Finding("manifest", "The manifest has no 'attack_type_counts' mapping.")]
    return [
        Finding(
            "manifest", f"{name} counts {actual} differ from the manifest's {recorded.get(name)}."
        )
        for name, records in splits.items()
        if (actual := _attack_type_counts(records)) != recorded.get(name)
    ]


def _manifest_quotas(manifest: Mapping[str, object]) -> tuple[dict[str, int] | None, list[Finding]]:
    quotas = manifest.get("golden_quotas")
    if isinstance(quotas, dict) and all(isinstance(v, int) for v in quotas.values()):
        return {str(k): v for k, v in quotas.items()}, []
    return None, [Finding("manifest", "The manifest has no 'golden_quotas' mapping of integers.")]


def validate_directory(
    data_dir: Path,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> ValidationReport:
    """Read the splits and manifest under ``data_dir`` and run every check."""
    splits: dict[str, list[PromptRecord]] = {}
    findings: list[Finding] = []
    for name, relative in SPLIT_FILES.items():
        path = data_dir / relative
        if not path.is_file():
            raise DataValidationError(f"{path} does not exist; run ingestion or 'dvc pull'.")
        splits[name], schema_findings = read_split(path, name)
        findings.extend(schema_findings)
    manifest_path = data_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise DataValidationError(f"{manifest_path} does not exist; run ingestion or 'dvc pull'.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise DataValidationError(f"{manifest_path} must hold a JSON object.")
    quotas, quota_findings = _manifest_quotas(manifest)
    report = validate_splits(splits, quotas, threshold)
    report.findings[:0] = [*findings, *quota_findings, *_check_manifest(manifest, splits)]
    return report


def write_report(report: ValidationReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the prepared guardrail datasets.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory that ingestion wrote (default: data)",
    )
    parser.add_argument("--report", type=Path, help="Write the validation report as JSON here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    report = validate_directory(args.data_dir)
    if args.report is not None:
        write_report(report, args.report)
    for finding in report.findings:
        logger.error("%s: %s", finding.check, finding.message)
    logger.info(
        "%s: %s rows, %d cross-split near-duplicate pair(s).",
        "PASS" if report.ok else "FAIL",
        report.split_rows,
        report.cross_split_near_duplicates,
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
