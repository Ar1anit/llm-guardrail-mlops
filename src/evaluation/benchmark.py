"""Release benchmark: score a verified export on the golden set and enforce release gates.

``python -m src.evaluation.benchmark --export-dir <dir> --report <file>`` loads the model with
``GoldenEvaluator.from_export``, scores the golden file, compares the metrics with the gates in
``configs/evaluation/default.yaml``, and writes a lineage report. It exits 0 when every gate
passes, 3 when a gate fails (the report is still written), and 2 when the export, golden file
or configuration cannot be verified (no report is written). An unexpected crash exits 1.
"""

import argparse
import json
import logging
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf

from src.evaluation.golden_evaluator import (
    DEFAULT_BATCH_SIZE,
    ArtifactVerificationError,
    GoldenEvaluator,
)

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path("configs/evaluation/default.yaml")
EXIT_PASSED = 0
EXIT_UNVERIFIED = 2
EXIT_GATE_FAILED = 3

Comparator = Literal[">=", "<="]
_COMPARE: Mapping[Comparator, Callable[[float, float], bool]] = {
    ">=": operator.ge,
    "<=": operator.le,
}
_SECTIONS: Mapping[str, Comparator] = {"minimums": ">=", "maximums": "<="}


class GateConfigError(ValueError):
    """The benchmark configuration is malformed or enforces nothing."""


@dataclass(frozen=True, slots=True)
class GateResult:
    metric: str
    comparator: Comparator
    threshold: float
    actual: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class Gate:
    metric: str
    comparator: Comparator
    threshold: float

    def check(self, metrics: Mapping[str, float]) -> GateResult:
        """A missing or non-finite metric fails, so an absent slice cannot pass."""
        actual = metrics.get(self.metric)
        passed = (
            actual is not None
            and math.isfinite(actual)
            and _COMPARE[self.comparator](actual, self.threshold)
        )
        return GateResult(self.metric, self.comparator, self.threshold, actual, passed)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    passed: bool
    gates: tuple[GateResult, ...]
    document: dict[str, object]


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    golden_file: Path
    batch_size: int
    gates: tuple[Gate, ...]


def load_config(path: Path) -> BenchmarkConfig:
    """Read and validate the benchmark configuration; an empty gate list raises."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict):
        raise GateConfigError(f"{path} must be a mapping.")
    golden_file = raw.get("golden_file")
    batch_size = raw.get("batch_size", DEFAULT_BATCH_SIZE)
    sections = raw.get("gates")
    if not isinstance(golden_file, str) or not golden_file:
        raise GateConfigError(f"{path}: 'golden_file' must be a path.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise GateConfigError(f"{path}: 'batch_size' must be a positive integer.")
    if not isinstance(sections, dict) or set(sections) - set(_SECTIONS):
        raise GateConfigError(f"{path}: 'gates' may only hold {sorted(_SECTIONS)}.")
    gates: list[Gate] = []
    for section, comparator in _SECTIONS.items():
        entries = sections.get(section) or {}
        if not isinstance(entries, dict):
            raise GateConfigError(f"{path}: 'gates.{section}' must map metrics to numbers.")
        for metric, threshold in entries.items():
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise GateConfigError(f"{path}: gate '{metric}' needs a numeric threshold.")
            if not math.isfinite(threshold):
                raise GateConfigError(f"{path}: gate '{metric}' has a non-finite threshold.")
            gates.append(Gate(str(metric), comparator, float(threshold)))
    if not gates:
        raise GateConfigError(f"{path} defines no gates; a benchmark must enforce something.")
    return BenchmarkConfig(Path(golden_file), batch_size, tuple(gates))


def run_benchmark(
    evaluator: GoldenEvaluator,
    config: BenchmarkConfig,
    *,
    config_path: Path,
) -> BenchmarkReport:
    """Score the golden set; ``document`` is the JSON lineage report."""
    if evaluator.provenance is None:
        raise ArtifactVerificationError(
            "The benchmark needs an evaluator built with GoldenEvaluator.from_export.",
        )
    result = evaluator.evaluate_file(config.golden_file, batch_size=config.batch_size)
    gate_results = tuple(gate.check(result.metrics) for gate in config.gates)
    passed = all(gate.passed for gate in gate_results)
    provenance = evaluator.provenance
    document: dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "passed": passed,
        "model": {
            "export_directory": str(provenance.export_directory),
            "model_sha256": result.model_sha256,
            "manifest_sha256": provenance.manifest_sha256,
            "lineage": dict(provenance.lineage),
            "inference_contract": dict(provenance.inference_contract),
        },
        "golden_set": {"path": str(config.golden_file), "sha256": result.golden_sha256},
        "config_path": str(config_path),
        "gates": [asdict(gate) for gate in gate_results],
        "metrics": result.metrics,
    }
    return BenchmarkReport(passed, gate_results, document)


def _write_report(report: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mode "x": release evidence is never overwritten.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark an exported model on the golden set.")
    parser.add_argument("--export-dir", type=Path, required=True, help="Training export directory")
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path for the JSON report; an existing file is never overwritten",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Gate configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--golden-file",
        type=Path,
        help="Override the golden file named in the configuration",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.report.exists():
        logger.error("Report %s already exists; choose a new path.", args.report)
        return EXIT_UNVERIFIED
    try:
        config = load_config(args.config)
        if args.golden_file is not None:
            config = BenchmarkConfig(args.golden_file, config.batch_size, config.gates)
        evaluator = GoldenEvaluator.from_export(args.export_dir)
        report = run_benchmark(evaluator, config, config_path=args.config)
    # ArtifactVerificationError and GateConfigError are ValueErrors.
    except (ValueError, OSError) as error:
        logger.error("Benchmark could not run: %s", error)
        return EXIT_UNVERIFIED

    _write_report(report.document, args.report)
    for gate in report.gates:
        logger.info(
            "%s %s %s %s: actual %s",
            "PASS" if gate.passed else "FAIL",
            gate.metric,
            gate.comparator,
            gate.threshold,
            gate.actual,
        )
    if not report.passed:
        logger.error("Release gates failed; see %s.", args.report)
        return EXIT_GATE_FAILED
    logger.info("All release gates passed; report written to %s.", args.report)
    return EXIT_PASSED


if __name__ == "__main__":
    raise SystemExit(main())
