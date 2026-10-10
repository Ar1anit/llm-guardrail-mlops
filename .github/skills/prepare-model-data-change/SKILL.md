---
name: prepare-model-data-change
description: "Implement or prepare training, evaluation, dataset, labeling, threshold, preprocessing, monitoring, DVC, Hydra configuration, or model-artifact changes with reproducibility, lineage, security evaluation, governance, and rollback evidence. Use for any model or data lifecycle change."
argument-hint: "Describe the requested model, data, threshold, or pipeline change"
---

# Prepare Model or Data Change

Implement the requested model or data change without treating this pre-alpha scaffold as a
production-ready security control.

## Procedure

1. Identify the affected pipeline stages, source files, configuration, DVC metadata, tests, and
   governance documentation.
2. Define the intended behavior, acceptance criteria, security impact, and rollback conditions
   before editing. Ask for clarification when thresholds, fail-open behavior, data policy, or API
   contracts are ambiguous.
3. Preserve reproducibility:
   - record dataset, configuration, and code revisions;
   - control and document random seeds;
   - keep preprocessing deterministic and aligned between online and offline paths;
   - keep exported artifacts immutable, checksummed, and traceable.
4. Protect repository and user data. Do not commit dataset contents, weights, checkpoints,
   experiment runs, private prompts, personal data, secrets, or local caches.
5. Implement the smallest complete change within the repository's package boundaries.
6. Add focused deterministic tests, including invalid inputs and security-relevant regressions.
7. Update architecture, threat-model, governance, README, or operational documentation whenever
   their documented behavior changes.
8. Run the smallest relevant checks. For broad changes, run Ruff lint and format checks, mypy, and
   pytest through `uv`.

## Completion Report

Report:

- files changed and the behavior they affect;
- dataset, configuration, code, artifact, and random-seed lineage;
- metrics and security slices evaluated or still required;
- threshold, false-positive, false-negative, operational, and rollback implications;
- validation commands and results;
- remaining blockers and required human approvals.
