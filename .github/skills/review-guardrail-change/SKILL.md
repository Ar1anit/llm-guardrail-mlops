---
name: review-guardrail-change
description: "Review guardrail classifier changes for correctness, regressions, security decision errors, unsafe fail-open behavior, data leakage, reproducibility, typing, documentation, and test coverage. Use for code review, change review, security-sensitive review, or pre-PR validation."
argument-hint: "Optional paths or change scope to review"
---

# Review Guardrail Change

Review the requested scope, or the current working-tree changes when no scope is supplied.

## Procedure

1. Read the changed files and the directly relevant tests, configuration, and project
   documentation before drawing conclusions.
2. Determine the intended behavior and whether the change crosses data, training, evaluation,
   serving, monitoring, API, or security-policy boundaries.
3. Inspect the change in this priority order:
   - incorrect or incomplete behavior and regressions;
   - security decision errors, unsafe fail-open behavior, input-validation gaps, and
     sensitive-data exposure;
   - broken reproducibility or missing dataset, model, and configuration lineage;
   - type-checking, formatting, and test gaps;
   - architecture, threat-model, governance, README, and pull-request checklist omissions.
4. Run the smallest relevant validation commands when execution is available. Do not claim a
   check passed unless it was run successfully.
5. Report only actionable findings supported by evidence. Distinguish confirmed defects from
   questions and optional suggestions.

## Output

Order findings by severity. For each finding, provide:

- severity and a concise title;
- file and line reference;
- evidence and user or security impact;
- a concrete, minimal fix.

If there are no findings, state that explicitly and list the checks run and any checks that could
not be run.
