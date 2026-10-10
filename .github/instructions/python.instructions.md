---
name: Python Implementation
description: "Use when editing Python source or tests in this repository. Covers typing, imports, errors, formatting, and test conventions."
applyTo: ["src/**/*.py", "tests/**/*.py"]
---

# Python implementation rules

- Target Python 3.11 syntax and keep compatibility with Python 3.12.
- Use absolute imports from the `src` package and keep imports Ruff-compatible.
- Add precise type annotations to functions, methods, module-level data, and public interfaces.
- Preserve strict mypy compatibility; avoid `Any`, broad casts, and type-ignore comments unless the
  boundary genuinely requires them and the reason is documented.
- Raise explicit, domain-appropriate exceptions for invalid input and unavailable dependencies.
  Do not catch `Exception` broadly or hide errors.
- Keep functions focused and deterministic. Make filesystem, network, model, and time dependencies
  injectable where that improves testability.
- Use the project’s Ruff configuration rather than local formatting exceptions.
- Tests should verify observable behavior, invalid inputs, and security-sensitive boundaries.
  Mock external model, filesystem, and service boundaries instead of requiring live infrastructure.
