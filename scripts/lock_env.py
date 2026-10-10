"""Regenerate requirements.lock from the *currently installed* environment.

This pins the core runtime + dev dependency closure (what ``pip install -e
.[dev]`` produces). The optional stacks (ml / deep / agents / db / viz) are
resolved from ``pyproject.toml`` extras and are not pinned here.

Usage::

    python scripts/lock_env.py
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CORE = [
    "numpy",
    "pandas",
    "pyarrow",
    "PyYAML",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "typing-extensions",
    "scipy",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "pandera",
    "typeguard",
    "typing-inspect",
    "mypy-extensions",
    "packaging",
    "python-dateutil",
    "pytz",
    "six",
    "tzdata",
]

DEV = ["pytest", "pytest-cov", "ruff", "ipykernel"]

HEADER = [
    "# Core runtime + dev lock for EnergyMind Phase 1.",
    "# Scope: the dependencies installed by `pip install -e .[dev]` (no ML/deep/agents/db extras).",
    "# Optional stacks are resolved from pyproject.toml extras:",
    "#   ml     = statsmodels, lightgbm, pyod, stumpy, shap, lime",
    "#   deep   = torch        agents = ollama        db = psycopg[binary], pgvector",
    "#   viz    = matplotlib, seaborn",
    "# Regenerate with: python scripts/lock_env.py",
    "",
]


def _pin(pkg: str) -> str:
    try:
        return f"{pkg}=={version(pkg)}"
    except PackageNotFoundError:
        return f"# {pkg}: not installed"


def build_lock() -> str:
    lines = [*HEADER, "# --- core ---"]
    lines += [_pin(p) for p in CORE]
    lines += ["", "# --- dev ---"]
    lines += [_pin(p) for p in DEV]
    return "\n".join(lines) + "\n"


def main() -> int:
    target = Path(__file__).resolve().parents[1] / "requirements.lock"
    target.write_text(build_lock(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
