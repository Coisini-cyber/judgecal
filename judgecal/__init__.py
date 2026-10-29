"""judgecal: reliability cards for LLM judges.

    >>> from judgecal import load, analyze, render_markdown
    >>> report = analyze(load("judgments.jsonl"))
    >>> print(report.verdict.grade)

The package answers four questions about a judge, and says "not measured"
when the data cannot answer them:

- agreement with human labels, corrected for chance
- calibration of the judge's confidence scores
- stability across repeated runs
- sensitivity to answer position and output length
"""

from __future__ import annotations

__version__ = "0.1.0"

from .card import render_html, render_json, render_markdown  # noqa: E402
from .metrics import Report, analyze  # noqa: E402
from .records import (  # noqa: E402
    Dataset,
    Record,
    RecordError,
    lint,
    load_records,
    load_records_from_string,
)
from .stats import (  # noqa: E402
    StatsError,
    bootstrap_ci,
    cohens_kappa,
    expected_calibration_error,
    krippendorff_alpha,
    spearman,
    weighted_kappa,
)


def load(path: str) -> Dataset:
    """Load a judgment file into a :class:`Dataset`."""
    return Dataset(load_records(path), source=path)


__all__ = [
    "__version__",
    "Dataset",
    "Record",
    "RecordError",
    "Report",
    "StatsError",
    "analyze",
    "bootstrap_ci",
    "cohens_kappa",
    "expected_calibration_error",
    "krippendorff_alpha",
    "lint",
    "load",
    "load_records",
    "load_records_from_string",
    "render_html",
    "render_json",
    "render_markdown",
    "spearman",
    "weighted_kappa",
]
