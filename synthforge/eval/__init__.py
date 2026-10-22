"""Evaluation package: fidelity, utility, validity, privacy evidence and reporting."""

from synthforge.eval.suite import (
    datasheet_markdown,
    evaluate_dataset,
    model_card_markdown,
    summarize_comparison,
    write_report,
)

__all__ = [
    "evaluate_dataset",
    "summarize_comparison",
    "write_report",
    "model_card_markdown",
    "datasheet_markdown",
]
