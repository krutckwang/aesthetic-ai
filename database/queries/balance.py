"""Class balance reporter and inverse-frequency weight calculator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sqlalchemy.orm import Session

from database.queries.training_set import query_training_set


@dataclass
class BalanceReport:
    counts: dict[str, int]    # treatment_category → pair count
    weights: dict[str, float] # treatment_category → inverse-frequency weight
    total: int                # total pairs (labelled + unlabelled)
    unlabelled: int           # pairs with no treatment_category


def compute_balance(
    session: Session,
    include_low_ordering: bool = False,
) -> BalanceReport:
    """Count pairs per treatment category and compute weighted-sampler weights."""
    records = query_training_set(session, include_low_ordering=include_low_ordering)

    counts: Counter[str] = Counter()
    unlabelled = 0
    for r in records:
        if r.treatment_category:
            counts[r.treatment_category] += 1
        else:
            unlabelled += 1

    labelled_total = sum(counts.values())
    weights: dict[str, float] = {}
    if labelled_total > 0:
        for cat, count in counts.items():
            weights[cat] = round(labelled_total / count, 4)

    return BalanceReport(
        counts=dict(counts),
        weights=weights,
        total=len(records),
        unlabelled=unlabelled,
    )
