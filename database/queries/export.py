"""DVC dataset export — writes a deterministic JSON manifest of all training pairs."""

from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy.orm import Session

from database.queries.training_set import query_training_set


def export_manifest(
    session: Session,
    output_path: str | Path,
    include_low_ordering: bool = False,
) -> int:
    """Write a sorted JSON manifest to output_path. Returns number of pairs written."""
    records = query_training_set(session, include_low_ordering=include_low_ordering)
    records.sort(key=lambda r: r.pair_id)

    manifest = [
        {
            "pair_id": r.pair_id,
            "before_path": r.before_path,
            "after_path": r.after_path,
            "treatment_category": r.treatment_category,
            "treatment_brand": r.treatment_brand,
            "zone_codes": r.zone_codes,
        }
        for r in records
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(manifest)
