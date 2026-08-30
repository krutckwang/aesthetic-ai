"""Query helpers for building the training set."""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import select, not_
from sqlalchemy.orm import Session

from database.models import Image, ImagePair, TreatmentLabel, ZoneLabel, Quarantine, OrderingConfidence


@dataclass
class TrainingRecord:
    pair_id: int
    before_path: str
    after_path: str
    treatment_category: str | None
    treatment_brand: str | None
    zone_codes: list[str] = field(default_factory=list)


def query_training_set(
    session: Session,
    include_low_ordering: bool = False,
) -> list[TrainingRecord]:
    """
    Return pairs suitable for training:
    - Neither before nor after image is in the quarantine table
    - ordering_confidence is not LOW (unless include_low_ordering=True)
    """
    quarantined_ids = select(Quarantine.image_id)

    stmt = select(ImagePair).where(
        not_(ImagePair.before_image_id.in_(quarantined_ids)),
        not_(ImagePair.after_image_id.in_(quarantined_ids)),
    )
    if not include_low_ordering:
        stmt = stmt.where(ImagePair.ordering_confidence != OrderingConfidence.LOW.value)

    pairs = session.execute(stmt).scalars().all()

    records: list[TrainingRecord] = []
    for pair in pairs:
        before_img = session.get(Image, pair.before_image_id)
        after_img = session.get(Image, pair.after_image_id)
        if before_img is None or after_img is None:
            continue

        cat = None
        brand = None
        if pair.treatment_label:
            cat = pair.treatment_label.treatment_category
            brand = pair.treatment_label.treatment_brand

        zone_codes = [zl.zone_code for zl in pair.zone_labels]

        records.append(TrainingRecord(
            pair_id=pair.id,
            before_path=before_img.file_path,
            after_path=after_img.file_path,
            treatment_category=cat,
            treatment_brand=brand,
            zone_codes=zone_codes,
        ))

    return records
