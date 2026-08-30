"""Write zone labels from a ZoneMappingResult to the database."""

from __future__ import annotations

from sqlalchemy.orm import Session

from database.models import ZoneLabel
from pipeline.segmentation.zone_mapper import ZoneMappingResult


class ZoneLabeller:
    """Persists zone presence from the pipeline into ZoneLabel rows."""

    def __init__(self, confidence_threshold: float = 0.5):
        self._threshold = confidence_threshold

    def write_labels(
        self,
        session: Session,
        pair_id: int,
        zone_result: ZoneMappingResult,
    ) -> int:
        """Add ZoneLabel rows for each zone above the confidence threshold.

        Returns the number of labels written.
        """
        if not zone_result.success:
            return 0

        written = 0
        for zone in zone_result.zones:
            if zone.confidence < self._threshold:
                continue
            label = ZoneLabel(
                pair_id=pair_id,
                zone_code=zone.zone_code,
                confidence=round(zone.confidence, 4),
                source="auto",
            )
            session.add(label)
            written += 1
        return written
