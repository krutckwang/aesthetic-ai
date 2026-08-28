"""
Validated pair writer — persists image files to block storage and records to the database.

Handles:
  - Image file download and storage to Oracle block volume filesystem (via Downloader)
  - SQLAlchemy record creation for Image, ImagePair, ConsentRecord, SourceMetadata
  - Tier 3 routing to the quarantine table
  - Idempotency: duplicate source URLs are caught by UniqueConstraint and skipped
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger
from sqlalchemy.exc import IntegrityError

from crawler.base import ConsentTier, RawImagePair
from crawler.consent.classifier import ConsentAssessment
from crawler.storage.downloader import Downloader
from database.models import (
    ConsentRecord,
    Image,
    ImagePair,
    Quarantine,
    SourceMetadata,
)
from database.session import get_session


class PairWriter:
    """
    Writes a validated, ordered RawImagePair to the database and filesystem.

    Usage:
        writer = PairWriter(storage_base_path="/mnt/block/aesthetic-ai")
        writer.write(pair, before_assessment, after_assessment)
    """

    def __init__(self, storage_base_path: str | Path | None = None) -> None:
        self.base_path = Path(
            storage_base_path
            or os.getenv("STORAGE_BASE_PATH", "/mnt/block/aesthetic-ai")
        )
        self._downloader = Downloader(storage_base_path=self.base_path)

    # ── Public interface ──────────────────────────────────────────────────────

    def write(
        self,
        pair: RawImagePair,
        before_assessment: ConsentAssessment,
        after_assessment: ConsentAssessment,
    ) -> bool:
        """
        Write a validated pair to the database and filesystem.

        Tier 3 images are written to the quarantine table only — never to
        the main image table. Returns True if written, False if skipped.
        """
        # If either image is Tier 3, quarantine the whole pair
        if (
            before_assessment.tier == ConsentTier.UNCERTAIN
            or after_assessment.tier == ConsentTier.UNCERTAIN
        ):
            self._write_quarantine(pair, reason="consent_tier3")
            return False

        before_result, after_result = self._downloader.download_pair(
            pair.before_url, pair.after_url, pair.source_name
        )
        if not before_result.success or not after_result.success:
            logger.warning(
                f"[writer] Download failed for {pair.source_url}: "
                f"before={before_result.failure_reason} after={after_result.failure_reason}"
            )
            return False

        before_path = before_result.path
        after_path = after_result.path

        try:
            self._write_pair_to_db(
                pair, before_path, after_path, before_assessment, after_assessment
            )
            return True
        except IntegrityError:
            logger.debug(f"[writer] Duplicate pair skipped: {pair.before_url}")
            return False
        except Exception as exc:
            logger.error(f"[writer] DB write failed: {exc}")
            return False

    # ── Private: DB writes ────────────────────────────────────────────────────

    def _write_pair_to_db(
        self,
        pair: RawImagePair,
        before_path: Path,
        after_path: Path,
        before_assessment: ConsentAssessment,
        after_assessment: ConsentAssessment,
    ) -> None:
        domain = urlparse(pair.source_url).netloc

        with get_session() as session:
            before_img = self._upsert_image(
                session, pair.before_url, before_path, domain,
                pair.language, before_assessment,
            )
            after_img = self._upsert_image(
                session, pair.after_url, after_path, domain,
                pair.language, after_assessment,
            )

            img_pair = ImagePair(
                before_image_id=before_img.id,
                after_image_id=after_img.id,
                layer1_score=pair.layer1_score,
                layer2_score=pair.layer2_score,
                layer3_score=pair.layer3_score,
                ordering_confidence=pair.ordering_confidence or "UNKNOWN",
            )
            session.add(img_pair)
            session.flush()  # get img_pair.id

            meta = SourceMetadata(
                pair_id=img_pair.id,
                practitioner_name=pair.metadata.get("provider_name"),
                clinic_name=pair.metadata.get("clinic_name"),
                date_posted=pair.metadata.get("date_posted"),
                language=pair.language,
                source_name=pair.source_name,
                raw_metadata=json.dumps(pair.metadata) if pair.metadata else None,
            )
            session.add(meta)

        logger.debug(f"[writer] Pair written: {pair.before_url[:60]} ↔ {pair.after_url[:60]}")

    def _upsert_image(
        self,
        session,
        url: str,
        file_path: Path,
        domain: str,
        language: str,
        assessment: ConsentAssessment,
    ) -> Image:
        """Return existing Image record or create a new one."""
        existing = session.query(Image).filter_by(source_url=url).first()
        if existing:
            return existing

        img = Image(
            file_path=str(file_path),
            source_url=url,
            domain=domain,
            language=language,
            consent_tier=int(assessment.tier),
            file_size_bytes=file_path.stat().st_size if file_path.exists() else None,
        )
        session.add(img)
        session.flush()

        consent = ConsentRecord(
            image_id=img.id,
            consent_tier=int(assessment.tier),
            signals_found=json.dumps(assessment.signals_found),
        )
        session.add(consent)
        return img

    def _write_quarantine(self, pair: RawImagePair, reason: str) -> None:
        """Write a Tier 3 pair's source URL to the quarantine table."""
        domain = urlparse(pair.source_url).netloc
        with get_session() as session:
            # We store a minimal Image record so quarantine has a valid FK
            for url in (pair.before_url, pair.after_url):
                existing = session.query(Image).filter_by(source_url=url).first()
                if existing:
                    img_id = existing.id
                else:
                    img = Image(
                        file_path="",  # not downloaded
                        source_url=url,
                        domain=domain,
                        language=pair.language,
                        consent_tier=int(ConsentTier.UNCERTAIN),
                    )
                    session.add(img)
                    session.flush()
                    img_id = img.id

                already = session.query(Quarantine).filter_by(image_id=img_id).first()
                if not already:
                    session.add(
                        Quarantine(
                            image_id=img_id,
                            reason=reason,
                            source_url=pair.source_url,
                        )
                    )
        logger.debug(f"[writer] Quarantined pair from: {pair.source_url[:80]}")
