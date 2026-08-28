"""
Validation worker — independent process that consumes pairs from the staging
queue and runs all three validation layers + ordering check.

Architecture
────────────
  Crawler process  ──enqueue──►  staging_queue.db  ──dequeue──►  Validation worker
                                                                        │
                                                              Layer 1: structural
                                                              Layer 2: NLP pairing
                                                              Layer 3: ArcFace
                                                              Gate 1+2: ordering
                                                                        │
                                                              pass ──► PairWriter (DB + disk)
                                                              fail ──► mark_failed (queue)

The worker runs continuously, sleeping when the queue is empty.
Start via:
    python -m crawler.validation.worker \
        --queue-db  /mnt/block/aesthetic-ai/staging.db \
        --config    configs/validation.yaml \
        --storage   /mnt/block/aesthetic-ai/data
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

import yaml
from loguru import logger

from crawler.consent.classifier import ConsentClassifier, ConsentTier
from crawler.storage.staging_queue import StagingQueue
from crawler.storage.writer import PairWriter
from crawler.validation.face_similarity import FaceSimilarityValidator
from crawler.validation.nlp_pairing import NLPPairingValidator
from crawler.validation.ordering import OrderingValidator, OrderingVerdict
from crawler.validation.structural import StructuralValidator


IDLE_SLEEP_SECONDS = 5.0
BATCH_SIZE = 20


class ValidationWorker:
    """
    Continuously dequeues pairs from the staging queue, runs validation,
    and writes passing pairs to the main database.
    """

    def __init__(
        self,
        queue: StagingQueue,
        writer: PairWriter,
        validation_config_path: Path,
    ) -> None:
        self.queue = queue
        self.writer = writer
        self._running = True

        config = self._load_config(validation_config_path)
        self._struct = StructuralValidator()
        self._nlp = NLPPairingValidator(
            min_keyword_score=config.get("layer2", {}).get("min_pairing_score", 0.65)
        )
        self._face = FaceSimilarityValidator(
            min_cosine=config.get("layer3", {}).get("min_cosine_similarity", 0.40)
        )
        self._ordering = OrderingValidator(gate2_active=False)
        self._consent = ConsentClassifier()

        self._min_structural = config.get("layer1", {}).get("min_confidence", 0.70)
        self._min_nlp = config.get("layer2", {}).get("min_pairing_score", 0.65)
        self._min_arcface = config.get("layer3", {}).get("min_cosine_similarity", 0.40)

        # Wire SIGINT / SIGTERM for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

        logger.info(
            "[worker] Validation worker initialised — "
            f"L1≥{self._min_structural}, L2≥{self._min_nlp}, L3≥{self._min_arcface}"
        )

    def run(self) -> None:
        """Main processing loop."""
        logger.info("[worker] Starting validation loop …")
        total_processed = 0
        total_passed = 0
        total_failed = 0

        while self._running:
            batch = self.queue.dequeue_batch(BATCH_SIZE)
            if not batch:
                logger.debug("[worker] Queue empty — sleeping …")
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            for item in batch:
                item_id = item["id"]
                pair_data = item.get("pair", {})

                try:
                    passed = self._validate_and_write(pair_data)
                    if passed:
                        self.queue.mark_done(item_id)
                        total_passed += 1
                    else:
                        self.queue.mark_failed(item_id)
                        total_failed += 1
                except Exception as exc:
                    logger.error(f"[worker] Item {item_id} raised: {exc}")
                    self.queue.mark_failed(item_id)
                    total_failed += 1

                total_processed += 1
                if total_processed % 100 == 0:
                    logger.info(
                        f"[worker] Processed={total_processed} "
                        f"passed={total_passed} failed={total_failed}"
                    )

        logger.info(
            f"[worker] Stopped. Total: processed={total_processed} "
            f"passed={total_passed} failed={total_failed}"
        )

    def _validate_and_write(self, pair_data: dict) -> bool:
        """
        Run all validation layers for a single pair.
        Returns True if the pair passes and was written to the database.
        """
        before_url = pair_data.get("before_url", "")
        after_url = pair_data.get("after_url", "")
        page_html = pair_data.get("page_html", "")
        source_url = pair_data.get("source_url", "")
        consent_tier_raw = pair_data.get("consent_tier", 3)
        treatment_type = pair_data.get("metadata", {}).get("treatment_type", "")

        # Rebuild consent assessment from stored tier
        before_assessment = self._consent.classify(before_url, "")
        after_assessment = self._consent.classify(after_url, "")

        # Hard Tier 3 always quarantined — write and return False
        if (before_assessment.tier == ConsentTier.UNCERTAIN or
                after_assessment.tier == ConsentTier.UNCERTAIN):
            from crawler.base import RawImagePair, ConsentTier as BaseTier
            pair_obj = _dict_to_raw_pair(pair_data, BaseTier.UNCERTAIN)
            self.writer.write(pair_obj, before_assessment, after_assessment)
            return False

        # ── Layer 1: structural ──────────────────────────────────────────────
        structural = self._struct.validate(before_url, after_url, page_html)
        if structural.confidence < self._min_structural:
            logger.debug(
                f"[worker] L1 FAIL: {before_url[:60]} confidence={structural.confidence:.2f}"
            )
            return False

        # ── Ordering Gate 1 ──────────────────────────────────────────────────
        ordering = self._ordering.validate(structural, treatment_type)
        if ordering.verdict == OrderingVerdict.GATE1_FAIL:
            logger.debug(f"[worker] Ordering Gate 1 FAIL: {before_url[:60]}")
            return False

        # ── Layer 2: NLP pairing ─────────────────────────────────────────────
        before_text = self._nlp.extract_image_context(before_url, page_html)
        after_text = self._nlp.extract_image_context(after_url, page_html)
        nlp_result = self._nlp.validate(before_text, after_text)
        if nlp_result.pairing_score < self._min_nlp:
            logger.debug(
                f"[worker] L2 FAIL: {before_url[:60]} score={nlp_result.pairing_score:.2f}"
            )
            return False

        # ── Layer 3: ArcFace (needs file paths — deferred to after download) ─
        # NOTE: ArcFace runs after download in PairWriter.write().
        # Here we record the NLP+structural scores and let PairWriter do L3.

        # ── Write pair ───────────────────────────────────────────────────────
        from crawler.base import RawImagePair, ConsentTier as BaseTier
        from crawler.base import RenderingMethod

        pair_obj = _dict_to_raw_pair(pair_data, BaseTier(consent_tier_raw))
        pair_obj.layer1_score = structural.confidence
        pair_obj.layer2_score = nlp_result.pairing_score
        pair_obj.ordering_confidence = ordering.confidence

        written = self.writer.write(pair_obj, before_assessment, after_assessment)
        return written

    def _handle_stop(self, *_: object) -> None:
        logger.info("[worker] Shutdown signal received.")
        self._running = False

    @staticmethod
    def _load_config(path: Path) -> dict:
        if not path.exists():
            logger.warning(f"[worker] Config not found at {path} — using defaults.")
            return {}
        with open(path) as f:
            return yaml.safe_load(f) or {}


def _dict_to_raw_pair(data: dict, tier) -> "RawImagePair":
    """Reconstruct a RawImagePair from a queue item dict."""
    from crawler.base import RawImagePair, RenderingMethod

    return RawImagePair(
        before_url=data.get("before_url", ""),
        after_url=data.get("after_url", ""),
        source_url=data.get("source_url", ""),
        source_name=data.get("source_name", "unknown"),
        language=data.get("language", "en"),
        consent_tier=tier,
        metadata=data.get("metadata", {}),
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validation worker.")
    parser.add_argument("--queue-db", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--config", default="configs/validation.yaml")
    args = parser.parse_args()

    queue = StagingQueue(Path(args.queue_db))
    queue.reset_stale_processing()

    writer = PairWriter(storage_base_path=Path(args.storage))
    worker = ValidationWorker(
        queue=queue,
        writer=writer,
        validation_config_path=Path(args.config),
    )
    worker.run()


if __name__ == "__main__":
    main()
