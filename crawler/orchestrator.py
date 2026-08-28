"""
Crawler orchestrator — runs all enabled source modules and feeds the staging queue.

The orchestrator is the crawler process. It runs independently from the
validation worker. Pairs discovered by sources are enqueued to the SQLite
staging queue; the worker drains the queue in a separate process.

Usage:
    python -m crawler.orchestrator
    python -m crawler.orchestrator --calibration-only
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from loguru import logger

from crawler.base import RawImagePair
from crawler.factory import build_calibration_source, build_sources
from crawler.storage.staging_queue import StagingQueue


class CrawlerOrchestrator:
    """
    Runs all enabled source modules sequentially, enqueuing discovered pairs.

    Designed to run as a long-lived background process on the Oracle VM.
    Handles SIGINT/SIGTERM for graceful shutdown.
    """

    def __init__(
        self,
        queue: StagingQueue,
        config_path: str | Path = "configs/crawler.yaml",
        calibration_only: bool = False,
    ) -> None:
        self.queue = queue
        self.config_path = config_path
        self.calibration_only = calibration_only
        self._shutdown = False
        self._stats: dict[str, dict[str, int]] = {}
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ── Public run interface ──────────────────────────────────────────────────

    def run(self) -> dict[str, dict[str, int]]:
        """
        Run all sources. Returns per-source stats dict:
            {source_name: {inserted: N, skipped: N, errors: N}}
        """
        if self.calibration_only:
            sources = [build_calibration_source(self.config_path)]
            logger.info("[orchestrator] Calibration-only mode — running RealSelf source.")
        else:
            sources = build_sources(self.config_path)

        if not sources:
            logger.warning("[orchestrator] No enabled sources found. Check crawler.yaml.")
            return {}

        logger.info(f"[orchestrator] Starting crawl across {len(sources)} source(s).")

        for source in sources:
            if self._shutdown:
                logger.info("[orchestrator] Shutdown requested — stopping before next source.")
                break
            self._run_source(source)

        self._log_summary()
        return self._stats

    # ── Private ───────────────────────────────────────────────────────────────

    def _run_source(self, source) -> None:
        name = source.config.name
        self._stats[name] = {"inserted": 0, "skipped": 0, "errors": 0}
        logger.info(f"[orchestrator] Starting source: {name}")
        start = time.monotonic()

        try:
            for pair in source.crawl():
                if self._shutdown:
                    break
                inserted = self.queue.enqueue(pair)
                if inserted:
                    self._stats[name]["inserted"] += 1
                else:
                    self._stats[name]["skipped"] += 1

                if (self._stats[name]["inserted"] % 100) == 0 and self._stats[name]["inserted"] > 0:
                    logger.info(
                        f"[orchestrator:{name}] {self._stats[name]['inserted']} pairs queued "
                        f"({self._stats[name]['skipped']} skipped)"
                    )

        except Exception as exc:
            logger.error(f"[orchestrator] Source '{name}' raised an error: {exc}")
            self._stats[name]["errors"] += 1

        elapsed = time.monotonic() - start
        logger.info(
            f"[orchestrator] Source '{name}' done in {elapsed:.1f}s — "
            f"inserted={self._stats[name]['inserted']} "
            f"skipped={self._stats[name]['skipped']} "
            f"errors={self._stats[name]['errors']}"
        )

    def _log_summary(self) -> None:
        total_inserted = sum(s["inserted"] for s in self._stats.values())
        total_skipped = sum(s["skipped"] for s in self._stats.values())
        queue_stats = self.queue.stats()
        logger.info(
            f"[orchestrator] Crawl complete — "
            f"total inserted: {total_inserted}, skipped: {total_skipped}. "
            f"Queue: {queue_stats}"
        )

    def _handle_shutdown(self, signum, frame) -> None:
        logger.info(f"[orchestrator] Signal {signum} received — shutting down gracefully.")
        self._shutdown = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Aesthetic AI crawler orchestrator")
    parser.add_argument(
        "--calibration-only",
        action="store_true",
        help="Run only the calibration bootstrap source (RealSelf).",
    )
    parser.add_argument(
        "--config",
        default="configs/crawler.yaml",
        help="Path to crawler config YAML.",
    )
    parser.add_argument(
        "--queue-db",
        default=None,
        help="Path to staging queue SQLite DB. Defaults to STORAGE_BASE_PATH/staging/queue.db.",
    )
    args = parser.parse_args()

    import os
    queue_db = args.queue_db or os.path.join(
        os.getenv("STORAGE_BASE_PATH", "/mnt/block/aesthetic-ai"),
        "staging",
        "queue.db",
    )
    Path(queue_db).parent.mkdir(parents=True, exist_ok=True)

    queue = StagingQueue(queue_db)
    queue.reset_stale_processing()

    orchestrator = CrawlerOrchestrator(
        queue=queue,
        config_path=args.config,
        calibration_only=args.calibration_only,
    )
    stats = orchestrator.run()
    sys.exit(0 if stats else 1)


if __name__ == "__main__":
    main()
