"""
SQLite-based staging queue for raw candidate image pairs.

The crawler process enqueues raw pairs here. The validation worker dequeues
and processes them independently. WAL mode allows both processes to access
the DB simultaneously without locking.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from loguru import logger

from crawler.base import ConsentTier, RawImagePair


# Queue item status values
PENDING = "pending"
PROCESSING = "processing"
DONE = "done"
FAILED = "failed"


class StagingQueue:
    """
    Thread-safe SQLite queue for raw crawler output.

    Each item represents one candidate before/after pair awaiting validation.
    Idempotent: duplicate (before_url, after_url) pairs are silently ignored.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS staging_queue (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    before_url   TEXT    NOT NULL,
                    after_url    TEXT    NOT NULL,
                    source_url   TEXT    NOT NULL,
                    source_name  TEXT    NOT NULL,
                    language     TEXT    NOT NULL DEFAULT 'en',
                    consent_tier INTEGER NOT NULL,
                    metadata     TEXT,
                    status       TEXT    NOT NULL DEFAULT 'pending',
                    failure_reason TEXT,
                    created_at   TEXT    NOT NULL,
                    processed_at TEXT,
                    UNIQUE(before_url, after_url)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_status ON staging_queue(status)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Write operations ──────────────────────────────────────────────────────

    def enqueue(self, pair: RawImagePair) -> bool:
        """
        Add a raw pair to the queue.

        Returns True if inserted, False if duplicate (silently skipped).
        Never raises on duplicate — idempotency is intentional.
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO staging_queue
                        (before_url, after_url, source_url, source_name,
                         language, consent_tier, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pair.before_url,
                        pair.after_url,
                        pair.source_url,
                        pair.source_name,
                        pair.language,
                        int(pair.consent_tier),
                        json.dumps(pair.metadata) if pair.metadata else None,
                        datetime.utcnow().isoformat(),
                    ),
                )
            return True
        except sqlite3.IntegrityError:
            # Duplicate (before_url, after_url) — silently skip
            return False

    def enqueue_batch(self, pairs: list[RawImagePair]) -> tuple[int, int]:
        """Enqueue multiple pairs. Returns (inserted, skipped) counts."""
        inserted = skipped = 0
        for pair in pairs:
            if self.enqueue(pair):
                inserted += 1
            else:
                skipped += 1
        return inserted, skipped

    def mark_processing(self, item_id: int) -> None:
        """Mark an item as currently being processed by the validation worker."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE staging_queue SET status=? WHERE id=?",
                (PROCESSING, item_id),
            )

    def mark_done(self, item_id: int) -> None:
        """Mark a successfully validated item as done."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE staging_queue SET status=?, processed_at=? WHERE id=?",
                (DONE, datetime.utcnow().isoformat(), item_id),
            )

    def mark_failed(self, item_id: int, reason: str) -> None:
        """Mark an item that failed validation with a reason string."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE staging_queue
                SET status=?, failure_reason=?, processed_at=?
                WHERE id=?
                """,
                (FAILED, reason[:512], datetime.utcnow().isoformat(), item_id),
            )

    def reset_stale_processing(self) -> int:
        """
        Reset items stuck in PROCESSING back to PENDING.
        Called on worker startup to recover from a crashed previous run.
        Returns the number of items reset.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE staging_queue SET status=? WHERE status=?",
                (PENDING, PROCESSING),
            )
            count = cursor.rowcount
        if count:
            logger.warning(f"[staging_queue] Reset {count} stale PROCESSING items to PENDING.")
        return count

    # ── Read operations ───────────────────────────────────────────────────────

    def dequeue_batch(self, batch_size: int = 10) -> list[dict]:
        """
        Fetch up to batch_size PENDING items and mark them PROCESSING.
        Returns list of row dicts with all queue fields.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM staging_queue
                WHERE status = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (PENDING, batch_size),
            ).fetchall()

            if not rows:
                return []

            ids = [row["id"] for row in rows]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE staging_queue SET status=? WHERE id IN ({placeholders})",
                [PROCESSING, *ids],
            )

        return [dict(row) for row in rows]

    def pending_count(self) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM staging_queue WHERE status=?", (PENDING,)
            ).fetchone()[0]

    def is_duplicate(self, before_url: str, after_url: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM staging_queue WHERE before_url=? AND after_url=?",
                (before_url, after_url),
            ).fetchone()
        return row is not None

    def stats(self) -> dict[str, int]:
        """Return counts by status."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as n FROM staging_queue GROUP BY status"
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}
