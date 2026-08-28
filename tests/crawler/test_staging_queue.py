"""Tests for crawler/storage/staging_queue.py."""

from __future__ import annotations

import pytest

from crawler.base import ConsentTier, RawImagePair
from crawler.storage.staging_queue import DONE, FAILED, PENDING, PROCESSING, StagingQueue


@pytest.fixture
def queue(tmp_queue_db) -> StagingQueue:
    return StagingQueue(tmp_queue_db)


@pytest.fixture
def pair_a() -> RawImagePair:
    return RawImagePair(
        before_url="https://example.com/before_a.jpg",
        after_url="https://example.com/after_a.jpg",
        source_url="https://example.com/gallery/1",
        source_name="test_source",
        language="en",
        consent_tier=ConsentTier.CONFIRMED,
        metadata={"treatment_name": "Botox"},
    )


@pytest.fixture
def pair_b() -> RawImagePair:
    return RawImagePair(
        before_url="https://example.com/before_b.jpg",
        after_url="https://example.com/after_b.jpg",
        source_url="https://example.com/gallery/2",
        source_name="test_source",
        language="en",
        consent_tier=ConsentTier.LIKELY,
        metadata={},
    )


class TestEnqueue:
    def test_enqueue_returns_true_on_insert(self, queue, pair_a):
        assert queue.enqueue(pair_a) is True

    def test_enqueue_item_is_pending(self, queue, pair_a):
        queue.enqueue(pair_a)
        assert queue.pending_count() == 1

    def test_enqueue_idempotent(self, queue, pair_a):
        queue.enqueue(pair_a)
        result = queue.enqueue(pair_a)  # duplicate
        assert result is False
        assert queue.pending_count() == 1

    def test_enqueue_different_pairs(self, queue, pair_a, pair_b):
        queue.enqueue(pair_a)
        queue.enqueue(pair_b)
        assert queue.pending_count() == 2

    def test_enqueue_batch(self, queue, pair_a, pair_b):
        inserted, skipped = queue.enqueue_batch([pair_a, pair_b])
        assert inserted == 2
        assert skipped == 0

    def test_enqueue_batch_duplicate_counted(self, queue, pair_a):
        queue.enqueue(pair_a)
        inserted, skipped = queue.enqueue_batch([pair_a])
        assert inserted == 0
        assert skipped == 1


class TestDequeue:
    def test_dequeue_returns_pending_items(self, queue, pair_a, pair_b):
        queue.enqueue(pair_a)
        queue.enqueue(pair_b)
        batch = queue.dequeue_batch(batch_size=10)
        assert len(batch) == 2

    def test_dequeue_marks_processing(self, queue, pair_a):
        queue.enqueue(pair_a)
        batch = queue.dequeue_batch(batch_size=1)
        assert len(batch) == 1
        assert queue.pending_count() == 0  # moved to PROCESSING

    def test_dequeue_empty_queue_returns_empty_list(self, queue):
        assert queue.dequeue_batch() == []

    def test_dequeue_respects_batch_size(self, queue, pair_a, pair_b):
        queue.enqueue(pair_a)
        queue.enqueue(pair_b)
        batch = queue.dequeue_batch(batch_size=1)
        assert len(batch) == 1

    def test_dequeued_item_has_correct_fields(self, queue, pair_a):
        queue.enqueue(pair_a)
        item = queue.dequeue_batch(batch_size=1)[0]
        assert item["before_url"] == pair_a.before_url
        assert item["after_url"] == pair_a.after_url
        assert item["source_name"] == pair_a.source_name
        assert item["consent_tier"] == int(pair_a.consent_tier)


class TestMarkStatus:
    def test_mark_done(self, queue, pair_a):
        queue.enqueue(pair_a)
        item = queue.dequeue_batch(batch_size=1)[0]
        queue.mark_done(item["id"])
        stats = queue.stats()
        assert stats.get(DONE, 0) == 1
        assert stats.get(PROCESSING, 0) == 0

    def test_mark_failed(self, queue, pair_a):
        queue.enqueue(pair_a)
        item = queue.dequeue_batch(batch_size=1)[0]
        queue.mark_failed(item["id"], reason="face_not_detected")
        stats = queue.stats()
        assert stats.get(FAILED, 0) == 1

    def test_mark_done_not_returned_by_dequeue(self, queue, pair_a):
        queue.enqueue(pair_a)
        item = queue.dequeue_batch(batch_size=1)[0]
        queue.mark_done(item["id"])
        assert queue.dequeue_batch() == []


class TestHelpers:
    def test_is_duplicate_true(self, queue, pair_a):
        queue.enqueue(pair_a)
        assert queue.is_duplicate(pair_a.before_url, pair_a.after_url) is True

    def test_is_duplicate_false(self, queue, pair_a):
        assert queue.is_duplicate(pair_a.before_url, pair_a.after_url) is False

    def test_stats_returns_counts_by_status(self, queue, pair_a, pair_b):
        queue.enqueue(pair_a)
        queue.enqueue(pair_b)
        item = queue.dequeue_batch(batch_size=1)[0]
        queue.mark_done(item["id"])
        stats = queue.stats()
        assert stats[PENDING] == 1
        assert stats[DONE] == 1

    def test_reset_stale_processing(self, queue, pair_a):
        queue.enqueue(pair_a)
        queue.dequeue_batch(batch_size=1)  # moves to PROCESSING
        count = queue.reset_stale_processing()
        assert count == 1
        assert queue.pending_count() == 1

    def test_persistence_across_reconnect(self, tmp_queue_db, pair_a):
        """Data in queue survives closing and reopening the connection."""
        q1 = StagingQueue(tmp_queue_db)
        q1.enqueue(pair_a)

        q2 = StagingQueue(tmp_queue_db)  # new connection to same file
        assert q2.pending_count() == 1
