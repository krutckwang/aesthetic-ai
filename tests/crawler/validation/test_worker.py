"""Tests for crawler/validation/worker.py — validation worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from crawler.validation.worker import ValidationWorker, _dict_to_raw_pair


@pytest.fixture
def validation_config(tmp_path: Path) -> Path:
    config = {
        "calibrated": False,
        "layer1": {"min_confidence": 0.70},
        "layer2": {"min_pairing_score": 0.65},
        "layer3": {"min_cosine_similarity": 0.40},
    }
    path = tmp_path / "validation.yaml"
    path.write_text(yaml.dump(config))
    return path


def _make_queue_item(
    before_url: str = "https://realself.com/before.jpg",
    after_url: str = "https://realself.com/after.jpg",
    consent_tier: int = 1,
    html: str = "",
) -> dict:
    return {
        "id": 1,
        "pair": {
            "before_url": before_url,
            "after_url": after_url,
            "after_url": after_url,
            "source_url": "https://realself.com/review/1",
            "source_name": "realself",
            "language": "en",
            "consent_tier": consent_tier,
            "page_html": html,
            "metadata": {"treatment_type": "botulinum_toxin"},
        },
    }


def _make_worker(tmp_path, validation_config) -> tuple[ValidationWorker, MagicMock, MagicMock]:
    queue = MagicMock()
    writer = MagicMock()
    writer.write.return_value = True

    worker = ValidationWorker(
        queue=queue,
        writer=writer,
        validation_config_path=validation_config,
    )
    return worker, queue, writer


class TestWorkerConfigLoading:
    def test_worker_loads_thresholds_from_config(self, tmp_path, validation_config):
        worker, _, _ = _make_worker(tmp_path, validation_config)
        assert worker._min_structural == 0.70
        assert worker._min_nlp == 0.65
        assert worker._min_arcface == 0.40

    def test_missing_config_uses_defaults(self, tmp_path):
        worker = ValidationWorker(
            queue=MagicMock(),
            writer=MagicMock(),
            validation_config_path=tmp_path / "nonexistent.yaml",
        )
        # Should not raise; uses hardcoded defaults
        assert worker._min_structural >= 0.0
        assert worker._min_nlp >= 0.0


class TestWorkerLoop:
    def test_empty_queue_does_not_process(self, tmp_path, validation_config):
        worker, queue, writer = _make_worker(tmp_path, validation_config)
        queue.dequeue_batch.return_value = []

        # Patch sleep and running flag so loop exits immediately
        worker._running = False
        with patch("time.sleep"):
            worker.run()

        writer.write.assert_not_called()

    def test_tier3_pair_quarantined_not_written_as_pass(self, tmp_path, validation_config):
        item = _make_queue_item(
            before_url="https://instagram.com/before.jpg",
            after_url="https://instagram.com/after.jpg",
            consent_tier=3,
        )
        worker, queue, writer = _make_worker(tmp_path, validation_config)

        # Queue returns one item then stops
        call_count = [0]

        def dequeue_side_effect(n):
            call_count[0] += 1
            if call_count[0] == 1:
                return [item]
            worker._running = False
            return []

        queue.dequeue_batch.side_effect = dequeue_side_effect
        writer.write.return_value = False  # quarantine returns False

        with patch("time.sleep"):
            worker.run()

        # Tier 3 should go to mark_failed (not pass)
        queue.mark_failed.assert_called_once()
        queue.mark_done.assert_not_called()

    def test_item_exception_marked_as_failed(self, tmp_path, validation_config):
        item = _make_queue_item()
        worker, queue, writer = _make_worker(tmp_path, validation_config)

        call_count = [0]

        def dequeue_side_effect(n):
            call_count[0] += 1
            if call_count[0] == 1:
                return [item]
            worker._running = False
            return []

        queue.dequeue_batch.side_effect = dequeue_side_effect
        worker._struct.validate = MagicMock(side_effect=RuntimeError("boom"))

        with patch("time.sleep"):
            worker.run()

        queue.mark_failed.assert_called_once_with(item["id"])


class TestDictToRawPair:
    def test_reconstructs_pair_from_dict(self):
        data = {
            "before_url": "https://ex.com/b.jpg",
            "after_url": "https://ex.com/a.jpg",
            "source_url": "https://ex.com/page",
            "source_name": "test",
            "language": "en",
            "consent_tier": 1,
            "metadata": {"treatment_type": "ha_filler"},
        }
        from crawler.base import ConsentTier

        pair = _dict_to_raw_pair(data, ConsentTier.CONFIRMED)
        assert pair.before_url == "https://ex.com/b.jpg"
        assert pair.source_name == "test"
        assert pair.language == "en"

    def test_missing_fields_use_defaults(self):
        from crawler.base import ConsentTier

        pair = _dict_to_raw_pair({}, ConsentTier.CONFIRMED)
        assert pair.before_url == ""
        assert pair.source_name == "unknown"
        assert pair.language == "en"
