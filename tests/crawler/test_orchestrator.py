"""Tests for crawler/orchestrator.py."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest
import yaml

from crawler.base import ConsentTier, RawImagePair, SourceConfig, RenderingMethod
from crawler.orchestrator import CrawlerOrchestrator
from crawler.storage.staging_queue import StagingQueue


def _make_mock_source(name: str, pairs: list[RawImagePair]) -> MagicMock:
    """Create a mock source that yields the given pairs from crawl()."""
    source = MagicMock()
    source.config.name = name
    source.config.enabled = True
    source.crawl.return_value = iter(pairs)
    return source


def _make_pair(n: int) -> RawImagePair:
    return RawImagePair(
        before_url=f"https://example.com/before_{n}.jpg",
        after_url=f"https://example.com/after_{n}.jpg",
        source_url=f"https://example.com/gallery/{n}",
        source_name="test_source",
        language="en",
        consent_tier=ConsentTier.CONFIRMED,
        metadata={},
    )


@pytest.fixture
def queue(tmp_queue_db) -> StagingQueue:
    return StagingQueue(tmp_queue_db)


@pytest.fixture
def minimal_config(tmp_path: Path) -> Path:
    config = {
        "global": {"user_agent": "TestBot/0.1", "default_rate_limit_seconds": 0.0},
        "sources": {
            "realself": {
                "enabled": True,
                "rendering": "static",
                "rate_limit_seconds": 0.0,
                "language": "en",
                "consent_tier": 1,
                "calibration_source": True,
                "base_urls": [],
            },
        },
    }
    config_path = tmp_path / "crawler.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestOrchestrator:
    def test_discovered_pairs_enqueued(self, queue, minimal_config):
        pairs = [_make_pair(i) for i in range(3)]
        mock_source = _make_mock_source("realself", pairs)

        with patch("crawler.orchestrator.build_sources", return_value=[mock_source]):
            orch = CrawlerOrchestrator(queue=queue, config_path=minimal_config)
            orch.run()

        assert queue.pending_count() == 3

    def test_disabled_sources_not_crawled(self, queue, minimal_config):
        """Sources not returned by build_sources are not processed."""
        with patch("crawler.orchestrator.build_sources", return_value=[]) as mock_build:
            orch = CrawlerOrchestrator(queue=queue, config_path=minimal_config)
            stats = orch.run()

        assert stats == {}
        assert queue.pending_count() == 0

    def test_stats_returned_per_source(self, queue, minimal_config):
        pairs = [_make_pair(i) for i in range(5)]
        mock_source = _make_mock_source("realself", pairs)

        with patch("crawler.orchestrator.build_sources", return_value=[mock_source]):
            orch = CrawlerOrchestrator(queue=queue, config_path=minimal_config)
            stats = orch.run()

        assert "realself" in stats
        assert stats["realself"]["inserted"] == 5
        assert stats["realself"]["skipped"] == 0

    def test_duplicate_pairs_counted_as_skipped(self, queue, minimal_config):
        pair = _make_pair(1)
        # Same pair twice from source
        mock_source = _make_mock_source("realself", [pair, pair])

        with patch("crawler.orchestrator.build_sources", return_value=[mock_source]):
            orch = CrawlerOrchestrator(queue=queue, config_path=minimal_config)
            stats = orch.run()

        assert stats["realself"]["inserted"] == 1
        assert stats["realself"]["skipped"] == 1

    def test_source_exception_does_not_crash_orchestrator(self, queue, minimal_config):
        bad_source = MagicMock()
        bad_source.config.name = "bad_source"
        bad_source.config.enabled = True
        bad_source.crawl.side_effect = RuntimeError("Network error")

        good_source = _make_mock_source("realself", [_make_pair(1)])

        with patch(
            "crawler.orchestrator.build_sources",
            return_value=[bad_source, good_source],
        ):
            orch = CrawlerOrchestrator(queue=queue, config_path=minimal_config)
            stats = orch.run()

        assert stats["bad_source"]["errors"] == 1
        assert stats["realself"]["inserted"] == 1  # good source still ran

    def test_calibration_only_mode_uses_calibration_source(self, queue, minimal_config):
        mock_source = _make_mock_source("realself", [_make_pair(1)])

        with patch("crawler.orchestrator.build_calibration_source", return_value=mock_source) as mock_cal:
            orch = CrawlerOrchestrator(
                queue=queue,
                config_path=minimal_config,
                calibration_only=True,
            )
            orch.run()

        mock_cal.assert_called_once()
        assert queue.pending_count() == 1
