"""Tests for crawler/storage/writer.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crawler.base import ConsentTier, RawImagePair
from crawler.consent.classifier import ConsentAssessment
from crawler.storage.downloader import DownloadResult
from crawler.storage.writer import PairWriter
from database.models import Image, ImagePair, Quarantine


def _make_assessment(tier: ConsentTier, signals: list[str] | None = None) -> ConsentAssessment:
    return ConsentAssessment(
        tier=tier,
        signals_found=signals or [f"tier{int(tier)}_domain"],
        domain="example.com",
    )


def _ok_result(path: Path) -> DownloadResult:
    return DownloadResult(url="https://example.com/img.jpg", path=path, success=True, file_size_bytes=12345)


def _fail_result() -> DownloadResult:
    return DownloadResult(url="https://example.com/img.jpg", path=None, success=False, failure_reason="http_404")


@pytest.fixture
def writer(tmp_db, tmp_storage) -> PairWriter:
    return PairWriter(storage_base_path=tmp_storage)


@pytest.fixture
def tier1_assessments():
    return (
        _make_assessment(ConsentTier.CONFIRMED),
        _make_assessment(ConsentTier.CONFIRMED),
    )


@pytest.fixture
def tier3_assessments():
    return (
        _make_assessment(ConsentTier.UNCERTAIN, ["hard_tier3_domain"]),
        _make_assessment(ConsentTier.UNCERTAIN, ["hard_tier3_domain"]),
    )


class TestWriteTier1Pair:
    def test_write_tier1_creates_image_records(
        self, writer, tmp_db, sample_pair, tier1_assessments
    ):
        before_path = Path("/tmp/before.jpg")
        after_path = Path("/tmp/after.jpg")

        with patch.object(writer._downloader, "download_pair",
                          return_value=(_ok_result(before_path), _ok_result(after_path))), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 12345
            result = writer.write(sample_pair, *tier1_assessments)

        assert result is True

    def test_write_tier3_goes_to_quarantine(
        self, writer, tmp_db, sample_pair_tier3, tier3_assessments
    ):
        from sqlalchemy.orm import sessionmaker
        result = writer.write(sample_pair_tier3, *tier3_assessments)
        assert result is False

        # Verify quarantine records exist
        from sqlalchemy.orm import Session as OrmSession
        with tmp_db.connect() as conn:
            count = conn.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM quarantine")
            ).scalar()
        assert count >= 1

    def test_quarantine_not_in_main_image_table(
        self, writer, tmp_db, sample_pair_tier3, tier3_assessments
    ):
        writer.write(sample_pair_tier3, *tier3_assessments)
        with tmp_db.connect() as conn:
            # Quarantine images have empty file_path — they should not be in image_pair
            pair_count = conn.execute(
                __import__("sqlalchemy").text("SELECT COUNT(*) FROM image_pair")
            ).scalar()
        assert pair_count == 0


class TestImageDownload:
    """Download-level tests now delegate to test_downloader.py.
    These tests verify writer delegates correctly to its internal Downloader."""

    def test_download_failure_returns_false(self, writer, tmp_db, sample_pair, tier1_assessments):
        """When the downloader returns failure, write() returns False."""
        with patch.object(writer._downloader, "download_pair",
                          return_value=(_fail_result(), _fail_result())):
            result = writer.write(sample_pair, *tier1_assessments)
        assert result is False

    def test_before_download_fail_returns_false(self, writer, tmp_db, sample_pair, tier1_assessments):
        after_path = Path("/tmp/after.jpg")
        with patch.object(writer._downloader, "download_pair",
                          return_value=(_fail_result(), _ok_result(after_path))):
            result = writer.write(sample_pair, *tier1_assessments)
        assert result is False


class TestWriteIdempotency:
    def test_duplicate_pair_does_not_raise(
        self, writer, tmp_db, sample_pair, tier1_assessments
    ):
        before_path = Path("/tmp/b.jpg")
        after_path = Path("/tmp/a.jpg")

        with patch.object(writer._downloader, "download_pair",
                          return_value=(_ok_result(before_path), _ok_result(after_path))), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 5000
            r1 = writer.write(sample_pair, *tier1_assessments)
            r2 = writer.write(sample_pair, *tier1_assessments)

        # First write succeeds, second is silently skipped (UniqueConstraint)
        assert r1 is True
        assert r2 is False
