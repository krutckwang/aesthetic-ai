"""Tests for crawler/storage/downloader.py — image pair downloader."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from crawler.storage.downloader import (
    DEFAULT_MIN_FILE_BYTES,
    Downloader,
    DownloadResult,
)


def _fake_jpeg(size: int = 4096) -> bytes:
    """Return a fake image payload above the minimum file-size threshold."""
    # JPEG magic bytes + padding
    return b"\xff\xd8\xff\xe0" + b"\x00" * (size - 4)


def _make_response(status: int = 200, content_type: str = "image/jpeg", body: bytes = b"") -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"content-type": content_type}
    resp.content = body or _fake_jpeg()
    return resp


@pytest.fixture
def downloader(tmp_path: Path) -> Downloader:
    return Downloader(
        storage_base_path=tmp_path,
        max_retries=2,
        backoff_base=0.0,   # no sleep in tests
        rate_limit_seconds=0.0,
    )


# ── Path determinism ──────────────────────────────────────────────────────────

class TestDestPath:
    def test_path_is_deterministic(self, downloader, tmp_path):
        url = "https://example.com/image.jpg"
        p1 = downloader._dest_path(url, "test_source")
        p2 = downloader._dest_path(url, "test_source")
        assert p1 == p2

    def test_path_uses_md5_shard(self, downloader, tmp_path):
        url = "https://example.com/image.jpg"
        url_hash = hashlib.md5(url.encode()).hexdigest()
        p = downloader._dest_path(url, "mysource")
        assert p.parent.name == url_hash[:2]        # shard dir
        assert p.name.startswith(url_hash)          # filename starts with hash
        assert "mysource" in str(p)                 # source_name in path

    def test_unknown_extension_defaults_to_jpg(self, downloader):
        url = "https://example.com/image"
        p = downloader._dest_path(url, "src")
        assert p.suffix == ".jpg"

    def test_webp_extension_preserved(self, downloader):
        url = "https://example.com/photo.webp"
        p = downloader._dest_path(url, "src")
        assert p.suffix == ".webp"


# ── Single download ──────────────────────────────────────────────────────────

class TestDownload:
    def test_successful_download_saves_file(self, downloader, tmp_path):
        url = "https://example.com/img.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response()):
            result = downloader.download(url, "test")
        assert result.success is True
        assert result.path is not None
        assert result.path.exists()
        assert result.file_size_bytes > 0
        assert result.from_cache is False

    def test_idempotent_second_call_returns_cached(self, downloader, tmp_path):
        url = "https://example.com/img.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response()) as mock_get:
            result1 = downloader.download(url, "test")
            result2 = downloader.download(url, "test")
        assert result1.success and result2.success
        assert result2.from_cache is True
        assert mock_get.call_count == 1   # second call hit cache, no HTTP request

    def test_http_404_returns_failure(self, downloader):
        url = "https://example.com/missing.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response(status=404)):
            result = downloader.download(url, "test")
        assert result.success is False
        assert "404" in result.failure_reason

    def test_bad_content_type_returns_failure(self, downloader):
        url = "https://example.com/page.html"
        with patch.object(
            downloader._http, "get",
            return_value=_make_response(content_type="text/html", body=b"<html></html>"),
        ):
            result = downloader.download(url, "test")
        assert result.success is False
        assert "bad_content_type" in result.failure_reason

    def test_file_too_small_returns_failure(self, downloader):
        url = "https://example.com/tiny.jpg"
        with patch.object(
            downloader._http, "get",
            return_value=_make_response(body=b"\xff\xd8\xff\xe0" + b"\x00" * 10),
        ):
            result = downloader.download(url, "test")
        assert result.success is False
        assert "too_small" in result.failure_reason

    def test_connection_error_is_retried(self, downloader):
        import httpx
        url = "https://example.com/retry.jpg"
        responses = [
            httpx.ConnectError("refused"),
            httpx.ConnectError("refused"),
            _make_response(),
        ]
        with patch.object(downloader._http, "get", side_effect=responses):
            result = downloader.download(url, "test")
        assert result.success is True
        assert downloader.stats.retried == 2

    def test_retryable_5xx_is_retried(self, downloader):
        url = "https://example.com/retry503.jpg"
        responses = [
            _make_response(status=503),
            _make_response(status=503),
            _make_response(),
        ]
        with patch.object(downloader._http, "get", side_effect=responses):
            result = downloader.download(url, "test")
        assert result.success is True

    def test_exceeding_max_retries_returns_failure(self, downloader):
        import httpx
        url = "https://example.com/unreachable.jpg"
        with patch.object(
            downloader._http, "get",
            side_effect=httpx.ConnectError("always fails"),
        ):
            result = downloader.download(url, "test")
        assert result.success is False

    def test_stats_updated_on_success(self, downloader):
        url = "https://example.com/stats.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response()):
            downloader.download(url, "test")
        assert downloader.stats.total == 1
        assert downloader.stats.succeeded == 1
        assert downloader.stats.failed == 0

    def test_stats_updated_on_failure(self, downloader):
        url = "https://example.com/fail.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response(status=404)):
            downloader.download(url, "test")
        assert downloader.stats.failed == 1


# ── Pair download ─────────────────────────────────────────────────────────────

class TestDownloadPair:
    def test_both_succeed(self, downloader):
        before_url = "https://example.com/before.jpg"
        after_url = "https://example.com/after.jpg"
        with patch.object(downloader._http, "get", return_value=_make_response()):
            b, a = downloader.download_pair(before_url, after_url, "test")
        assert b.success is True
        assert a.success is True

    def test_after_fails_removes_before_file(self, downloader, tmp_path):
        before_url = "https://example.com/before.jpg"
        after_url = "https://example.com/after_missing.jpg"

        def side_effect(url, **kwargs):
            if "before" in url:
                return _make_response()
            return _make_response(status=404)

        with patch.object(downloader._http, "get", side_effect=side_effect):
            b, a = downloader.download_pair(before_url, after_url, "test")

        assert b.success is True
        assert a.success is False
        # Before file should have been cleaned up
        assert b.path is None or not b.path.exists()

    def test_before_fails_does_not_download_after(self, downloader):
        """If before fails, after may still be attempted independently."""
        before_url = "https://example.com/before_fail.jpg"
        after_url = "https://example.com/after_ok.jpg"

        def side_effect(url, **kwargs):
            if "before_fail" in url:
                return _make_response(status=404)
            return _make_response()

        with patch.object(downloader._http, "get", side_effect=side_effect):
            b, a = downloader.download_pair(before_url, after_url, "test")

        assert b.success is False
        # After download was attempted but should be cleaned up
        assert a.path is None or not a.path.exists()


# ── Directory creation ─────────────────────────────────────────────────────────

class TestDirectoryCreation:
    def test_shard_directory_created(self, downloader, tmp_path):
        url = "https://example.com/dir_test.jpg"
        url_hash = hashlib.md5(url.encode()).hexdigest()
        expected_dir = tmp_path / "images" / "raw" / "src" / url_hash[:2]

        with patch.object(downloader._http, "get", return_value=_make_response()):
            downloader.download(url, "src")

        assert expected_dir.exists()


# ── Content-type validation ───────────────────────────────────────────────────

class TestContentTypeValidation:
    @pytest.mark.parametrize("ct", [
        "image/jpeg", "image/png", "image/webp", "image/gif",
        "application/octet-stream", "image/x-custom",
    ])
    def test_valid_content_types_accepted(self, downloader, ct):
        assert downloader._is_valid_content_type(ct) is True

    @pytest.mark.parametrize("ct", [
        "text/html", "application/json", "application/pdf", "video/mp4",
    ])
    def test_invalid_content_types_rejected(self, downloader, ct):
        assert downloader._is_valid_content_type(ct) is False


# ── Context manager ───────────────────────────────────────────────────────────

class TestContextManager:
    def test_close_called_on_exit(self, tmp_path):
        with Downloader(storage_base_path=tmp_path, rate_limit_seconds=0) as dl:
            pass
        # After context exit, _http should be closed; a second close should not raise
        dl.close()
