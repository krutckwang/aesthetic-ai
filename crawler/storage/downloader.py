"""
Image pair downloader with per-domain rate limiting, retry/backoff, and
content-type validation.

Design principles
─────────────────
- Deterministic file paths: MD5(url) → 2-char shard / hash.ext
  Same URL always maps to same file — idempotent across restarts.
- Per-domain rate limiting: avoids hammering a single host.
- Exponential backoff: retries on transient 5xx / connection errors only.
- Content-type guard: rejects HTML, PDF, or any non-image response.
- Minimum file-size guard: rejects 1×1 pixel tracking images (< 1 KB).
- Atomic pair download: if either image in a pair fails, the partial file
  is cleaned up so the DB is never written with a half-pair.

Used by:
  crawler/storage/writer.py   — replaces inline _download_image
  crawler/validation/worker.py — direct access for pre-download checks
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger


# ── Configuration defaults ────────────────────────────────────────────────────
DEFAULT_TIMEOUT = 60
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0     # seconds; wait = base ** attempt
DEFAULT_MIN_FILE_BYTES = 1024  # 1 KB minimum — rejects tracking pixels

VALID_CONTENT_TYPES = frozenset([
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "image/gif", "application/octet-stream",
])

VALID_EXTENSIONS = frozenset([".jpg", ".jpeg", ".png", ".webp", ".gif"])

RETRY_STATUS_CODES = frozenset([429, 500, 502, 503, 504])


@dataclass
class DownloadResult:
    """Result of a single image download attempt."""

    url: str
    path: Path | None           # None = failed
    success: bool
    file_size_bytes: int = 0
    from_cache: bool = False    # True = file already existed on disk
    failure_reason: str = ""


@dataclass
class DownloadStats:
    """Aggregate statistics for a Downloader instance's lifetime."""

    total: int = 0
    succeeded: int = 0
    from_cache: int = 0
    failed: int = 0
    retried: int = 0
    bytes_downloaded: int = 0

    def record(self, result: DownloadResult) -> None:
        self.total += 1
        if result.success:
            self.succeeded += 1
            self.bytes_downloaded += result.file_size_bytes
            if result.from_cache:
                self.from_cache += 1
        else:
            self.failed += 1


class Downloader:
    """
    Downloads image URLs to block storage with rate limiting and retries.

    Args:
        storage_base_path: Root path for all downloaded files.
            Files are stored at: <base>/images/raw/<source_name>/<shard>/<hash>.<ext>
        timeout:            HTTP request timeout in seconds.
        max_retries:        Maximum retry attempts for retryable errors.
        backoff_base:       Exponential backoff base in seconds.
        min_file_bytes:     Minimum acceptable file size. Smaller = rejected.
        rate_limit_seconds: Default inter-request delay for a domain (0 = no limit).
    """

    def __init__(
        self,
        storage_base_path: str | Path,
        timeout: int = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        min_file_bytes: int = DEFAULT_MIN_FILE_BYTES,
        rate_limit_seconds: float = 0.5,
    ) -> None:
        self.base_path = Path(storage_base_path)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.min_file_bytes = min_file_bytes
        self.rate_limit_seconds = rate_limit_seconds
        self.stats = DownloadStats()

        self._http = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "AestheticAI-Research-Bot/0.1 (academic)"},
        )
        self._domain_last_request: dict[str, float] = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def download(self, url: str, source_name: str) -> DownloadResult:
        """
        Download a single image URL to block storage.

        Returns:
            DownloadResult with path set on success, None on failure.
        """
        dest_path = self._dest_path(url, source_name)

        # Idempotency check
        if dest_path.exists() and dest_path.stat().st_size >= self.min_file_bytes:
            result = DownloadResult(
                url=url,
                path=dest_path,
                success=True,
                file_size_bytes=dest_path.stat().st_size,
                from_cache=True,
            )
            self.stats.record(result)
            return result

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        result = self._fetch_with_retry(url, dest_path, source_name)
        self.stats.record(result)
        return result

    def download_pair(
        self,
        before_url: str,
        after_url: str,
        source_name: str,
    ) -> tuple[DownloadResult, DownloadResult]:
        """
        Download a before/after pair atomically.

        If either download fails, any partially-written file is removed so
        the database is never written with an incomplete pair.

        Returns:
            (before_result, after_result) — check .success on each.
        """
        before_result = self.download(before_url, source_name)
        after_result = self.download(after_url, source_name)

        # Atomic cleanup: if either failed, clean up the successful one
        if before_result.success and not after_result.success:
            self._cleanup(before_result.path, cached=before_result.from_cache)
        elif after_result.success and not before_result.success:
            self._cleanup(after_result.path, cached=after_result.from_cache)

        return before_result, after_result

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Downloader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ── Private: fetch with retry ─────────────────────────────────────────────

    def _fetch_with_retry(
        self, url: str, dest_path: Path, source_name: str
    ) -> DownloadResult:
        last_reason = ""
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                wait = self.backoff_base ** attempt
                logger.debug(f"[downloader] Retry {attempt}/{self.max_retries} in {wait:.1f}s: {url}")
                time.sleep(wait)
                self.stats.retried += 1

            self._rate_limit(url)

            try:
                resp = self._http.get(url)
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as exc:
                last_reason = f"connection_error: {exc}"
                continue

            if resp.status_code in RETRY_STATUS_CODES:
                last_reason = f"http_{resp.status_code}"
                continue

            if resp.status_code != 200:
                last_reason = f"http_{resp.status_code}"
                break  # non-retryable (4xx except 429)

            content_type = resp.headers.get("content-type", "").lower().split(";")[0].strip()
            if not self._is_valid_content_type(content_type):
                last_reason = f"bad_content_type:{content_type}"
                break

            content = resp.content
            if len(content) < self.min_file_bytes:
                last_reason = f"file_too_small:{len(content)}_bytes"
                break

            dest_path.write_bytes(content)
            logger.debug(f"[downloader] Saved {len(content):,} bytes → {dest_path.name}")
            return DownloadResult(
                url=url,
                path=dest_path,
                success=True,
                file_size_bytes=len(content),
                from_cache=False,
            )

        logger.warning(f"[downloader] Failed after {self.max_retries} retries: {url} — {last_reason}")
        return DownloadResult(
            url=url,
            path=None,
            success=False,
            failure_reason=last_reason,
        )

    # ── Private: helpers ──────────────────────────────────────────────────────

    def _dest_path(self, url: str, source_name: str) -> Path:
        """Deterministic file path from URL hash."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        shard = url_hash[:2]
        ext = Path(urlparse(url).path).suffix.lower()
        if ext not in VALID_EXTENSIONS:
            ext = ".jpg"
        return self.base_path / "images" / "raw" / source_name / shard / f"{url_hash}{ext}"

    def _rate_limit(self, url: str) -> None:
        """Sleep if necessary to respect per-domain rate limit."""
        if self.rate_limit_seconds <= 0:
            return
        domain = urlparse(url).netloc
        now = time.monotonic()
        last = self._domain_last_request.get(domain, 0.0)
        gap = now - last
        if gap < self.rate_limit_seconds:
            time.sleep(self.rate_limit_seconds - gap)
        self._domain_last_request[domain] = time.monotonic()

    @staticmethod
    def _is_valid_content_type(content_type: str) -> bool:
        if content_type in VALID_CONTENT_TYPES:
            return True
        # Accept anything starting with "image/"
        return content_type.startswith("image/")

    @staticmethod
    def _cleanup(path: Path | None, cached: bool) -> None:
        """Remove a freshly-downloaded file (not cached) on pair failure."""
        if path is None or cached:
            return
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"[downloader] Cleaned up partial download: {path.name}")
        except OSError:
            pass
