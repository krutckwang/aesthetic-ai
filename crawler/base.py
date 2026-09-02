"""Abstract base class for all crawler source modules."""

from __future__ import annotations

import time
import urllib.robotparser
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import httpx
import yaml
from loguru import logger


class RenderingMethod(str, Enum):
    STATIC = "static"
    HEADLESS = "headless"


class ConsentTier(int, Enum):
    CONFIRMED = 1
    LIKELY = 2
    UNCERTAIN = 3


@dataclass
class RawImagePair:
    """A candidate before/after image pair extracted from a source page."""

    before_url: str
    after_url: str
    source_url: str
    source_name: str
    language: str
    consent_tier: ConsentTier
    metadata: dict = field(default_factory=dict)
    # Populated by validation worker — not by the source module
    layer1_score: float | None = None
    layer2_score: float | None = None
    layer3_score: float | None = None
    ordering_confidence: str | None = None  # HIGH | LOW | None


@dataclass
class SourceConfig:
    """Runtime config for a single source, loaded from crawler.yaml."""

    name: str
    enabled: bool
    rendering: RenderingMethod
    rate_limit_seconds: float
    language: str
    consent_tier: ConsentTier
    base_urls: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


class BaseSource(ABC):
    """
    Abstract base for all crawler source modules.

    Subclasses implement `iter_page_urls` and `extract_pairs_from_page`.
    The base class handles robots.txt compliance, rate limiting, retrying,
    and HTTP/headless fetching.
    """

    def __init__(self, config: SourceConfig) -> None:
        self.config = config
        self._robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_request_time: dict[str, float] = {}
        import os
        self._client = httpx.Client(
            headers={"User-Agent": self._load_global_user_agent()},
            timeout=30,
            follow_redirects=True,
            verify=os.getenv("CRAWLER_SSL_VERIFY", "1") != "0",
        )
        # Secondary pages discovered during extraction (two-stage crawl)
        self._discovered_pages: list[str] = []
        self._visited_urls: set[str] = set()

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def iter_page_urls(self) -> Iterator[str]:
        """Yield page URLs to crawl for this source."""

    @abstractmethod
    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        """Parse HTML and return candidate image pairs found on the page."""

    # ── Public crawl entry point ──────────────────────────────────────────────

    def crawl(self) -> Iterator[RawImagePair]:
        """
        Main crawl loop. Yields RawImagePair candidates.

        Supports two-stage crawling: sources can call `_queue_page(url)` inside
        `extract_pairs_from_page()` to schedule secondary pages (e.g. individual
        blog posts discovered from a listing page). The queue is drained after
        all primary pages are exhausted.
        """
        if not self.config.enabled:
            logger.info(f"[{self.config.name}] Source disabled in config — skipping.")
            return

        # Load primary URLs into a mutable queue
        page_queue: list[str] = list(self.iter_page_urls())
        self._visited_urls.clear()

        while page_queue:
            page_url = page_queue.pop(0)
            if not page_url or page_url in self._visited_urls:
                continue
            self._visited_urls.add(page_url)

            if self.config.extra.get("respect_robots_txt", True) and \
                    not self._is_allowed_by_robots(page_url):
                logger.warning(f"[{self.config.name}] robots.txt disallows: {page_url}")
                continue

            self._rate_limit(page_url)

            html = self._fetch(page_url)
            if html is None:
                continue

            pairs = self.extract_pairs_from_page(html, page_url)
            logger.debug(f"[{self.config.name}] {page_url} → {len(pairs)} candidate pairs")
            yield from pairs

            # Drain any secondary pages queued during extraction
            secondary = self._pop_queued_pages()
            for url in secondary:
                if url not in self._visited_urls:
                    page_queue.append(url)

    # ── Two-stage crawl helpers ───────────────────────────────────────────────

    def _queue_page(self, url: str) -> None:
        """
        Queue a secondary page URL to be crawled after the current page.
        Call this from extract_pairs_from_page() to schedule follow-up pages
        (e.g. individual post pages found on a listing page).
        """
        if url and url not in self._visited_urls:
            self._discovered_pages.append(url)

    def _pop_queued_pages(self) -> list[str]:
        """Drain and return all queued secondary page URLs."""
        pages = list(self._discovered_pages)
        self._discovered_pages.clear()
        return pages

    # ── robots.txt compliance ─────────────────────────────────────────────────

    def _is_allowed_by_robots(self, url: str) -> bool:
        """Return True if the URL is permitted by the domain's robots.txt."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        if base not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception as exc:
                logger.warning(f"[{self.config.name}] Could not read robots.txt for {base}: {exc}")
                rp = None
            self._robots_cache[base] = rp

        rp = self._robots_cache[base]
        if rp is None:
            return True  # if robots.txt unreadable, allow but log

        ua = self._load_global_user_agent()
        allowed = rp.can_fetch(ua, url)
        return allowed

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _rate_limit(self, url: str) -> None:
        """Block until the per-domain rate limit interval has elapsed."""
        domain = urlparse(url).netloc
        last = self._last_request_time.get(domain, 0.0)
        elapsed = time.monotonic() - last
        wait = self.config.rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_time[domain] = time.monotonic()

    # ── HTTP fetch ────────────────────────────────────────────────────────────

    def _fetch(self, url: str, retries: int = 4) -> str | None:
        """
        Fetch page HTML. Dispatches to headless Playwright for SPA sources.
        Retries with exponential backoff on 429/503.
        Returns None on unrecoverable failure.
        """
        if self.config.rendering == RenderingMethod.HEADLESS:
            return self._fetch_headless(url)

        delay = 2.0
        for attempt in range(retries):
            try:
                resp = self._client.get(url)
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code in (429, 503):
                    logger.warning(
                        f"[{self.config.name}] {resp.status_code} on {url} — "
                        f"backing off {delay}s (attempt {attempt + 1}/{retries})"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.warning(f"[{self.config.name}] HTTP {resp.status_code}: {url}")
                    return None
            except httpx.RequestError as exc:
                logger.error(f"[{self.config.name}] Request error on {url}: {exc}")
                time.sleep(delay)
                delay *= 2
        return None

    def _fetch_headless(self, url: str) -> str | None:
        """
        Fetch a JavaScript-rendered SPA page using Playwright headless Chromium.
        Waits for network idle before returning HTML.
        Falls back to static httpx fetch if Playwright is not installed.
        """
        try:
            from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        except ImportError:
            logger.warning(
                f"[{self.config.name}] Playwright not installed — "
                "falling back to static fetch (SPA content may be incomplete)."
            )
            return self._fetch_static(url)

        import os
        ssl_verify = os.getenv("CRAWLER_SSL_VERIFY", "1") != "0"
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1920, "height": 1080},
                    locale="en-US",
                    timezone_id="America/New_York",
                    extra_http_headers={
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    },
                    ignore_https_errors=not ssl_verify,
                )
                context.add_init_script("""
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    window.chrome = { runtime: {} };
                    Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US','en'] });
                """)
                page = context.new_page()
                page.goto(url, wait_until="networkidle", timeout=30_000)
                html = page.content()
                browser.close()
                return html
        except PWTimeout:
            logger.warning(f"[{self.config.name}] Playwright timeout for {url}")
            return None
        except Exception as exc:
            logger.error(f"[{self.config.name}] Playwright error for {url}: {exc}")
            return None

    def _fetch_static(self, url: str) -> str | None:
        """Plain httpx fetch — used as Playwright fallback."""
        try:
            resp = self._client.get(url)
            return resp.text if resp.status_code == 200 else None
        except httpx.RequestError:
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_global_user_agent() -> str:
        config_path = Path("configs/crawler.yaml")
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("global", {}).get("user_agent", "AestheticAI-Bot/0.1")
        return "AestheticAI-Bot/0.1"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r}, enabled={self.config.enabled})"
