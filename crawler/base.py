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
        self._client = httpx.Client(
            headers={"User-Agent": self._load_global_user_agent()},
            timeout=30,
            follow_redirects=True,
        )

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
        Respects robots.txt, rate limits, and handles transient failures.
        """
        if not self.config.enabled:
            logger.info(f"[{self.config.name}] Source disabled in config — skipping.")
            return

        for page_url in self.iter_page_urls():
            if not self._is_allowed_by_robots(page_url):
                logger.warning(f"[{self.config.name}] robots.txt disallows: {page_url}")
                continue

            self._rate_limit(page_url)

            html = self._fetch(page_url)
            if html is None:
                continue

            pairs = self.extract_pairs_from_page(html, page_url)
            logger.debug(f"[{self.config.name}] {page_url} → {len(pairs)} candidate pairs")

            yield from pairs

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
        Fetch page HTML. Retries with exponential backoff on 429/503.
        Returns None on unrecoverable failure.
        """
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
