"""Reddit public subreddit crawler for aesthetic before/after posts."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


# Title keywords that signal a before/after post
TITLE_KEYWORDS = frozenset([
    "before", "after", "results", "progress", "transformation",
    "botox", "filler", "juvederm", "restylane", "dysport",
    "1 month", "2 months", "3 months", "6 months", "1 year",
    "weeks later", "day 1", "day 14",
])

# Image URL patterns that indicate direct images
IMAGE_URL_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)
REDDIT_IMAGE_HOSTS = frozenset(["i.redd.it", "i.imgur.com", "preview.redd.it"])


class RedditSource(BaseSource):
    """
    Crawls aesthetic subreddits (r/PlasticSurgery, r/Injectables, r/Botox).
    Uses old.reddit.com for static HTML listing pages.
    Post pages are queued for second-stage extraction of gallery images.
    Consent tier: 2 — users self-post but no explicit release language.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for base_url in self.config.base_urls:
            old_url = base_url.replace("www.reddit.com", "old.reddit.com")
            # Yield up to 10 paginated listing pages per subreddit
            for _ in range(10):
                yield old_url
                break  # pagination token injected via _next_page_token() after first fetch

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")

        # Listing page: queue candidate post URLs for second-stage extraction
        if self._is_listing_page(page_url):
            return self._process_listing_page(soup, page_url)

        # Post page: extract before/after images from the post itself
        return self._process_post_page(soup, page_url)

    # ── Listing page ──────────────────────────────────────────────────────────

    def _process_listing_page(self, soup: BeautifulSoup, page_url: str) -> list[RawImagePair]:
        """Find candidate posts and queue their URLs for second-stage crawl."""
        queued = 0
        for post in soup.select(".thing.link"):
            title_el = post.select_one("a.title")
            if not title_el:
                continue
            title_text = title_el.get_text(strip=True).lower()

            if not any(kw in title_text for kw in TITLE_KEYWORDS):
                continue

            post_url = post.get("data-url", "")
            permalink = post.get("data-permalink", "")

            # Direct image links: build a synthetic pair from post data-url
            if post_url and IMAGE_URL_RE.search(post_url):
                # Single image post — can't form a pair from listing page alone
                # Queue the comments page to look for a paired image
                if permalink:
                    full_permalink = urljoin("https://old.reddit.com", permalink)
                    self._queue_page(full_permalink)
                    queued += 1
            elif permalink:
                # Gallery or self-post — queue the post page
                full_permalink = urljoin("https://old.reddit.com", permalink)
                self._queue_page(full_permalink)
                queued += 1

        # Pagination: follow "next" link on listing pages
        next_link = soup.select_one("span.next-button a")
        if next_link:
            next_url = next_link.get("href", "")
            if next_url:
                self._queue_page(next_url)

        logger.debug(f"[reddit] Queued {queued} candidate posts from {page_url}")
        return []

    # ── Post page ─────────────────────────────────────────────────────────────

    def _process_post_page(self, soup: BeautifulSoup, page_url: str) -> list[RawImagePair]:
        """Extract before/after image pairs from a Reddit post page."""
        pairs: list[RawImagePair] = []

        # Strategy 1: Reddit gallery (multiple images in one post)
        gallery_imgs = self._extract_gallery_images(soup, page_url)
        if len(gallery_imgs) >= 2:
            # Pair first half as before, second half as after
            mid = len(gallery_imgs) // 2
            for b, a in zip(gallery_imgs[:mid], gallery_imgs[mid:]):
                pairs.append(self._make_pair(b, a, page_url, source="gallery"))

        # Strategy 2: Two direct image links with before/after alt text or title
        if not pairs:
            pairs = self._extract_titled_image_pairs(soup, page_url)

        logger.debug(f"[reddit] {len(pairs)} pairs extracted from post {page_url}")
        return pairs

    def _extract_gallery_images(self, soup: BeautifulSoup, page_url: str) -> list[str]:
        """Extract image URLs from a Reddit gallery post."""
        urls: list[str] = []

        # Reddit gallery items are in <a> tags pointing to i.redd.it
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            parsed = urlparse(href)
            if parsed.netloc in REDDIT_IMAGE_HOSTS and IMAGE_URL_RE.search(href):
                if href not in urls:
                    urls.append(href)

        # Also check <img> tags with reddit CDN sources
        for img in soup.select("img[src]"):
            src = img.get("src", "")
            parsed = urlparse(src)
            if parsed.netloc in REDDIT_IMAGE_HOSTS and IMAGE_URL_RE.search(src):
                if src not in urls:
                    urls.append(src)

        return urls

    def _extract_titled_image_pairs(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Look for two images where title/alt text indicates before and after."""
        pairs: list[RawImagePair] = []
        before_url: str | None = None
        after_url: str | None = None

        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src = img.get("src", "")
            if not src or not IMAGE_URL_RE.search(src):
                continue
            if "before" in alt and before_url is None:
                before_url = src
            elif "after" in alt and after_url is None:
                after_url = src

        if before_url and after_url:
            pairs.append(self._make_pair(before_url, after_url, page_url, "titled_alt"))
        return pairs

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_listing_page(url: str) -> bool:
        path = urlparse(url).path.rstrip("/")
        # Listing pages end in /r/<subreddit> or /r/<subreddit>/top etc.
        return bool(re.match(r"^/r/[^/]+(/\w+)?$", path))

    def _make_pair(
        self, before_url: str, after_url: str, source_url: str, source: str = ""
    ) -> RawImagePair:
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={"extraction_method": source},
        )
