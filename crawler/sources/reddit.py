"""Reddit public subreddit crawler — uses the JSON API.

Avoids old.reddit.com (which blocks all bots via robots.txt) by hitting
/r/{sub}.json directly on www.reddit.com.  The JSON API is publicly
accessible, returns gallery metadata including full-resolution image URLs,
and supports cursor-based pagination via the `after` token.
"""

from __future__ import annotations

import json
import re
from typing import Iterator
from urllib.parse import urlparse

from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


TITLE_KEYWORDS = frozenset([
    "before", "after", "results", "progress", "transformation",
    "botox", "filler", "juvederm", "restylane", "dysport", "sculptra",
    "rhinoplasty", "rhino", "blepharoplasty", "facelift",
    "1 month", "2 months", "3 months", "6 months", "1 year",
    "weeks later", "day 1", "day 14", "week 1", "week 2",
])

IMAGE_URL_RE = re.compile(r"\.(jpe?g|png|webp)(\?.*)?$", re.IGNORECASE)

# Maps subreddit name (lowercase) → treatment category stored in metadata
SUBREDDIT_TREATMENT: dict[str, str] = {
    "rhinoplasty":            "rhinoplasty",
    "jawsurgery":             "jawline_filler",
    "facialplasticsurgery":   "facelift",
    "eyelidsurgery":          "blepharoplasty",
    "fillers":                "dermal_filler",
    "injectables":            "dermal_filler",
    "botox":                  "botox",
    "plasticsurgery":         None,           # mixed — no single label
    "plasticsurgeryrecovery": None,
    "skincareaddiction":      None,
}

# Crawl hot + top-all for each subreddit to maximise coverage
LISTING_SUFFIXES = [
    ".json?limit=100",
    "/top.json?t=all&limit=100",
]

# Max pagination pages per listing (100 posts each)
MAX_PAGES = 5


class RedditSource(BaseSource):
    """
    Crawls aesthetic subreddits via the Reddit JSON API.

    Stage 1: Listing pages (.json) — filter posts by title keywords,
             queue gallery posts for pair extraction.
    Stage 2: Gallery posts embedded in listing JSON — extract image pairs
             directly from media_metadata without a second HTTP request.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for base_url in self.config.base_urls:
            clean = base_url.rstrip("/")
            # Strip any existing path suffix so we always start from sub root
            # e.g. https://www.reddit.com/r/rhinoplasty
            for suffix in LISTING_SUFFIXES:
                yield f"{clean}{suffix}"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        try:
            data = json.loads(html)
        except (json.JSONDecodeError, ValueError):
            logger.warning(f"[reddit] Non-JSON response from {page_url}")
            return []

        listing = data.get("data", {})
        posts = listing.get("children", [])
        after_token = listing.get("after")

        subreddit = self._subreddit_from_url(page_url)
        treatment = SUBREDDIT_TREATMENT.get(subreddit.lower()) if subreddit else None

        pairs: list[RawImagePair] = []
        for wrapper in posts:
            post = wrapper.get("data", {})
            title = post.get("title", "").lower()
            if not any(kw in title for kw in TITLE_KEYWORDS):
                continue
            pairs.extend(self._pairs_from_post(post, treatment))

        # Paginate: queue next page if cursor exists and we haven't gone too deep
        page_num = self._page_number(page_url)
        if after_token and page_num < MAX_PAGES:
            base = page_url.split("&after=")[0]
            self._queue_page(f"{base}&after={after_token}")

        logger.debug(f"[reddit] {len(pairs)} pairs from {page_url}")
        return pairs

    # ── Post extraction ───────────────────────────────────────────────────────

    def _pairs_from_post(self, post: dict, treatment: str | None) -> list[RawImagePair]:
        source_url = f"https://www.reddit.com{post.get('permalink', '')}"

        # Reddit gallery post: images stored in media_metadata
        if post.get("is_gallery") and post.get("media_metadata"):
            images = self._gallery_images(post["media_metadata"])
            if len(images) >= 2:
                mid = len(images) // 2
                return [
                    self._make_pair(b, a, source_url, "gallery", treatment)
                    for b, a in zip(images[:mid], images[mid:])
                ]

        return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _gallery_images(media_metadata: dict) -> list[str]:
        """Return full-resolution URLs from a Reddit gallery's media_metadata."""
        urls: list[str] = []
        for item in media_metadata.values():
            if item.get("status") != "valid":
                continue
            src = item.get("s", {})
            url = src.get("u", "") or src.get("gif", "")
            url = url.replace("&amp;", "&")  # Reddit escapes & in JSON
            if url and IMAGE_URL_RE.search(url.split("?")[0]):
                urls.append(url)
        return urls

    @staticmethod
    def _subreddit_from_url(url: str) -> str:
        """Extract subreddit name from a .json listing URL."""
        path = urlparse(url).path  # e.g. /r/rhinoplasty/top.json
        match = re.search(r"/r/([^/]+)", path)
        return match.group(1) if match else ""

    @staticmethod
    def _page_number(url: str) -> int:
        """Count how many &after= params are chained (proxy for page depth)."""
        return url.count("after=")

    def _make_pair(
        self,
        before_url: str,
        after_url: str,
        source_url: str,
        method: str,
        treatment: str | None = None,
    ) -> RawImagePair:
        metadata: dict = {"extraction_method": method}
        if treatment:
            metadata["treatment_category"] = treatment
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata=metadata,
        )
