"""Reddit public subreddit crawler for aesthetic before/after posts."""

from __future__ import annotations

from typing import Iterator

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


class RedditSource(BaseSource):
    """
    Crawls aesthetic subreddits (r/PlasticSurgery, r/Injectables, r/Botox).
    Uses Reddit's old.reddit.com interface for static HTML access.
    Consent tier: 2 — users self-post but no explicit release language.
    """

    def iter_page_urls(self) -> Iterator[str]:
        for base_url in self.config.base_urls:
            # Convert to old.reddit for static HTML
            old_url = base_url.replace("www.reddit.com", "old.reddit.com")
            after_token: str | None = None
            for _ in range(20):  # max 20 pages per subreddit
                url = f"{old_url}?after={after_token}" if after_token else old_url
                yield url
                # after_token updated during extraction — handled via metadata passthrough
                break  # pagination token extracted in extract_pairs_from_page; stub for now

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")
        pairs: list[RawImagePair] = []

        for post in soup.select(".thing.link"):
            title = post.select_one("a.title")
            if not title:
                continue
            title_text = title.get_text(strip=True).lower()

            # Only process posts with before/after signals in title
            if not any(kw in title_text for kw in ["before", "after", "results", "progress"]):
                continue

            # Stub: full image extraction requires expanding gallery posts
            # Implementation: follow post link, parse gallery/image embeds
            post_url = post.get("data-url", "")
            if post_url:
                metadata = {
                    "title": title.get_text(strip=True),
                    "subreddit": post.get("data-subreddit", ""),
                    "score": post.get("data-score", ""),
                }
                # Placeholder — real extraction follows post_url and parses images
                logger.debug(f"[reddit] Found candidate post: {post_url}")

        return pairs
