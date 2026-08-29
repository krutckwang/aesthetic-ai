"""Integration tests for RedditSource."""

import pytest
from unittest.mock import MagicMock, patch
from crawler.sources.reddit import RedditSource
from crawler.base import SourceConfig, RenderingMethod, ConsentTier


LISTING_HTML = """
<html><body>
  <div class="thing link" data-url="/r/PlasticSurgery/comments/abc/before_and_after_botox/"
       data-permalink="/r/PlasticSurgery/comments/abc/before_and_after_botox/">
    <a class="title" href="/r/PlasticSurgery/comments/abc/">Before and after botox results</a>
  </div>
  <div class="thing link" data-url="https://i.redd.it/photo.jpg"
       data-permalink="/r/PlasticSurgery/comments/xyz/my_transformation/">
    <a class="title" href="/r/PlasticSurgery/comments/xyz/">My transformation after filler</a>
  </div>
  <div class="thing link" data-url="/r/PlasticSurgery/comments/zzz/unrelated/"
       data-permalink="/r/PlasticSurgery/comments/zzz/unrelated/">
    <a class="title" href="/r/PlasticSurgery/comments/zzz/">Completely unrelated post</a>
  </div>
  <span class="next-button"><a href="https://old.reddit.com/r/PlasticSurgery/?after=t3_abc">next</a></span>
</body></html>
"""

POST_GALLERY_HTML = """
<html><body>
  <a href="https://i.redd.it/before_photo.jpg">Before</a>
  <a href="https://i.redd.it/after_photo.jpg">After</a>
  <img src="https://i.redd.it/before_photo.jpg" alt="before treatment">
  <img src="https://i.redd.it/after_photo.jpg" alt="after treatment">
</body></html>
"""

POST_ALT_TEXT_HTML = """
<html><body>
  <img src="https://i.redd.it/before_filler.jpg" alt="Before filler">
  <img src="https://i.redd.it/after_filler.jpg" alt="After filler 2 weeks">
</body></html>
"""


@pytest.fixture
def source():
    cfg = SourceConfig(
        name="reddit_test",
        enabled=True,
        rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0,
        language="en",
        consent_tier=ConsentTier.LIKELY,
        base_urls=["https://old.reddit.com/r/PlasticSurgery/"],
    )
    return RedditSource(cfg)


def test_listing_page_queues_candidate_posts(source):
    result = source.extract_pairs_from_page(LISTING_HTML, "https://old.reddit.com/r/PlasticSurgery/")
    assert result == []  # listing page yields no pairs directly
    queued = source._pop_queued_pages()
    assert len(queued) >= 2  # at least 2 candidate posts + pagination


def test_listing_page_queues_next_page(source):
    source.extract_pairs_from_page(LISTING_HTML, "https://old.reddit.com/r/PlasticSurgery/")
    queued = source._pop_queued_pages()
    next_pages = [u for u in queued if "after=t3_abc" in u]
    assert len(next_pages) == 1


def test_listing_page_ignores_unrelated_posts(source):
    source.extract_pairs_from_page(LISTING_HTML, "https://old.reddit.com/r/PlasticSurgery/")
    queued = source._pop_queued_pages()
    unrelated = [u for u in queued if "unrelated" in u]
    assert len(unrelated) == 0


def test_post_gallery_extracts_pair(source):
    pairs = source.extract_pairs_from_page(
        POST_GALLERY_HTML, "https://old.reddit.com/r/PlasticSurgery/comments/abc/"
    )
    assert len(pairs) >= 1
    assert all(p.before_url for p in pairs)
    assert all(p.after_url for p in pairs)


def test_post_alt_text_pair_extraction(source):
    pairs = source.extract_pairs_from_page(
        POST_ALT_TEXT_HTML, "https://old.reddit.com/r/PlasticSurgery/comments/xyz/"
    )
    assert len(pairs) >= 1
    assert "before" in pairs[0].before_url.lower() or "before" in pairs[0].metadata.get("extraction_method", "").lower()


def test_pair_has_correct_metadata(source):
    pairs = source.extract_pairs_from_page(
        POST_ALT_TEXT_HTML, "https://old.reddit.com/r/PlasticSurgery/comments/xyz/"
    )
    assert pairs[0].source_name == "reddit_test"
    assert pairs[0].consent_tier == ConsentTier.LIKELY
    assert pairs[0].language == "en"


def test_is_listing_page_detection(source):
    assert source._is_listing_page("https://old.reddit.com/r/PlasticSurgery/") is True
    assert source._is_listing_page("https://old.reddit.com/r/PlasticSurgery/top/") is True
    assert source._is_listing_page("https://old.reddit.com/r/PlasticSurgery/comments/abc/title/") is False
