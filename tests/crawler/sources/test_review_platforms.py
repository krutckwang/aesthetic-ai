"""Integration tests for ReviewPlatformsSource."""

import pytest
from crawler.sources.review_platforms import ReviewPlatformsSource
from crawler.base import SourceConfig, RenderingMethod, ConsentTier


HEALTHGRADES_HTML = """
<html><body>
  <div class="before-after-photos">
    <img src="https://hg.com/img/before1.jpg">
    <img src="https://hg.com/img/after1.jpg">
  </div>
  <a href="https://www.healthgrades.com/provider/dr-smith/photos">Photos</a>
</body></html>
"""

HEALTHGRADES_GENERIC_HTML = """
<html><body>
  <img src="https://hg.com/before_treatment.jpg" alt="before botox">
  <img src="https://hg.com/after_treatment.jpg" alt="after botox results">
</body></html>
"""

YELP_HTML = """
<html><body>
  <div class="beforeAfterSection" data-photo-type="before-after">
    <img src="https://yelp.com/img/b.jpg">
    <img src="https://yelp.com/img/a.jpg">
  </div>
  <a href="https://www.yelp.com/biz/spa-nyc/photos">Photos</a>
</body></html>
"""

RATEMDS_HTML = """
<html><body>
  <div class="photo-gallery">
    <img src="https://ratemds.com/img/b.jpg" alt="before treatment">
    <img src="https://ratemds.com/img/a.jpg" alt="after treatment results">
  </div>
</body></html>
"""


@pytest.fixture
def source():
    cfg = SourceConfig(
        name="review_test",
        enabled=True,
        rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0,
        language="en",
        consent_tier=ConsentTier.LIKELY,
        base_urls=[],
    )
    return ReviewPlatformsSource(cfg)


def test_healthgrades_section_extraction(source):
    pairs = source.extract_pairs_from_page(
        HEALTHGRADES_HTML, "https://www.healthgrades.com/provider/dr-smith"
    )
    assert len(pairs) >= 1
    assert "hg.com/img" in pairs[0].before_url
    assert pairs[0].metadata["extraction_method"] == "healthgrades_section"


def test_healthgrades_queues_photos_page(source):
    source.extract_pairs_from_page(
        HEALTHGRADES_HTML, "https://www.healthgrades.com/provider/dr-smith"
    )
    queued = source._pop_queued_pages()
    assert any("photos" in u for u in queued)


def test_healthgrades_generic_fallback(source):
    pairs = source.extract_pairs_from_page(
        HEALTHGRADES_GENERIC_HTML, "https://www.healthgrades.com/provider/dr-jones/photos"
    )
    assert len(pairs) >= 1


def test_yelp_before_after_section(source):
    pairs = source.extract_pairs_from_page(
        YELP_HTML, "https://www.yelp.com/biz/spa-nyc"
    )
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "yelp_before_after"


def test_yelp_queues_photos_page(source):
    source.extract_pairs_from_page(YELP_HTML, "https://www.yelp.com/biz/spa-nyc")
    queued = source._pop_queued_pages()
    assert any("photos" in u for u in queued)


def test_ratemds_gallery_extraction(source):
    pairs = source.extract_pairs_from_page(
        RATEMDS_HTML, "https://www.ratemds.com/doctor-ratings/dr-abc"
    )
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "ratemds_gallery"


def test_pair_source_name_and_tier(source):
    pairs = source.extract_pairs_from_page(
        RATEMDS_HTML, "https://www.ratemds.com/doctor-ratings/dr-abc"
    )
    if pairs:
        assert pairs[0].source_name == "review_test"
        assert pairs[0].consent_tier == ConsentTier.LIKELY
