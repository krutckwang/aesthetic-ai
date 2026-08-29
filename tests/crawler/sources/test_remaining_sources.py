"""Integration tests for OpenAccessPubs, BeautyMedia, OpenDatasets,
AcademyPortals, ProfessionalSocieties, BrandGalleries, Pinterest."""

import json
import pytest
from pathlib import Path
from crawler.base import SourceConfig, RenderingMethod, ConsentTier
from crawler.sources.open_access_pubs import OpenAccessPubsSource
from crawler.sources.beauty_media import BeautyMediaSource
from crawler.sources.open_datasets import OpenDatasetsSource
from crawler.sources.academy_portals import AcademyPortalsSource
from crawler.sources.professional_societies import ProfessionalSocietiesSource
from crawler.sources.brand_galleries import BrandGalleriesSource
from crawler.sources.pinterest import PinterestSource


def make_config(name, consent_tier=ConsentTier.CONFIRMED, language="en"):
    return SourceConfig(
        name=name,
        enabled=True,
        rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0,
        language=language,
        consent_tier=consent_tier,
        base_urls=[],
    )


# ── OpenAccessPubs ────────────────────────────────────────────────────────────

PMC_ARTICLE_HTML = """
<html><body>
  <figure>
    <img src="https://pmc.ncbi.nlm.nih.gov/img/fig1a.jpg">
    <img src="https://pmc.ncbi.nlm.nih.gov/img/fig1b.jpg">
    <figcaption>Figure 1. Patient before (A) and after (B) treatment.</figcaption>
  </figure>
</body></html>
"""

PMC_SEARCH_HTML = """
<html><body>
  <a class="article-title" href="/pmc/articles/PMC123/">Botox treatment outcomes</a>
  <a href="/pmc/articles/PMC456/">Filler results study</a>
</body></html>
"""


def test_open_access_pubs_figure_caption():
    source = OpenAccessPubsSource(make_config("oap_test"))
    pairs = source.extract_pairs_from_page(PMC_ARTICLE_HTML, "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/")
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "figure_caption"


def test_open_access_pubs_search_queues_articles():
    source = OpenAccessPubsSource(make_config("oap_test"))
    result = source.extract_pairs_from_page(
        PMC_SEARCH_HTML, "https://www.ncbi.nlm.nih.gov/pmc/search/?query=botox"
    )
    assert result == []
    queued = source._pop_queued_pages()
    assert any("PMC" in u for u in queued)


# ── BeautyMedia ───────────────────────────────────────────────────────────────

BEAUTY_LISTING_HTML = """
<html><body>
  <a href="/beauty/botox-before-after-transformation">Botox Before and After</a>
  <a href="/beauty/best-moisturizers">Best Moisturizers</a>
  <a href="/gallery/lip-filler-results">Lip filler results gallery</a>
  <a href="/beauty/skincare-tips">Skincare Tips</a>
  <a href="/beauty/retinol-guide">Retinol Guide</a>
  <a href="/beauty/before-after-filler">Before and After Filler Photos</a>
</body></html>
"""

BEAUTY_ARTICLE_HTML = """
<html><body>
  <figure>
    <img src="https://allure.com/img/before.jpg">
    <img src="https://allure.com/img/after.jpg">
    <figcaption>Before and after lip filler treatment</figcaption>
  </figure>
</body></html>
"""

BEAUTY_SLIDESHOW_HTML = """
<html><body>
  <div class="slide">
    <img src="https://allure.com/slide1_before.jpg" alt="before">
  </div>
  <div class="slide">
    <img src="https://allure.com/slide2_after.jpg" alt="after treatment result">
  </div>
</body></html>
"""


def test_beauty_media_listing_queues_articles():
    source = BeautyMediaSource(make_config("beauty_test"))
    result = source.extract_pairs_from_page(BEAUTY_LISTING_HTML, "https://www.allure.com/")
    queued = source._pop_queued_pages()
    gallery_links = [u for u in queued if "before-after" in u or "gallery" in u or "results" in u]
    assert len(gallery_links) >= 1


def test_beauty_media_figure_caption_pair():
    source = BeautyMediaSource(make_config("beauty_test"))
    pairs = source.extract_pairs_from_page(BEAUTY_ARTICLE_HTML, "https://www.allure.com/article/botox")
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "figure_caption"


def test_beauty_media_slideshow_pair():
    source = BeautyMediaSource(make_config("beauty_test"))
    pairs = source.extract_pairs_from_page(BEAUTY_SLIDESHOW_HTML, "https://www.allure.com/slideshow/botox")
    assert len(pairs) >= 1


# ── OpenDatasets ──────────────────────────────────────────────────────────────

def test_open_datasets_hf_card_processed():
    source = OpenDatasetsSource(make_config("datasets_test"))
    html = "<html><body><a href='/resolve/main/sample.jpg'>Sample</a></body></html>"
    result = source.extract_pairs_from_page(html, "https://huggingface.co/datasets/test")
    assert isinstance(result, list)  # no pairs from card page, but no crash


def test_open_datasets_local_split_dirs(tmp_path):
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    (before_dir / "patient01.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
    (after_dir / "patient01.jpg").write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

    cfg = SourceConfig(
        name="datasets_test", enabled=True, rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0, language="en", consent_tier=ConsentTier.CONFIRMED,
        base_urls=[], extra={"local_path": str(tmp_path)},
    )
    source = OpenDatasetsSource(cfg)
    pairs = list(source._scan_local_dataset(tmp_path))
    assert len(pairs) == 1
    assert "patient01" in pairs[0].before_url


def test_open_datasets_local_pairs_dir(tmp_path):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "001_before.jpg").write_bytes(b"\x00" * 10)
    (pairs_dir / "001_after.jpg").write_bytes(b"\x00" * 10)

    pairs = list(OpenDatasetsSource(make_config("d"))._pair_from_pairs_dir(pairs_dir))
    assert len(pairs) == 1
    assert "before" in pairs[0].before_url


# ── AcademyPortals ────────────────────────────────────────────────────────────

ACADEMY_HTML = """
<html><body>
  <div class="before-after-photos">
    <img src="https://asps.org/img/before1.jpg">
    <img src="https://asps.org/img/after1.jpg">
  </div>
  <a href="/case-study/lip-augmentation">Case study: lip augmentation</a>
  <a href="/photo-gallery/rhinoplasty">Photo gallery</a>
</body></html>
"""


def test_academy_portals_section_extraction():
    source = AcademyPortalsSource(make_config("academy_test"))
    pairs = source.extract_pairs_from_page(ACADEMY_HTML, "https://www.plasticsurgery.org/patient-safety")
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "section_container"


def test_academy_portals_queues_case_study_links():
    source = AcademyPortalsSource(make_config("academy_test"))
    source.extract_pairs_from_page(ACADEMY_HTML, "https://www.plasticsurgery.org/patient-safety")
    queued = source._pop_queued_pages()
    assert any("case-study" in u or "gallery" in u for u in queued)


# ── ProfessionalSocieties ─────────────────────────────────────────────────────

SOCIETY_HTML = """
<html><body>
  <div class="patient-gallery">
    <img src="https://plasticsurgery.org/img/before.jpg" alt="before rhinoplasty">
    <img src="https://plasticsurgery.org/img/after.jpg" alt="after rhinoplasty result">
  </div>
  <a href="/photo-gallery/before-after">Before and after photos</a>
  <a class="next" href="/gallery?page=2">Next</a>
</body></html>
"""


def test_professional_societies_container_extraction():
    source = ProfessionalSocietiesSource(make_config("societies_test"))
    pairs = source.extract_pairs_from_page(SOCIETY_HTML, "https://www.plasticsurgery.org/procedures")
    assert len(pairs) >= 1


def test_professional_societies_queues_gallery_links():
    source = ProfessionalSocietiesSource(make_config("societies_test"))
    source.extract_pairs_from_page(SOCIETY_HTML, "https://www.plasticsurgery.org/procedures")
    queued = source._pop_queued_pages()
    assert any("gallery" in u or "before-after" in u for u in queued)


def test_professional_societies_queues_next_page():
    source = ProfessionalSocietiesSource(make_config("societies_test"))
    source.extract_pairs_from_page(SOCIETY_HTML, "https://www.plasticsurgery.org/gallery")
    queued = source._pop_queued_pages()
    assert any("page=2" in u for u in queued)


# ── BrandGalleries ────────────────────────────────────────────────────────────

BRAND_HTML = """
<html><body>
  <div class="before-after-result">
    <img src="https://juvederm.com/img/before1.jpg">
    <img src="https://juvederm.com/img/after1.jpg">
  </div>
  <a href="/real-results/lips">See real lip results</a>
  <a href="/before-after-photos">Before &amp; After</a>
</body></html>
"""

BRAND_SLIDER_HTML = """
<html><body>
  <div class="comparison-slider">
    <img src="https://botox.com/slider_before.jpg">
    <img src="https://botox.com/slider_after.jpg">
  </div>
</body></html>
"""


def test_brand_galleries_container_extraction():
    source = BrandGalleriesSource(make_config("brand_test", consent_tier=ConsentTier.LIKELY))
    pairs = source.extract_pairs_from_page(BRAND_HTML, "https://www.juvederm.com/treatments")
    assert len(pairs) >= 1


def test_brand_galleries_slider_extraction():
    source = BrandGalleriesSource(make_config("brand_test", consent_tier=ConsentTier.LIKELY))
    pairs = source.extract_pairs_from_page(BRAND_SLIDER_HTML, "https://www.botox.com/results")
    assert len(pairs) >= 1
    assert pairs[0].metadata["extraction_method"] == "slider"


def test_brand_galleries_queues_gallery_links():
    source = BrandGalleriesSource(make_config("brand_test", consent_tier=ConsentTier.LIKELY))
    source.extract_pairs_from_page(BRAND_HTML, "https://www.juvederm.com/")
    queued = source._pop_queued_pages()
    assert any("real-results" in u or "before-after" in u for u in queued)


# ── Pinterest ─────────────────────────────────────────────────────────────────

PINTEREST_SCRIPT_HTML = """
<html><body>
  <script>
    var data = {
      "description": "botox before treatment",
      "image": "https://i.pinimg.com/736x/before_botox.jpg"
    };
  </script>
  <script>
    var pins = [
      {"description": "before lip filler", "originals": "https://i.pinimg.com/originals/b.jpg"},
      {"description": "after lip filler result", "originals": "https://i.pinimg.com/originals/a.jpg"}
    ];
  </script>
</body></html>
"""

PINTEREST_IMG_HTML = """
<html><body>
  <img src="https://i.pinimg.com/736x/before_botox.jpg" alt="before botox">
  <img src="https://i.pinimg.com/736x/after_botox.jpg" alt="after botox result">
</body></html>
"""


def test_pinterest_img_tag_extraction():
    cfg = SourceConfig(
        name="pinterest_test", enabled=True, rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0, language="en",
        consent_tier=ConsentTier.UNCERTAIN, base_urls=[],
    )
    source = PinterestSource(cfg)
    pairs = source.extract_pairs_from_page(PINTEREST_IMG_HTML, "https://www.pinterest.com/board/aesthetic/")
    assert len(pairs) >= 1
    assert pairs[0].consent_tier == ConsentTier.UNCERTAIN


def test_pinterest_inline_script_extraction():
    cfg = SourceConfig(
        name="pinterest_test", enabled=True, rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0, language="en",
        consent_tier=ConsentTier.UNCERTAIN, base_urls=[],
    )
    source = PinterestSource(cfg)
    # Should not crash even if no pairs found from scripts
    pairs = source.extract_pairs_from_page(PINTEREST_SCRIPT_HTML, "https://www.pinterest.com/search/")
    assert isinstance(pairs, list)
