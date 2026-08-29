"""Integration tests for KoreanSourcesSource."""

import pytest
from crawler.sources.korean_sources import KoreanSourcesSource
from crawler.base import SourceConfig, RenderingMethod, ConsentTier


SEARCH_HTML = """
<html><body>
  <a class="api_txt_lines" href="https://blog.naver.com/clinic123/222">보톡스 시술 전후 사진</a>
  <a class="api_txt_lines" href="https://blog.naver.com/user456/333">필러 전후 결과</a>
  <div class="total_wrap">
    <a href="https://blog.naver.com/doc789/444">입술 필러 전후 사진 공유</a>
  </div>
  <a class="btn_next" href="/search.naver?where=blog&query=%EB%B3%B4%ED%86%A1%EC%8A%A4&start=11">다음</a>
</body></html>
"""

BLOG_POST_ALT_HTML = """
<html><body>
  <div class="se-main-container">
    <img src="https://postfiles.pstatic.net/before.jpg" alt="시술전">
    <img src="https://postfiles.pstatic.net/after.jpg" alt="시술후">
  </div>
</body></html>
"""

BLOG_POST_CAPTION_HTML = """
<html><body>
  <div class="post-view">
    <img src="https://postfiles.pstatic.net/img1.jpg">
    <p>시술전 사진입니다</p>
    <img src="https://postfiles.pstatic.net/img2.jpg">
    <p>시술후 결과입니다</p>
  </div>
</body></html>
"""

BLOG_POST_FALLBACK_HTML = """
<html><body>
  <title>보톡스 전후 사진 공유</title>
  <div class="post-view">
    <img src="https://postfiles.pstatic.net/a.jpg">
    <img src="https://postfiles.pstatic.net/b.jpg">
    <img src="https://postfiles.pstatic.net/c.jpg">
    <img src="https://postfiles.pstatic.net/d.jpg">
  </div>
</body></html>
"""


@pytest.fixture
def source():
    cfg = SourceConfig(
        name="korean_test",
        enabled=True,
        rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0,
        language="ko",
        consent_tier=ConsentTier.LIKELY,
        base_urls=[],
    )
    return KoreanSourcesSource(cfg)


def test_search_results_queues_blog_posts(source):
    result = source.extract_pairs_from_page(
        SEARCH_HTML, "https://search.naver.com/search.naver?where=blog&query=test"
    )
    assert result == []
    queued = source._pop_queued_pages()
    blog_urls = [u for u in queued if "m.blog.naver.com" in u]
    assert len(blog_urls) >= 2


def test_search_results_converts_to_mobile_url(source):
    source.extract_pairs_from_page(
        SEARCH_HTML, "https://search.naver.com/search.naver?where=blog&query=test"
    )
    queued = source._pop_queued_pages()
    desktop_urls = [u for u in queued if "blog.naver.com" in u and "m.blog.naver.com" not in u]
    assert len(desktop_urls) == 0  # all should be mobile


def test_search_results_queues_pagination(source):
    source.extract_pairs_from_page(
        SEARCH_HTML, "https://search.naver.com/search.naver?where=blog&query=test"
    )
    queued = source._pop_queued_pages()
    next_pages = [u for u in queued if "start=11" in u]
    assert len(next_pages) == 1


def test_blog_post_alt_text_extraction(source):
    pairs = source.extract_pairs_from_page(
        BLOG_POST_ALT_HTML, "https://m.blog.naver.com/clinic123/222"
    )
    assert len(pairs) >= 1
    assert "before" in pairs[0].before_url or "postfiles" in pairs[0].before_url


def test_naver_url_resolver_strips_cdm_params(source):
    from bs4 import BeautifulSoup
    img_html = '<img src="https://postfiles.pstatic.net/photo.jpg?type=w966">'
    img = BeautifulSoup(img_html, "lxml").find("img")
    result = source._resolve_naver_img_src(img)
    assert "?type=w966" not in result


def test_to_mobile_url_conversion(source):
    desktop = "https://blog.naver.com/user/123"
    mobile = source._to_mobile_url(desktop)
    assert "m.blog.naver.com" in mobile
    assert "blog.naver.com" in mobile


def test_pair_metadata(source):
    pairs = source.extract_pairs_from_page(
        BLOG_POST_ALT_HTML, "https://m.blog.naver.com/clinic123/222"
    )
    if pairs:
        assert pairs[0].source_name == "korean_test"
        assert pairs[0].language == "ko"
