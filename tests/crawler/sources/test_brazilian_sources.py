"""Integration tests for BrazilianSourcesSource."""

import pytest
from crawler.sources.brazilian_sources import BrazilianSourcesSource
from crawler.base import SourceConfig, RenderingMethod, ConsentTier


BING_RESULTS_HTML = """
<html><body>
  <li class="b_algo">
    <h2><a href="https://clinicaestetica.com.br/botox-antes-depois">Botox antes e depois</a></h2>
  </li>
  <li class="b_algo">
    <h2><a href="https://clinicaxyz.com.br/resultados">Resultados tratamentos estéticos</a></h2>
  </li>
  <li class="b_algo">
    <h2><a href="https://facebook.com/clinic">Clinic Facebook</a></h2>
  </li>
</body></html>
"""

CLINIC_DATA_ATTR_HTML = """
<html><body>
  <div data-before="https://clinica.com/img/before1.jpg"
       data-after="https://clinica.com/img/after1.jpg"></div>
</body></html>
"""

CLINIC_ALT_TEXT_HTML = """
<html><body>
  <img src="https://clinica.com/antes.jpg" alt="antes do procedimento">
  <img src="https://clinica.com/depois.jpg" alt="depois da aplicação">
</body></html>
"""

CLINIC_FIGURE_HTML = """
<html><body>
  <figure>
    <img src="https://clinica.com/a.jpg">
    <img src="https://clinica.com/b.jpg">
    <figcaption>Antes e depois do preenchimento labial</figcaption>
  </figure>
</body></html>
"""

CLINIC_HEADING_HTML = """
<html><body>
  <h2>Antes e Depois</h2>
  <div>
    <img src="https://clinica.com/a.jpg">
    <img src="https://clinica.com/b.jpg">
  </div>
</body></html>
"""

CLINIC_GALLERY_LINK_HTML = """
<html><body>
  <a href="/antes-depois-botox">Ver galeria antes e depois</a>
  <a href="/resultados-preenchimento">Resultados</a>
  <a href="/about">Sobre nós</a>
</body></html>
"""


@pytest.fixture
def source():
    cfg = SourceConfig(
        name="brazilian_test",
        enabled=True,
        rendering=RenderingMethod.STATIC,
        rate_limit_seconds=0,
        language="pt-BR",
        consent_tier=ConsentTier.LIKELY,
        base_urls=[],
    )
    return BrazilianSourcesSource(cfg)


def test_bing_search_queues_clinic_pages(source):
    result = source.extract_pairs_from_page(
        BING_RESULTS_HTML, "https://www.bing.com/search?q=botox+antes+e+depois&setlang=pt-BR"
    )
    assert result == []
    queued = source._pop_queued_pages()
    assert any("clinicaestetica.com.br" in u for u in queued)


def test_bing_search_skips_social_media(source):
    source.extract_pairs_from_page(
        BING_RESULTS_HTML, "https://www.bing.com/search?q=botox+antes+e+depois&setlang=pt-BR"
    )
    queued = source._pop_queued_pages()
    social = [u for u in queued if "facebook.com" in u]
    assert len(social) == 0


def test_data_attr_extraction(source):
    pairs = source.extract_pairs_from_page(
        CLINIC_DATA_ATTR_HTML, "https://clinica.com.br/botox"
    )
    assert len(pairs) == 1
    assert pairs[0].before_url == "https://clinica.com/img/before1.jpg"
    assert pairs[0].after_url == "https://clinica.com/img/after1.jpg"
    assert pairs[0].metadata["extraction_method"] == "data_attr"


def test_alt_text_pt_extraction(source):
    pairs = source.extract_pairs_from_page(
        CLINIC_ALT_TEXT_HTML, "https://clinica.com.br/botox"
    )
    assert len(pairs) == 1
    assert "antes" in pairs[0].before_url or "antes" in pairs[0].metadata.get("extraction_method", "")


def test_figure_caption_extraction(source):
    pairs = source.extract_pairs_from_page(
        CLINIC_FIGURE_HTML, "https://clinica.com.br/preenchimento"
    )
    assert len(pairs) >= 1


def test_section_heading_extraction(source):
    pairs = source.extract_pairs_from_page(
        CLINIC_HEADING_HTML, "https://clinica.com.br/resultados"
    )
    assert len(pairs) >= 1


def test_gallery_links_queued(source):
    source.extract_pairs_from_page(
        CLINIC_GALLERY_LINK_HTML, "https://clinica.com.br/"
    )
    queued = source._pop_queued_pages()
    assert any("antes" in u or "resultados" in u for u in queued)
