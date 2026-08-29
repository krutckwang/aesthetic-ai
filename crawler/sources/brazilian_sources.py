"""Brazilian Portuguese aesthetic content crawler."""

from __future__ import annotations

import re
from typing import Iterator
from urllib.parse import urljoin, urlparse, quote

from bs4 import BeautifulSoup
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


BEFORE_PT = frozenset([
    "antes", "antes do", "antes da", "pré-tratamento", "pre-tratamento",
    "antes do procedimento", "antes da aplicação",
])
AFTER_PT = frozenset([
    "depois", "após", "pós-tratamento", "pos-tratamento", "resultado",
    "resultado final", "depois do procedimento", "após aplicação",
])

IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif)(\?.*)?$", re.IGNORECASE)

# Search terms for Brazilian clinic sites via Bing (avoids aggressive Google blocking)
SEARCH_QUERIES_PT = [
    "botox antes e depois clínica",
    "preenchimento labial antes e depois resultado",
    "ácido hialurônico lábios antes e depois",
    "dysport antes e depois resultado",
    "juvederm preenchimento resultado antes depois",
    "preenchimento facial nasolabial antes e depois",
]


class BrazilianSourcesSource(BaseSource):
    """
    Two-stage crawler for Brazilian Portuguese aesthetic content.

    Stage 1: Bing image search for Brazilian clinic result pages (public HTML).
    Stage 2: Individual clinic/blog pages → extract antes/depois image pairs.

    Also processes any base_urls directly as clinic site seeds.
    Consent tier: 2 — clinic-posted marketing content.
    """

    def iter_page_urls(self) -> Iterator[str]:
        # Direct seed URLs from config (clinic sites)
        for url in self.config.base_urls:
            yield url

        # Bing search for Brazilian clinic result pages
        for query in SEARCH_QUERIES_PT:
            encoded = quote(query)
            yield f"https://www.bing.com/search?q={encoded}&setlang=pt-BR&cc=BR"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        soup = BeautifulSoup(html, "lxml")

        if "bing.com/search" in page_url:
            return self._process_search_results(soup, page_url)

        return self._process_clinic_page(soup, page_url)

    # ── Stage 1: search results ───────────────────────────────────────────────

    def _process_search_results(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Collect clinic page URLs from Bing search and queue them."""
        queued = 0
        seen_domains: set[str] = set()

        for result in soup.select("li.b_algo h2 a, div.b_algo h2 a"):
            href = result.get("href", "")
            if not href or not href.startswith("http"):
                continue
            domain = urlparse(href).netloc
            # Skip major aggregators — target clinic and blog sites
            if any(skip in domain for skip in [
                "google", "facebook", "instagram", "youtube",
                "wikipedia", "bing", "yahoo",
            ]):
                continue
            # One page per domain to avoid over-crawling any single site
            if domain not in seen_domains:
                seen_domains.add(domain)
                self._queue_page(href)
                queued += 1

        logger.debug(f"[brazilian_sources] Queued {queued} clinic pages from {page_url}")
        return []

    # ── Stage 2: clinic/blog pages ────────────────────────────────────────────

    def _process_clinic_page(
        self, soup: BeautifulSoup, page_url: str
    ) -> list[RawImagePair]:
        """Extract antes/depois pairs from a Brazilian clinic or blog page."""
        pairs: list[RawImagePair] = []
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        # Strategy 1: explicit data-before / data-after attributes
        for el in soup.select("[data-before][data-after]"):
            b = self._abs(el.get("data-before", ""), base)
            a = self._abs(el.get("data-after", ""), base)
            if b and a:
                pairs.append(self._make_pair(b, a, page_url, "data_attr"))

        # Strategy 2: alt text with Portuguese before/after keywords
        if not pairs:
            pairs = self._pair_by_alt(soup, page_url, base)

        # Strategy 3: figure pairs where caption contains antes/depois
        if not pairs:
            pairs = self._pair_by_figure_caption(soup, page_url, base)

        # Strategy 4: adjacent images in a container with keyword in heading
        if not pairs:
            pairs = self._pair_by_section_heading(soup, page_url, base)

        # Queue internal links that likely lead to before/after galleries
        self._queue_gallery_links(soup, page_url, base)

        logger.debug(f"[brazilian_sources] {len(pairs)} pairs from {page_url}")
        return pairs

    def _pair_by_alt(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        before_urls: list[str] = []
        after_urls: list[str] = []
        for img in soup.find_all("img", src=True):
            alt = (img.get("alt") or "").lower()
            src = self._abs(img.get("src") or img.get("data-src", ""), base)
            if not src or not IMAGE_EXT_RE.search(src):
                continue
            if any(kw in alt for kw in BEFORE_PT):
                before_urls.append(src)
            elif any(kw in alt for kw in AFTER_PT):
                after_urls.append(src)
        return [
            self._make_pair(b, a, page_url, "alt_text_pt")
            for b, a in zip(before_urls, after_urls)
        ]

    def _pair_by_figure_caption(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        for fig in soup.find_all("figure"):
            cap = fig.find("figcaption")
            if not cap:
                continue
            cap_text = cap.get_text().lower()
            if any(b in cap_text for b in BEFORE_PT) or any(a in cap_text for a in AFTER_PT):
                imgs = fig.find_all("img")
                if len(imgs) >= 2:
                    b = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
                    a = self._abs(imgs[1].get("src") or imgs[1].get("data-src", ""), base)
                    if b and a:
                        pairs.append(self._make_pair(b, a, page_url, "figure_caption_pt"))
        return pairs

    def _pair_by_section_heading(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> list[RawImagePair]:
        pairs: list[RawImagePair] = []
        antes_depois_re = re.compile(r"antes\s*e\s*depois|before\s*and?\s*after", re.I)
        for heading in soup.find_all(re.compile(r"h[1-6]")):
            if antes_depois_re.search(heading.get_text()):
                # Look for images in the next sibling container
                container = heading.find_next_sibling()
                if container:
                    imgs = container.find_all("img", src=True)
                    if len(imgs) >= 2:
                        b = self._abs(imgs[0].get("src") or imgs[0].get("data-src", ""), base)
                        a = self._abs(imgs[1].get("src") or imgs[1].get("data-src", ""), base)
                        if b and a:
                            pairs.append(self._make_pair(b, a, page_url, "section_heading_pt"))
        return pairs

    def _queue_gallery_links(
        self, soup: BeautifulSoup, page_url: str, base: str
    ) -> None:
        """Queue internal links that likely lead to gallery or result pages."""
        gallery_re = re.compile(
            r"(antes.?depois|resultado|galeria|before.?after|fotos)", re.I
        )
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = a.get_text().lower()
            full_url = self._abs(href, base)
            if not full_url:
                continue
            # Only follow same-domain links that look like gallery pages
            if urlparse(full_url).netloc == urlparse(page_url).netloc:
                if gallery_re.search(href) or gallery_re.search(text):
                    self._queue_page(full_url)

    @staticmethod
    def _abs(url: str, base: str) -> str:
        if not url:
            return ""
        if url.startswith("http"):
            return url
        return urljoin(base, url)

    def _make_pair(
        self, before_url: str, after_url: str, source_url: str, method: str
    ) -> RawImagePair:
        return RawImagePair(
            before_url=before_url,
            after_url=after_url,
            source_url=source_url,
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={"extraction_method": method},
        )
