"""
Consent tier classifier for crawled images.

Assigns each image a ConsentTier (1=CONFIRMED, 2=LIKELY, 3=UNCERTAIN)
based on the source's base tier and signals found in the page content.

Rules:
  - Tier 3 sources always produce Tier 3 output — never upgraded.
  - Tier 2 sources are upgraded to Tier 1 if explicit consent language is found.
  - Tier 1 sources stay Tier 1 regardless of page content.
  - All classification decisions and signals are recorded for auditing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from crawler.base import ConsentTier


# ── Tier 1 signals ────────────────────────────────────────────────────────────
# Explicit language indicating patient gave informed consent for publication.

TIER1_DOMAIN_SUBSTRINGS: frozenset[str] = frozenset([
    "realself.com",
    "ncbi.nlm.nih.gov",
    "pubmedcentral",
    "pmc.ncbi",
    "plasticsurgery.org",
    "aad.org",
    "asds.net",
    "baaps.org.uk",
    "isaps.org",
    "allerganmedicalinstitute.com",
    "galderma-institute.com",
    "sinclairacademy.com",
])

TIER1_TEXT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"with\s+patient\s+consent",
        r"patient\s+consented",
        r"written\s+consent",
        r"informed\s+consent",
        r"model\s+release",
        r"consent\s+to\s+publish",
        r"permission\s+to\s+use",
        r"creative\s+commons",
        r"cc[\s\-]by",
        r"open\s+access",
        r"irb[\s\-]approved",
        r"institutional\s+review\s+board",
        # Korean: 동의를 받아, 동의 하에
        r"동의를?\s*받",
        r"동의\s*하에",
        # Portuguese: com consentimento, autorizado pelo paciente
        r"com\s+consentimento",
        r"autorizado\s+pelo\s+paciente",
        r"com\s+autorização",
    ]
]

# ── Tier 2 signals ────────────────────────────────────────────────────────────
# Implied consent — clinical context suggests patient approved but no explicit statement.

TIER2_DOMAIN_SUBSTRINGS: frozenset[str] = frozenset([
    "healthgrades.com",
    "ratemds.com",
    "yelp.com",
    "newbeauty.com",
    "allure.com",
    "byrdie.com",
    "refinery29.com",
    "old.reddit.com",
    "reddit.com",
    "blog.naver.com",
    "allergan.com",
    "galderma.com",
    "merzaesthetics.com",
    "revance.com",
    "evolus.com",
    "soltamedical.com",
    "inmodemd.com",
    "candelamedical.com",
    "cutera.com",
    "lumenis.com",
    "fotona.com",
    "cynosure.com",
    "teoxane.com",
])

TIER2_TEXT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE) for p in [
        r"results?\s+may\s+vary",
        r"individual\s+results?\s+may\s+vary",
        r"actual\s+patient",
        r"real\s+patient\s+results?",
        r"patient\s+results?",
        r"before\s+and\s+after\s+photos?",
        # Korean: 실제 환자, 실제 결과
        r"실제\s*환자",
        r"실제\s*결과",
        # Portuguese: resultado real, paciente real
        r"resultado\s+real",
        r"paciente\s+real",
    ]
]

# ── Hard Tier 3 domains — never upgraded regardless of page content ────────────

HARD_TIER3_DOMAIN_SUBSTRINGS: frozenset[str] = frozenset([
    "instagram.com",
    "tiktok.com",
    "facebook.com",
    "twitter.com",
    "x.com",
])


@dataclass
class ConsentAssessment:
    """Result of a consent classification."""

    tier: ConsentTier
    signals_found: list[str]
    domain: str


class ConsentClassifier:
    """
    Classifies consent tier for a crawled image based on source URL,
    page HTML content, and the source module's configured base tier.
    """

    def classify(
        self,
        source_url: str,
        page_html: str,
        base_tier: ConsentTier,
        metadata: dict | None = None,
    ) -> ConsentAssessment:
        """
        Determine consent tier.

        Args:
            source_url:  The page URL where the image was found.
            page_html:   Full page HTML (may be empty string if unavailable).
            base_tier:   The source module's configured default tier.
            metadata:    Optional metadata dict from the crawler.

        Returns:
            ConsentAssessment with final tier and list of signals found.
        """
        domain = self._extract_domain(source_url)
        signals: list[str] = []

        # Hard Tier 3 domains — cannot be upgraded
        if any(s in domain for s in HARD_TIER3_DOMAIN_SUBSTRINGS):
            return ConsentAssessment(
                tier=ConsentTier.UNCERTAIN,
                signals_found=["hard_tier3_domain"],
                domain=domain,
            )

        # Source already Tier 3 — no upgrade possible
        if base_tier == ConsentTier.UNCERTAIN:
            return ConsentAssessment(
                tier=ConsentTier.UNCERTAIN,
                signals_found=["source_base_tier3"],
                domain=domain,
            )

        # Source already Tier 1 — no further analysis needed
        if base_tier == ConsentTier.CONFIRMED:
            return ConsentAssessment(
                tier=ConsentTier.CONFIRMED,
                signals_found=["source_base_tier1"],
                domain=domain,
            )

        # Tier 1 domain check
        if any(s in domain for s in TIER1_DOMAIN_SUBSTRINGS):
            signals.append("tier1_domain")
            return ConsentAssessment(
                tier=ConsentTier.CONFIRMED,
                signals_found=signals,
                domain=domain,
            )

        # Tier 1 text pattern check (upgrades Tier 2 → Tier 1)
        for pattern in TIER1_TEXT_PATTERNS:
            if pattern.search(page_html):
                signals.append(f"tier1_text:{pattern.pattern}")

        if signals:
            return ConsentAssessment(
                tier=ConsentTier.CONFIRMED,
                signals_found=signals,
                domain=domain,
            )

        # Tier 2 domain check
        if any(s in domain for s in TIER2_DOMAIN_SUBSTRINGS):
            signals.append("tier2_domain")
            return ConsentAssessment(
                tier=ConsentTier.LIKELY,
                signals_found=signals,
                domain=domain,
            )

        # Tier 2 text pattern check
        for pattern in TIER2_TEXT_PATTERNS:
            if pattern.search(page_html):
                signals.append(f"tier2_text:{pattern.pattern}")

        if signals:
            return ConsentAssessment(
                tier=ConsentTier.LIKELY,
                signals_found=signals,
                domain=domain,
            )

        # Base tier was Tier 2 but no signals found — keep as Tier 2
        # (source configuration provides the baseline guarantee)
        return ConsentAssessment(
            tier=base_tier,
            signals_found=["source_base_tier_fallback"],
            domain=domain,
        )

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""
