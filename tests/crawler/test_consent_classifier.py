"""Tests for crawler/consent/classifier.py."""

from __future__ import annotations

import pytest

from crawler.base import ConsentTier
from crawler.consent.classifier import ConsentClassifier


@pytest.fixture
def clf() -> ConsentClassifier:
    return ConsentClassifier()


class TestTier1Domain:
    def test_realself_domain_is_tier1(self, clf):
        result = clf.classify(
            source_url="https://www.realself.com/botox/reviews/123",
            page_html="",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED
        assert "tier1_domain" in result.signals_found

    def test_pmc_domain_is_tier1(self, clf):
        result = clf.classify(
            source_url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC123/",
            page_html="",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED

    def test_asps_domain_is_tier1(self, clf):
        result = clf.classify(
            source_url="https://www.plasticsurgery.org/patient-safety/case-1",
            page_html="",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED


class TestTier1ExplicitLanguage:
    def test_explicit_patient_consent_upgrades_tier2(self, clf):
        result = clf.classify(
            source_url="https://www.someclinic.com/gallery",
            page_html="All photos shown with patient consent and written authorization.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED

    def test_model_release_language_upgrades(self, clf):
        result = clf.classify(
            source_url="https://www.someclinic.com/gallery",
            page_html="Images used under model release agreement.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED

    def test_creative_commons_upgrades(self, clf):
        result = clf.classify(
            source_url="https://journal.example.com/article/1",
            page_html="This article is licensed under Creative Commons CC-BY 4.0.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED

    def test_korean_consent_language_upgrades(self, clf):
        result = clf.classify(
            source_url="https://blog.naver.com/some-post",
            page_html="모든 사진은 환자의 동의를 받아 게시되었습니다.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED

    def test_portuguese_consent_language_upgrades(self, clf):
        result = clf.classify(
            source_url="https://clinica.com.br/galeria",
            page_html="Fotos publicadas com consentimento do paciente.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.CONFIRMED


class TestTier2:
    def test_tier2_domain_stays_tier2(self, clf):
        result = clf.classify(
            source_url="https://www.allure.com/story/botox-results",
            page_html="These are real patient results from a board-certified injector.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.LIKELY

    def test_results_may_vary_is_tier2(self, clf):
        result = clf.classify(
            source_url="https://www.someclinic.com/gallery",
            page_html="Results may vary. Individual results may vary by patient.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.LIKELY

    def test_tier2_base_with_no_signals_stays_tier2(self, clf):
        result = clf.classify(
            source_url="https://www.unknownclinic.com/gallery",
            page_html="Welcome to our gallery.",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.LIKELY


class TestTier3:
    def test_no_signals_and_tier3_base_stays_tier3(self, clf):
        result = clf.classify(
            source_url="https://www.unknownsite.com/post/123",
            page_html="Check out my transformation!",
            base_tier=ConsentTier.UNCERTAIN,
        )
        assert result.tier == ConsentTier.UNCERTAIN

    def test_instagram_domain_is_hard_tier3(self, clf):
        result = clf.classify(
            source_url="https://www.instagram.com/p/abc123/",
            page_html="with patient consent and model release",  # explicit language present
            base_tier=ConsentTier.LIKELY,
        )
        # Hard Tier 3 domain overrides even explicit consent language
        assert result.tier == ConsentTier.UNCERTAIN
        assert "hard_tier3_domain" in result.signals_found

    def test_tiktok_domain_is_hard_tier3(self, clf):
        result = clf.classify(
            source_url="https://www.tiktok.com/@user/video/123",
            page_html="",
            base_tier=ConsentTier.LIKELY,
        )
        assert result.tier == ConsentTier.UNCERTAIN


class TestTier1BasePassthrough:
    def test_tier1_base_stays_tier1_without_page_analysis(self, clf):
        result = clf.classify(
            source_url="https://www.someclinic.com/gallery",
            page_html="",
            base_tier=ConsentTier.CONFIRMED,
        )
        assert result.tier == ConsentTier.CONFIRMED
        assert "source_base_tier1" in result.signals_found
