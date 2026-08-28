"""Tests for crawler/factory.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from crawler.base import ConsentTier, RenderingMethod
from crawler.factory import SOURCE_REGISTRY, build_calibration_source, build_sources
from crawler.sources.instagram import InstagramSource
from crawler.sources.realself import RealSelfSource
from crawler.sources.tiktok_static import TikTokStaticSource


@pytest.fixture
def minimal_crawler_config(tmp_path: Path) -> Path:
    """A minimal crawler.yaml with 2 enabled sources and 2 disabled."""
    config = {
        "global": {
            "default_rate_limit_seconds": 3.0,
            "user_agent": "TestBot/0.1",
        },
        "sources": {
            "realself": {
                "enabled": True,
                "rendering": "static",
                "rate_limit_seconds": 4.0,
                "language": "en",
                "consent_tier": 1,
                "calibration_source": True,
                "base_urls": ["https://www.realself.com/botulinum-toxin/reviews"],
            },
            "reddit": {
                "enabled": True,
                "rendering": "static",
                "rate_limit_seconds": 3.0,
                "language": "en",
                "consent_tier": 2,
                "base_urls": ["https://www.reddit.com/r/PlasticSurgery"],
            },
            "instagram": {
                "enabled": False,
                "rendering": "static",
                "rate_limit_seconds": 10.0,
                "language": "en",
                "consent_tier": 3,
            },
            "tiktok_static": {
                "enabled": False,
                "rendering": "static",
                "rate_limit_seconds": 5.0,
                "language": "en",
                "consent_tier": 3,
            },
        },
    }
    config_path = tmp_path / "crawler.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


class TestBuildSources:
    def test_returns_only_enabled_sources(self, minimal_crawler_config):
        sources = build_sources(minimal_crawler_config)
        names = [s.config.name for s in sources]
        assert "realself" in names
        assert "reddit" in names
        assert "instagram" not in names
        assert "tiktok_static" not in names

    def test_include_disabled_returns_all(self, minimal_crawler_config):
        sources = build_sources(minimal_crawler_config, include_disabled=True)
        names = [s.config.name for s in sources]
        assert "instagram" in names
        assert "tiktok_static" in names

    def test_source_config_values_populated(self, minimal_crawler_config):
        sources = build_sources(minimal_crawler_config)
        realself = next(s for s in sources if s.config.name == "realself")
        assert realself.config.rate_limit_seconds == 4.0
        assert realself.config.language == "en"
        assert realself.config.consent_tier == ConsentTier.CONFIRMED
        assert realself.config.rendering == RenderingMethod.STATIC
        assert len(realself.config.base_urls) == 1

    def test_realself_instantiated_as_correct_class(self, minimal_crawler_config):
        sources = build_sources(minimal_crawler_config)
        realself = next(s for s in sources if s.config.name == "realself")
        assert isinstance(realself, RealSelfSource)

    def test_instagram_is_disabled_by_default_in_real_config(self):
        """Verify the real configs/crawler.yaml has instagram disabled."""
        config_path = Path("configs/crawler.yaml")
        if not config_path.exists():
            pytest.skip("configs/crawler.yaml not found — skipping real-config test.")
        sources = build_sources(config_path)
        names = [s.config.name for s in sources]
        assert "instagram" not in names, "Instagram must be disabled in default config."

    def test_tiktok_is_disabled_by_default_in_real_config(self):
        config_path = Path("configs/crawler.yaml")
        if not config_path.exists():
            pytest.skip("configs/crawler.yaml not found — skipping real-config test.")
        sources = build_sources(config_path)
        names = [s.config.name for s in sources]
        assert "tiktok_static" not in names, "TikTok must be disabled in default config."

    def test_missing_config_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_sources(tmp_path / "nonexistent.yaml")


class TestBuildCalibrationSource:
    def test_returns_realself_source(self, minimal_crawler_config):
        source = build_calibration_source(minimal_crawler_config)
        assert isinstance(source, RealSelfSource)
        assert source.config.name == "realself"

    def test_raises_if_no_calibration_source(self, tmp_path):
        config = {
            "sources": {
                "reddit": {
                    "enabled": True,
                    "rendering": "static",
                    "rate_limit_seconds": 3.0,
                    "language": "en",
                    "consent_tier": 2,
                }
            }
        }
        config_path = tmp_path / "crawler.yaml"
        config_path.write_text(yaml.dump(config))
        with pytest.raises(RuntimeError, match="No source with calibration_source"):
            build_calibration_source(config_path)


class TestSourceRegistry:
    def test_all_15_sources_registered(self):
        expected = {
            "realself", "clinic_sites", "instagram", "reddit", "pinterest",
            "tiktok_static", "brand_galleries", "open_access_pubs",
            "review_platforms", "professional_societies", "beauty_media",
            "academy_portals", "korean_sources", "brazilian_sources",
            "open_datasets",
        }
        assert set(SOURCE_REGISTRY.keys()) == expected
