"""
Source factory — reads crawler.yaml and returns instantiated source modules.

Maps each source name to its class, builds a SourceConfig from the YAML,
and returns only enabled sources unless include_disabled=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Type

import yaml
from loguru import logger

from crawler.base import BaseSource, ConsentTier, RenderingMethod, SourceConfig
from crawler.sources.academy_portals import AcademyPortalsSource
from crawler.sources.beauty_media import BeautyMediaSource
from crawler.sources.brand_galleries import BrandGalleriesSource
from crawler.sources.brazilian_sources import BrazilianSourcesSource
from crawler.sources.clinic_sites import ClinicSitesSource
from crawler.sources.instagram import InstagramSource
from crawler.sources.korean_sources import KoreanSourcesSource
from crawler.sources.open_access_pubs import OpenAccessPubsSource
from crawler.sources.open_datasets import OpenDatasetsSource
from crawler.sources.pinterest import PinterestSource
from crawler.sources.professional_societies import ProfessionalSocietiesSource
from crawler.sources.realself import RealSelfSource
from crawler.sources.reddit import RedditSource
from crawler.sources.review_platforms import ReviewPlatformsSource
from crawler.sources.tiktok_static import TikTokStaticSource

# Registry: config key → source class
SOURCE_REGISTRY: dict[str, Type[BaseSource]] = {
    "realself": RealSelfSource,
    "clinic_sites": ClinicSitesSource,
    "instagram": InstagramSource,
    "reddit": RedditSource,
    "pinterest": PinterestSource,
    "tiktok_static": TikTokStaticSource,
    "brand_galleries": BrandGalleriesSource,
    "open_access_pubs": OpenAccessPubsSource,
    "review_platforms": ReviewPlatformsSource,
    "professional_societies": ProfessionalSocietiesSource,
    "beauty_media": BeautyMediaSource,
    "academy_portals": AcademyPortalsSource,
    "korean_sources": KoreanSourcesSource,
    "brazilian_sources": BrazilianSourcesSource,
    "open_datasets": OpenDatasetsSource,
}


def _build_source_config(name: str, raw: dict) -> SourceConfig:
    """Convert a raw YAML dict for one source into a SourceConfig dataclass."""
    rendering_str = raw.get("rendering", "static")
    try:
        rendering = RenderingMethod(rendering_str)
    except ValueError:
        logger.warning(f"[factory] Unknown rendering method '{rendering_str}' for {name} — defaulting to static.")
        rendering = RenderingMethod.STATIC

    consent_tier_val = raw.get("consent_tier", 3)
    try:
        consent_tier = ConsentTier(int(consent_tier_val))
    except ValueError:
        consent_tier = ConsentTier.UNCERTAIN

    # Collect all known scalar fields; everything else goes into extra
    known_keys = {
        "enabled", "rendering", "rate_limit_seconds", "language",
        "consent_tier", "base_urls",
    }
    extra = {k: v for k, v in raw.items() if k not in known_keys}

    return SourceConfig(
        name=name,
        enabled=bool(raw.get("enabled", False)),
        rendering=rendering,
        rate_limit_seconds=float(raw.get("rate_limit_seconds", 3.0)),
        language=str(raw.get("language", "en")),
        consent_tier=consent_tier,
        base_urls=list(raw.get("base_urls", [])),
        extra=extra,
    )


def build_sources(
    config_path: str | Path = "configs/crawler.yaml",
    include_disabled: bool = False,
) -> list[BaseSource]:
    """
    Read crawler.yaml and return instantiated source objects.

    Args:
        config_path:       Path to crawler.yaml.
        include_disabled:  If True, also return disabled sources (for testing).

    Returns:
        List of BaseSource instances in registry order.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Crawler config not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    sources_raw: dict = raw_config.get("sources", {})
    sources: list[BaseSource] = []

    for name, source_class in SOURCE_REGISTRY.items():
        raw = sources_raw.get(name, {})
        if not raw:
            logger.debug(f"[factory] No config entry for '{name}' — skipping.")
            continue

        cfg = _build_source_config(name, raw)

        if not cfg.enabled and not include_disabled:
            logger.debug(f"[factory] Source '{name}' disabled — skipping.")
            continue

        instance = source_class(cfg)
        sources.append(instance)
        status = "enabled" if cfg.enabled else "disabled"
        logger.debug(f"[factory] Loaded source: {name} ({status}, {cfg.rendering.value})")

    logger.info(f"[factory] {len(sources)} sources loaded from {config_path}.")
    return sources


def build_calibration_source(
    config_path: str | Path = "configs/crawler.yaml",
) -> BaseSource:
    """
    Return only the calibration bootstrap source (RealSelf).
    Used by the calibration script before the full crawl begins.
    """
    config_path = Path(config_path)
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    sources_raw = raw_config.get("sources", {})
    for name, raw in sources_raw.items():
        if raw.get("calibration_source", False):
            cfg = _build_source_config(name, raw)
            source_class = SOURCE_REGISTRY.get(name)
            if source_class is None:
                raise ValueError(f"No source class registered for calibration source '{name}'.")
            logger.info(f"[factory] Calibration source: {name}")
            return source_class(cfg)

    raise RuntimeError("No source with calibration_source: true found in crawler.yaml.")
