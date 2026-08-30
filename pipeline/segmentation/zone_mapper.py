"""Zone mapper — maps MediaPipe landmarks to treatment zones from zones.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from pipeline.landmarks.extractor import LandmarkResult


@dataclass
class ZonePresence:
    zone_code: str
    confidence: float  # 0.0–1.0, based on landmark coverage
    landmark_count: int
    centroid_x: float  # normalised 0–1
    centroid_y: float  # normalised 0–1


@dataclass
class ZoneMappingResult:
    zones: list[ZonePresence]
    success: bool
    total_zones_checked: int


class ZoneMapper:
    """
    Maps a set of 478 MediaPipe landmarks to named treatment zones
    defined in configs/zones.yaml.

    Each zone has a list of landmark indices. Confidence is computed as
    the fraction of zone landmarks that are present in the landmark result
    (MediaPipe occasionally omits landmarks near image boundaries).
    """

    def __init__(self, config_path: str | Path = "configs/zones.yaml") -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self._zones: dict[str, list[int]] = {}
        for zone_name, zone_data in cfg.get("zones", {}).items():
            indices = zone_data.get("landmark_indices", [])
            # Deduplicate indices (zones.yaml sometimes has repeated indices)
            self._zones[zone_name] = list(dict.fromkeys(indices))

    @property
    def zone_names(self) -> list[str]:
        return list(self._zones.keys())

    def map(self, landmark_result: LandmarkResult) -> ZoneMappingResult:
        """
        Compute zone presence for all zones defined in zones.yaml.
        Zones with no landmark indices (whole-face zones) are skipped.
        """
        if not landmark_result.success:
            return ZoneMappingResult(zones=[], success=False, total_zones_checked=0)

        lm_map = {lm.index: lm for lm in landmark_result.landmarks}
        zone_presences: list[ZonePresence] = []
        checked = 0

        for zone_code, indices in self._zones.items():
            if not indices:
                # Whole-face zone — no landmark indices to check
                continue
            checked += 1

            present = [i for i in indices if i in lm_map]
            confidence = len(present) / len(indices) if indices else 0.0

            if not present:
                continue

            pts = np.array([[lm_map[i].x, lm_map[i].y] for i in present])
            centroid_x = float(pts[:, 0].mean())
            centroid_y = float(pts[:, 1].mean())

            zone_presences.append(ZonePresence(
                zone_code=zone_code,
                confidence=confidence,
                landmark_count=len(present),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
            ))

        return ZoneMappingResult(
            zones=zone_presences,
            success=True,
            total_zones_checked=checked,
        )

    def zones_for_treatment(self, treatment_category: str) -> list[str]:
        """
        Return the primary zone codes relevant to a treatment category.
        Used to focus training attention on the right facial area.
        """
        mapping: dict[str, list[str]] = {
            "botulinum_toxin": ["forehead_lines", "glabellar_complex", "brow_position", "crow_feet"],
            "botox": ["forehead_lines", "glabellar_complex", "brow_position"],
            "dysport": ["forehead_lines", "glabellar_complex"],
            "lip_filler": ["lips", "perioral_lines"],
            "cheek_filler": ["cheek_malar", "midface_volume"],
            "nasolabial_filler": ["nasolabial_folds"],
            "chin_filler": ["chin"],
            "jawline_filler": ["jawline"],
            "masseter_botox": ["masseter"],
            "tear_trough_filler": ["periorbital"],
            "temporal_filler": ["temporal_region"],
            "nose_filler": ["nose"],
            "neck_botox": ["platysmal_bands", "neck_laxity"],
            "skin_booster": ["skin_texture", "overall_skin_tone"],
        }
        return mapping.get(treatment_category.lower(), [])

    def dominant_zones(
        self,
        result: ZoneMappingResult,
        min_confidence: float = 0.6,
        top_n: int = 5,
    ) -> list[ZonePresence]:
        """Return the top-N zones by confidence, above min_confidence threshold."""
        eligible = [z for z in result.zones if z.confidence >= min_confidence]
        return sorted(eligible, key=lambda z: z.confidence, reverse=True)[:top_n]
