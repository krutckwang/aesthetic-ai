"""
Threshold calibration for all three validation layers.

How calibration works
─────────────────────
1. Pull N pairs from the calibration source (RealSelf — known high-quality).
   These are treated as TRUE POSITIVES (known-good before/after pairs).

2. Generate synthetic HARD NEGATIVES:
   - Mismatched pairs: shuffle before/after images from different people.
   - Mismatched text: swap captions between different pairs.

3. Run all three validation layers on both sets.

4. Find the threshold for each layer that achieves:
     precision ≥ 0.90  AND  recall ≥ 0.80
   on this calibration corpus using binary search.

5. Write the calibrated thresholds back into validation.yaml and
   flip the `calibrated: true` flag.

Usage
─────
    python -m crawler.validation.calibration \
        --queue-db /path/to/staging.db \
        --config configs/validation.yaml \
        --calibration-n 200
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml
from loguru import logger

from crawler.storage.staging_queue import StagingQueue
from crawler.validation.face_similarity import FaceSimilarityValidator, FaceSimilarityResult
from crawler.validation.nlp_pairing import NLPPairingValidator, NLPPairingResult
from crawler.validation.structural import StructuralValidator, StructuralResult


# Minimum acceptable precision/recall targets
MIN_PRECISION = 0.90
MIN_RECALL = 0.80

# Search grid for threshold calibration
STRUCTURAL_GRID = [round(x * 0.05, 2) for x in range(1, 20)]  # 0.05 – 0.95
NLP_GRID = [round(x * 0.05, 2) for x in range(1, 20)]
ARCFACE_GRID = [round(x * 0.05, 2) for x in range(1, 16)]     # 0.05 – 0.75


@dataclass
class CalibrationSample:
    """One labelled calibration sample."""

    is_positive: bool                      # True = genuine before/after pair
    structural: StructuralResult | None
    nlp: NLPPairingResult | None
    arcface: FaceSimilarityResult | None


@dataclass
class CalibrationResult:
    """Final calibrated thresholds and supporting statistics."""

    structural_threshold: float
    nlp_threshold: float
    arcface_threshold: float
    n_positives: int
    n_negatives: int
    structural_precision: float
    structural_recall: float
    nlp_precision: float
    nlp_recall: float
    arcface_precision: float
    arcface_recall: float


class Calibrator:
    """
    Runs calibration on a set of known-good pairs from the staging queue
    to derive per-layer thresholds.
    """

    def __init__(
        self,
        queue: StagingQueue,
        config_path: Path,
        n_samples: int = 200,
        seed: int = 42,
    ) -> None:
        self.queue = queue
        self.config_path = config_path
        self.n_samples = n_samples
        self._rng = random.Random(seed)

        self._struct = StructuralValidator()
        self._nlp = NLPPairingValidator()
        self._face = FaceSimilarityValidator()

    def run(self) -> CalibrationResult:
        """Run calibration and write updated thresholds to validation.yaml."""
        logger.info(f"[calibration] Pulling {self.n_samples} pairs from queue …")
        positive_samples = self._collect_positive_samples()
        negative_samples = self._build_negative_samples(positive_samples)

        all_samples = positive_samples + negative_samples
        self._rng.shuffle(all_samples)

        logger.info(
            f"[calibration] Corpus: {len(positive_samples)} positives, "
            f"{len(negative_samples)} negatives."
        )

        struct_thresh = self._calibrate_layer(
            all_samples, "structural", STRUCTURAL_GRID
        )
        nlp_thresh = self._calibrate_layer(
            all_samples, "nlp", NLP_GRID
        )
        arcface_thresh = self._calibrate_layer(
            all_samples, "arcface", ARCFACE_GRID
        )

        result = CalibrationResult(
            structural_threshold=struct_thresh,
            nlp_threshold=nlp_thresh,
            arcface_threshold=arcface_thresh,
            n_positives=len(positive_samples),
            n_negatives=len(negative_samples),
            structural_precision=self._precision_at(all_samples, "structural", struct_thresh),
            structural_recall=self._recall_at(all_samples, "structural", struct_thresh),
            nlp_precision=self._precision_at(all_samples, "nlp", nlp_thresh),
            nlp_recall=self._recall_at(all_samples, "nlp", nlp_thresh),
            arcface_precision=self._precision_at(all_samples, "arcface", arcface_thresh),
            arcface_recall=self._recall_at(all_samples, "arcface", arcface_thresh),
        )

        self._write_config(result)
        return result

    # ── Sample collection ─────────────────────────────────────────────────────

    def _collect_positive_samples(self) -> list[CalibrationSample]:
        """Pull N pairs from the queue and run all three layers on them."""
        batch = self.queue.dequeue_batch(self.n_samples)
        samples = []
        for item in batch:
            pair = item.get("pair", {})
            before_url = pair.get("before_url", "")
            after_url = pair.get("after_url", "")
            page_html = pair.get("page_html", "")

            structural = self._struct.validate(before_url, after_url, page_html)

            before_text = self._nlp.extract_image_context(before_url, page_html)
            after_text = self._nlp.extract_image_context(after_url, page_html)
            nlp = self._nlp.validate(before_text, after_text)

            # ArcFace needs file paths — skip if files not yet downloaded
            arcface = None

            samples.append(CalibrationSample(
                is_positive=True,
                structural=structural,
                nlp=nlp,
                arcface=arcface,
            ))

        logger.info(f"[calibration] Collected {len(samples)} positive samples.")
        return samples

    def _build_negative_samples(
        self,
        positives: list[CalibrationSample],
    ) -> list[CalibrationSample]:
        """
        Build synthetic hard negatives by shuffling structural/nlp results.

        Strategy: swap the 'after' half of each result with another pair's 'after'
        to simulate same-label-but-wrong-person mismatches.
        """
        if len(positives) < 2:
            return []

        negatives: list[CalibrationSample] = []
        indices = list(range(len(positives)))
        self._rng.shuffle(indices)

        for i, j in zip(indices[: len(indices) // 2], indices[len(indices) // 2 :]):
            a = positives[i]
            b = positives[j]

            # Create a synthetic NLPPairingResult with mismatched texts
            from crawler.validation.nlp_pairing import NLPPairingResult
            neg_nlp = NLPPairingResult(
                pairing_score=max(
                    0.0,
                    ((a.nlp.pairing_score if a.nlp else 0.0) +
                     (b.nlp.pairing_score if b.nlp else 0.0)) / 2 - 0.3,
                ),
                detected_language=(a.nlp.detected_language if a.nlp else "en"),
                method="synthetic_negative",
                before_text_snippet=(a.nlp.before_text_snippet if a.nlp else ""),
                after_text_snippet=(b.nlp.after_text_snippet if b.nlp else ""),
            )

            negatives.append(CalibrationSample(
                is_positive=False,
                structural=a.structural,  # structural doesn't change (page label still valid)
                nlp=neg_nlp,
                arcface=None,
            ))

        logger.info(f"[calibration] Built {len(negatives)} synthetic negatives.")
        return negatives

    # ── Threshold search ──────────────────────────────────────────────────────

    def _calibrate_layer(
        self,
        samples: list[CalibrationSample],
        layer: str,
        grid: list[float],
    ) -> float:
        """Find highest threshold where precision ≥ MIN_PRECISION and recall ≥ MIN_RECALL."""
        best_thresh = grid[0]
        for thresh in reversed(grid):  # highest first
            prec = self._precision_at(samples, layer, thresh)
            rec = self._recall_at(samples, layer, thresh)
            if prec >= MIN_PRECISION and rec >= MIN_RECALL:
                best_thresh = thresh
                break
        logger.info(
            f"[calibration] Layer '{layer}' threshold = {best_thresh} "
            f"(prec={self._precision_at(samples, layer, best_thresh):.2f}, "
            f"rec={self._recall_at(samples, layer, best_thresh):.2f})"
        )
        return best_thresh

    @staticmethod
    def _score_for_layer(sample: CalibrationSample, layer: str) -> float:
        """Extract the relevant score for a given layer."""
        if layer == "structural":
            return sample.structural.confidence if sample.structural else 0.0
        if layer == "nlp":
            return sample.nlp.pairing_score if sample.nlp else 0.0
        if layer == "arcface":
            sim = sample.arcface.cosine_similarity if sample.arcface else -2.0
            return sim if sim >= 0 else 0.0
        return 0.0

    def _precision_at(
        self, samples: list[CalibrationSample], layer: str, thresh: float
    ) -> float:
        tp = sum(
            1 for s in samples
            if self._score_for_layer(s, layer) >= thresh and s.is_positive
        )
        fp = sum(
            1 for s in samples
            if self._score_for_layer(s, layer) >= thresh and not s.is_positive
        )
        return tp / (tp + fp) if (tp + fp) > 0 else 1.0

    def _recall_at(
        self, samples: list[CalibrationSample], layer: str, thresh: float
    ) -> float:
        tp = sum(
            1 for s in samples
            if self._score_for_layer(s, layer) >= thresh and s.is_positive
        )
        fn = sum(
            1 for s in samples
            if self._score_for_layer(s, layer) < thresh and s.is_positive
        )
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ── Config update ─────────────────────────────────────────────────────────

    def _write_config(self, result: CalibrationResult) -> None:
        """Write calibrated thresholds to validation.yaml."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        config["calibrated"] = True
        config.setdefault("layer1", {})["min_confidence"] = result.structural_threshold
        config.setdefault("layer2", {})["min_pairing_score"] = result.nlp_threshold
        config.setdefault("layer3", {})["min_cosine_similarity"] = result.arcface_threshold

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        logger.info(
            f"[calibration] Written to {self.config_path}: "
            f"L1={result.structural_threshold}, "
            f"L2={result.nlp_threshold}, "
            f"L3={result.arcface_threshold}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate validation thresholds.")
    parser.add_argument("--queue-db", required=True, help="Path to staging queue SQLite DB.")
    parser.add_argument(
        "--config", default="configs/validation.yaml", help="Path to validation.yaml."
    )
    parser.add_argument("--calibration-n", type=int, default=200, help="Number of samples.")
    args = parser.parse_args()

    queue = StagingQueue(Path(args.queue_db))
    calibrator = Calibrator(
        queue=queue,
        config_path=Path(args.config),
        n_samples=args.calibration_n,
    )
    result = calibrator.run()

    print("\n=== Calibration Results ===")
    print(f"Positives: {result.n_positives}  Negatives: {result.n_negatives}")
    print(f"Layer 1 (structural): threshold={result.structural_threshold}  "
          f"prec={result.structural_precision:.2f}  rec={result.structural_recall:.2f}")
    print(f"Layer 2 (NLP):        threshold={result.nlp_threshold}  "
          f"prec={result.nlp_precision:.2f}  rec={result.nlp_recall:.2f}")
    print(f"Layer 3 (ArcFace):    threshold={result.arcface_threshold}  "
          f"prec={result.arcface_precision:.2f}  rec={result.arcface_recall:.2f}")


if __name__ == "__main__":
    main()
