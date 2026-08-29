"""Open dataset source (Kaggle, HuggingFace Hub, Zenodo public datasets)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from loguru import logger

from crawler.base import BaseSource, ConsentTier, RawImagePair


IMAGE_EXT_RE = re.compile(r"\.(jpe?g|png|webp|gif|tiff?)$", re.IGNORECASE)

# Known public aesthetic image pair datasets indexed by short key
KNOWN_DATASETS = {
    "ffhq_aging": {
        "description": "FFHQ-Aging synthetic aging progression dataset",
        "hf_repo": "datasets/FluxAI/ffhq-aging",
        "pair_method": "metadata_csv",
    },
    "celeba_hq": {
        "description": "CelebA-HQ high-resolution face dataset",
        "hf_repo": "datasets/mattymchen/celeba-hq",
        "pair_method": "folder_structure",
    },
    "scut_fbp5500": {
        "description": "SCUT-FBP5500 facial beauty prediction dataset",
        "hf_repo": "datasets/SCUT-FBP5500",
        "pair_method": "folder_structure",
    },
}


class OpenDatasetsSource(BaseSource):
    """
    Indexes and yields image pairs from known open public datasets.

    Supports two modes:
      folder_structure: before/ and after/ subdirectories with matching filenames.
      metadata_csv:     reads a CSV/TSV with before_path and after_path columns.

    Also handles HuggingFace Hub dataset cards queued from config base_urls.
    Consent tier: 1 — explicitly open-licensed research datasets.
    """

    def iter_page_urls(self) -> Iterator[str]:
        # Yield HuggingFace dataset search pages
        for url in self.config.base_urls:
            yield url

        # Yield dataset card pages for known datasets
        for key, info in KNOWN_DATASETS.items():
            if info.get("hf_repo"):
                yield f"https://huggingface.co/{info['hf_repo']}"

    def extract_pairs_from_page(self, html: str, page_url: str) -> list[RawImagePair]:
        from bs4 import BeautifulSoup
        from urllib.parse import urlparse

        soup = BeautifulSoup(html, "lxml")
        base = f"{urlparse(page_url).scheme}://{urlparse(page_url).netloc}"

        if "huggingface.co" in page_url:
            return self._process_hf_dataset_card(soup, page_url, base)

        # Local dataset directory from config extra.local_path
        local_path = self.config.extra.get("local_path")
        if local_path:
            return list(self._scan_local_dataset(Path(local_path)))

        return []

    # ── HuggingFace dataset card processing ───────────────────────────────────

    def _process_hf_dataset_card(
        self, soup, page_url: str, base: str
    ) -> list[RawImagePair]:
        """
        Queue download links for dataset files found on the dataset card.
        In production, HuggingFace datasets are accessed via the datasets library,
        not via HTTP scraping. This method queues any sample image links.
        """
        from urllib.parse import urljoin
        pairs: list[RawImagePair] = []

        # Queue file viewer pages that may contain sample images
        for link in soup.select("a[href*='/resolve/main/'], a[href*='/blob/main/']"):
            href = link.get("href", "")
            if IMAGE_EXT_RE.search(href):
                full = urljoin(base, href)
                self._queue_page(full)

        # Queue related dataset search pages
        for link in soup.select("a[href*='/datasets?']"):
            href = link.get("href", "")
            full = urljoin(base, href)
            if "aesthetic" in full.lower() or "before" in full.lower():
                self._queue_page(full)

        logger.debug(f"[open_datasets] Processed HF card: {page_url}")
        return pairs

    # ── Local dataset directory scanner ───────────────────────────────────────

    def _scan_local_dataset(self, root: Path) -> Iterator[RawImagePair]:
        """
        Scan a local directory for before/after image pairs.
        Expects either:
          root/before/*.jpg + root/after/*.jpg  (matching stems)
          root/pairs/NNN_before.jpg + root/pairs/NNN_after.jpg
        """
        before_dir = root / "before"
        after_dir = root / "after"
        pairs_dir = root / "pairs"

        if before_dir.is_dir() and after_dir.is_dir():
            yield from self._pair_from_split_dirs(before_dir, after_dir)

        elif pairs_dir.is_dir():
            yield from self._pair_from_pairs_dir(pairs_dir)

        else:
            # Try root-level before_*.jpg / after_*.jpg pattern
            yield from self._pair_from_filename_pattern(root)

    def _pair_from_split_dirs(
        self, before_dir: Path, after_dir: Path
    ) -> Iterator[RawImagePair]:
        """Match before/after images by filename stem."""
        before_map = {
            f.stem: f
            for f in before_dir.iterdir()
            if IMAGE_EXT_RE.search(f.name)
        }
        for after_file in after_dir.iterdir():
            if not IMAGE_EXT_RE.search(after_file.name):
                continue
            before_file = before_map.get(after_file.stem)
            if before_file:
                yield self._local_pair(str(before_file), str(after_file), "split_dirs")

    def _pair_from_pairs_dir(self, pairs_dir: Path) -> Iterator[RawImagePair]:
        """Match NNN_before.ext + NNN_after.ext patterns."""
        before_re = re.compile(r"^(.+)_before\.(jpe?g|png|webp)$", re.I)
        after_re = re.compile(r"^(.+)_after\.(jpe?g|png|webp)$", re.I)

        before_map: dict[str, Path] = {}
        after_map: dict[str, Path] = {}

        for f in pairs_dir.iterdir():
            m = before_re.match(f.name)
            if m:
                before_map[m.group(1)] = f
                continue
            m = after_re.match(f.name)
            if m:
                after_map[m.group(1)] = f

        for key, before_file in before_map.items():
            after_file = after_map.get(key)
            if after_file:
                yield self._local_pair(str(before_file), str(after_file), "pairs_dir")

    def _pair_from_filename_pattern(self, root: Path) -> Iterator[RawImagePair]:
        """Fallback: scan root for before_*.ext + after_*.ext pairs."""
        before_files = [
            f for f in root.iterdir()
            if f.is_file() and IMAGE_EXT_RE.search(f.name)
            and "before" in f.name.lower()
        ]
        for before_file in before_files:
            stem = before_file.stem.lower().replace("before", "after")
            after_file = root / f"{stem}{before_file.suffix}"
            if after_file.exists():
                yield self._local_pair(
                    str(before_file), str(after_file), "filename_pattern"
                )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _local_pair(
        self, before_path: str, after_path: str, method: str
    ) -> RawImagePair:
        return RawImagePair(
            before_url=f"file://{before_path}",
            after_url=f"file://{after_path}",
            source_url="local",
            source_name=self.config.name,
            language=self.config.language,
            consent_tier=ConsentTier(self.config.consent_tier),
            metadata={"extraction_method": method, "local": True},
        )
