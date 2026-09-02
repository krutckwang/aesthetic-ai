"""
Shared treatment-label extraction used by all crawler sources.

Every RawImagePair yielded by BaseSource.crawl() passes through
extract_treatment_from_url() so that treatment_category is written
into metadata at crawl time, not inferred later in Kaggle Cell 4.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

SLUG_MAP: dict[str, str] = {
    # Botox / neurotoxins
    "botulinum-toxin":                "botox",
    "botox":                          "botox",
    "botox-cosmetic":                 "botox",
    "dysport":                        "botox",
    "xeomin":                         "botox",
    "jeuveau":                        "botox",
    "daxi":                           "botox",
    # Lip filler
    "lip-augmentation":               "lip_filler",
    "lip-augmentation---enhancement": "lip_filler",
    "lip-enhancement":                "lip_filler",
    "lip-filler":                     "lip_filler",
    "lip-fillers":                    "lip_filler",
    # Dermal fillers
    "dermal-fillers":                 "dermal_filler",
    "dermal-filler":                  "dermal_filler",
    "fillers":                        "dermal_filler",
    "juvederm":                       "dermal_filler",
    "restylane":                      "dermal_filler",
    "sculptra":                       "dermal_filler",
    "radiesse":                       "dermal_filler",
    "belotero":                       "dermal_filler",
    "cheek-augmentation":             "dermal_filler",
    "cheek-filler":                   "dermal_filler",
    "cheek-fillers":                  "dermal_filler",
    # Jawline / chin
    "chin-augmentation":              "jawline_filler",
    "chin-filler":                    "jawline_filler",
    "chin-implants":                  "jawline_filler",
    "jawline-filler":                 "jawline_filler",
    # Under-eye
    "under-eye-filler":               "under_eye_filler",
    "tear-trough":                    "under_eye_filler",
    # Kybella
    "kybella":                        "kybella",
    # Surgical face
    "facelift":                       "facelift",
    "face-lift":                      "facelift",
    "mini-facelift":                  "facelift",
    "brow-lift":                      "facelift",
    "browlift":                       "facelift",
    "forehead-lift":                  "facelift",
    "neck-lift":                      "facelift",
    "necklift":                       "facelift",
    "eyelid-surgery":                 "blepharoplasty",
    "blepharoplasty":                 "blepharoplasty",
    "upper-blepharoplasty":           "blepharoplasty",
    "lower-blepharoplasty":           "blepharoplasty",
    "upper-eyelid-surgery":           "blepharoplasty",
    "lower-eyelid-surgery":           "blepharoplasty",
    "rhinoplasty":                    "rhinoplasty",
    "nose-surgery":                   "rhinoplasty",
    "nose-reshaping":                 "rhinoplasty",
    "otoplasty":                      "otoplasty",
    "ear-surgery":                    "otoplasty",
    "fat-transfer-to-face":           "fat_transfer",
    "fat-transfer":                   "fat_transfer",
    # Skin treatments
    "laser-skin-resurfacing":         "laser_resurfacing",
    "laser-resurfacing":              "laser_resurfacing",
    "laser-treatment":                "laser_resurfacing",
    "chemical-peel":                  "chemical_peel",
    "chemical-peels":                 "chemical_peel",
    "microneedling":                  "microneedling",
    "thread-lift":                    "thread_lift",
    "microdermabrasion":              "microdermabrasion",
    "ipl-photofacial":                "ipl_photofacial",
    "ipl":                            "ipl_photofacial",
    "prp":                            "prp",
    "platelet-rich-plasma":           "prp",
}

SUBREDDIT_MAP: dict[str, str] = {
    "rhinoplasty":          "rhinoplasty",
    "jawsurgery":           "jawline_filler",
    "facialplasticsurgery": "facelift",
    "eyelidsurgery":        "blepharoplasty",
    "fillers":              "dermal_filler",
    "injectables":          "dermal_filler",
    "botox":                "botox",
}


def extract_treatment_from_url(url: str) -> str | None:
    """
    Derive a treatment_category from a source URL.

    Checks (in order):
    1. Reddit subreddit name  (/r/<sub>)
    2. URL path segments matched against SLUG_MAP
    """
    url_lower = url.lower()

    # Reddit subreddit
    m = re.search(r"/r/([^/?#]+)", url_lower)
    if m:
        sub = m.group(1)
        if sub in SUBREDDIT_MAP:
            return SUBREDDIT_MAP[sub]

    # Path segments
    path = urlparse(url_lower).path
    for part in path.strip("/").split("/"):
        part = part.split("?")[0]
        if part in SLUG_MAP:
            return SLUG_MAP[part]

    return None
