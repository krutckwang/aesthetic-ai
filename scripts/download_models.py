"""
Pre-download all model weights to local cache before pipeline runs.

Run once on the Oracle Arm A1 VM after environment setup:
    python scripts/download_models.py

Downloads:
  - paraphrase-multilingual-MiniLM-L12-v2  (Layer 2 NLP validation)
  - InsightFace buffalo_l / ArcFace         (Layer 3 face similarity + identity loss)
  - MTCNN weights                           (face detection)
  - timbrooks/instruct-pix2pix             (Phase 4 base model)
  - h94/IP-Adapter face weights            (identity preservation)

All weights are cached to ~/.cache/huggingface/ and ~/.insightface/
No GPU required — this runs on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def download_sentence_transformer() -> None:
    logger.info("Downloading paraphrase-multilingual-MiniLM-L12-v2 ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    # Smoke test
    embedding = model.encode(["before treatment", "after treatment"])
    assert embedding.shape == (2, 384), f"Unexpected embedding shape: {embedding.shape}"
    logger.success("paraphrase-multilingual-MiniLM-L12-v2 ready.")


def download_insightface() -> None:
    logger.info("Downloading InsightFace buffalo_l (ArcFace) ...")
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    logger.success("InsightFace buffalo_l ready.")


def download_mtcnn() -> None:
    logger.info("Downloading MTCNN weights ...")
    from facenet_pytorch import MTCNN
    import torch
    mtcnn = MTCNN(device="cpu")
    logger.success("MTCNN ready.")


def download_instruct_pix2pix() -> None:
    logger.info("Downloading timbrooks/instruct-pix2pix ...")
    logger.warning(
        "This model is ~7GB. Ensure sufficient disk space on Oracle block volume."
    )
    from huggingface_hub import snapshot_download
    path = snapshot_download(
        repo_id="timbrooks/instruct-pix2pix",
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # PyTorch only
    )
    logger.success(f"instruct-pix2pix cached at: {path}")


def download_ip_adapter() -> None:
    logger.info("Downloading h94/IP-Adapter (face model) ...")
    from huggingface_hub import hf_hub_download
    # Download only the face-specific weights, not the full repo
    path = hf_hub_download(
        repo_id="h94/IP-Adapter",
        subfolder="models",
        filename="ip-adapter-full-face_sd15.bin",
    )
    logger.success(f"IP-Adapter weights cached at: {path}")


def download_mediapipe() -> None:
    """MediaPipe downloads its model files on first use — trigger that here."""
    logger.info("Warming up MediaPipe face mesh ...")
    import mediapipe as mp
    import numpy as np
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
    ) as face_mesh:
        # Process a blank image to trigger model download
        dummy = np.zeros((256, 256, 3), dtype=np.uint8)
        face_mesh.process(dummy)
    logger.success("MediaPipe face mesh ready.")


def main() -> None:
    steps = [
        ("sentence-transformers NLP model", download_sentence_transformer),
        ("MediaPipe face mesh", download_mediapipe),
        ("MTCNN face detector", download_mtcnn),
        ("InsightFace ArcFace (buffalo_l)", download_insightface),
        ("IP-Adapter identity weights", download_ip_adapter),
        ("InstructPix2Pix base model", download_instruct_pix2pix),
    ]

    # Allow skipping the large model downloads with --skip-large flag
    skip_large = "--skip-large" in sys.argv
    if skip_large:
        steps = [(name, fn) for name, fn in steps if fn != download_instruct_pix2pix]
        logger.info("--skip-large: skipping instruct-pix2pix download (run on Kaggle instead).")

    failed: list[str] = []
    for name, fn in steps:
        try:
            fn()
        except Exception as exc:
            logger.error(f"FAILED: {name} — {exc}")
            failed.append(name)

    print()
    if failed:
        logger.error(f"Downloads failed: {failed}")
        sys.exit(1)
    else:
        logger.success("All model weights downloaded and verified.")


if __name__ == "__main__":
    main()
