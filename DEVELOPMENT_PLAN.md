# Development Plan
## Facial Aesthetic Treatment Prediction System

**Version:** 0.1  
**Date:** 2026-08-28  
**Status:** Active  

All decisions, constraints, and architecture references are in [PRD.md](../classifier/PRD.md).

---

## Infrastructure Setup (Before Phase 1)

Must be completed before any code runs in production.

| Task | Detail | Done |
|---|---|---|
| Provision Oracle Arm A1 instance | 4 OCPUs, 24GB RAM, Always Free. Do NOT select AMD E2.1.Micro. | [ ] |
| Attach and mount 200GB block volume | Mount at `/mnt/block/aesthetic-ai` | [ ] |
| Create directory structure on block volume | `/mnt/block/aesthetic-ai/{images,processed,aligned,staging}` | [ ] |
| Install system dependencies | Python 3.11, git, pip, Playwright Chromium, PostgreSQL | [ ] |
| Clone repo and install requirements | `pip install -r requirements.txt && playwright install chromium` | [ ] |
| Initialise DVC | `dvc init && dvc remote add -d oracle /mnt/block/aesthetic-ai/dvc-store` | [ ] |
| Start MLflow tracking server | `mlflow server --host 0.0.0.0 --port 5000` (systemd service) | [ ] |
| Create `.env` file | `HF_TOKEN`, `HF_REPO_ID`, `DATABASE_URL` | [ ] |
| Initialise Alembic and run first migration | `alembic upgrade head` | [ ] |
| Verify Kaggle account GPU quota | Confirm T4 access and ~30 hrs/week limit | [ ] |

---

## Phase 1 — Web Crawler

**Goal:** Collect 5,000–10,000 clean validated image pairs.  
**Location:** `crawler/`  
**Tests:** `tests/crawler/`

### Step 1.1 — Core infrastructure (build first, everything depends on this) ✅ COMPLETE

| Task | File | Test |
|---|---|---|
| Implement staging queue (SQLite) | `crawler/storage/staging_queue.py` ✅ | `tests/crawler/test_staging_queue.py` ✅ |
| Implement DB + filesystem writer | `crawler/storage/writer.py` ✅ | `tests/crawler/test_writer.py` ✅ |
| Implement consent classifier (Tier 1/2/3) | `crawler/consent/classifier.py` ✅ | `tests/crawler/test_consent_classifier.py` ✅ |
| Implement source factory (loads config, instantiates sources) | `crawler/factory.py` ✅ | `tests/crawler/test_factory.py` ✅ |
| Implement crawler orchestrator (runs all enabled sources) | `crawler/orchestrator.py` ✅ | `tests/crawler/test_orchestrator.py` ✅ |

**Acceptance criteria:**
- Staging queue persists across process restarts
- Writer is idempotent (duplicate URL raises, does not duplicate record)
- Consent classifier correctly tiers all test fixtures
- Factory reads `configs/crawler.yaml` and returns correct source instances

---

### Step 1.2 — Calibration bootstrap (run before full crawl) ✅ COMPLETE

| Task | File | Test |
|---|---|---|
| Implement Layer 1 structural heuristics | `crawler/validation/structural.py` ✅ | `tests/crawler/validation/test_structural.py` ✅ |
| Implement Layer 2 multilingual NLP pairing | `crawler/validation/nlp_pairing.py` ✅ | `tests/crawler/validation/test_nlp_pairing.py` ✅ |
| Implement Layer 3 ArcFace face similarity | `crawler/validation/face_similarity.py` ✅ | `tests/crawler/validation/test_face_similarity.py` ✅ |
| Implement ordering gate (two-gate hybrid) | `crawler/validation/ordering.py` ✅ | `tests/crawler/validation/test_ordering.py` ✅ |
| Implement calibration script | `crawler/validation/calibration.py` ✅ | `tests/crawler/validation/test_calibration.py` ✅ |

**Calibration procedure:**
1. Enable RealSelf source only (`calibration_source: true`)
2. Run: `python -m crawler.validation.calibration`
3. Script crawls 200 RealSelf pairs (known-good)
4. Generates known-bad pairs by cross-pairing
5. Sweeps thresholds to ≥95% precision
6. Writes results to `configs/validation.yaml` with `calibrated: true`
7. Full crawl only proceeds after `calibrated: true`

**Acceptance criteria:**
- All three validation layers pass their unit tests with known fixtures
- Calibration script runs without error and produces valid YAML output
- `configs/validation.yaml` `calibrated` flag set to `true`

---

### Step 1.3 — Validation worker ✅ COMPLETE

| Task | File | Test |
|---|---|---|
| Implement async validation worker | `crawler/validation/worker.py` ✅ | `tests/crawler/validation/test_worker.py` ✅ |
| Implement pair downloader (downloads image files to block storage) | `crawler/storage/downloader.py` ✅ | `tests/crawler/test_downloader.py` ✅ |

**Architecture:** Crawler process writes to staging queue → worker polls queue → runs three-layer validation → writes validated pairs to main DB → downloads image files to `/mnt/block/aesthetic-ai/images/`.

**Acceptance criteria:**
- Worker and crawler run as independent processes with no shared state
- Worker correctly routes Tier 3 images to quarantine
- Quarantine table has no join path to training tables (verified by test)

---

### Step 1.4 — Source modules (all built simultaneously) ✅ COMPLETE

Build all 13 enabled sources in parallel. Each source requires:
- Implementation of `iter_page_urls()` and `extract_pairs_from_page()`
- Integration test: crawl 5 real pages, verify output schema
- robots.txt compliance test

| Source | File | Priority | Notes |
|---|---|---|---|
| RealSelf | `crawler/sources/realself.py` | HIGHEST | Calibration source — complete first |
| Open Access Pubs | `crawler/sources/open_access_pubs.py` | HIGH | Tier 1, high quality |
| Brand Galleries | `crawler/sources/brand_galleries.py` | HIGH | Tier 1, Playwright required |
| Academy Portals | `crawler/sources/academy_portals.py` | HIGH | Tier 1, Playwright required |
| Professional Societies | `crawler/sources/professional_societies.py` | HIGH | Tier 1 |
| Clinic Sites | `crawler/sources/clinic_sites.py` | MEDIUM | Tier 2, variable structure |
| Reddit | `crawler/sources/reddit.py` | MEDIUM | Tier 2 |
| Beauty Media | `crawler/sources/beauty_media.py` | MEDIUM | Tier 2 |
| Review Platforms | `crawler/sources/review_platforms.py` | MEDIUM | Tier 2 |
| Korean Sources | `crawler/sources/korean_sources.py` | MEDIUM | KO language |
| Brazilian Sources | `crawler/sources/brazilian_sources.py` | MEDIUM | PT language |
| Open Datasets | `crawler/sources/open_datasets.py` | MEDIUM | Tier 1 |
| Pinterest | `crawler/sources/pinterest.py` | LOW | Needs headless upgrade |
| Instagram | `crawler/sources/instagram.py` | DISABLED | Build only, do not enable |
| TikTok Static | `crawler/sources/tiktok_static.py` | DISABLED | Build only, do not enable |

---

### Step 1.5 — Video source scaffold (deferred)

| Task | File | Activation |
|---|---|---|
| Scaffold YouTube source module | `crawler/sources/youtube.py` | Auto if pairs < 5,000 |
| Scaffold TikTok video source module | `crawler/sources/tiktok_video.py` | Auto if pairs < 5,000 |
| Implement frame extractor (yt-dlp + OpenCV) | `crawler/video/frame_extractor.py` | Same gate |
| Implement audio transcriber (Whisper) | `crawler/video/transcriber.py` | Same gate |

---

### Phase 1 Completion Gate

Run: `python -m crawler.status`

- [ ] `configs/validation.yaml` → `calibrated: true`
- [ ] Clean validated pairs in DB ≥ 5,000 (target: 10,000)
- [ ] Quarantine table populated but isolated
- [ ] All source integration tests pass
- [ ] DVC snapshot tagged: `dvc tag raw-v1`

---

## Phase 2 — Image Processing Pipeline ✅ COMPLETE

**Goal:** Quality-filter, align, and annotate all collected image pairs.  
**Location:** `pipeline/`  
**Tests:** `tests/pipeline/`  
**Dependency:** Phase 1 complete gate passed.

### Step 2.1 — Quality scoring

| Task | File | Test |
|---|---|---|
| Laplacian variance blur detector | `pipeline/quality/scorer.py` | `tests/pipeline/test_scorer.py` |
| Lighting uniformity scorer | `pipeline/quality/scorer.py` | same |
| Resolution checker | `pipeline/quality/scorer.py` | same |
| Quality grade aggregator (PASS/FAIL) | `pipeline/quality/scorer.py` | same |

**Thresholds:** read from `configs/pipeline.yaml`. No hardcoded values.

---

### Step 2.2 — Face detection and cropping

| Task | File | Test |
|---|---|---|
| MTCNN face detector | `pipeline/detection/detector.py` | `tests/pipeline/test_detector.py` |
| RetinaFace fallback | `pipeline/detection/detector.py` | same |
| Multi-face / no-face discard logic | `pipeline/detection/detector.py` | same |

---

### Step 2.3 — Landmark extraction

| Task | File | Test |
|---|---|---|
| MediaPipe 478-point mesh extraction | `pipeline/landmarks/extractor.py` | `tests/pipeline/test_extractor.py` |
| Landmark DB write | `pipeline/landmarks/extractor.py` | same |

---

### Step 2.4 — Alignment and normalisation

| Task | File | Test |
|---|---|---|
| Eye-distance normalisation to 512×512 | `pipeline/alignment/normaliser.py` | `tests/pipeline/test_normaliser.py` |
| In-plane rotation correction | `pipeline/alignment/normaliser.py` | same |
| Aligned image write to `/mnt/block/aesthetic-ai/aligned/` | `pipeline/alignment/normaliser.py` | same |

---

### Step 2.5 — Pair-level validation (post-pipeline)

| Task | File | Test |
|---|---|---|
| Head pose angle extractor (yaw/pitch/roll) | `pipeline/alignment/pose.py` | `tests/pipeline/test_pose.py` |
| Pair angle match validator (±15° threshold) | `pipeline/alignment/pose.py` | same |

---

### Step 2.6 — Zone segmentation

| Task | File | Test |
|---|---|---|
| Landmark-to-zone mapper (reads `configs/zones.yaml`) | `pipeline/segmentation/zone_mapper.py` | `tests/pipeline/test_zone_mapper.py` |
| Zone label writer to DB | `pipeline/segmentation/zone_mapper.py` | same |

---

### Step 2.7 — Pipeline orchestrator

| Task | File |
|---|---|
| End-to-end pipeline runner (processes all validated pairs in DB) | `pipeline/runner.py` |
| Progress tracking and resume (skip already-processed pairs) | `pipeline/runner.py` |

---

### Phase 2 Completion Gate

- [ ] All pairs in DB have quality scores
- [ ] All PASS pairs have aligned images at 512×512 in block storage
- [ ] All PASS pairs have 478 landmarks stored
- [ ] All PASS pairs have zone labels
- [ ] Post-pipeline clean pair count ≥ 5,000
- [ ] DVC snapshot tagged: `dvc tag processed-v1`

---

## Phase 3 — Database and Annotation

**Goal:** Finalize schema, auto-label treatments, verify training set integrity.  
**Location:** `database/`  
**Tests:** `tests/database/`  
**Dependency:** Phase 1 core infrastructure (Step 1.1) complete.

> Phase 3 runs in parallel with Phases 1 and 2 — schema and migrations are built first.

### Step 3.1 — Schema and migrations

| Task | File | Test |
|---|---|---|
| SQLAlchemy models | `database/models/__init__.py` | `tests/database/test_models.py` |
| Alembic initial migration | `database/migrations/versions/001_initial.py` | `tests/database/test_migrations.py` |
| Database session factory | `database/session.py` | — |

---

### Step 3.2 — Auto-labelling

| Task | File | Test |
|---|---|---|
| Treatment type extractor (multilingual NLP on source text) | `database/labelling/treatment_labeller.py` | `tests/database/test_treatment_labeller.py` |
| Treatment brand extractor | `database/labelling/treatment_labeller.py` | same |
| Zone auto-labeller (from landmark data) | `database/labelling/zone_labeller.py` | `tests/database/test_zone_labeller.py` |

**Rule:** Labels below confidence threshold written as NULL — never guessed.

---

### Step 3.3 — Query helpers and DVC export

| Task | File | Test |
|---|---|---|
| Training set query (excludes quarantine, LOW ordering confidence) | `database/queries/training_set.py` | `tests/database/test_training_set.py` |
| DVC dataset export script | `database/queries/export.py` | `tests/database/test_export.py` |
| Class balance reporter | `database/queries/balance.py` | `tests/database/test_balance.py` |

---

### Phase 3 Completion Gate

- [ ] `alembic upgrade head` runs cleanly
- [ ] All model CRUD tests pass
- [ ] Quarantine isolation test passes
- [ ] Training set query excludes all quarantine records
- [ ] DVC export produces deterministic snapshot
- [ ] Class distribution report generated — weighted sampler configured if needed

---

## Phase 4 — Model Development

**Goal:** Train InstructPix2Pix + LoRA + IP-Adapter to predict aesthetic treatment outcomes.  
**Location:** `model/`  
**Tests:** `tests/model/`  
**Dependency:** Phase 2 and 3 completion gates.

### Step 4.1 — Dataset and data loader

| Task | File | Test |
|---|---|---|
| PyTorch Dataset class (reads from DB + block storage) | `model/training/dataset.py` | `tests/model/test_dataset.py` |
| Data augmentation pipeline | `model/training/augmentation.py` | `tests/model/test_augmentation.py` |
| Weighted sampler for class balance | `model/training/dataset.py` | `tests/model/test_dataset.py` |
| Train/val/test split (80/10/10, stratified) | `model/training/dataset.py` | same |

---

### Step 4.2 — InstructPix2Pix LoRA fine-tuning

| Task | File | Test |
|---|---|---|
| LoRA adapter setup | `model/instruct_pix2pix/lora.py` | `tests/model/test_lora.py` |
| IP-Adapter integration | `model/ip_adapter/adapter.py` | `tests/model/test_ip_adapter.py` |
| Training loop with gradient checkpointing + fp16 | `model/training/trainer.py` | `tests/model/test_trainer.py` |
| MLflow logging (loss, metrics, DVC hash, config hash) | `model/training/trainer.py` | `tests/model/test_mlflow_logging.py` |
| Epoch checkpoint save + HuggingFace Hub push | `model/training/trainer.py` | `tests/model/test_checkpoint.py` |

**Run on Kaggle:** Upload DVC-pinned dataset, run `model/training/train.py`. Checkpoint pushed to HuggingFace Hub after each run.

---

### Step 4.3 — CycleGAN comparison baseline

| Task | File | Test |
|---|---|---|
| Generator and discriminator | `model/cyclegan/model.py` | `tests/model/test_cyclegan.py` |
| CycleGAN training loop | `model/cyclegan/trainer.py` | `tests/model/test_cyclegan.py` |

**Note:** CycleGAN is comparison-only. Not the primary product model.

---

### Step 4.4 — Evaluation

| Task | File | Test |
|---|---|---|
| FID scorer | `model/evaluation/metrics.py` | `tests/model/test_metrics.py` |
| LPIPS scorer | `model/evaluation/metrics.py` | same |
| SSIM scorer | `model/evaluation/metrics.py` | same |
| Evaluation runner on held-out test set | `model/evaluation/evaluator.py` | `tests/model/test_evaluator.py` |
| Clinical rating protocol document | `model/evaluation/clinical_rating_protocol.md` | — |

**Clinical evaluation:** Run before production release. 3 practitioners, 60-pair blind test (30 per treatment), 3 dimensions (Realism / Treatment accuracy / Identity preservation), ≥3.5 mean threshold.

---

### Phase 4 Completion Gate

- [ ] Training smoke test passes (1 epoch, 10 batches)
- [ ] FID, LPIPS, SSIM computed on test set; results logged to MLflow
- [ ] Primary model meets minimum thresholds (FID < 80, LPIPS < 0.35, SSIM > 0.65)
- [ ] Clinical expert rating ≥ 3.0 mean across all dimensions
- [ ] Best checkpoint tagged `best` on HuggingFace Hub
- [ ] CycleGAN comparison metrics logged alongside primary model

---

## Phase 5 — Inference API

**Goal:** Deploy FastAPI endpoint on Oracle Arm A1 accepting patient photo + treatment, returning predicted outcome.  
**Location:** `api/`  
**Tests:** `tests/api/`  
**Dependency:** Phase 4 completion gate + `best` checkpoint on HuggingFace Hub.

### Step 5.1 — Pydantic schemas

| Task | File | Test |
|---|---|---|
| Request schema (image + treatment + optional zone) | `api/schemas/request.py` | `tests/api/test_schemas.py` |
| Response schema (image bytes + confidence + metadata) | `api/schemas/response.py` | `tests/api/test_schemas.py` |

---

### Step 5.2 — Preprocessing

| Task | File | Test |
|---|---|---|
| Image preprocessing pipeline (quality → detect → align) | `api/inference/preprocessor.py` | `tests/api/test_preprocessor.py` |
| Structured error responses for failed preprocessing | `api/inference/preprocessor.py` | same |

---

### Step 5.3 — Model loader and inference

| Task | File | Test |
|---|---|---|
| Model loader (loads from HuggingFace Hub at startup) | `api/inference/loader.py` | `tests/api/test_loader.py` |
| Inference engine (preprocessed image → prediction) | `api/inference/engine.py` | `tests/api/test_engine.py` |
| Post-processing (blend, JPEG encode, confidence score) | `api/inference/engine.py` | same |

---

### Step 5.4 — API routes

| Task | File | Test |
|---|---|---|
| `POST /predict` route | `api/routers/predict.py` | `tests/api/test_predict.py` |
| `GET /health` route | `api/routers/health.py` | `tests/api/test_health.py` |
| Rate limiting middleware (10 req/min per IP) | `api/middleware.py` | `tests/api/test_rate_limit.py` |
| FastAPI app assembly | `api/main.py` | — |

---

### Step 5.5 — Deployment on Oracle Arm A1

| Task | Detail |
|---|---|
| Create systemd service for FastAPI | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| Configure Oracle VCN security rules | Open port 8000 (or 443 with nginx reverse proxy) |
| Load test | Verify p95 latency < 10s under 5 concurrent requests |

---

### Phase 5 Completion Gate

- [ ] All API tests pass
- [ ] `POST /predict` happy-path returns 200 with valid image
- [ ] Rate limiting returns 429 on 11th request in 60s
- [ ] `GET /health` returns `model_loaded: true`
- [ ] p95 latency < 10s on Oracle Arm A1 (measured with locust or wrk)
- [ ] Service survives restart (model reloads from HuggingFace Hub)

---

## Testing Checkpoints Summary

| After | Run | Must pass |
|---|---|---|
| Step 1.1 | `pytest tests/crawler/test_staging_queue.py tests/crawler/test_writer.py tests/crawler/test_consent_classifier.py` | 100% |
| Step 1.2 | `pytest tests/crawler/validation/` | 100% |
| Step 1.4 (each source) | `pytest tests/crawler/sources/test_{source}.py` | 100% |
| Phase 1 gate | `pytest tests/crawler/` | 100% |
| Phase 2 gate | `pytest tests/pipeline/` | 100% |
| Phase 3 gate | `pytest tests/database/` | 100% |
| Phase 4 gate | `pytest tests/model/` | 100% |
| Phase 5 gate | `pytest tests/api/` | 100% |
| Full suite | `pytest --cov` | ≥ 80% coverage |

---

## Key File References

| Purpose | File |
|---|---|
| Abstract crawler base | [crawler/base.py](crawler/base.py) |
| Crawler config | [configs/crawler.yaml](configs/crawler.yaml) |
| Validation thresholds (written by calibration) | [configs/validation.yaml](configs/validation.yaml) |
| Pipeline config | [configs/pipeline.yaml](configs/pipeline.yaml) |
| Zone taxonomy | [configs/zones.yaml](configs/zones.yaml) |
| Model hyperparameters | [configs/model.yaml](configs/model.yaml) |
| API config | [configs/api.yaml](configs/api.yaml) |
| ORM models | [database/models/__init__.py](database/models/__init__.py) |
| Clinical rating protocol | [model/evaluation/clinical_rating_protocol.md](model/evaluation/clinical_rating_protocol.md) |
| PRD | [PRD.md](../classifier/PRD.md) |
