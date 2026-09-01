#!/usr/bin/env python3
"""Run the aesthetic-ai crawler locally on Windows.

Points the staging queue to data/staging.db inside the project folder.
Disables headless sources (brand_galleries, academy_portals) since they
require Playwright — those run on the cloud config.

Usage:
    python scripts/run_local.py
    python scripts/run_local.py --calibration-only
"""
import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DB = PROJECT_ROOT / "data" / "staging.db"
CONFIG = PROJECT_ROOT / "configs" / "crawler_local.yaml"

os.chdir(PROJECT_ROOT)

# Point the orchestrator's env-based fallback to the local data folder
os.environ["STORAGE_BASE_PATH"] = str(PROJECT_ROOT / "data")

# Skip SSL cert verification errors on corporate/VPN networks
os.environ["CRAWLER_SSL_VERIFY"] = "0"

QUEUE_DB.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-m", "crawler.orchestrator",
    "--queue-db", str(QUEUE_DB),
    "--config",   str(CONFIG),
] + sys.argv[1:]

print(f"Queue DB : {QUEUE_DB}")
print(f"Config   : {CONFIG}")
print(f"Running  : {' '.join(cmd)}\n")

sys.exit(subprocess.call(cmd))
