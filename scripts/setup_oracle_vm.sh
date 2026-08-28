#!/usr/bin/env bash
# Oracle Arm A1 VM setup script.
# Run once as the default ubuntu user after first SSH login.
# Usage: bash scripts/setup_oracle_vm.sh

set -euo pipefail

BLOCK_MOUNT="/mnt/block/aesthetic-ai"
REPO_URL="https://github.com/YOUR_USERNAME/aesthetic-ai.git"   # update before running
PYTHON="python3.11"

echo "=== [1/9] System packages ==="
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  python3.11 python3.11-dev python3.11-venv python3-pip \
  git curl wget unzip \
  postgresql postgresql-contrib libpq-dev \
  build-essential cmake pkg-config \
  libgl1-mesa-glx libglib2.0-0 \
  libsm6 libxext6 libxrender-dev \
  ffmpeg \
  nginx \
  jq

echo "=== [2/9] Mount Oracle block volume ==="
# Assumes the block volume is already attached as /dev/sdb in Oracle Console.
# Skip if already mounted.
if ! mountpoint -q "$BLOCK_MOUNT"; then
  sudo mkfs.ext4 -F /dev/sdb 2>/dev/null || true   # skip if already formatted
  sudo mkdir -p "$BLOCK_MOUNT"
  sudo mount /dev/sdb "$BLOCK_MOUNT"
  # Persist across reboots
  echo "/dev/sdb $BLOCK_MOUNT ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
fi

echo "=== [3/9] Create block volume directories ==="
sudo mkdir -p \
  "$BLOCK_MOUNT/images/raw" \
  "$BLOCK_MOUNT/images/processed" \
  "$BLOCK_MOUNT/images/aligned" \
  "$BLOCK_MOUNT/staging" \
  "$BLOCK_MOUNT/dvc-store" \
  "$BLOCK_MOUNT/mlruns"
sudo chown -R "$(whoami):$(whoami)" "$BLOCK_MOUNT"

echo "=== [4/9] Clone repository ==="
cd ~
if [ ! -d "aesthetic-ai" ]; then
  git clone "$REPO_URL" aesthetic-ai
fi
cd aesthetic-ai

echo "=== [5/9] Python virtual environment ==="
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

echo "=== [6/9] Install Python packages (CPU torch first) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "=== [7/9] Playwright Chromium ==="
playwright install chromium
playwright install-deps chromium

echo "=== [8/9] Initialise DVC and Alembic ==="
dvc init --no-scm 2>/dev/null || true
dvc remote add -d oracle "$BLOCK_MOUNT/dvc-store" 2>/dev/null || true
alembic upgrade head

echo "=== [9/9] Pre-commit hooks ==="
pre-commit install
detect-secrets scan > .secrets.baseline

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Copy .env file: cp /path/to/.env ~/aesthetic-ai/.env"
echo "  2. Set HF_TOKEN, HF_REPO_ID, DATABASE_URL in .env"
echo "  3. Run: python scripts/download_models.py --skip-large"
echo "  4. Start MLflow: mlflow server --host 0.0.0.0 --port 5000 &"
echo "  5. Run calibration: python -m crawler.validation.calibration"
