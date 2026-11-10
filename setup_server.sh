#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
source ./env.sh
if [ ! -x .python310/bin/python ]; then
  conda create --prefix "$PWD/.python310" python=3.10.22 -y
fi
if [ ! -x .venv/bin/python ]; then
  .python310/bin/python -m venv .venv
fi
.venv/bin/python -m pip install 'pip==25.0.1' 'setuptools==75.8.0'
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python train_dual.py --help > logs/train-cli-check.log
.venv/bin/python check_project.py
echo 'SETUP VERIFIED. No training started.'
if [ -f .secrets/download-url ]; then
  .venv/bin/python prepare_data.py --download
fi
