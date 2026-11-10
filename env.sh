#!/usr/bin/env bash
export TORCH_HOME="$PWD/.cache/torch"
export HF_HOME="$PWD/.cache/huggingface"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export YOLOV5_CONFIG_DIR="$PWD/.cache/yolo"
export XDG_CACHE_HOME="$PWD/.cache"
export PIP_CACHE_DIR="$PWD/.cache/pip"
export CONDA_PKGS_DIRS="$PWD/.cache/conda-pkgs"
export TMPDIR="$PWD/.cache/tmp"
export WANDB_MODE=disabled
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" logs
