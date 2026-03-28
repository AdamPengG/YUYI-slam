#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/peng/isacc_slam/reference/OVO"
CONDA_SH="/home/peng/miniconda3/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "conda init script not found: ${CONDA_SH}" >&2
  exit 1
fi

source "${CONDA_SH}"
conda activate ovo5090

# Default to the RTX 5090. Callers can still override CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${ROOT_DIR}"
exec python -u run_eval.py "$@"
