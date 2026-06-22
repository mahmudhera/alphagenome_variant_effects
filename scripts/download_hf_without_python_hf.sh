#!/usr/bin/env bash
set -euo pipefail

FILE_NAME="${1:-model_all_folds.safetensors}"
OUT_PATH="${2:-artifacts/${FILE_NAME}}"
REPO="${HF_REPO:-gtca/alphagenome_pytorch}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Set HF_TOKEN before running this script." >&2
  echo "Example: export HF_TOKEN='hf_...'" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUT_PATH}")"
URL="https://huggingface.co/${REPO}/resolve/main/${FILE_NAME}"

curl -L \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "${URL}" \
  -o "${OUT_PATH}"

echo "Wrote ${OUT_PATH}"
