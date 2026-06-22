#!/usr/bin/env bash
set -euo pipefail
mkdir -p results
python -m ag_variant_effects \
  --backend mock \
  --input examples/input.tsv \
  --input-sep auto \
  --ref-col ref_seq \
  --alt-col alt_seq \
  --id-col variant_id \
  --effects-out results/mock_variant_effects.tsv \
  --predictions-out results/mock_all_predictions.zip \
  --device cpu \
  --batch-size 2
