#!/bin/bash
#SBATCH --job-name=AG
#SBATCH --mem=48G
#SBATCH --time=24:00:00

#SBATCH --output=logs/AG.out
#SBATCH --error=logs/AG.err

# Optional but recommended
#SBATCH --partition=main-redhat

# Ensure log directory exists
mkdir -p logs

# Make stdout line-buffered / unbuffered
export PYTHONUNBUFFERED=1

source activate alphagenome
export PYTHONPATH=$(pwd)

CMD=(
	python feature_extractor.py \
        --ref-fasta toy_ref.fa \
        --alt-fasta toy_alt.fa \
        --weights artifacts/model_all_folds.safetensors \
        --output variant_features.tsv \
        --batch-size 8 \
        --device cpu
)

echo "===== JOB COMMAND ====="
printf ' %q' "${CMD[@]}"
echo
echo "========================"

srun --unbuffered "${CMD[@]}"
