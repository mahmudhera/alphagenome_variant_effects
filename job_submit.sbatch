#!/bin/bash
#SBATCH --job-name=AGK562MPRAVARDB
#SBATCH --mem=48G
#SBATCH --time=24:00:00

#SBATCH --output=logs/AGK562MPRAVARDB.out
#SBATCH --error=logs/AGK562MPRAVARDB.err

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
    --ref-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-refs.fa \
    --alt-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPRAVARDB_K562_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu
)

echo "===== JOB COMMAND ====="
printf ' %q' "${CMD[@]}"
echo
echo "========================"

srun --unbuffered "${CMD[@]}"
