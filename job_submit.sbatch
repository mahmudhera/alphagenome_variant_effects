#!/bin/bash
#SBATCH --job-name=AGHEPG2
#SBATCH --mem=48G
#SBATCH --time=24:00:00

#SBATCH --output=logs/AGHEPG2.out
#SBATCH --error=logs/AGHEPG2.err

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
    --ref-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_HEPG2_refs.fa \
    --alt-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_HEPG2_alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPAC_emvar_HEPG2_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu
)

echo "===== JOB COMMAND ====="
printf ' %q' "${CMD[@]}"
echo
echo "========================"

srun --unbuffered "${CMD[@]}"
