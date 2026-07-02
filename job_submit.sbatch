#!/bin/bash
#SBATCH --job-name=xgboost_mpac_emvar_sknsh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

#SBATCH --output=logs/xgboost_mpac_emvar_sknsh.out
#SBATCH --error=logs/xgboost_mpac_emvar_sknsh.err

# Optional but recommended
#SBATCH --partition=main-redhat

# Ensure log directory exists
mkdir -p logs

# Make stdout line-buffered / unbuffered
export PYTHONUNBUFFERED=1

source activate malinois_inference
export PYTHONPATH=$(pwd)

CMD=(
	python xgboost_train_and_test.py \
        --feature-file MPAC_emvar_SKNSH_alphagenome_diff_features.tsv \
        --target-file /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_SKNSH_combined.tsv \
        --output-dir xgb_results_mpac_emvar_sknsh \
        --feature-id-col id \
        --target-id-col variant \
        --target-col log2FC_skew \
        --n-estimators 1000 \
        --max-depth 5 \
        --learning-rate 0.03 \
        --early-stopping-rounds 100 \
        --n-jobs 1
)

echo "===== JOB COMMAND ====="
printf ' %q' "${CMD[@]}"
echo
echo "========================"

srun --unbuffered "${CMD[@]}"
