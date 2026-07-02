# toy
python feature_extractor.py \
    --ref-fasta toy_ref.fa \
    --alt-fasta toy_alt.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output variant_features.tsv \
    --batch-size 8 \
    --device cpu

# mpac emvar K562
python feature_extractor.py \
    --ref-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_K562_refs.fa \
    --alt-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_K562_alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPAC_emvar_K562_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu

# mpac emvar HEPG2
python feature_extractor.py \
    --ref-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_HEPG2_refs.fa \
    --alt-fasta /projectsp/f_ak1833_1/Mahmudur/files_to_predict/MPAC_emvar_HEPG2_alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPAC_emvar_HEPG2_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu

# mpac emvar SKNSH
python feature_extractor.py \
    --ref-fasta /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_SKNSH_refs.fa \
    --alt-fasta /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_SKNSH_alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPAC_emvar_SKNSH_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu


# MPRAVARDB ref alt locations:
# K562: /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-refs.fa, /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-alts.fa
# HepG2: /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-HEPG2-refs.fa, /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-HEPG2-alts.fa
# SKNSH: /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-SKNSH-refs.fa, /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-SKNSH-alts.fa


# extract features for MPRAVARDB K562
python feature_extractor.py \
    --ref-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-refs.fa \
    --alt-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-K562-alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPRAVARDB_K562_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu

# extract features for MPRAVARDB HEPG2
python feature_extractor.py \
    --ref-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-HEPG2-refs.fa \
    --alt-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-HEPG2-alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPRAVARDB_HEPG2_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu


# extract features for MPRAVARDB SKNSH
python feature_extractor.py \
    --ref-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-SKNSH-refs.fa \
    --alt-fasta /home/mr2320/malinois_inference_minimal/data/MPRAVARDB-SKNSH-alts.fa \
    --weights artifacts/model_all_folds.safetensors \
    --output MPRAVARDB_SKNSH_alphagenome_diff_features.tsv \
    --batch-size 32 \
    --device cpu


# train xgboost model using K562 data
python xgboost_train_and_test.py \
  --feature-file MPAC_emvar_K562_alphagenome_diff_features.tsv \
  --target-file /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv \
  --output-dir xgb_results_mpac_emvar_k562 \
  --feature-id-col id \
  --target-id-col variant \
  --target-col log2FC_skew \
  --n-estimators 1000 \
  --max-depth 5 \
  --learning-rate 0.03 \
  --early-stopping-rounds 100 \
  --n-jobs 1


# train xgboost model using HEPG2 data
python xgboost_train_and_test.py \
  --feature-file MPAC_emvar_HEPG2_alphagenome_diff_features.tsv \
  --target-file /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_HEPG2_combined.tsv \
  --output-dir xgb_results_mpac_emvar_hepg2 \
  --feature-id-col id \
  --target-id-col variant \
  --target-col log2FC_skew \
  --n-estimators 1000 \
  --max-depth 5 \
  --learning-rate 0.03 \
  --early-stopping-rounds 100 \
  --n-jobs 1


# make predictions on MPAC_emvar_K562_alphagenome_diff_features.tsv
python xgboost_make_predictions.py \
  --model-file xgb_results_mpac_emvar_k562/xgboost_model.json \
  --feature-file MPAC_emvar_K562_alphagenome_diff_features.tsv \
  --target-file /home/mr2320/malinois_inference_minimal/data/MPAC_emvar_K562_combined.tsv \
  --target-col log2FC_skew \
  --output-file xgb_results_mpac_emvar_k562/MPAC_emvar_K562_predictions.tsv


# make predictions on MPRAVARDB_K562_alphagenome_diff_features.tsv
python xgboost_make_predictions.py \
    --model-file xgb_results_mpac_emvar_k562/xgboost_model.json \
    --feature-file MPRAVARDB_K562_alphagenome_diff_features.tsv \
    --target-file /home/mr2320/malinois_inference_minimal/data/K562_processed.csv \
    --target-col log2FC \
    --output-file xgb_results_mpac_emvar_k562/MPRAVARDB_K562_predictions.tsv