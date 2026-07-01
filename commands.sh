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
