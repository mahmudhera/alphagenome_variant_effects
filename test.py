from alphagenome_pytorch import AlphaGenome
from alphagenome_pytorch.utils.sequence import sequence_to_onehot_tensor

import torch


allowed_modalities = ['atac', 'dnase', 'rna_seq', 'chip_tf', 'chip_histone']


def batch_variant_feature_extraction(pred_ref, pred_alt) -> torch.Tensor:
    """
    Return difference between ALT and REF predictions for each sample in the batch.

    Returns
    -------
    torch.Tensor
        Shape: [batch_size]
    """

    diff_predictions = []

    pred_modalities = pred_ref.keys()
    for modality in pred_modalities:
        if modality in allowed_modalities:
            ref_tensor_128bp_res = pred_ref[modality][128]
            alt_tensor_128bp_res = pred_alt[modality][128]
            diff = alt_tensor_128bp_res - ref_tensor_128bp_res

            # the tensors are of shape [batch_size, num_positions, num_tracks]
            # get positions 0 and 1 only
            diff = diff[:, :3, :]

            diff_predictions.append(diff.reshape(diff.shape[0], -1))  # flatten to 1D per sample

    if not diff_predictions:
        raise ValueError("No tensors found in predictions")

    batch_size = diff_predictions[0].shape[0]
    for s in diff_predictions:
        if s.shape[0] != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_size} and {s.shape[0]}"
            )

    # list of tensors -- simply make them one dimensional by concatenating
    all_diff_predictions = torch.cat(diff_predictions, dim=1)

    return all_diff_predictions



model = AlphaGenome.from_pretrained("artifacts/model_all_folds.safetensors")

# random 200bp sequence
sequence_ref = 'AGCAGCGACGTGACATGACT'*10
# make it 4096bp by padding with Ns
sequence = sequence_ref + 'N'*(2048-len(sequence_ref))

sequence_alt = str(sequence)
sequence_alt = list(sequence_alt)
sequence_alt[100] = 'A' if sequence[100] != 'A' else 'C'
sequence_alt = ''.join(sequence_alt)

ref_sequences = [sequence, sequence, sequence, sequence, sequence]
alt_sequences = [sequence_alt, sequence_alt, sequence_alt, sequence_alt, sequence_alt]

batch_size = 128
num_sequences = len(ref_sequences)

# create batch of one-hot encoded sequences
for i in range(num_sequences // batch_size + (1 if num_sequences % batch_size else 0)):
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_sequences)
    batch_ref_sequences = ref_sequences[start_idx:end_idx]
    batch_alt_sequences = alt_sequences[start_idx:end_idx]

    ref_sequences_onehot = [sequence_to_onehot_tensor(seq) for seq in batch_ref_sequences]
    alt_sequences_onehot = [sequence_to_onehot_tensor(seq) for seq in batch_alt_sequences]

    # create batch
    batch_ref = torch.stack(ref_sequences_onehot)
    batch_alt = torch.stack(alt_sequences_onehot)

    # make predictions for the batch
    preds_ref = model.predict(batch_ref, organism_index=0)
    preds_alt = model.predict(batch_alt, organism_index=0)
    diff_features = batch_variant_feature_extraction(preds_ref, preds_alt)
    print(f"Variant features: {diff_features.shape}")