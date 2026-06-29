#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from alphagenome_pytorch import AlphaGenome


ALLOWED_MODALITIES = [
    "atac",
    "dnase",
    "rna_seq",
    "chip_tf",
    "chip_histone",
]


def read_fasta(path: str) -> Tuple[List[str], List[str]]:
    """
    Read FASTA file.

    Returns
    -------
    ids : list[str]
    seqs : list[str]
    """

    ids = []
    seqs = []

    current_id = None
    current_seq = []

    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None:
                    ids.append(current_id)
                    seqs.append("".join(current_seq).upper())

                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

    if current_id is not None:
        ids.append(current_id)
        seqs.append("".join(current_seq).upper())

    if not ids:
        raise ValueError(f"No sequences found in FASTA: {path}")

    return ids, seqs


def validate_ref_alt_fastas(
    ref_ids: List[str],
    ref_seqs: List[str],
    alt_ids: List[str],
    alt_seqs: List[str],
    require_matching_ids: bool = True,
) -> None:
    if len(ref_seqs) != len(alt_seqs):
        raise ValueError(
            f"Different number of sequences: REF={len(ref_seqs)}, ALT={len(alt_seqs)}"
        )

    if require_matching_ids:
        for i, (rid, aid) in enumerate(zip(ref_ids, alt_ids)):
            if rid != aid:
                raise ValueError(
                    f"FASTA IDs differ at index {i}: REF={rid}, ALT={aid}"
                )


def make_onehot_batch(sequences: List[str], device: torch.device) -> torch.Tensor:
    """
    Convert list of DNA sequences into AlphaGenome input tensor.

    Returns
    -------
    torch.Tensor
        Shape: [batch_size, sequence_length, 4]
    """

    if len(sequences) == 0:
        raise ValueError("Empty sequence batch")

    # A: 0, C: 1, G: 2, T: 3, N: all zeros
    base_to_idx = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
    }

    seq_len = len(sequences[0])

    batch = torch.zeros(
        (len(sequences), seq_len, 4),
        dtype=torch.float32,
        device=device,
    )

    for i, seq in enumerate(sequences):
        seq = seq.upper()
        for j, base in enumerate(seq):
            idx = base_to_idx.get(base)
            if idx is not None:
                batch[i, j, idx] = 1.0
            # N or unknown bases remain all zeros

    return batch.to(device)


def batch_variant_feature_extraction(
    pred_ref,
    pred_alt,
    allowed_modalities: List[str],
    num_bins: int = 3,
    resolution: int = 128,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Extract flattened ALT - REF features.

    For each allowed modality:
        pred[modality][resolution] has shape:
            [batch_size, num_positions, num_tracks]

    This function keeps the first `num_bins` bins and all tracks.

    Returns
    -------
    features : torch.Tensor
        Shape: [batch_size, total_features]

    feature_names : list[str]
        Names for each feature column.
    """

    diff_predictions = []
    feature_names = []

    for modality in pred_ref.keys():
        if modality not in allowed_modalities:
            continue

        if modality not in pred_alt:
            raise KeyError(f"Missing modality in ALT predictions: {modality}")

        if resolution not in pred_ref[modality]:
            raise KeyError(f"Missing resolution {resolution} for REF modality {modality}")

        if resolution not in pred_alt[modality]:
            raise KeyError(f"Missing resolution {resolution} for ALT modality {modality}")

        ref_tensor = pred_ref[modality][resolution]
        alt_tensor = pred_alt[modality][resolution]

        if ref_tensor.shape != alt_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {modality}[{resolution}]: "
                f"REF={ref_tensor.shape}, ALT={alt_tensor.shape}"
            )

        if ref_tensor.ndim != 3:
            raise ValueError(
                f"Expected tensor shape [batch, positions, tracks] for "
                f"{modality}[{resolution}], got {ref_tensor.shape}"
            )

        batch_size, n_positions, n_tracks = ref_tensor.shape

        if n_positions < num_bins:
            raise ValueError(
                f"{modality}[{resolution}] has only {n_positions} bins, "
                f"but num_bins={num_bins}"
            )

        diff = alt_tensor - ref_tensor

        # Keep first num_bins bins and all tracks.
        diff = diff[:, :num_bins, :]

        # Shape: [batch_size, num_bins * n_tracks]
        diff_flat = diff.reshape(batch_size, -1)

        diff_predictions.append(diff_flat)

        for bin_idx in range(num_bins):
            for track_idx in range(n_tracks):
                feature_names.append(
                    f"{modality}_res{resolution}_bin{bin_idx}_track{track_idx}"
                )

    if not diff_predictions:
        raise ValueError("No allowed prediction tensors found")

    batch_size = diff_predictions[0].shape[0]
    for x in diff_predictions:
        if x.shape[0] != batch_size:
            raise ValueError(
                f"Inconsistent batch sizes found: {batch_size} and {x.shape[0]}"
            )

    all_features = torch.cat(diff_predictions, dim=1)

    if all_features.shape[1] != len(feature_names):
        raise RuntimeError(
            f"Feature-name mismatch: features={all_features.shape[1]}, "
            f"names={len(feature_names)}"
        )

    return all_features, feature_names


def save_npy(path: str, features: np.ndarray) -> None:
    np.save(path, features)


def save_npz(
    path: str,
    features: np.ndarray,
    ids: List[str],
    feature_names: List[str],
) -> None:
    np.savez_compressed(
        path,
        features=features,
        ids=np.array(ids, dtype=object),
        feature_names=np.array(feature_names, dtype=object),
    )


def save_tsv(
    path: str,
    features: np.ndarray,
    ids: List[str],
    feature_names: List[str],
) -> None:
    with open(path, "w") as out:
        out.write("id\t" + "\t".join(feature_names) + "\n")

        for seq_id, row in zip(ids, features):
            row_str = "\t".join(str(float(x)) for x in row)
            out.write(f"{seq_id}\t{row_str}\n")


def parse_modalities(s: str) -> List[str]:
    if s.lower() == "default":
        return list(ALLOWED_MODALITIES)

    modalities = [x.strip() for x in s.split(",") if x.strip()]
    if not modalities:
        raise ValueError("No modalities provided")

    return modalities


def predict_batch(
    model,
    batch: torch.Tensor,
    organism_index: int,
    heads: List[str],
    resolution: int,
):
    """
    Run AlphaGenome prediction.

    Uses only requested heads and 128bp resolution by default.
    """

    organism = torch.full(
        (batch.shape[0],),
        int(organism_index),
        dtype=torch.long,
        device=batch.device,
    )

    with torch.no_grad():
        try:
            return model.predict(
                batch,
                organism_index=organism,
                heads=tuple(heads),
                resolutions=(resolution,),
            )
        except TypeError:
            # Fallback for package versions that do not accept heads/resolutions.
            return model.predict(
                batch,
                organism_index=organism,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Extract AlphaGenome ALT-REF variant features from paired FASTA files."
    )

    parser.add_argument(
        "--ref-fasta",
        required=True,
        help="FASTA file containing REF sequences.",
    )

    parser.add_argument(
        "--alt-fasta",
        required=True,
        help="FASTA file containing ALT sequences.",
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to AlphaGenome PyTorch .safetensors checkpoint.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path. Extension determines format: .npy, .npz, or .tsv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for prediction.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use.",
    )

    parser.add_argument(
        "--organism-index",
        type=int,
        default=0,
        help="Organism index. Usually 0 for human.",
    )

    parser.add_argument(
        "--num-bins",
        type=int,
        default=3,
        help="Number of 128bp bins to keep from each modality.",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=128,
        help="Prediction resolution to use.",
    )

    parser.add_argument(
        "--modalities",
        default="default",
        help=(
            "Comma-separated modalities to use. "
            "Default: atac,dnase,rna_seq,chip_tf,chip_histone"
        ),
    )

    parser.add_argument(
        "--allow-mismatched-ids",
        action="store_true",
        help="Allow REF and ALT FASTA IDs to differ. Matching is then by order.",
    )

    args = parser.parse_args()

    allowed_modalities = parse_modalities(args.modalities)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Using modalities: {allowed_modalities}")

    ref_ids, ref_seqs = read_fasta(args.ref_fasta)
    alt_ids, alt_seqs = read_fasta(args.alt_fasta)

    validate_ref_alt_fastas(
        ref_ids,
        ref_seqs,
        alt_ids,
        alt_seqs,
        require_matching_ids=not args.allow_mismatched_ids,
    )

    print(f"Loaded {len(ref_seqs)} REF/ALT sequence pairs")

    model = AlphaGenome.from_pretrained(args.weights)
    model = model.to(device)
    model.eval()

    all_features = []
    feature_names = None

    n = len(ref_seqs)

    iteration = 0
    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)

        batch_ref_seqs = ref_seqs[start:end]
        batch_alt_seqs = alt_seqs[start:end]

        print(f"Processing batch {start}:{end}")

        # make all sequences 2048 length
        batch_ref_seqs = [seq + 'N'*(2048-len(seq)) for seq in batch_ref_seqs]
        batch_alt_seqs = [seq + 'N'*(2048-len(seq)) for seq in batch_alt_seqs]

        batch_ref = make_onehot_batch(batch_ref_seqs, device=device)
        batch_alt = make_onehot_batch(batch_alt_seqs, device=device)

        print(batch_ref)
        preds_ref = predict_batch(
            model,
            batch_ref,
            organism_index=args.organism_index,
            heads=allowed_modalities,
            resolution=args.resolution,
        )

        preds_alt = predict_batch(
            model,
            batch_alt,
            organism_index=args.organism_index,
            heads=allowed_modalities,
            resolution=args.resolution,
        )

        batch_features, batch_feature_names = batch_variant_feature_extraction(
            preds_ref,
            preds_alt,
            allowed_modalities=allowed_modalities,
            num_bins=args.num_bins,
            resolution=args.resolution,
        )

        if feature_names is None:
            feature_names = batch_feature_names
        else:
            if feature_names != batch_feature_names:
                raise RuntimeError("Feature names changed across batches")

        all_features.append(batch_features.detach().cpu())

        print(f"Batch {iteration} complete. Remaining: {n - end} sequences...")

    features = torch.cat(all_features, dim=0).numpy()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npy":
        save_npy(str(output_path), features)

    elif output_path.suffix == ".npz":
        save_npz(str(output_path), features, ref_ids, feature_names)

    elif output_path.suffix == ".tsv":
        save_tsv(str(output_path), features, ref_ids, feature_names)

    else:
        raise ValueError(
            f"Unsupported output extension: {output_path.suffix}. "
            "Use .npy, .npz, or .tsv"
        )

    print(f"Saved features to: {output_path}")
    print(f"Final feature matrix shape: {features.shape}")
    print(features)


if __name__ == "__main__":
    main()