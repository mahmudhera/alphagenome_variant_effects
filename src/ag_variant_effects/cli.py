from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import torch

from .archive import PredictionZipWriter
from .backends import BackendConfig, create_backend
from .io_utils import DictTableWriter, read_input_table
from .predictions import collect_prediction_arrays, split_batch_array
from .scoring import EFFECT_FIELDNAMES, score_prediction_pair
from .sequence import batch_onehot, clean_sequence


@dataclass
class InputRecord:
    row_index: int
    input_id: str
    ref_seq: str
    alt_seq: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ag-variant-effects",
        description="Run AlphaGenome-style predictions on many REF/ALT sequence pairs and write variant effects plus raw predictions.",
    )
    parser.add_argument("--input", required=True, help="Input CSV/TSV path")
    parser.add_argument("--input-sep", default="auto", help="auto, csv, tsv, or a literal separator")
    parser.add_argument("--ref-col", required=True, help="Column containing REF sequences")
    parser.add_argument("--alt-col", required=True, help="Column containing ALT sequences")
    parser.add_argument("--id-col", default=None, help="Optional input ID column")

    parser.add_argument("--backend", choices=["mock", "alphagenome-pytorch"], default="alphagenome-pytorch")
    parser.add_argument("--weights", default=None, help="Local model artifact path for alphagenome-pytorch backend")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--organism", default="human", choices=["human", "mouse", "0", "1"])
    parser.add_argument("--batch-size", type=int, default=1, help="Batch equal-length sequence pairs together")
    parser.add_argument("--compile-model", action="store_true", help="Apply torch.compile to the model when supported")

    parser.add_argument("--effects-out", required=True, help="Output TSV/CSV for variant-effect summaries")
    parser.add_argument("--predictions-out", required=True, help="Output zip archive for all REF/ALT prediction tensors")
    parser.add_argument("--prediction-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--zip-compression", choices=["deflated", "stored"], default="deflated")

    parser.add_argument("--ambiguous", choices=["zeros", "uniform", "raise"], default="zeros")
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--channels-last", dest="channels_last", action="store_true", default=True)
    parser.add_argument("--channels-first", dest="channels_last", action="store_false")
    parser.add_argument("--raw-outputs", dest="named_outputs", action="store_false", help="Do not request named outputs from backend")
    parser.add_argument("--named-outputs", dest="named_outputs", action="store_true", default=True, help="Request named outputs when backend supports them")
    parser.add_argument("--include-padding", action="store_true", help="Ask backend to include padded tracks when named outputs support it")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def run(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.pseudocount < 0:
        raise ValueError("--pseudocount must be >= 0")

    df = read_input_table(args.input, input_sep=args.input_sep)
    missing = [c for c in [args.ref_col, args.alt_col] if c not in df.columns]
    if args.id_col and args.id_col not in df.columns:
        missing.append(args.id_col)
    if missing:
        raise ValueError(f"Missing column(s) in input: {missing}. Available columns: {list(df.columns)}")

    records = _records_from_dataframe(df, args.ref_col, args.alt_col, args.id_col)
    if not records:
        raise ValueError("No input rows found")

    backend = create_backend(
        BackendConfig(
            backend=args.backend,
            weights=args.weights,
            device=args.device,
            organism=args.organism,
            named_outputs=args.named_outputs,
            include_padding=args.include_padding,
            compile_model=args.compile_model,
        )
    )

    groups: dict[tuple[int, int], list[InputRecord]] = defaultdict(list)
    for rec in records:
        groups[(len(rec.ref_seq), len(rec.alt_seq))].append(rec)

    with DictTableWriter(args.effects_out, EFFECT_FIELDNAMES) as effects_writer, PredictionZipWriter(
        args.predictions_out, compression=args.zip_compression
    ) as pred_writer:
        for _length_key, group_records in groups.items():
            for batch_records in _chunks(group_records, args.batch_size):
                _process_batch(
                    batch_records=batch_records,
                    backend=backend,
                    effects_writer=effects_writer,
                    pred_writer=pred_writer,
                    ambiguous=args.ambiguous,
                    prediction_dtype=args.prediction_dtype,
                    pseudocount=args.pseudocount,
                    channels_last=args.channels_last,
                )


def _records_from_dataframe(df, ref_col: str, alt_col: str, id_col: Optional[str]) -> list[InputRecord]:
    records: list[InputRecord] = []
    seen_ids: dict[str, int] = {}
    for row_index, row in df.iterrows():
        ref_seq = clean_sequence(row[ref_col])
        alt_seq = clean_sequence(row[alt_col])
        if id_col:
            input_id = str(row[id_col]).strip() or f"row_{row_index}"
        else:
            input_id = f"row_{row_index}"
        # Make IDs unique for archive paths while preserving the visible base ID.
        count = seen_ids.get(input_id, 0)
        seen_ids[input_id] = count + 1
        if count:
            input_id = f"{input_id}__dup{count}"
        records.append(InputRecord(row_index=int(row_index), input_id=input_id, ref_seq=ref_seq, alt_seq=alt_seq))
    return records


def _chunks(items: list[InputRecord], size: int) -> Iterable[list[InputRecord]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _process_batch(
    *,
    batch_records: list[InputRecord],
    backend,
    effects_writer: DictTableWriter,
    pred_writer: PredictionZipWriter,
    ambiguous: str,
    prediction_dtype: str,
    pseudocount: float,
    channels_last: bool,
) -> None:
    ref_batch_np = batch_onehot([r.ref_seq for r in batch_records], ambiguous=ambiguous)
    alt_batch_np = batch_onehot([r.alt_seq for r in batch_records], ambiguous=ambiguous)

    ref_batch = torch.from_numpy(ref_batch_np).float()
    alt_batch = torch.from_numpy(alt_batch_np).float()

    ref_outputs = collect_prediction_arrays(backend.predict(ref_batch))
    alt_outputs = collect_prediction_arrays(backend.predict(alt_batch))

    expected_batch_size = len(batch_records)
    all_keys = sorted(set(ref_outputs) | set(alt_outputs))

    for batch_idx, rec in enumerate(batch_records):
        per_record_ref: dict[tuple[str, str], tuple[np.ndarray, object]] = {}
        per_record_alt: dict[tuple[str, str], tuple[np.ndarray, object]] = {}

        for key in all_keys:
            head, resolution = key
            if key in ref_outputs:
                arr, tracks = ref_outputs[key]
                arr_i = split_batch_array(arr, batch_idx, expected_batch_size)
                per_record_ref[key] = (arr_i, tracks)
                pred_writer.write_array(
                    input_id=rec.input_id,
                    row_index=rec.row_index,
                    allele="ref",
                    head=head,
                    resolution=resolution,
                    arr=arr_i,
                    dtype=prediction_dtype,
                )
            if key in alt_outputs:
                arr, tracks = alt_outputs[key]
                arr_i = split_batch_array(arr, batch_idx, expected_batch_size)
                per_record_alt[key] = (arr_i, tracks)
                pred_writer.write_array(
                    input_id=rec.input_id,
                    row_index=rec.row_index,
                    allele="alt",
                    head=head,
                    resolution=resolution,
                    arr=arr_i,
                    dtype=prediction_dtype,
                )

        common_keys = sorted(set(per_record_ref) & set(per_record_alt))
        for key in common_keys:
            head, resolution = key
            ref_arr, ref_tracks = per_record_ref[key]
            alt_arr, alt_tracks = per_record_alt[key]
            tracks = alt_tracks if alt_tracks is not None else ref_tracks
            rows = score_prediction_pair(
                input_id=rec.input_id,
                row_index=rec.row_index,
                head=head,
                resolution=resolution,
                ref=ref_arr,
                alt=alt_arr,
                tracks=tracks,
                pseudocount=pseudocount,
                channels_last=channels_last,
            )
            effects_writer.write_rows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
