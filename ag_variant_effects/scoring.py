from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from .predictions import track_metadata_json, track_name_from_metadata

EFFECT_FIELDNAMES = [
    "input_id",
    "row_index",
    "head",
    "resolution",
    "track_index",
    "track_name",
    "ref_mean",
    "alt_mean",
    "delta_mean",
    "ref_sum",
    "alt_sum",
    "delta_sum",
    "log2fc_sum",
    "ref_max",
    "alt_max",
    "delta_max",
    "max_abs_delta",
    "l2_delta",
    "ref_shape",
    "alt_shape",
    "comparable_by_position",
    "track_metadata_json",
]


def score_prediction_pair(
    *,
    input_id: str,
    row_index: int,
    head: str,
    resolution: str,
    ref: np.ndarray,
    alt: np.ndarray,
    tracks: Optional[tuple[Any, ...]] = None,
    pseudocount: float = 1.0,
    channels_last: bool = True,
) -> Iterable[dict[str, object]]:
    """Yield generic per-track REF/ALT effect summaries for one output tensor pair."""
    ref_arr = _as_float32(ref)
    alt_arr = _as_float32(alt)

    ref_norm = _normalize_track_axis(ref_arr, channels_last=channels_last)
    alt_norm = _normalize_track_axis(alt_arr, channels_last=channels_last)

    n_tracks = max(ref_norm.shape[-1], alt_norm.shape[-1])
    ref_shape = "x".join(str(x) for x in ref_arr.shape)
    alt_shape = "x".join(str(x) for x in alt_arr.shape)
    comparable = ref_norm.shape == alt_norm.shape

    for track_idx in range(n_tracks):
        ref_track = _get_track(ref_norm, track_idx)
        alt_track = _get_track(alt_norm, track_idx)

        ref_mean = _safe_stat(ref_track, np.mean)
        alt_mean = _safe_stat(alt_track, np.mean)
        ref_sum = _safe_stat(ref_track, np.sum)
        alt_sum = _safe_stat(alt_track, np.sum)
        ref_max = _safe_stat(ref_track, np.max)
        alt_max = _safe_stat(alt_track, np.max)

        if ref_track is not None and alt_track is not None and ref_track.shape == alt_track.shape:
            delta = alt_track - ref_track
            delta_mean = _safe_stat(delta, np.mean)
            delta_sum = _safe_stat(delta, np.sum)
            delta_max = _safe_stat(delta, np.max)
            max_abs_delta = _safe_stat(np.abs(delta), np.max)
            l2_delta = float(np.sqrt(np.mean(np.square(delta)))) if delta.size else math.nan
            comparable_track = True
        else:
            delta_mean = _nan_diff(alt_mean, ref_mean)
            delta_sum = _nan_diff(alt_sum, ref_sum)
            delta_max = _nan_diff(alt_max, ref_max)
            max_abs_delta = math.nan
            l2_delta = math.nan
            comparable_track = False

        log2fc_sum = _safe_log2_ratio(alt_sum, ref_sum, pseudocount)

        track = tracks[track_idx] if tracks is not None and track_idx < len(tracks) else None
        yield {
            "input_id": input_id,
            "row_index": row_index,
            "head": head,
            "resolution": resolution,
            "track_index": track_idx,
            "track_name": track_name_from_metadata(track),
            "ref_mean": _format_float(ref_mean),
            "alt_mean": _format_float(alt_mean),
            "delta_mean": _format_float(delta_mean),
            "ref_sum": _format_float(ref_sum),
            "alt_sum": _format_float(alt_sum),
            "delta_sum": _format_float(delta_sum),
            "log2fc_sum": _format_float(log2fc_sum),
            "ref_max": _format_float(ref_max),
            "alt_max": _format_float(alt_max),
            "delta_max": _format_float(delta_max),
            "max_abs_delta": _format_float(max_abs_delta),
            "l2_delta": _format_float(l2_delta),
            "ref_shape": ref_shape,
            "alt_shape": alt_shape,
            "comparable_by_position": str(bool(comparable and comparable_track)).lower(),
            "track_metadata_json": track_metadata_json(track),
        }


def _as_float32(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.float32:
        return arr
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float32, copy=False)
    raise TypeError(f"Prediction tensor has non-numeric dtype {arr.dtype}")


def _normalize_track_axis(arr: np.ndarray, channels_last: bool = True) -> np.ndarray:
    """Return an array with track/channel axis last.

    Scalars become shape (1,). 1D arrays are treated as one track over positions.
    """
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        # Treat a vector as a spatial signal with one track.
        return arr.reshape(arr.shape[0], 1)
    if channels_last:
        return arr
    # After removing batch dimension, channel-first predictions are Cx...
    axes = list(range(arr.ndim))
    return np.moveaxis(arr, 0, -1)


def _get_track(arr: np.ndarray, track_idx: int) -> Optional[np.ndarray]:
    if arr.ndim == 0:
        return arr.reshape(1) if track_idx == 0 else None
    if arr.shape[-1] <= track_idx:
        return None
    return arr[..., track_idx]


def _safe_stat(arr: Optional[np.ndarray], fn) -> float:
    if arr is None or arr.size == 0:
        return math.nan
    try:
        return float(fn(arr))
    except Exception:
        return math.nan


def _nan_diff(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return a - b


def _safe_log2_ratio(alt_sum: float, ref_sum: float, pseudocount: float) -> float:
    if math.isnan(alt_sum) or math.isnan(ref_sum):
        return math.nan
    numerator = alt_sum + pseudocount
    denominator = ref_sum + pseudocount
    if numerator <= 0.0 or denominator <= 0.0:
        return math.nan
    return float(np.log2(numerator / denominator))


def _format_float(x: float) -> str:
    if x is None or math.isnan(float(x)):
        return "nan"
    if math.isinf(float(x)):
        return "inf" if x > 0 else "-inf"
    return f"{float(x):.8g}"
