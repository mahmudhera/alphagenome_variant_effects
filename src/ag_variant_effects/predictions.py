from __future__ import annotations

import dataclasses
import json
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch

PredictionKey = Tuple[str, str]

KNOWN_HEADS = (
    "atac",
    "dnase",
    "procap",
    "cage",
    "rna_seq",
    "rnaseq",
    "chip_tf",
    "chip_histone",
    "contact_maps",
    "splice_sites",
    "splice_junctions",
    "splice_site_usage",
    "splice_sites_classification",
    "splice_sites_usage",
    "splice_sites_junction",
)

KNOWN_RESOLUTIONS = (1, 128, "probs", "usage", "junction", "pairwise", "64x64", "native")


def tensor_to_numpy(x: Any) -> np.ndarray:
    """Convert torch/NumPy tensor-like output to NumPy float32 where possible."""
    if torch.is_tensor(x):
        # bfloat16 cannot be directly converted to NumPy on some Torch versions.
        if x.dtype in {torch.bfloat16, torch.float16}:
            x = x.float()
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == np.dtype("O"):
        raise TypeError(f"Object array is not a numeric prediction tensor: {type(x)}")
    return arr


def is_tensor_like(x: Any) -> bool:
    return torch.is_tensor(x) or isinstance(x, np.ndarray)


def track_to_dict(track: Any) -> dict[str, Any]:
    """Best-effort serialization for track metadata objects."""
    if track is None:
        return {}
    if hasattr(track, "to_dict"):
        try:
            d = track.to_dict()
            if isinstance(d, dict):
                return _jsonable_dict(d)
        except Exception:
            pass
    if dataclasses.is_dataclass(track):
        try:
            return _jsonable_dict(dataclasses.asdict(track))
        except Exception:
            pass
    if hasattr(track, "__dict__"):
        return _jsonable_dict({k: v for k, v in vars(track).items() if not k.startswith("_")})
    return {"repr": repr(track)}


def _jsonable_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        elif isinstance(v, (list, tuple)):
            out[str(k)] = [str(x) if not isinstance(x, (str, int, float, bool)) and x is not None else x for x in v]
        else:
            out[str(k)] = str(v)
    return out


def track_name_from_metadata(track: Any) -> str:
    d = track_to_dict(track)
    for key in ("track_name", "name", "description", "assay_title"):
        value = d.get(key)
        if value:
            return str(value)
    return ""


def track_metadata_json(track: Any) -> str:
    d = track_to_dict(track)
    if not d:
        return ""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def collect_prediction_arrays(outputs: Any) -> Dict[PredictionKey, tuple[np.ndarray, Optional[tuple[Any, ...]]]]:
    """Convert model outputs into a mapping: (head, resolution) -> (array, tracks).

    The function supports:
    - plain dict outputs: outputs['atac'][128]
    - contact-map tensors without a resolution dict
    - NamedOutputs-like objects exposing .items() or head attributes
    - NamedTrackTensor-like objects exposing .tensor and .tracks
    """
    collected: Dict[PredictionKey, tuple[np.ndarray, Optional[tuple[Any, ...]]]] = {}
    for head, resolution, tensor_obj, tracks in iter_prediction_tensors(outputs):
        arr = tensor_to_numpy(tensor_obj)
        key = (str(head), str(resolution))
        collected[key] = (arr, tracks)
    if not collected:
        raise ValueError(
            "Could not find prediction tensors in model output. "
            "Expected a dict of tensors, a dict of resolution->tensor, or a NamedOutputs-like object."
        )
    return collected


def iter_prediction_tensors(outputs: Any) -> Iterator[tuple[str, str, Any, Optional[tuple[Any, ...]]]]:
    if is_tensor_like(outputs) or _has_tensor_attr(outputs):
        yield from _yield_from_obj(outputs, head="output", resolution="native")
        return

    if isinstance(outputs, dict):
        for key, value in outputs.items():
            yield from _yield_from_key_value(key, value)
        return

    # NamedOutputs from alphagenome_pytorch may expose items(). Try it first.
    items = getattr(outputs, "items", None)
    if callable(items):
        yielded = False
        try:
            for key, value in items():
                yielded = True
                yield from _yield_from_key_value(key, value)
            if yielded:
                return
        except Exception:
            pass

    # Fall back to known output-head attributes.
    for head in KNOWN_HEADS:
        if hasattr(outputs, head):
            value = getattr(outputs, head)
            yield from _yield_from_obj(value, head=head, resolution="native")


def _yield_from_key_value(key: Any, value: Any) -> Iterator[tuple[str, str, Any, Optional[tuple[Any, ...]]]]:
    if isinstance(key, tuple) and len(key) >= 2:
        head, resolution = key[0], key[1]
        yield from _yield_from_obj(value, head=str(head), resolution=str(resolution))
    else:
        yield from _yield_from_obj(value, head=str(key), resolution="native")


def _yield_from_obj(obj: Any, head: str, resolution: str) -> Iterator[tuple[str, str, Any, Optional[tuple[Any, ...]]]]:
    if _has_tensor_attr(obj):
        tracks = getattr(obj, "tracks", None)
        if tracks is not None and not isinstance(tracks, tuple):
            try:
                tracks = tuple(tracks)
            except TypeError:
                tracks = None
        yield head, resolution, getattr(obj, "tensor"), tracks
        return

    if is_tensor_like(obj):
        yield head, resolution, obj, None
        return

    if isinstance(obj, dict):
        for sub_key, sub_value in obj.items():
            yield from _yield_from_obj(sub_value, head=head, resolution=str(sub_key))
        return

    items = getattr(obj, "items", None)
    if callable(items):
        try:
            yielded = False
            for sub_key, sub_value in items():
                yielded = True
                yield from _yield_from_obj(sub_value, head=head, resolution=str(sub_key))
            if yielded:
                return
        except Exception:
            pass

    # NamedOutputHead-like objects may support __getitem__ for common resolutions.
    for res in KNOWN_RESOLUTIONS:
        try:
            sub = obj[res]  # type: ignore[index]
        except Exception:
            continue
        yield from _yield_from_obj(sub, head=head, resolution=str(res))


def _has_tensor_attr(obj: Any) -> bool:
    try:
        tensor = getattr(obj, "tensor")
    except Exception:
        return False
    return is_tensor_like(tensor)


def split_batch_array(arr: np.ndarray, batch_index: int, expected_batch_size: int) -> np.ndarray:
    """Return one sample from a batch-first prediction array when possible."""
    if arr.ndim >= 1 and arr.shape[0] == expected_batch_size:
        return arr[batch_index]
    if expected_batch_size == 1:
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr
    raise ValueError(
        f"Prediction array shape {arr.shape} does not appear to contain batch size {expected_batch_size}."
    )
