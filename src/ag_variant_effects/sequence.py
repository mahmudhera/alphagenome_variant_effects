from __future__ import annotations

from typing import Iterable

import numpy as np

_BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def clean_sequence(seq: object) -> str:
    """Return an uppercase DNA sequence string without whitespace."""
    if seq is None:
        raise ValueError("Sequence value is None")
    s = str(seq).strip().upper()
    # Remove common whitespace characters that may appear in pasted sequences.
    for ch in (" ", "\n", "\r", "\t"):
        s = s.replace(ch, "")
    if not s:
        raise ValueError("Empty sequence")
    return s


def sequence_to_onehot(seq: object, ambiguous: str = "zeros") -> np.ndarray:
    """Encode a DNA string as an NLC one-hot array with channel order A,C,G,T.

    Parameters
    ----------
    seq:
        Input DNA sequence. U is treated as T.
    ambiguous:
        How to encode non-ACGT bases.
        - "zeros": leave ambiguous rows as all zeros.
        - "uniform": encode ambiguous rows as [0.25, 0.25, 0.25, 0.25].
        - "raise": raise ValueError when an ambiguous base is present.
    """
    if ambiguous not in {"zeros", "uniform", "raise"}:
        raise ValueError(f"Unsupported ambiguous={ambiguous!r}")

    s = clean_sequence(seq)
    arr = np.zeros((len(s), 4), dtype=np.float32)
    for i, base in enumerate(s):
        idx = _BASE_TO_INDEX.get(base)
        if idx is not None:
            arr[i, idx] = 1.0
        elif ambiguous == "uniform":
            arr[i, :] = 0.25
        elif ambiguous == "raise":
            raise ValueError(f"Ambiguous or invalid base {base!r} at position {i} in sequence")
        # ambiguous == "zeros": leave zeros
    return arr


def batch_onehot(sequences: Iterable[object], ambiguous: str = "zeros") -> np.ndarray:
    """Encode equal-length sequences into shape (batch, length, 4)."""
    arrays = [sequence_to_onehot(seq, ambiguous=ambiguous) for seq in sequences]
    if not arrays:
        raise ValueError("No sequences to encode")
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All sequences in a batch must have the same length; got {sorted(lengths)}")
    return np.stack(arrays, axis=0)
