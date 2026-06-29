from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_path_component(value: object, fallback: str) -> str:
    s = str(value) if value is not None and str(value) else fallback
    s = _SAFE_CHARS_RE.sub("_", s).strip("._")
    return s or fallback


class PredictionZipWriter:
    """Write prediction tensors as .npy entries in one zip archive."""

    def __init__(self, path: str | Path, compression: str = "deflated"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        method = zipfile.ZIP_DEFLATED if compression == "deflated" else zipfile.ZIP_STORED
        self._zip = zipfile.ZipFile(self.path, mode="w", compression=method)
        self._manifest_rows: list[dict[str, str]] = []

    def write_array(
        self,
        *,
        input_id: str,
        row_index: int,
        allele: str,
        head: str,
        resolution: str,
        arr: np.ndarray,
        dtype: str = "float32",
    ) -> None:
        if dtype == "float16":
            arr_to_write = np.asarray(arr, dtype=np.float16)
        elif dtype == "float32":
            arr_to_write = np.asarray(arr, dtype=np.float32) if np.issubdtype(np.asarray(arr).dtype, np.floating) else np.asarray(arr)
        else:
            raise ValueError(f"Unsupported prediction dtype: {dtype}")

        safe_id = safe_path_component(input_id, fallback=f"row_{row_index}")
        safe_head = safe_path_component(head, fallback="head")
        safe_res = safe_path_component(resolution, fallback="native")
        archive_path = f"predictions/{safe_id}/{allele}/{safe_head}__{safe_res}.npy"

        with self._zip.open(archive_path, "w") as fh:
            np.save(fh, arr_to_write, allow_pickle=False)

        self._manifest_rows.append(
            {
                "input_id": str(input_id),
                "row_index": str(row_index),
                "allele": allele,
                "head": str(head),
                "resolution": str(resolution),
                "shape": "x".join(str(x) for x in arr_to_write.shape),
                "dtype": str(arr_to_write.dtype),
                "archive_path": archive_path,
            }
        )

    def close(self) -> None:
        manifest = io.StringIO()
        fieldnames = ["input_id", "row_index", "allele", "head", "resolution", "shape", "dtype", "archive_path"]
        writer = csv.DictWriter(manifest, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in self._manifest_rows:
            writer.writerow(row)
        self._zip.writestr("manifest.tsv", manifest.getvalue())
        self._zip.writestr(
            "README.txt",
            "This archive contains AlphaGenome prediction tensors saved as .npy files. "
            "Use manifest.tsv to map input IDs, alleles, heads, and resolutions to archive paths.\n",
        )
        self._zip.close()

    def __enter__(self) -> "PredictionZipWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
