from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Optional

import pandas as pd


def infer_sep(path: str | Path, input_sep: str) -> str:
    if input_sep == "csv":
        return ","
    if input_sep == "tsv":
        return "\t"
    if input_sep != "auto":
        return input_sep
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return ","
    return "\t"


def read_input_table(path: str | Path, input_sep: str = "auto") -> pd.DataFrame:
    sep = infer_sep(path, input_sep)
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def output_sep_from_path(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    return "," if suffix == ".csv" else "\t"


class DictTableWriter:
    """Streaming dict writer for TSV/CSV with a fixed header."""

    def __init__(self, path: str | Path, fieldnames: list[str], sep: Optional[str] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sep = sep if sep is not None else output_sep_from_path(self.path)
        self.fieldnames = fieldnames
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._fh,
            fieldnames=self.fieldnames,
            delimiter=self.sep,
            extrasaction="ignore",
            lineterminator="\n",
        )
        self._writer.writeheader()

    def write_rows(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            safe = {k: _stringify(row.get(k, "")) for k in self.fieldnames}
            self._writer.writerow(safe)

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "DictTableWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _stringify(value: object) -> object:
    if value is None:
        return ""
    return value
