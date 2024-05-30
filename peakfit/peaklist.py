from collections.abc import Callable, Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from peakfit.clustering import Spectra

Reader = Callable[[Path, Spectra], pd.DataFrame]

READERS: dict[str, Reader] = {}


def register_reader(file_types: str | Iterable[str]) -> Callable[[Reader], Reader]:
    if isinstance(file_types, str):
        file_types = [file_types]

    def decorator(fn: Reader) -> Reader:
        for ft in file_types:
            READERS[ft] = fn
        return fn

    return decorator


def _complete_peak_infos(peaks: pd.DataFrame, spectra: Spectra) -> pd.DataFrame:
    nz, ny, nx = spectra.data.shape
    peaks["x0_pt"] = spectra.ucx.f(peaks["x0_ppm"], "ppm")
    peaks["y0_pt"] = spectra.ucy.f(peaks["y0_ppm"], "ppm")
    peaks["x0_int"] = peaks["x0_pt"].round().astype(int) % nx
    peaks["y0_int"] = peaks["y0_pt"].round().astype(int) % ny
    peaks["x0_hz"] = spectra.ucx.hz(peaks["x0_pt"])
    peaks["y0_hz"] = spectra.ucy.hz(peaks["y0_pt"])
    return peaks


@register_reader("list")
def read_sparky_list(path: Path, spectra: Spectra) -> pd.DataFrame:
    """Read a Sparky list file and return a list of peaks."""
    with path.open() as f:
        text = "\n".join([line for line in f if "Ass" not in line])
    peaks = pd.read_table(
        StringIO(text),
        sep=r"\s+",
        comment="#",
        header=None,
        encoding="utf-8",
        names=("name", "y0_ppm", "x0_ppm"),
    )
    return _complete_peak_infos(peaks, spectra)


@np.vectorize
def _make_names(f1name: str, f2name: str, peak_id: int) -> str:
    """Make a name from the indirect and direct dimension names."""
    if not isinstance(f1name, str) and not isinstance(f2name, str):
        return f"{peak_id}"
    items1 = f1name.split(".")
    items2 = f2name.split(".")
    if len(items1) != 4 or len(items2) != 4:
        return f"{peak_id}"
    if items1[1] == items2[1] and items1[2] == items2[2]:
        items2[1] = ""
        items2[2] = ""
    return f"{items1[2]}{items1[1]}{items1[3]}-{items2[2]}{items2[1]}{items2[3]}"


def _read_ccpn_list(
    path: Path, spectra: Spectra, read_func: Callable[[Path], pd.DataFrame]
) -> pd.DataFrame:
    """Read a generic list file and return a list of peaks."""
    peaks_csv = read_func(path)
    names = _make_names(peaks_csv["Assign F2"], peaks_csv["Assign F1"], peaks_csv["#"])
    peaks = pd.DataFrame(
        {"name": names, "y0_ppm": peaks_csv["Pos F2"], "x0_ppm": peaks_csv["Pos F1"]}
    )
    return _complete_peak_infos(peaks, spectra)


@register_reader("csv")
def read_csv_list(path: Path, spectra: Spectra) -> pd.DataFrame:
    return _read_ccpn_list(path, spectra, pd.read_csv)


@register_reader("json")
def read_json_list(path: Path, spectra: Spectra) -> pd.DataFrame:
    return _read_ccpn_list(path, spectra, pd.read_json)


@register_reader(["xlsx", "xls"])
def read_excel_list(path: Path, spectra: Spectra) -> pd.DataFrame:
    return _read_ccpn_list(path, spectra, pd.read_excel)


def read_list(path: Path, spectra: Spectra) -> pd.DataFrame:
    extension = path.suffix.lstrip(".")
    reader = READERS.get(extension)
    if reader is None:
        msg = f"No reader registered for extension: {extension}"
        raise ValueError(msg)
    return reader(path, spectra)
