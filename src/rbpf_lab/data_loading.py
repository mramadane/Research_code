from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


DWD_NA = ["-999", "-9999", "-999.0", "", "NA", "NaN"]


@dataclass
class WeatherSeries:
    timestamps: np.ndarray  # datetime64[s]
    t_out_c: np.ndarray
    t_out_k: np.ndarray
    u_s: np.ndarray
    grid_seconds: np.ndarray


def load_temperature(path: Path) -> pd.DataFrame:
    """Load a DWD temperature file and normalize column names."""
    df = pd.read_csv(path, sep=";", decimal=",", dtype=str, na_values=DWD_NA)
    if "MESS_DATUM" not in df.columns:
        raise ValueError("Temperature file missing 'MESS_DATUM'.")
    if "TT_10" not in df.columns:
        cand = [c for c in df.columns if c.upper().startswith("TT")]
        if not cand:
            raise ValueError("Temperature file missing 'TT_10'.")
        df = df.rename(columns={cand[0]: "TT_10"})

    df["Timestamp"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce")
    df["TT_10"] = pd.to_numeric(df["TT_10"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "TT_10"]).reset_index(drop=True)
    return df[["Timestamp", "TT_10"]].sort_values("Timestamp")


def load_radiation(path: Path) -> pd.DataFrame:
    """Load a DWD global radiation file and convert to W/m^2."""
    df = pd.read_csv(path, sep=";", decimal=",", dtype=str, na_values=DWD_NA)
    if "MESS_DATUM" not in df.columns:
        raise ValueError("Solar file missing 'MESS_DATUM'.")
    if "GS_10" not in df.columns:
        cand = [c for c in df.columns if c.upper().startswith("GS")]
        if not cand:
            raise ValueError("Solar file missing 'GS_10'.")
        df = df.rename(columns={cand[0]: "GS_10"})

    df["Timestamp"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H%M", errors="coerce")
    df["GS_10"] = pd.to_numeric(df["GS_10"], errors="coerce")
    df = df.dropna(subset=["Timestamp", "GS_10"]).reset_index(drop=True)

    # DWD provides J/cm^2 over 10 minutes. Convert to W/m^2.
    df["U_s"] = df["GS_10"] * 10000.0 / 600.0
    return df[["Timestamp", "U_s"]].sort_values("Timestamp")


def _filter_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["Timestamp"])
    mask = (ts >= pd.to_datetime(start)) & (ts < pd.to_datetime(end))
    return df.loc[mask].copy()


def _dedupe_and_sort(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x, kind="mergesort")
    x_sorted, y_sorted = x[order], y[order]
    uniq_x, uniq_idx = np.unique(x_sorted, return_index=True)
    return uniq_x, y_sorted[uniq_idx]


def align_weather(temp_df: pd.DataFrame, rad_df: pd.DataFrame, start: str, end: str, dt_seconds: int) -> WeatherSeries:
    """Align irregular DWD temperature and solar series onto a uniform grid."""
    t_df = _filter_window(temp_df, start, end)
    s_df = _filter_window(rad_df, start, end)
    if t_df.empty or s_df.empty:
        raise ValueError("One of the weather inputs is empty in the requested window.")

    merged = pd.merge_asof(
        t_df.sort_values("Timestamp"),
        s_df.sort_values("Timestamp"),
        on="Timestamp",
        direction="nearest",
        tolerance=pd.to_timedelta(dt_seconds / 2, unit="s"),
    ).dropna(subset=["TT_10", "U_s"]).set_index("Timestamp").sort_index()

    start_np = np.datetime64(pd.to_datetime(start), "s")
    end_np = np.datetime64(pd.to_datetime(end), "s")
    step = np.timedelta64(int(dt_seconds), "s")
    ts_grid = np.arange(start_np, end_np, step, dtype="datetime64[s]")
    t_grid_s = ts_grid.astype("int64")

    t_in = merged.index.values.astype("datetime64[s]")
    x_in = t_in.astype("int64")
    tt_in = merged["TT_10"].to_numpy(float)
    us_in = merged["U_s"].to_numpy(float)

    good = np.isfinite(tt_in) & np.isfinite(us_in)
    x_in, tt_in, us_in = x_in[good], tt_in[good], us_in[good]
    x_in, tt_in = _dedupe_and_sort(x_in, tt_in)
    _, us_in = _dedupe_and_sort(x_in, us_in)

    t_out_c = np.interp(t_grid_s, x_in, tt_in, left=tt_in[0], right=tt_in[-1])
    u_s = np.interp(t_grid_s, x_in, us_in, left=us_in[0], right=us_in[-1])

    return WeatherSeries(
        timestamps=ts_grid,
        t_out_c=t_out_c,
        t_out_k=t_out_c + 273.15,
        u_s=u_s,
        grid_seconds=t_grid_s,
    )
