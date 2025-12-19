from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        dtype=str,
        na_values=DWD_NA,
        on_bad_lines="skip",
    )
    if "MESS_DATUM" not in df.columns:
        df.columns = df.columns.str.strip()
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
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        dtype=str,
        na_values=DWD_NA,
        on_bad_lines="skip",
    )
    if "MESS_DATUM" not in df.columns:
        df.columns = df.columns.str.strip()
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


def _align_to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, col_name: str, dt_seconds: int) -> np.ndarray:
    df = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"])
    df = df.set_index("Timestamp")
    tol = pd.Timedelta(seconds=dt_seconds / 2)
    aligned = df.reindex(grid, method="nearest", tolerance=tol)
    aligned[col_name] = aligned[col_name].ffill().bfill()
    return aligned[col_name].to_numpy(float)


def align_weather(temp_df: pd.DataFrame, rad_df: pd.DataFrame, start: str, end: str, dt_seconds: int) -> WeatherSeries:
    """Align irregular DWD temperature and solar series onto a uniform grid."""
    t_df = _filter_window(temp_df, start, end)
    s_df = _filter_window(rad_df, start, end)
    if t_df.empty or s_df.empty:
        raise ValueError("One of the weather inputs is empty in the requested window.")

    ts_grid_pd = pd.date_range(start=start, end=end, freq=f"{dt_seconds}s", inclusive="left")
    t_out_c = _align_to_grid(t_df, ts_grid_pd, "TT_10", dt_seconds)
    u_s = _align_to_grid(s_df, ts_grid_pd, "U_s", dt_seconds)

    ts_grid = ts_grid_pd.values.astype("datetime64[s]")
    t_grid_s = ts_grid.astype("int64")

    return WeatherSeries(
        timestamps=ts_grid,
        t_out_c=t_out_c,
        t_out_k=t_out_c + 273.15,
        u_s=u_s,
        grid_seconds=t_grid_s,
    )
