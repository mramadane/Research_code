from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Event:
    start_idx: int
    end_idx: int
    duration_s: float
    peak_w: float | None = None
    peak_idx: int | None = None
    mode: str | None = None


def generate_square_heating(
    N: int,
    dt_seconds: int,
    power: float = 6000.0,
    p_start: float = 0.002,
    min_duration_s: int = 30 * 60,
    max_duration_s: int = 4 * 60 * 60,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[Event]]:
    """Generate a square ON/OFF signal with random episode lengths."""
    rng = np.random.default_rng(seed)
    min_steps = int(np.ceil(min_duration_s / dt_seconds))
    max_steps = int(np.ceil(max_duration_s / dt_seconds))
    min_steps = max(min_steps, 1)
    max_steps = max(max_steps, min_steps)

    f_true = np.zeros(N, dtype=float)
    log: List[Event] = []

    on_remaining = 0
    start_idx = None

    for k in range(N):
        if on_remaining > 0:
            f_true[k] = power
            on_remaining -= 1
            if on_remaining == 0 and start_idx is not None:
                log.append(
                    Event(
                        start_idx=start_idx,
                        end_idx=k,
                        duration_s=(k - start_idx + 1) * dt_seconds,
                    )
                )
                start_idx = None
        else:
            if rng.random() < p_start:
                dur_steps = rng.integers(min_steps, max_steps + 1)
                on_remaining = dur_steps
                start_idx = k
                f_true[k] = power
                on_remaining -= 1

    if start_idx is not None and (len(log) == 0 or log[-1].end_idx != N - 1):
        log.append(
            Event(
                start_idx=start_idx,
                end_idx=N - 1,
                duration_s=(N - start_idx) * dt_seconds,
            )
        )

    return f_true, log


def _sample_log_space(rng: np.random.Generator, lo: float, hi: float) -> float:
    lo = max(lo, 1.0)
    hi = max(hi, lo + 1.0)
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def generate_pulse_heating(
    N: int,
    dt_seconds: int,
    mode: str = "alpha",
    p_start: float = 0.0015,
    amp_range: tuple[float, float] = (1500.0, 8000.0),
    tau_r_range_s: tuple[float, float] = (5 * 60, 40 * 60),
    tau_d_range_s: tuple[float, float] = (45 * 60, 4 * 60 * 60),
    sigma_range_s: tuple[float, float] = (15 * 60, 90 * 60),
    kill_rel: float = 1e-3,
    max_active: int = 6,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[Event]]:
    """Generate a smooth, positive latent heat signal as a sum of pulses."""
    rng = np.random.default_rng(seed)
    f_true = np.zeros(N, dtype=float)
    log: List[Event] = []
    active = []

    for k in range(N):
        if len(active) < max_active and rng.random() < p_start:
            A = rng.uniform(*amp_range)
            if mode == "alpha":
                tau_r = _sample_log_space(rng, *tau_r_range_s)
                tau_d = _sample_log_space(rng, *tau_d_range_s)
                if tau_d <= tau_r:
                    tau_d = tau_r * rng.uniform(1.5, 3.0)
                active.append(
                    dict(
                        mode="alpha",
                        A=A,
                        tau_r=tau_r,
                        tau_d=tau_d,
                        sigma=None,
                        age_s=0.0,
                        start_idx=k,
                        peak_w=0.0,
                        peak_idx=k,
                    )
                )
            elif mode == "gauss":
                sigma = _sample_log_space(rng, *sigma_range_s)
                active.append(
                    dict(
                        mode="gauss",
                        A=A,
                        tau_r=None,
                        tau_d=None,
                        sigma=sigma,
                        age_s=0.0,
                        start_idx=k,
                        peak_w=0.0,
                        peak_idx=k,
                    )
                )
            else:
                raise ValueError("mode must be 'alpha' or 'gauss'")

        val_k = 0.0
        keep = []
        for p in active:
            age = p["age_s"]
            A = p["A"]
            if p["mode"] == "alpha":
                tr = p["tau_r"]
                td = p["tau_d"]
                f = A * max(np.exp(-age / td) - np.exp(-age / tr), 0.0)
            else:
                sg = p["sigma"]
                f = A * np.exp(-0.5 * (age / sg) ** 2)

            val_k += f
            if f > p["peak_w"]:
                p["peak_w"] = f
                p["peak_idx"] = k

            if (age == 0.0) or (f >= kill_rel * A):
                p["age_s"] = age + dt_seconds
                keep.append(p)
            else:
                log.append(
                    Event(
                        start_idx=p["start_idx"],
                        end_idx=k,
                        duration_s=(k - p["start_idx"]) * dt_seconds,
                        peak_w=p["peak_w"],
                        peak_idx=p["peak_idx"],
                        mode=p["mode"],
                    )
                )

        active = keep
        f_true[k] = val_k

    for p in active:
        log.append(
            Event(
                start_idx=p["start_idx"],
                end_idx=N - 1,
                duration_s=(N - p["start_idx"]) * dt_seconds,
                peak_w=p["peak_w"],
                peak_idx=p["peak_idx"],
                mode=p["mode"],
            )
        )

    return f_true, log
