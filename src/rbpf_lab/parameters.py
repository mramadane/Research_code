from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


THETA_NAMES: Tuple[str, ...] = (
    "Ci",
    "Cm",
    "Ce",
    "Ch",
    "Cs",
    "g_im",
    "g_ie",
    "g_ih",
    "g_is",
    "g_ea",
)


@dataclass(frozen=True)
class ParameterBounds:
    theta_lo: np.ndarray
    theta_hi: np.ndarray
    free_idx: np.ndarray
    fixed_idx: np.ndarray

    @property
    def n_free(self) -> int:
        return int(self.free_idx.size)


def build_parameter_bounds(param_ranges: Dict[str, float | Tuple[float, float]]) -> ParameterBounds:
    theta_lo = np.empty(len(THETA_NAMES), float)
    theta_hi = np.empty(len(THETA_NAMES), float)
    is_fixed = np.zeros(len(THETA_NAMES), dtype=bool)

    for j, nm in enumerate(THETA_NAMES):
        spec = param_ranges[nm]
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            lo, hi = float(spec[0]), float(spec[1])
        else:
            lo = hi = float(spec)
        lo = max(lo, 1e-20)
        hi = max(hi, 1e-20)
        theta_lo[j] = lo
        theta_hi[j] = hi
        is_fixed[j] = (hi == lo)

    free_idx = np.nonzero(~is_fixed)[0]
    fixed_idx = np.nonzero(is_fixed)[0]
    return ParameterBounds(theta_lo=theta_lo, theta_hi=theta_hi, free_idx=free_idx, fixed_idx=fixed_idx)


def theta_from_phi(phi_log: np.ndarray, theta_base: np.ndarray, free_idx: Iterable[int]) -> np.ndarray:
    theta = theta_base.copy()
    if len(free_idx) > 0:
        theta[np.asarray(free_idx)] = np.exp(phi_log)
    return theta


def initial_theta(bounds: ParameterBounds) -> np.ndarray:
    theta = bounds.theta_lo.copy()
    if bounds.n_free > 0:
        theta[bounds.free_idx] = 0.5 * (bounds.theta_lo[bounds.free_idx] + bounds.theta_hi[bounds.free_idx])
    return theta
