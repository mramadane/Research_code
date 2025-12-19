from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.signal import cont2discrete

from .config import ModelConfig
from .data_loading import WeatherSeries


def _to_conductance(R: float) -> float:
    return 0.0 if not np.isfinite(R) else 1000.0 / R


@dataclass
class ThermalParams:
    Ci: float
    Cm: float
    Ce: float
    Ch: float
    Cs: float
    g_im: float
    g_ie: float
    g_ih: float
    g_is: float
    g_ea: float
    g_ia: float = 0.0
    aw: float = 0.0
    ae: float = 0.0
    eta_h: float = 1.0

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "ThermalParams":
        cap_j = {k: v * 3_600_000.0 for k, v in cfg.capacities_kwh_per_k.items()}
        r = cfg.resistances_k_per_kw
        return cls(
            Ci=cap_j["Ci"],
            Cm=cap_j["Cm"],
            Ce=cap_j["Ce"],
            Ch=cap_j["Ch"],
            Cs=cap_j["Cs"],
            g_im=_to_conductance(r["Rim"]),
            g_ie=_to_conductance(r["Rie"]),
            g_ih=_to_conductance(r["Rih"]),
            g_is=_to_conductance(r["Ris"]),
            g_ea=_to_conductance(r["Rea"]),
            g_ia=_to_conductance(r.get("Ria", float("inf"))),
            aw=float(cfg.aw_m2),
            ae=float(cfg.ae_m2),
            eta_h=float(cfg.eta_h),
        )

    @property
    def theta_vector(self) -> np.ndarray:
        return np.array(
            [self.Ci, self.Cm, self.Ce, self.Ch, self.Cs, self.g_im, self.g_ie, self.g_ih, self.g_is, self.g_ea],
            dtype=float,
        )


def default_param_ranges(params: ThermalParams) -> Dict[str, float | Tuple[float, float]]:
    """Reproduce the loose parameter box from the notebook."""
    return {
        "Ci": params.Ci,
        "Cm": (0.1 * params.Cm, 3 * params.Cm),
        "Ce": (0.1 * params.Ce, 3 * params.Ce),
        "Ch": params.Ch,
        "Cs": params.Cs,
        "g_im": (0.1 * params.g_im, 3 * params.g_im),
        "g_ie": (0.1 * params.g_ie, 3 * params.g_ie),
        "g_ih": (0.1 * params.g_ih, 3 * params.g_ih),
        "g_is": params.g_is,
        "g_ea": (0.1 * params.g_ea, 3 * params.g_ea),
    }


def _conductance_matrix(theta: np.ndarray) -> np.ndarray:
    Ci, Cm, Ce, Ch, Cs, gim, gie, gih, gis, gea = theta
    gia = 0.0
    invC = np.array([1.0 / Ci, 1.0 / Cm, 1.0 / Ce, 1.0 / Ch, 1.0 / Cs])

    G00 = -(gim + gie + gih + gis + gia)
    G01 = gim
    G02 = gie
    G03 = gih
    G04 = gis
    G10 = gim
    G11 = -gim
    G20 = gie
    G22 = -(gie + gea)
    G30 = gih
    G33 = -gih
    G40 = gis
    G44 = -gis

    A_c = np.array(
        [
            [G00, G01, G02, G03, G04],
            [G10, G11, 0.0, 0.0, 0.0],
            [G20, 0.0, G22, 0.0, 0.0],
            [G30, 0.0, 0.0, G33, 0.0],
            [G40, 0.0, 0.0, 0.0, G44],
        ],
        dtype=float,
    )
    return (A_c.T * invC).T


def _continuous_inputs(theta: np.ndarray, aw: float, ae: float, eta_h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Ci, Cm, Ce, Ch, Cs, gim, gie, gih, gis, gea = theta
    B_u = np.zeros((5,), float)
    B_u[3] = eta_h / Ch
    B_To = np.zeros((5,), float)
    B_To[0] = 0.0
    B_To[2] = gea / Ce
    B_Us = np.zeros((5,), float)
    B_Us[0] = aw / Ci
    B_Us[2] = ae / Ce
    B_F = np.zeros((5,), float)
    B_F[0] = 1.0 / Ci
    return _conductance_matrix(theta), B_u, B_To, B_Us, B_F


def discrete_from_theta_single(theta: np.ndarray, dt_seconds: float, aw: float, ae: float, eta_h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact ZOH discretization for a single parameter vector."""
    A_c, B_u, B_To, B_Us, _ = _continuous_inputs(theta, aw, ae, eta_h)
    B_c = np.column_stack((B_u, B_To, B_Us))
    C_dummy = np.zeros((1, 5))
    D_dummy = np.zeros((1, 3))
    Phi, B_d, _, _, _ = cont2discrete((A_c, B_c, C_dummy, D_dummy), dt_seconds, method="zoh")
    return Phi, B_d[:, 0], B_d[:, 1], B_d[:, 2]


def discrete_from_theta(theta_arr: np.ndarray, dt_seconds: float, aw: float, ae: float, eta_h: float):
    """Vectorized Euler discretization used inside the RBPF."""
    Ci = theta_arr[:, 0]
    Cm = theta_arr[:, 1]
    Ce = theta_arr[:, 2]
    Ch = theta_arr[:, 3]
    Cs = theta_arr[:, 4]
    gim = theta_arr[:, 5]
    gie = theta_arr[:, 6]
    gih = theta_arr[:, 7]
    gis = theta_arr[:, 8]
    gea = theta_arr[:, 9]
    gia = np.zeros_like(gea)

    invC = np.stack([1.0 / Ci, 1.0 / Cm, 1.0 / Ce, 1.0 / Ch, 1.0 / Cs], axis=1)
    G00 = -(gim + gie + gih + gis + gia)
    G01 = gim
    G02 = gie
    G03 = gih
    G04 = gis
    G10 = gim
    G11 = -gim
    G20 = gie
    G22 = -(gie + gea)
    G30 = gih
    G33 = -gih
    G40 = gis
    G44 = -gis
    Z0 = np.zeros_like(G00)

    A0 = np.stack([G00, G01, G02, G03, G04], axis=1) * invC[:, [0]]
    A1 = np.stack([G10, G11, Z0, Z0, Z0], axis=1) * invC[:, [1]]
    A2 = np.stack([G20, Z0, G22, Z0, Z0], axis=1) * invC[:, [2]]
    A3 = np.stack([G30, Z0, Z0, G33, Z0], axis=1) * invC[:, [3]]
    A4 = np.stack([G40, Z0, Z0, Z0, G44], axis=1) * invC[:, [4]]
    A_c = np.stack([A0, A1, A2, A3, A4], axis=1)

    B_u = np.zeros((theta_arr.shape[0], 5), float)
    B_u[:, 3] = eta_h / Ch
    B_To = np.zeros((theta_arr.shape[0], 5), float)
    B_To[:, 0] = 0.0
    B_To[:, 2] = gea * (1.0 / Ce)
    B_Us = np.zeros((theta_arr.shape[0], 5), float)
    B_Us[:, 0] = aw * (1.0 / Ci)
    B_Us[:, 2] = ae * (1.0 / Ce)

    Phi = np.eye(5)[None, :, :] + A_c * dt_seconds
    Gam_u = B_u * dt_seconds
    Gam_To = B_To * dt_seconds
    Gam_Us = B_Us * dt_seconds
    return Phi, Gam_u, Gam_To, Gam_Us


def continuous_from_theta(theta_arr: np.ndarray, aw: float, ae: float, eta_h: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized continuous-time matrices used inside the RBPF."""
    Ci = theta_arr[:, 0]
    Cm = theta_arr[:, 1]
    Ce = theta_arr[:, 2]
    Ch = theta_arr[:, 3]
    Cs = theta_arr[:, 4]
    gim = theta_arr[:, 5]
    gie = theta_arr[:, 6]
    gih = theta_arr[:, 7]
    gis = theta_arr[:, 8]
    gea = theta_arr[:, 9]
    gia = np.zeros_like(gea)

    invC = np.stack([1.0 / Ci, 1.0 / Cm, 1.0 / Ce, 1.0 / Ch, 1.0 / Cs], axis=1)
    G00 = -(gim + gie + gih + gis + gia)
    G01 = gim
    G02 = gie
    G03 = gih
    G04 = gis
    G10 = gim
    G11 = -gim
    G20 = gie
    G22 = -(gie + gea)
    G30 = gih
    G33 = -gih
    G40 = gis
    G44 = -gis
    Z0 = np.zeros_like(G00)

    A0 = np.stack([G00, G01, G02, G03, G04], axis=1) * invC[:, [0]]
    A1 = np.stack([G10, G11, Z0, Z0, Z0], axis=1) * invC[:, [1]]
    A2 = np.stack([G20, Z0, G22, Z0, Z0], axis=1) * invC[:, [2]]
    A3 = np.stack([G30, Z0, Z0, G33, Z0], axis=1) * invC[:, [3]]
    A4 = np.stack([G40, Z0, Z0, Z0, G44], axis=1) * invC[:, [4]]
    A_c = np.stack([A0, A1, A2, A3, A4], axis=1)

    B_u = np.zeros((theta_arr.shape[0], 5), float)
    B_u[:, 3] = eta_h / Ch
    B_To = np.zeros((theta_arr.shape[0], 5), float)
    B_To[:, 0] = 0.0
    B_To[:, 2] = gea * (1.0 / Ce)
    B_Us = np.zeros((theta_arr.shape[0], 5), float)
    B_Us[:, 0] = aw * (1.0 / Ci)
    B_Us[:, 2] = ae * (1.0 / Ce)
    return A_c, B_u, B_To, B_Us


def simulate_states(
    params: ThermalParams,
    weather: WeatherSeries,
    heater_w: np.ndarray,
    latent_w: np.ndarray,
    dt_seconds: int,
    q_std: np.ndarray | float = 0.03,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate the 5-state RC model with latent and heater inputs."""
    if np.isscalar(q_std):
        q_std_arr = np.full(5, float(q_std))
    else:
        q_std_arr = np.asarray(q_std, float)

    theta = params.theta_vector
    A_c, B_u, B_To, B_Us, B_F = _continuous_inputs(theta, params.aw, params.ae, params.eta_h)
    B_total = np.column_stack((B_u, B_To, B_Us, B_F))
    C_dummy = np.zeros((1, 5))
    D_dummy = np.zeros((1, 4))
    Phi, B_d, _, _, _ = cont2discrete((A_c, B_total, C_dummy, D_dummy), dt_seconds, method="zoh")
    Gu, GTo, GUs, GF = B_d[:, 0], B_d[:, 1], B_d[:, 2], B_d[:, 3]

    x = np.array([weather.t_out_k[0]] * 5, float)
    X = np.zeros((weather.timestamps.size, 5), float)
    X[0] = x
    rng = np.random.default_rng(seed)

    for k in range(weather.timestamps.size - 1):
        x = (
            Phi @ x
            + heater_w[k] * Gu
            + weather.t_out_k[k] * GTo
            + weather.u_s[k] * GUs
            + latent_w[k] * GF
        )
        x = x + rng.normal(0.0, q_std_arr)
        X[k + 1] = x

    y_meas = X[:, 0]
    return X, y_meas
