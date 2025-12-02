from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import EKFConfig
from .model import ThermalParams, default_param_ranges, discrete_from_theta_single
from .parameters import ParameterBounds, build_parameter_bounds, theta_from_phi


@dataclass
class EKFResult:
    ti_mean: np.ndarray  # Kelvin
    ti_var: np.ndarray
    theta_traj: np.ndarray
    y_pred: np.ndarray
    innov_var: np.ndarray

    @property
    def ti_mean_c(self) -> np.ndarray:
        return self.ti_mean - 273.15


def _init_state(y0: float, bounds: ParameterBounds, cfg: EKFConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_base = bounds.theta_lo.copy()
    theta_init = theta_base.copy()
    if bounds.n_free > 0:
        theta_init[bounds.free_idx] = 0.5 * (bounds.theta_lo[bounds.free_idx] + bounds.theta_hi[bounds.free_idx])
    phi0 = np.log(theta_init[bounds.free_idx]) if bounds.n_free > 0 else np.array([], dtype=float)

    x0 = np.array([y0] * 5, float)
    z0 = np.concatenate([x0, phi0])

    nz = z0.size
    P = np.zeros((nz, nz), float)
    P[:5, :5] = np.diag(np.full(5, cfg.init_state_std**2))
    if bounds.n_free > 0:
        P[5:, 5:] = np.diag(np.full(bounds.n_free, cfg.init_param_log_std**2))

    Q = np.zeros_like(P)
    q_states = np.asarray(cfg.q_states, float)
    Q[:5, :5] = np.diag(q_states**2)
    if bounds.n_free > 0:
        Q[5:, 5:] = np.diag(np.full(bounds.n_free, cfg.sigma_log_theta**2))

    return z0, P, Q


def run_ekf(
    y_meas: np.ndarray,
    u: np.ndarray,
    t_out_k: np.ndarray,
    u_s: np.ndarray,
    params: ThermalParams,
    dt_seconds: int,
    cfg: EKFConfig,
    param_ranges: Dict[str, float | Tuple[float, float]] | None = None,
) -> EKFResult:
    """Run the log-parameter EKF from the notebook in a reusable form."""

    y = np.asarray(y_meas, float).ravel()
    u_arr = np.asarray(u, float).ravel()
    T_out = np.asarray(t_out_k, float).ravel()
    U_s_arr = np.asarray(u_s, float).ravel()
    N = int(len(y))
    param_ranges = param_ranges or default_param_ranges(params)

    bounds = build_parameter_bounds(param_ranges)
    theta_base = bounds.theta_lo.copy()
    z, P, Q = _init_state(y0=y[0], bounds=bounds, cfg=cfg)
    nz = z.size

    R = np.array([[cfg.sigma_y**2]], float)
    H = np.zeros((1, nz), float)
    H[0, 0] = 1.0

    ti_mean = np.zeros(N)
    ti_var = np.zeros(N)
    theta_traj = np.zeros((N, len(theta_base)))
    y_pred = np.zeros(N)
    innov_var = np.zeros(N)

    theta_init = theta_from_phi(z[5:], theta_base, bounds.free_idx) if bounds.n_free > 0 else theta_base.copy()
    ti_mean[0] = z[0]
    ti_var[0] = P[0, 0]
    theta_traj[0] = theta_init
    y_pred[0] = z[0]
    innov_var[0] = P[0, 0] + R[0, 0]

    eps = cfg.jacobian_eps

    I = np.eye(nz)

    for k in range(1, N):
        phi = z[5:]
        theta_full = theta_from_phi(phi, theta_base, bounds.free_idx) if bounds.n_free > 0 else theta_base.copy()
        Phi, Gu, GTo, GUs = discrete_from_theta_single(theta_full, dt_seconds, params.aw, params.ae, params.eta_h)

        x = z[:5]
        uk = u_arr[k - 1]
        Tok = T_out[k - 1]
        Usk = U_s_arr[k - 1]
        x_pred = Phi @ x + uk * Gu + Tok * GTo + Usk * GUs
        z_pred = np.concatenate([x_pred, phi])

        F = np.eye(nz)
        F[:5, :5] = Phi

        if bounds.n_free > 0:
            for i in range(bounds.n_free):
                phi_pert = phi.copy()
                phi_pert[i] += eps
                theta_p = theta_from_phi(phi_pert, theta_base, bounds.free_idx)
                Phi_p, Gu_p, GTo_p, GUs_p = discrete_from_theta_single(theta_p, dt_seconds, params.aw, params.ae, params.eta_h)
                x_pred_p = Phi_p @ x + uk * Gu_p + Tok * GTo_p + Usk * GUs_p
                F[:5, 5 + i] = (x_pred_p - x_pred) / eps

        P_pred = F @ P @ F.T + Q
        yk_pred = z_pred[0]
        S = float(H @ P_pred @ H.T + R)
        K = (P_pred @ H.T) / S

        residual = y[k] - yk_pred
        z = z_pred + (K.flatten() * residual)
        P = (I - K @ H) @ P_pred

        ti_mean[k] = z[0]
        ti_var[k] = P[0, 0]
        theta_traj[k] = theta_from_phi(z[5:], theta_base, bounds.free_idx) if bounds.n_free > 0 else theta_base.copy()
        y_pred[k] = yk_pred
        innov_var[k] = S

    return EKFResult(
        ti_mean=ti_mean,
        ti_var=ti_var,
        theta_traj=theta_traj,
        y_pred=y_pred,
        innov_var=innov_var,
    )
