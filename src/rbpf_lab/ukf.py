from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

from .config import UKFConfig
from .model import ThermalParams, default_param_ranges, discrete_from_theta_single
from .parameters import build_parameter_bounds, theta_from_phi


@dataclass
class UKFResult:
    ti_mean: np.ndarray  # Kelvin
    ti_var: np.ndarray
    theta_traj: np.ndarray
    y_pred: np.ndarray
    innov_var: np.ndarray

    @property
    def ti_mean_c(self) -> np.ndarray:
        return self.ti_mean - 273.15


def run_ukf(
    y_meas: np.ndarray,
    u: np.ndarray,
    t_out_k: np.ndarray,
    u_s: np.ndarray,
    params: ThermalParams,
    dt_seconds: int,
    cfg: UKFConfig,
    param_ranges: Dict[str, float | Tuple[float, float]] | None = None,
) -> UKFResult:
    """Run the UKF from the notebook in reusable form."""

    y = np.asarray(y_meas, float).ravel()
    u_arr = np.asarray(u, float).ravel()
    T_out = np.asarray(t_out_k, float).ravel()
    U_s_arr = np.asarray(u_s, float).ravel()
    N = int(len(y))
    param_ranges = param_ranges or default_param_ranges(params)

    bounds = build_parameter_bounds(param_ranges)
    theta_base = bounds.theta_lo.copy()

    theta_init = theta_base.copy()
    if bounds.n_free > 0:
        theta_init[bounds.free_idx] = 0.5 * (bounds.theta_lo[bounds.free_idx] + bounds.theta_hi[bounds.free_idx])
        phi0 = np.log(theta_init[bounds.free_idx])
    else:
        phi0 = np.array([], dtype=float)

    x0 = np.array([y[0]] * 5, float)
    z0 = np.concatenate([x0, phi0])
    nz = z0.size

    P0 = np.zeros((nz, nz), float)
    P0[:5, :5] = np.diag(np.full(5, cfg.init_state_std**2))
    if bounds.n_free > 0:
        P0[5:, 5:] = np.diag(np.full(bounds.n_free, cfg.init_param_log_std**2))

    Q = np.zeros_like(P0)
    q_states = np.asarray(cfg.q_states, float)
    Q[:5, :5] = np.diag(q_states**2)
    if bounds.n_free > 0:
        Q[5:, 5:] = np.diag(np.full(bounds.n_free, cfg.sigma_log_theta**2))

    points = MerweScaledSigmaPoints(
        nz,
        alpha=cfg.points_alpha,
        beta=cfg.points_beta,
        kappa=cfg.points_kappa,
    )

    ukf = UnscentedKalmanFilter(dim_x=nz, dim_z=1, dt=dt_seconds, fx=None, hx=None, points=points)
    ukf.x = z0.copy()
    ukf.P = P0.copy()
    ukf.Q = Q
    ukf.R = np.array([[cfg.sigma_y**2]], float)

    state_idx = {"k": 0}

    def fx_augmented(z: np.ndarray, dt: float) -> np.ndarray:
        k = state_idx["k"]
        x = z[:5]
        phi_free = z[5:]
        theta_full = theta_from_phi(phi_free, theta_base, bounds.free_idx) if bounds.n_free > 0 else theta_base.copy()
        Phi, Gu, GTo, GUs = discrete_from_theta_single(theta_full, dt_seconds, params.aw, params.ae, params.eta_h)
        uk = u_arr[k]
        Tok = T_out[k]
        Usk = U_s_arr[k]
        x_next = Phi @ x + uk * Gu + Tok * GTo + Usk * GUs
        return np.concatenate([x_next, phi_free])

    def hx_augmented(z: np.ndarray) -> np.ndarray:
        return np.array([z[0]])

    ukf.fx = fx_augmented
    ukf.hx = hx_augmented

    ti_mean = np.zeros(N)
    ti_var = np.zeros(N)
    theta_traj = np.zeros((N, len(theta_base)))
    y_pred = np.zeros(N)
    innov_var = np.zeros(N)

    ti_mean[0] = ukf.x[0]
    ti_var[0] = ukf.P[0, 0]
    theta_traj[0] = theta_init
    y_pred[0] = ukf.x[0]
    innov_var[0] = ukf.P[0, 0] + ukf.R[0, 0]

    for k in range(1, N):
        state_idx["k"] = k - 1
        ukf.predict()

        y_pred[k] = ukf.x[0]
        innov_var[k] = ukf.P[0, 0] + ukf.R[0, 0]
        ukf.update(np.array([y[k]]))

        ti_mean[k] = ukf.x[0]
        ti_var[k] = ukf.P[0, 0]
        theta_traj[k] = theta_from_phi(ukf.x[5:], theta_base, bounds.free_idx) if bounds.n_free > 0 else theta_base.copy()

    return UKFResult(
        ti_mean=ti_mean,
        ti_var=ti_var,
        theta_traj=theta_traj,
        y_pred=y_pred,
        innov_var=innov_var,
    )
