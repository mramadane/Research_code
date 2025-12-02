from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import RBPFConfig
from .model import ThermalParams, default_param_ranges, discrete_from_theta


@dataclass
class RBPFResult:
    ti_lo: np.ndarray
    ti_md: np.ndarray
    ti_hi: np.ndarray
    theta_lo: np.ndarray
    theta_md: np.ndarray
    theta_hi: np.ndarray
    alpha_lo: np.ndarray
    alpha_md: np.ndarray
    alpha_hi: np.ndarray
    regime_flag: np.ndarray
    y_pred_mean: np.ndarray
    y_pred_std: np.ndarray
    weights_last: np.ndarray
    tau_lo: np.ndarray | None = None
    tau_md: np.ndarray | None = None
    tau_hi: np.ndarray | None = None


def _systematic_resample(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = w.size
    u0 = (rng.random() + np.arange(N)) / N
    cs = np.cumsum(w)
    idx = np.empty(N, int)
    i = j = 0
    while i < N:
        if u0[i] < cs[j]:
            idx[i] = j
            i += 1
        else:
            j += 1
    return idx


def _wquantile(x: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    i = np.argsort(x)
    xs, ws = x[i], w[i]
    cw = np.cumsum(ws)
    cw = cw / cw[-1] if cw[-1] > 0 else np.linspace(1 / len(ws), 1, len(ws))
    return np.interp(qs, cw, xs)


def _logsumexp(v: np.ndarray) -> float:
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))


def _sample_gaussian_mixture(means: np.ndarray, variances: np.ndarray, weights: np.ndarray, nsamp: int, rng: np.random.Generator) -> np.ndarray:
    cw = np.cumsum(weights)
    cw = cw / cw[-1] if cw[-1] > 0 else np.linspace(1 / len(weights), 1, len(weights))
    u = (rng.random(nsamp) + np.arange(nsamp)) / nsamp
    idx = np.searchsorted(cw, u, side="right")
    idx = np.clip(idx, 0, len(weights) - 1)
    stds = np.sqrt(np.maximum(variances[idx], 1e-12))
    return means[idx] + stds * rng.standard_normal(nsamp)


def _build_bounds(param_ranges: Dict[str, float | Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_names = ["Ci", "Cm", "Ce", "Ch", "Cs", "g_im", "g_ie", "g_ih", "g_is", "g_ea"]
    theta_lo = np.empty(len(theta_names), float)
    theta_hi = np.empty(len(theta_names), float)
    is_fixed = np.zeros(len(theta_names), dtype=bool)
    for j, nm in enumerate(theta_names):
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
    return theta_lo, theta_hi, free_idx, fixed_idx


def run_rbpf(
    y_meas: np.ndarray,
    u: np.ndarray,
    t_out_k: np.ndarray,
    u_s: np.ndarray,
    params: ThermalParams,
    dt_seconds: int,
    cfg: RBPFConfig,
    param_ranges: Dict[str, float | Tuple[float, float]] | None = None,
) -> RBPFResult:
    """Run the Rao-Blackwellized particle filter with the gating logic from the notebook."""
    y_meas = np.asarray(y_meas, float).ravel()
    u_arr = np.asarray(u, float).ravel()
    T_out = np.asarray(t_out_k, float).ravel()
    U_s_arr = np.asarray(u_s, float).ravel()
    N = int(len(y_meas))
    param_ranges = param_ranges or default_param_ranges(params)

    theta_lo, theta_hi, free_idx, fixed_idx = _build_bounds(param_ranges)
    theta_eps = 1e-20

    rng = np.random.default_rng(cfg.random_seed)
    Np = cfg.nparticles
    q_states = np.asarray(cfg.q_states, float)

    C_y = np.array([[1.0, 0, 0, 0, 0]], float)
    R = np.array([[cfg.sigma_y**2]], float)

    t0 = float(y_meas[0]) if np.isfinite(y_meas[0]) else float(T_out[0])
    m = np.tile(np.array([t0] * 5, float), (Np, 1))
    P = np.tile(np.diag(np.full(5, 0.2**2)), (Np, 1, 1))

    log_alpha = rng.normal(0.0, 0.1, size=Np)
    log_alpha = np.clip(log_alpha, np.log(cfg.alpha_min), np.log(cfg.alpha_max))
    alpha = np.exp(log_alpha)

    phi_lo = np.log(np.clip(theta_lo, theta_eps, None))
    phi_hi = np.log(np.clip(theta_hi, theta_eps, None))
    phi = rng.uniform(low=phi_lo[None, :], high=phi_hi[None, :], size=(Np, len(theta_lo)))
    if fixed_idx.size > 0:
        phi[:, fixed_idx] = np.log(np.clip(theta_lo[fixed_idx], theta_eps, None))
    theta = np.exp(phi)

    logw = np.full(Np, -np.log(Np), float)
    Q0_diag = q_states**2

    regime_flag = np.zeros(N, dtype=int)
    z_std_innov = np.zeros(N)

    n_bins_alpha = 30
    log_alpha_min = np.log(cfg.alpha_min)
    log_alpha_max = np.log(cfg.alpha_max)
    alpha_bin_edges = np.linspace(log_alpha_min, log_alpha_max, n_bins_alpha + 1)
    pseudo_count = 1.0
    alpha_hist_counts = np.full(n_bins_alpha, pseudo_count, dtype=float)
    F_quiet, F_burst = 0.50, 0.80

    q_lo, q_md, q_hi = 0.025, 0.50, 0.975
    Ti_lo = np.zeros(N)
    Ti_md = np.zeros(N)
    Ti_hi = np.zeros(N)
    Alpha_lo = np.zeros(N)
    Alpha_md = np.zeros(N)
    Alpha_hi = np.zeros(N)
    Th_lo = np.zeros((N, len(theta_lo)))
    Th_md = np.zeros((N, len(theta_lo)))
    Th_hi = np.zeros((N, len(theta_lo)))
    Y_pred_mean = np.full(N, np.nan)
    Y_pred_std = np.full(N, np.nan)

    tau_defs = [
        ("Ci/g_im", 0, 5),
        ("Ci/g_ie", 0, 6),
        ("Ci/g_ih", 0, 7),
        ("Ci/g_is", 0, 8),
        ("Cm/g_im", 1, 5),
        ("Ce/g_ie", 2, 6),
        ("Ce/g_ea", 2, 9),
        ("Ch/g_ih", 3, 7),
        ("Cs/g_is", 4, 8),
    ]
    Tau_lo = np.zeros((N, len(tau_defs)))
    Tau_md = np.zeros((N, len(tau_defs)))
    Tau_hi = np.zeros((N, len(tau_defs)))

    Nsamp_state = 2000
    Nsamp_predH = 2000
    H = cfg.horizon

    # t = 0 summaries
    w_lin = np.exp(logw - _logsumexp(logw))
    Ti_draws_0 = _sample_gaussian_mixture(m[:, 0], P[:, 0, 0], w_lin, Nsamp_state, rng)
    TiC0 = Ti_draws_0 - 273.15
    Ti_lo[0], Ti_md[0], Ti_hi[0] = np.quantile(TiC0, [q_lo, q_md, q_hi])
    Alpha_lo[0], Alpha_md[0], Alpha_hi[0] = _wquantile(alpha, w_lin, [q_lo, q_md, q_hi])
    for j in range(len(theta_lo)):
        Th_lo[0, j], Th_md[0, j], Th_hi[0, j] = _wquantile(theta[:, j], w_lin, [q_lo, q_md, q_hi])
    for t_idx, (_, c_idx, g_idx) in enumerate(tau_defs):
        tau_vals = theta[:, c_idx] / np.maximum(theta[:, g_idx], 1e-300)
        Tau_lo[0, t_idx], Tau_md[0, t_idx], Tau_hi[0, t_idx] = _wquantile(tau_vals, w_lin, [q_lo, q_md, q_hi])

    alpha_med_0 = Alpha_md[0]
    log_alpha_med_0 = np.log(np.clip(alpha_med_0, cfg.alpha_min, cfg.alpha_max))
    bin_idx_0 = np.searchsorted(alpha_bin_edges, log_alpha_med_0, side="right") - 1
    bin_idx_0 = np.clip(bin_idx_0, 0, n_bins_alpha - 1)
    alpha_hist_counts[bin_idx_0] += 1.0

    I5_batch = np.eye(5)[None, :, :]
    const_ll = -0.5 * np.log(2 * np.pi)

    for k in range(1, N):
        uk, Tok, Usk = u_arr[k - 1], T_out[k - 1], U_s_arr[k - 1]
        yk = float(y_meas[k])
        w_lin_prior = np.exp(logw - _logsumexp(logw))

        log_alpha = log_alpha + rng.normal(0.0, cfg.sigma_eta, size=Np)
        log_alpha = np.clip(log_alpha, np.log(cfg.alpha_min), np.log(cfg.alpha_max))
        alpha = np.exp(log_alpha)

        alpha_med_k = _wquantile(alpha, w_lin_prior, [0.5])[0]
        log_alpha_med_k = np.log(np.clip(alpha_med_k, cfg.alpha_min, cfg.alpha_max))
        bin_idx = np.searchsorted(alpha_bin_edges, log_alpha_med_k, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, n_bins_alpha - 1)
        alpha_hist_counts[bin_idx] += 1.0
        cdf_bins = np.cumsum(alpha_hist_counts)
        cdf_bins /= cdf_bins[-1]
        F_k = cdf_bins[bin_idx]
        if F_k <= F_quiet:
            g_alpha = 1.0
            regime_flag[k] = 0
        elif F_k >= F_burst:
            g_alpha = cfg.g_min
            regime_flag[k] = 1
        else:
            r = (F_burst - F_k) / (F_burst - F_quiet + 1e-12)
            g_alpha = cfg.g_min + (1.0 - cfg.g_min) * r
            regime_flag[k] = 0
        g_k = max(g_alpha, cfg.g_min)

        phi = np.log(np.clip(theta, theta_eps, None))
        if free_idx.size > 0:
            a_k = 1.0 - g_k * (1.0 - cfg.a_lw)
            h2_k = g_k * (1.0 - cfg.a_lw**2)
            phi_bar = np.sum(phi[:, free_idx] * w_lin_prior[:, None], axis=0)
            diff = phi[:, free_idx] - phi_bar[None, :]
            V = diff.T @ (diff * w_lin_prior[:, None])
            V = 0.5 * (V + V.T) + 1e-12 * np.eye(free_idx.size)
            L = np.linalg.cholesky(V)
            eps = rng.standard_normal((Np, free_idx.size)) @ (np.sqrt(h2_k) * L.T)
            phi[:, free_idx] = a_k * phi[:, free_idx] + (1.0 - a_k) * phi_bar[None, :] + eps
            phi[:, free_idx] = np.clip(phi[:, free_idx], phi_lo[None, free_idx], phi_hi[None, free_idx])
        if fixed_idx.size > 0:
            phi[:, fixed_idx] = np.log(np.clip(theta_lo[fixed_idx], theta_eps, None))
        theta = np.exp(phi)

        Phi_i, Gu_i, GTo_i, GUs_i = discrete_from_theta(theta, dt_seconds, params.aw, params.ae, params.eta_h)

        m_pred = np.einsum("ni,nji->nj", m, Phi_i)
        m_pred += uk * Gu_i
        m_pred += Tok * GTo_i
        m_pred += Usk * GUs_i

        P_pred = np.einsum("nij,njk,nlk->nil", Phi_i, P, Phi_i)
        scale_sq = np.exp(2.0 * log_alpha)
        qdiag = scale_sq[:, None] * Q0_diag[None, :]
        for s in range(5):
            P_pred[:, s, s] += qdiag[:, s]

        y_pred = (m_pred @ C_y.T).ravel()
        S = np.einsum("ij,njk,kl->n", C_y, P_pred, C_y.T) + R[0, 0]
        S = np.maximum(S, 1e-12)

        mu_k = np.sum(w_lin_prior * y_pred)
        var_k = np.sum(w_lin_prior * (y_pred**2 + S)) - mu_k**2
        var_k = max(var_k, 1e-12)
        Y_pred_mean[k] = mu_k
        Y_pred_std[k] = np.sqrt(var_k)
        z_std_innov[k] = (yk - mu_k) / np.sqrt(var_k)

        innov = yk - y_pred
        ll = const_ll - 0.5 * np.log(S) - 0.5 * (innov**2) / S
        logw = logw + ll
        logw = logw - _logsumexp(logw)
        w_lin_post = np.exp(logw)

        K = (P_pred @ C_y.T) / S[:, None, None]
        m = m_pred + (K.reshape(Np, 5) * innov[:, None])
        KC = np.matmul(K, C_y)
        P = np.matmul(I5_batch - KC, P_pred)

        if H >= 1:
            m_h = m.copy()
            P_h = P.copy()
            scale_sq_h = np.exp(2.0 * log_alpha)
            qdiag_h = scale_sq_h[:, None] * Q0_diag[None, :]
            for h in range(1, H + 1):
                kh = min(k + h - 1, N - 1)
                uk_h = u_arr[kh]
                Tok_h = T_out[kh]
                Usk_h = U_s_arr[kh]
                m_h = np.einsum("ni,nji->nj", m_h, Phi_i)
                m_h += uk_h * Gu_i
                m_h += Tok_h * GTo_i
                m_h += Usk_h * GUs_i
                P_h = np.einsum("nij,njk,nlk->nil", Phi_i, P_h, Phi_i)
                for s in range(5):
                    P_h[:, s, s] += qdiag_h[:, s]

        Ti_draws = _sample_gaussian_mixture(m[:, 0], P[:, 0, 0], w_lin_post, Nsamp_state, rng)
        TiC = Ti_draws - 273.15
        Ti_lo[k], Ti_md[k], Ti_hi[k] = np.quantile(TiC, [q_lo, q_md, q_hi])
        Alpha_lo[k], Alpha_md[k], Alpha_hi[k] = _wquantile(alpha, w_lin_post, [q_lo, q_md, q_hi])
        for j in range(len(theta_lo)):
            Th_lo[k, j], Th_md[k, j], Th_hi[k, j] = _wquantile(theta[:, j], w_lin_post, [q_lo, q_md, q_hi])
        for t_idx, (_, c_idx, g_idx) in enumerate(tau_defs):
            tau_vals = theta[:, c_idx] / np.maximum(theta[:, g_idx], 1e-300)
            Tau_lo[k, t_idx], Tau_md[k, t_idx], Tau_hi[k, t_idx] = _wquantile(tau_vals, w_lin_post, [q_lo, q_md, q_hi])

        Neff = 1.0 / np.sum(w_lin_post * w_lin_post)
        if Neff < cfg.resample_frac * Np:
            idx = _systematic_resample(w_lin_post, rng)
            m = m[idx]
            P = P[idx]
            log_alpha = log_alpha[idx]
            alpha = alpha[idx]
            theta = theta[idx]
            logw[:] = -np.log(Np)

    return RBPFResult(
        ti_lo=Ti_lo,
        ti_md=Ti_md,
        ti_hi=Ti_hi,
        theta_lo=Th_lo,
        theta_md=Th_md,
        theta_hi=Th_hi,
        alpha_lo=Alpha_lo,
        alpha_md=Alpha_md,
        alpha_hi=Alpha_hi,
        regime_flag=regime_flag,
        y_pred_mean=Y_pred_mean,
        y_pred_std=Y_pred_std,
        weights_last=np.exp(logw - _logsumexp(logw)),
        tau_lo=Tau_lo,
        tau_md=Tau_md,
        tau_hi=Tau_hi,
    )
