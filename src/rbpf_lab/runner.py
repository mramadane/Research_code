from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml

from .config import RunConfig, SignalConfig, load_config
from .data_loading import WeatherSeries, align_weather, load_radiation, load_temperature
from .ekf import EKFResult, run_ekf
from .model import ThermalParams, simulate_states
from .rbpf import RBPFResult, run_rbpf
from .signals import Event, generate_pulse_heating, generate_square_heating
from .ukf import UKFResult, run_ukf


def _to_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items()}
    return obj


@dataclass
class Signals:
    heater_w: np.ndarray
    latent_w: np.ndarray
    heater_events: Tuple[Event, ...]
    latent_events: Tuple[Event, ...]


@dataclass
class GroundTruth:
    states: np.ndarray
    measurements: np.ndarray


@dataclass
class RunOutputs:
    config: RunConfig
    weather: WeatherSeries
    signals: Signals
    truth: GroundTruth
    y_meas: np.ndarray
    rbpf: RBPFResult | None = None
    ekf: EKFResult | None = None
    ukf: UKFResult | None = None
    out_dir: Path


def _build_signal(cfg: SignalConfig, N: int, dt_seconds: int) -> Tuple[np.ndarray, Tuple[Event, ...]]:
    if cfg.kind == "square":
        sig, events = generate_square_heating(
            N=N,
            dt_seconds=dt_seconds,
            power=cfg.power,
            p_start=cfg.p_start,
            min_duration_s=cfg.min_duration_s,
            max_duration_s=cfg.max_duration_s,
            seed=cfg.seed,
        )
    elif cfg.kind == "pulse":
        sig, events = generate_pulse_heating(
            N=N,
            dt_seconds=dt_seconds,
            mode="alpha",
            p_start=cfg.p_start,
            amp_range=cfg.amp_range or (1500.0, 8000.0),
            tau_r_range_s=cfg.tau_r_range_s or (5 * 60, 40 * 60),
            tau_d_range_s=cfg.tau_d_range_s or (45 * 60, 4 * 60 * 60),
            sigma_range_s=cfg.sigma_range_s or (15 * 60, 90 * 60),
            kill_rel=cfg.kill_rel,
            max_active=cfg.max_active,
            seed=cfg.seed,
        )
    else:
        raise ValueError(f"Unknown signal kind: {cfg.kind}")
    return sig, tuple(events)


def run_experiment(cfg: RunConfig, *, output_dir_override: Path | None = None, filters: Tuple[str, ...] | None = None) -> RunOutputs:
    temp_df = load_temperature(cfg.files.temperature_file)
    rad_df = load_radiation(cfg.files.solar_file)
    weather = align_weather(temp_df, rad_df, cfg.window.start, cfg.window.end, cfg.window.dt_seconds)
    N = weather.timestamps.size

    latent, latent_events = _build_signal(cfg.latent, N, cfg.window.dt_seconds)
    heater, heater_events = _build_signal(cfg.heater, N, cfg.window.dt_seconds)

    params = ThermalParams.from_config(cfg.model)
    X_true, y_true = simulate_states(
        params,
        weather,
        heater_w=heater,
        latent_w=latent,
        dt_seconds=cfg.window.dt_seconds,
        q_std=0.03,
        seed=cfg.rbpf.random_seed + 11,
    )

    rng = np.random.default_rng(cfg.rbpf.random_seed + 7)
    y_meas = y_true + rng.normal(0.0, cfg.rbpf.sigma_y, size=y_true.shape)

    filter_set = {f.lower() for f in filters} if filters else {"rbpf", "ekf", "ukf"}

    rbpf_result = None
    if cfg.rbpf.enabled and "rbpf" in filter_set:
        rbpf_result = run_rbpf(
            y_meas=y_meas,
            u=heater,
            t_out_k=weather.t_out_k,
            u_s=weather.u_s,
            params=params,
            dt_seconds=cfg.window.dt_seconds,
            cfg=cfg.rbpf,
        )

    ekf_result = None
    if cfg.ekf.enabled and "ekf" in filter_set:
        ekf_result = run_ekf(
            y_meas=y_meas,
            u=heater,
            t_out_k=weather.t_out_k,
            u_s=weather.u_s,
            params=params,
            dt_seconds=cfg.window.dt_seconds,
            cfg=cfg.ekf,
        )

    ukf_result = None
    if cfg.ukf.enabled and "ukf" in filter_set:
        ukf_result = run_ukf(
            y_meas=y_meas,
            u=heater,
            t_out_k=weather.t_out_k,
            u_s=weather.u_s,
            params=params,
            dt_seconds=cfg.window.dt_seconds,
            cfg=cfg.ukf,
        )

    out_dir = output_dir_override or cfg.output_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    return RunOutputs(
        config=cfg,
        weather=weather,
        signals=Signals(
            heater_w=heater,
            latent_w=latent,
            heater_events=heater_events,
            latent_events=latent_events,
        ),
        truth=GroundTruth(states=X_true, measurements=y_true),
        y_meas=y_meas,
        rbpf=rbpf_result,
        ekf=ekf_result,
        ukf=ukf_result,
        out_dir=out_dir,
    )


def save_outputs(outputs: RunOutputs) -> Path:
    """Persist a run to disk; returns the directory used."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs.out_dir / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = run_dir / "config_used.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            _to_jsonable(outputs.config),
            f,
            sort_keys=False,
        )

    if outputs.config.save_npz:
        payload = {
            "timestamps": outputs.weather.timestamps,
            "t_out_c": outputs.weather.t_out_c,
            "t_out_k": outputs.weather.t_out_k,
            "u_s": outputs.weather.u_s,
            "heater": outputs.signals.heater_w,
            "latent": outputs.signals.latent_w,
            "y_true": outputs.truth.measurements,
            "y_meas": outputs.y_meas,
        }

        if outputs.rbpf is not None:
            payload.update(
                dict(
                    rbpf_ti_lo=outputs.rbpf.ti_lo,
                    rbpf_ti_md=outputs.rbpf.ti_md,
                    rbpf_ti_hi=outputs.rbpf.ti_hi,
                    rbpf_theta_lo=outputs.rbpf.theta_lo,
                    rbpf_theta_md=outputs.rbpf.theta_md,
                    rbpf_theta_hi=outputs.rbpf.theta_hi,
                    rbpf_alpha_lo=outputs.rbpf.alpha_lo,
                    rbpf_alpha_md=outputs.rbpf.alpha_md,
                    rbpf_alpha_hi=outputs.rbpf.alpha_hi,
                    rbpf_regime_flag=outputs.rbpf.regime_flag,
                )
            )

        if outputs.ekf is not None:
            payload.update(
                dict(
                    ekf_ti_mean=outputs.ekf.ti_mean,
                    ekf_ti_var=outputs.ekf.ti_var,
                    ekf_theta_traj=outputs.ekf.theta_traj,
                    ekf_y_pred=outputs.ekf.y_pred,
                    ekf_innov_var=outputs.ekf.innov_var,
                )
            )

        if outputs.ukf is not None:
            payload.update(
                dict(
                    ukf_ti_mean=outputs.ukf.ti_mean,
                    ukf_ti_var=outputs.ukf.ti_var,
                    ukf_theta_traj=outputs.ukf.theta_traj,
                    ukf_y_pred=outputs.ukf.y_pred,
                    ukf_innov_var=outputs.ukf.innov_var,
                )
            )

        np.savez_compressed(run_dir / "filters_outputs.npz", **payload)

    events_path = run_dir / "events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "heater": [e.__dict__ for e in outputs.signals.heater_events],
                "latent": [e.__dict__ for e in outputs.signals.latent_events],
            },
            f,
            indent=2,
        )

    return run_dir


def load_and_run(config_path: Path, *, output_dir: Path | None = None, filters: Tuple[str, ...] | None = None, persist: bool = True) -> RunOutputs:
    cfg = load_config(config_path)
    outputs = run_experiment(cfg, output_dir_override=output_dir, filters=filters)
    if persist:
        run_dir = save_outputs(outputs)
        outputs.out_dir = run_dir
    return outputs
