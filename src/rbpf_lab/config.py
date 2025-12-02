from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class FileConfig:
    """Input file locations."""

    temperature_file: Path
    solar_file: Path


@dataclass
class WindowConfig:
    """Time window and grid spacing."""

    start: str
    end: str
    dt_seconds: int = 120


@dataclass
class SignalConfig:
    """Configuration for a latent or heater signal generator."""

    kind: str = "square"  # square | pulse
    power: float = 0.0
    p_start: float = 0.001
    min_duration_s: int = 1200
    max_duration_s: int = 3600
    seed: Optional[int] = None

    # Pulse-only fields
    amp_range: Optional[tuple[float, float]] = None
    tau_r_range_s: Optional[tuple[float, float]] = None
    tau_d_range_s: Optional[tuple[float, float]] = None
    sigma_range_s: Optional[tuple[float, float]] = None
    kill_rel: float = 1e-3
    max_active: int = 6


@dataclass
class ModelConfig:
    """Thermal model parameters."""

    capacities_kwh_per_k: Dict[str, float] = field(
        default_factory=lambda: {
            "Ci": 2.745148488,
            "Cm": 19.76698557,
            "Ce": 2.412573281,
            "Ch": 0.101200218,
            "Cs": 0.106406602,
        }
    )
    resistances_k_per_kw: Dict[str, float] = field(
        default_factory=lambda: {
            "Rie": 0.532152894,
            "Rea": 7.115085934,
            "Ria": float("inf"),
            "Rim": 0.552472387,
            "Rih": 0.269801371,
            "Ris": 2.349904019,
        }
    )
    aw_m2: float = 4.542558125
    ae_m2: float = 0.0
    eta_h: float = 1.0


@dataclass
class RBPFConfig:
    """RBPF hyper-parameters."""

    nparticles: int = 4000
    sigma_y: float = 0.15
    q_states: tuple[float, float, float, float, float] = (0.01, 0.01, 0.01, 0.01, 0.01)
    alpha_min: float = 0.01
    alpha_max: float = 12.0
    sigma_eta: float = 0.06
    a_lw: float = 0.99
    g_min: float = 0.05
    resample_frac: float = 0.5
    horizon: int = 1
    random_seed: int = 123


@dataclass
class RunConfig:
    """Top-level configuration for a single run."""

    files: FileConfig
    window: WindowConfig
    heater: SignalConfig
    latent: SignalConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    rbpf: RBPFConfig = field(default_factory=RBPFConfig)
    output_dir: Path = Path("runs")
    save_npz: bool = True


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_signal(data: Dict[str, Any], defaults: SignalConfig) -> SignalConfig:
    merged = {**defaults.__dict__, **(data or {})}
    return SignalConfig(**merged)


def load_config(path: Path) -> RunConfig:
    """Load a YAML config file into strongly typed dataclasses."""

    raw = _load_yaml(path)

    files = FileConfig(
        temperature_file=Path(raw["files"]["temperature_file"]),
        solar_file=Path(raw["files"]["solar_file"]),
    )

    window = WindowConfig(**raw["window"])
    heater = _parse_signal(raw.get("heater", {}), SignalConfig(kind="square", power=800.0, p_start=0.0035,
                                                               min_duration_s=20 * 60, max_duration_s=3600,
                                                               seed=456))
    latent = _parse_signal(raw.get("latent", {}), SignalConfig(kind="square", power=7000.0, p_start=0.0011,
                                                               min_duration_s=20 * 60, max_duration_s=8 * 3600,
                                                               seed=321))

    model = ModelConfig(**raw.get("model", {}))
    rbpf_cfg = RBPFConfig(**raw.get("rbpf", {}))
    output_dir = Path(raw.get("output_dir", "runs"))
    save_npz = bool(raw.get("save_npz", True))

    return RunConfig(
        files=files,
        window=window,
        heater=heater,
        latent=latent,
        model=model,
        rbpf=rbpf_cfg,
        output_dir=output_dir,
        save_npz=save_npz,
    )
