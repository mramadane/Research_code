# RBPF Research Repository

Clean, configurable code to reproduce the experiments from the paper  
**“Calibrated Online RC Modeling with a Rao–Blackwellized Particle Filter and Latent Volatility.”**

The original Colab notebook export lives in `Raw_files` for provenance; this repository provides a small, structured package that you can run as-is to reproduce the paper's results or adapt to your own data.

## Layout
- `Raw_files/` – original notebook exports (not modified)
- `Data/` – weather inputs used in the paper (DWD temperature and radiation)
- `configs/` – YAML configs (edit or copy for new runs)
- `src/rbpf_lab/` – reusable Python package (data loading, signal generators, RC model, RBPF / EKF / UKF filters, CLI)
- `runs/` – auto-created outputs (NPZ + metadata)

## Setup
1. Python 3.10+ recommended.
2. Install deps (editable install):  
   `python -m pip install -e .`  
   - or quick: `python -m pip install -r requirements.txt` and run with `PYTHONPATH=src`

## Quick start
Run the full pipeline (RBPF + EKF + UKF) with the provided data and defaults:

```bash
python -m rbpf_lab.cli --config configs/demo.yaml
```

Artifacts land under `runs/<timestamp>/`:
- `config_used.yaml` – resolved config for the run
- `filters_outputs.npz` – weather grid, signals, measurements, and any enabled filter outputs
- `events.json` – heater / latent event logs

Use `--filters rbpf,ekf` to limit which filters run, `--no-save` to skip writing files, or `--output-dir my_runs` to change where results go.

## Configuring your own experiment
Edit or copy `configs/demo.yaml`. Key fields:
- `files.temperature_file` / `files.solar_file`: paths to DWD-like CSVs.
- `window.start`, `window.end`, `window.dt_seconds`: time range and grid spacing.
- `heater` and `latent`: choose `kind: square` or `kind: pulse` plus power / probability / duration options.
- `rbpf`, `ekf`, `ukf`: enable / disable each filter and tune their noise settings.
- `model` (optional): override default capacities / resistances / areas if testing a different building.

All parameters are resolved into strongly typed dataclasses (`rbpf_lab.config`) before running.

## How it works
1. Load weather data, align it to a uniform grid, and interpolate gaps (`rbpf_lab.data_loading`).
2. Generate heater / latent heat signals (`rbpf_lab.signals`).
3. Simulate the 5-state RC model to produce “true” indoor temperature and noisy measurements (`rbpf_lab.model`).
4. Run the RBPF (with Liu–West adaptation), EKF, and/or UKF from the notebook, each sharing the same parameter boxes and inputs (`rbpf_lab.rbpf`, `rbpf_lab.ekf`, `rbpf_lab.ukf`).
5. Save compact artifacts for downstream analysis (`rbpf_lab.runner`).

## Notes
- The raw Colab export (`Raw_files/rbpf_hist_q_v2.py`) is preserved intact; new code lives under `src/`.
- Dependencies stay light (NumPy, pandas, FilterPy, PyYAML). Add plotting or control-toolbox packages as needed for your analysis scripts.
- If you point the config at your own data, ensure timestamps follow the DWD format or adapt the loaders in `rbpf_lab.data_loading`.
