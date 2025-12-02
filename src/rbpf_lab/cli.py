from __future__ import annotations

import argparse
from pathlib import Path

from .runner import load_and_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RBPF experiments on weather-driven heating data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/demo.yaml"),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing artifacts to disk (still runs the filter).",
    )
    parser.add_argument(
        "--filters",
        type=str,
        default=None,
        help="Comma-separated list of filters to run (rbpf,ekf,ukf). Default: all enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filters = None
    if args.filters:
        filters = tuple(f.strip().lower() for f in args.filters.split(",") if f.strip())

    outputs = load_and_run(args.config, output_dir=args.output_dir, filters=filters, persist=not args.no_save)

    print(f"Completed run with config: {args.config}")
    print(f"Time steps: {outputs.weather.timestamps.size}, dt={outputs.config.window.dt_seconds}s")
    if outputs.rbpf is not None:
        print(f"RBPF final Ti median: {outputs.rbpf.ti_md[-1]:.3f} C")
    if outputs.ekf is not None:
        print(f"EKF final Ti mean: {outputs.ekf.ti_mean[-1] - 273.15:.3f} C")
    if outputs.ukf is not None:
        print(f"UKF final Ti mean: {outputs.ukf.ti_mean[-1] - 273.15:.3f} C")
    if not args.no_save:
        print(f"Artifacts saved under: {outputs.out_dir}")


if __name__ == "__main__":
    main()
