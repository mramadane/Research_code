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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = load_and_run(args.config, output_dir=args.output_dir, persist=not args.no_save)

    ti_final = outputs.rbpf.ti_md[-1]
    print(f"Completed run with config: {args.config}")
    print(f"Time steps: {outputs.weather.timestamps.size}, dt={outputs.config.window.dt_seconds}s")
    print(f"Final indoor temperature median estimate: {ti_final - 273.15:.3f} C")
    if not args.no_save:
        print(f"Artifacts saved under: {outputs.out_dir}")


if __name__ == "__main__":
    main()
