from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import load_raw_data, prepare_dataset, validation_report
from .download import download_bts_export


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flight-delay-milp",
        description="Reproduce leakage-aware flight-delay prediction and MILP scheduling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-bts", help="Download an official BTS ZIP export.")
    download.add_argument("--url", required=True)
    download.add_argument("--output-dir", type=Path, default=Path("data/raw"))

    validate = subparsers.add_parser("validate-data", help="Validate a BTS CSV/ZIP export.")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--config", type=Path, default=Path("configs/paper.yaml"))

    run = subparsers.add_parser("run", help="Run a smoke or formal experiment contract.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--input", type=Path)

    tune = subparsers.add_parser("tune", help="Run or resume the frozen nested-CV contract.")
    tune.add_argument("--config", type=Path, required=True)
    tune.add_argument("--input", type=Path)
    tune.add_argument("--resume-run", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "download-bts":
        manifest = download_bts_export(args.url, args.output_dir)
        print(json.dumps(manifest, indent=2))
        return
    if args.command == "validate-data":
        config = load_config(args.config)
        raw = load_raw_data(args.input)
        _, summary = prepare_dataset(raw, config["data"])
        report = validation_report(summary, config["data"])
        print(json.dumps(report, indent=2))
        return
    if args.command == "run":
        from .run import run_experiment

        run_dir = run_experiment(load_config(args.config), args.input)
        print(run_dir.resolve())
        return
    if args.command == "tune":
        from .tuning_run import run_tuning_experiment

        run_dir = run_tuning_experiment(
            load_config(args.config), args.input, resume_run=args.resume_run
        )
        print(run_dir.resolve())
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
