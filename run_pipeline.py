#!/usr/bin/env python3
"""Run one module or the complete MS-FFSD generation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"


def run(script: Path, *arguments: str) -> None:
    command = [sys.executable, str(script), *map(str, arguments)]
    print("$ " + " ".join(command), flush=True)
    subprocess.check_call(command, cwd=ROOT)


def run_timestamp() -> None:
    run(ROOT / "timestamp_synthesis" / "run_ffsd_timestamp_synthesis.py")


def run_initialization() -> None:
    run(ROOT / "initialization" / "source_location_province_initialization.py")
    run(ROOT / "initialization" / "user_information_initialization.py")
    run(ROOT / "initialization" / "merchant_initialization.py")


def run_optimization() -> None:
    # Fraud-aware category refinement consumes the optimizer's profile analysis.
    optimizer_arguments: list[str] = []
    run(
        ROOT / "consistency_optimization" / "code" / "run_dataset_consistency_optimization.py",
        *optimizer_arguments,
    )
    run(ROOT / "consistency_optimization" / "code" / "apply_merchant_fraud_semantics.py")


def require_api_configuration(args: argparse.Namespace) -> None:
    missing = []
    for name, value in [
        ("base URL", args.base_url),
        ("API key", args.api_key),
        ("model", args.model),
    ]:
        if not value:
            missing.append(name)
    if missing:
        raise ValueError(
            "Module 4 requires " + ", ".join(missing) + ". "
            "Pass the command-line options or set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL."
        )


def api_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
    ]


def resume_arguments(output: Path, checkpoint: Path) -> list[str]:
    return ["--resume"] if output.exists() or checkpoint.exists() else []


def run_descriptions(args: argparse.Namespace) -> None:
    require_api_configuration(args)
    generation_dir = ROOT / "long_text_generation"
    checkpoint_dir = WORK_DIR / "checkpoints"
    run(generation_dir / "run_long_text_generation.py")
    run(generation_dir / "generate_single_transaction_source_descriptions.py")
    run(
        generation_dir / "generate_target_descriptions.py",
        *api_arguments(args),
        *resume_arguments(WORK_DIR / "target_long_text.csv", checkpoint_dir / "target_long_text.jsonl"),
    )
    run(generation_dir / "deduplicate_merchant_names.py", *api_arguments(args))
    run(
        generation_dir / "generate_multi_transaction_source_descriptions.py",
        *api_arguments(args),
        *resume_arguments(
            WORK_DIR / "source_multi_transaction_long_text.csv",
            checkpoint_dir / "source_multi_transaction_long_text.jsonl",
        ),
    )
    run(generation_dir / "merge_user_descriptions.py")


def run_publish() -> None:
    run(ROOT / "publish_dataset.py")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "module",
        choices=["timestamp", "initialize", "optimize", "describe", "publish", "all"],
        help="Pipeline module to run.",
    )
    parser.add_argument("--base-url", default=os.getenv("LLM_BASE_URL", ""))
    parser.add_argument("--api-key", default=os.getenv("LLM_API_KEY", ""))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL", ""))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.module in {"timestamp", "all"}:
        run_timestamp()
    if args.module in {"initialize", "all"}:
        run_initialization()
    if args.module in {"optimize", "all"}:
        run_optimization()
    if args.module in {"describe", "all"}:
        run_descriptions(args)
    if args.module in {"publish", "all"}:
        run_publish()


if __name__ == "__main__":
    main()
