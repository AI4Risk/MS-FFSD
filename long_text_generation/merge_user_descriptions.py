#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge single- and multi-transaction user long-text files into one CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
DEFAULT_SINGLE_INPUT = WORK_DIR / "source_single_transaction_long_text.csv"
DEFAULT_MULTI_INPUT = WORK_DIR / "source_multi_transaction_long_text.csv"
DEFAULT_OUTPUT = WORK_DIR / "source_long_text.csv"
REQUIRED_COLUMNS = ["Source_id", "Source_description"]


def read_descriptions(path: Path) -> pd.DataFrame:
    descriptions = pd.read_csv(path, encoding="utf-8-sig")
    descriptions.columns = [str(column).replace("\ufeff", "").strip() for column in descriptions.columns]
    missing = set(REQUIRED_COLUMNS) - set(descriptions.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return descriptions[REQUIRED_COLUMNS]


def merge_descriptions(single_input: Path, multi_input: Path, output: Path) -> None:
    merged = pd.concat([read_descriptions(single_input), read_descriptions(multi_input)], ignore_index=True)
    if merged["Source_id"].astype(str).duplicated().any():
        raise ValueError("The two input files contain duplicate Source_id values.")
    if merged["Source_description"].isna().any() or merged["Source_description"].astype(str).str.strip().eq("").any():
        raise ValueError("Input files contain empty Source_description values.")
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Merged {len(merged)} user descriptions into {output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge single- and multi-transaction user descriptions.")
    parser.add_argument("--single-input", type=Path, default=DEFAULT_SINGLE_INPUT)
    parser.add_argument("--multi-input", type=Path, default=DEFAULT_MULTI_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_descriptions(args.single_input, args.multi_input, args.output)
