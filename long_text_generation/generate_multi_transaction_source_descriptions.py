#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate long-text descriptions for users with multiple transactions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from generate_target_descriptions import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    WORK_DIR,
    generate_english_model_output,
    strip_thinking,
)
from user_description_prompts import build_source_description_prompt


DEFAULT_INPUT = WORK_DIR / "multi_transaction_source.jsonl"
DEFAULT_OUTPUT = WORK_DIR / "source_multi_transaction_long_text.csv"
DEFAULT_CHECKPOINT = WORK_DIR / "checkpoints" / "source_multi_transaction_long_text.jsonl"
DEFAULT_SOURCE_LONG_TEXT_OUTPUT = WORK_DIR / "source_long_text.csv"


def read_profiles(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    return pd.DataFrame(
        {
            "Source_id": record["Source_id"],
            "basic_profile": json.dumps(record, ensure_ascii=False, indent=2),
        }
        for record in records
    )


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_output_files(output: Path, checkpoint: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["Source_id", "Source_description"]).to_csv(output, index=False, encoding="utf-8-sig")
    checkpoint.write_text("", encoding="utf-8")


def append_csv(path: Path, record: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Source_id", "Source_description"])
        writer.writerow(record)


def restore_output_from_checkpoint(checkpoint: Path, output: Path) -> None:
    if output.exists() or not checkpoint.exists():
        return
    with checkpoint.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if records:
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).drop_duplicates(subset=["Source_id"], keep="last").to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )


def parse_source_description(model_output: str) -> str:
    text = strip_thinking(model_output)
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Source_description:"):
            return line.split(":", 1)[1].strip()
    return text.strip()


def source_id_sort_key(source_id: object) -> tuple[int, str]:
    """Sort conventional IDs such as S10002 by their numeric suffix."""
    value = str(source_id)
    suffix = value[1:] if value.startswith("S") else value
    return (int(suffix), value) if suffix.isdigit() else (float("inf"), value)


def word_count(text: str) -> int:
    """Count English word tokens, treating hyphenated words as one token."""
    return len(re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text))


def replace_descriptions_in_source_long_text(path: Path, records: list[Mapping[str, str]]) -> None:
    """Replace only the targeted records in the combined file and preserve all others."""
    if not records:
        return
    if not path.exists():
        raise FileNotFoundError(f"Combined long-text file does not exist: {path}")

    existing = pd.read_csv(path, encoding="utf-8-sig", dtype={"Source_id": str})
    existing.columns = [str(column).replace("\ufeff", "").strip() for column in existing.columns]
    required_columns = {"Source_id", "Source_description"}
    if missing := required_columns - set(existing.columns):
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    existing = existing[["Source_id", "Source_description"]]

    new_records = pd.DataFrame(records, columns=["Source_id", "Source_description"])
    new_records["Source_id"] = new_records["Source_id"].astype(str)
    if new_records["Source_id"].duplicated().any():
        raise ValueError("The targeted generation produced duplicate Source_id values.")
    existing_ids = set(existing["Source_id"].astype(str))
    missing_ids = sorted(set(new_records["Source_id"]) - existing_ids)
    if missing_ids:
        raise ValueError(
            "Cannot replace descriptions because Source_id values are missing from the combined file: "
            + ", ".join(missing_ids)
        )

    replacements = new_records.set_index("Source_id")["Source_description"]
    existing["Source_description"] = existing["Source_id"].astype(str).map(replacements).fillna(existing["Source_description"])
    existing["_source_id_sort_key"] = existing["Source_id"].map(source_id_sort_key)
    existing = existing.sort_values("_source_id_sort_key", kind="stable").drop(columns="_source_id_sort_key")
    existing.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Replaced {len(new_records)} descriptions in {path}; total rows: {len(existing)}", flush=True)


def generate_descriptions(args: argparse.Namespace) -> None:
    if not args.base_url or not args.api_key or not args.model:
        raise ValueError(
            "Missing LLM configuration. Pass --base-url, --api-key, and --model, "
            "or set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: openai. Install it with: pip install openai") from exc

    profiles = read_profiles(args.input)
    profiles["Source_id"] = profiles["Source_id"].astype(str)
    if profiles["Source_id"].astype(str).duplicated().any():
        raise ValueError(f"{args.input} contains duplicate Source_id values.")

    targeted_generation = args.regenerate_under_min_words is not None
    if targeted_generation:
        if args.regenerate_under_min_words < 1:
            raise ValueError("--regenerate-under-min-words must be at least 1.")
        if not args.source_long_text_output.exists():
            raise FileNotFoundError(f"Combined long-text file does not exist: {args.source_long_text_output}")
        combined = pd.read_csv(args.source_long_text_output, encoding="utf-8-sig", dtype={"Source_id": str})
        combined.columns = [str(column).replace("\ufeff", "").strip() for column in combined.columns]
        required_columns = {"Source_id", "Source_description"}
        if missing := required_columns - set(combined.columns):
            raise ValueError(
                f"{args.source_long_text_output} is missing required columns: {', '.join(sorted(missing))}"
            )
        requested_ids = set(
            combined.loc[
                combined["Source_description"].fillna("").astype(str).map(word_count)
                < args.regenerate_under_min_words,
                "Source_id",
            ].astype(str)
        )
        known_ids = set(profiles["Source_id"])
        non_multi_ids = sorted(requested_ids - known_ids)
        if non_multi_ids:
            raise ValueError(
                "Some short descriptions are not multi-transaction users and cannot be regenerated by this script: "
                + ", ".join(non_multi_ids)
            )
        profiles = profiles[profiles["Source_id"].isin(requested_ids)].copy()
        profiles["_source_id_sort_key"] = profiles["Source_id"].map(source_id_sort_key)
        profiles = profiles.sort_values("_source_id_sort_key", kind="stable").drop(columns="_source_id_sort_key")
        if args.resume:
            raise ValueError(
                "--resume cannot be combined with --regenerate-under-min-words; "
                "selected descriptions are always generated anew."
            )

    if not args.resume and not targeted_generation:
        print(f"Resetting output files: {args.output} and {args.checkpoint}", flush=True)
        reset_output_files(args.output, args.checkpoint)
    elif args.resume:
        restore_output_from_checkpoint(args.checkpoint, args.output)

    done_ids = set()
    if args.resume and args.output.exists():
        done_ids = set(pd.read_csv(args.output, encoding="utf-8-sig")["Source_id"].astype(str))
    pending = [row for row in profiles.to_dict(orient="records") if str(row["Source_id"]) not in done_ids]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Loaded {len(profiles)} multi-transaction user base profiles from {args.input}", flush=True)
    if targeted_generation:
        print(
            f"Regenerating descriptions with fewer than {args.regenerate_under_min_words} English words.",
            flush=True,
        )
    print(f"Pending user descriptions: {len(pending)}", flush=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    generated_records: list[Mapping[str, str]] = []

    for index, row in enumerate(pending, 1):
        source_id = str(row["Source_id"])
        print(f"[Source] {index}/{len(pending)}: generating {source_id}", flush=True)
        model_output = generate_english_model_output(
            client=client,
            model=args.model,
            prompt=build_source_description_prompt(row),
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
        )
        description = parse_source_description(model_output)
        record = {"Source_id": source_id, "Source_description": description}
        if targeted_generation:
            generated_records.append(record)
        else:
            append_jsonl(args.checkpoint, record)
            append_csv(args.output, record)

    if targeted_generation:
        replace_descriptions_in_source_long_text(args.source_long_text_output, generated_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LLM descriptions for users with multiple transactions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from existing output instead of restarting.")
    parser.add_argument(
        "--regenerate-under-min-words",
        type=int,
        metavar="WORD_COUNT",
        help=(
            "Regenerate every multi-transaction description in source_long_text.csv with fewer than WORD_COUNT "
            "English words. Only selected descriptions are replaced; the file remains sorted by Source_id."
        ),
    )
    parser.add_argument(
        "--source-long-text-output",
        type=Path,
        default=DEFAULT_SOURCE_LONG_TEXT_OUTPUT,
        help="Combined long-text CSV read and updated by --regenerate-under-min-words.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_descriptions(parse_args())
