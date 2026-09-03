#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rename duplicate virtual merchant names with targeted LLM post-processing."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from generate_target_descriptions import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    WORK_DIR,
    stream_chat_completion,
    strip_thinking,
)
from merchant_name_deduplication_prompts import build_merchant_rename_prompt


DEFAULT_INPUT = WORK_DIR / "target_long_text.csv"


def normalize_name(value: object) -> str:
    """Normalize case, whitespace, and punctuation for uniqueness checks."""
    if pd.isna(value):
        return ""
    return re.sub(r"[\W_]+", "", str(value or "").strip().casefold(), flags=re.UNICODE)


def text_value(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def validate_input(df: pd.DataFrame, path: Path) -> None:
    required = {"Target_id", "Merchant", "Merchant_description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    if df["Target_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate Target_id values.")


def duplicate_indexes(df: pd.DataFrame) -> list[int]:
    """Return all but the first row for each non-empty duplicate merchant name."""
    seen: set[str] = set()
    duplicates: list[int] = []
    for index, value in df["Merchant"].items():
        name = normalize_name(value)
        if not name or name in seen:
            duplicates.append(index)
        else:
            seen.add(name)
    return duplicates


def parse_name(model_output: str) -> str:
    text = strip_thinking(model_output).strip()
    if text.startswith("```") and text.endswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return ""
    return text_value(payload.get("new_merchant", "")) if isinstance(payload, dict) else ""


def write_csv_atomically(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(temporary_path, index=False, encoding="utf-8-sig")
    temporary_path.replace(path)


def deduplicate_names(args: argparse.Namespace) -> None:
    if not args.base_url or not args.api_key or not args.model:
        raise ValueError(
            "Missing LLM configuration. Pass --base-url, --api-key, and --model, "
            "or set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Missing dependency: openai. Install it with: pip install openai") from exc

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.columns = [str(column).replace("\ufeff", "").strip() for column in df.columns]
    validate_input(df, args.input)
    output = args.output or args.input
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    round_index = 0
    while args.max_rounds is None or round_index < args.max_rounds:
        round_index += 1
        pending = duplicate_indexes(df)
        if not pending:
            if output != args.input:
                write_csv_atomically(df, output)
            print(f"Completed: {len(df)} merchant names are unique.", flush=True)
            return

        print(f"Round {round_index}: {len(pending)} duplicate or empty merchant names need renaming.", flush=True)
        used_names = {normalize_name(value) for value in df["Merchant"] if normalize_name(value)}
        changed = 0
        for position, index in enumerate(pending, 1):
            target_id = str(df.at[index, "Target_id"])
            old_name = text_value(df.at[index, "Merchant"])
            description = text_value(df.at[index, "Merchant_description"])
            for attempt in range(args.name_retries + 1):
                print(f"[Rename] {position}/{len(pending)}: {target_id} (attempt {attempt + 1})", flush=True)
                result = stream_chat_completion(
                    client=client,
                    model=args.model,
                    prompt=build_merchant_rename_prompt(old_name, description),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                )
                new_name = parse_name(result)
                normalized = normalize_name(new_name)
                if normalized and normalized not in used_names:
                    df.at[index, "Merchant"] = new_name
                    used_names.add(normalized)
                    changed += 1
                    write_csv_atomically(df, output)
                    break
                if not normalized:
                    preview = strip_thinking(result).replace("\n", " ").strip()[:300]
                    print(f"  Rejected: response does not contain a valid JSON new_merchant: {preview!r}", flush=True)
                else:
                    print(f"  Rejected: generated name already exists: {new_name!r}", flush=True)
            else:
                print(
                    f"  Deferred: {target_id} did not receive a unique name after "
                    f"{args.name_retries + 1} attempts; it will be retried in the next round.",
                    flush=True,
                )

            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

        print(f"Round {round_index}: renamed {changed} merchant names.", flush=True)

    remaining = len(duplicate_indexes(df))
    raise RuntimeError(f"Stopped after {args.max_rounds} rounds; {remaining} duplicate or empty names remain.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rename duplicate merchant names in target_long_text.csv.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to overwriting --input safely.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for each model API call.")
    parser.add_argument("--name-retries", type=int, default=4, help="Retries when a generated name is still duplicated.")
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional positive limit for full-table retry rounds; omitted by default.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.max_rounds is not None and args.max_rounds < 1:
        parser.error("--max-rounds must be a positive integer when provided.")
    return args


if __name__ == "__main__":
    deduplicate_names(parse_args())
