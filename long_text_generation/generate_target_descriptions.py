#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate merchant long-text descriptions from structured merchant profiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from merchant_description_prompts import build_target_description_prompt


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
DEFAULT_INPUT = WORK_DIR / "target_basic_profile.jsonl"
DEFAULT_OUTPUT = WORK_DIR / "target_long_text.csv"
DEFAULT_CHECKPOINT = WORK_DIR / "checkpoints" / "target_long_text.jsonl"
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "")
DEFAULT_MODEL = os.getenv("LLM_MODEL", "")
DEFAULT_API_KEY = os.getenv("LLM_API_KEY", "")
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path, id_column: str) -> set[str]:
    if not path.exists():
        return set()
    return set(read_csv(path)[id_column].astype(str))


def reset_output_files(output_path: Path, checkpoint_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["Target_id", "Merchant", "Merchant_description"]).to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )
    checkpoint_path.write_text("", encoding="utf-8")


def append_csv(path: Path, record: Mapping[str, str]) -> None:
    with path.open("a", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["Target_id", "Merchant", "Merchant_description"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow({field: record.get(field, "") for field in fieldnames})


def restore_output_from_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    if output_path.exists() or not checkpoint_path.exists():
        return
    with checkpoint_path.open("r", encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = pd.DataFrame(records).drop_duplicates(subset=["Target_id"], keep="last")
        output[["Target_id", "Merchant", "Merchant_description"]].to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )


def read_profile_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            records = [json.loads(line) for line in file if line.strip()]
        rows = []
        for record in records:
            category = record.get("merchant_category", {})
            rows.append(
                {
                    "Target_id": record["Target_id"],
                    "behavior_level1": category.get("MCC_level1", ""),
                    "behavior_level2": category.get("MCC_level2", ""),
                    "basic_profile": json.dumps(record, ensure_ascii=False, indent=2),
                }
            )
        return pd.DataFrame(rows)
    return read_csv(path)


def stream_chat_completion(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    max_tokens: int,
    max_retries: int,
    enable_thinking: bool = True,
) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            return "".join(
                getattr(chunk.choices[0].delta, "content", "") or ""
                for chunk in response
                if chunk.choices
            ).strip()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Model call failed after {max_retries + 1} attempts: {last_error}") from last_error


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def generate_english_model_output(
    client: Any,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    presence_penalty: float,
    max_tokens: int,
    max_retries: int,
) -> str:
    response = stream_chat_completion(
        client, model, prompt, temperature, top_p, presence_penalty, max_tokens, max_retries
    )
    if not CJK_PATTERN.search(response):
        return response

    correction = (
        prompt
        + "\n\nYour previous response contained Chinese characters. Regenerate the required output entirely in English, "
        "with no Chinese characters or mixed Chinese-English terms."
    )
    response = stream_chat_completion(
        client, model, correction, temperature, top_p, presence_penalty, max_tokens, max_retries
    )
    if CJK_PATTERN.search(response):
        raise RuntimeError("Model output still contains Chinese characters after an English-only retry.")
    return response


def normalize_memory_keywords(value: object) -> list[str]:
    parts = [part.strip() for part in re.split(r"[;,|]", str(value)) if part.strip()]
    unique: list[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    if not 3 <= len(unique) <= 5:
        raise ValueError("Memory_keywords must contain 3 to 5 semicolon-separated English concept labels.")
    if any(CJK_PATTERN.search(part) for part in unique):
        raise ValueError("Memory_keywords contains Chinese characters.")
    return unique


def parse_model_output(target_id: str, text: str) -> Dict[str, str]:
    text = strip_thinking(text)
    result = {
        "Target_id": target_id,
        "Merchant": "",
        "Merchant_description": "",
        "Memory_keywords": "",
    }
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Merchant:"):
            result["Merchant"] = line.split(":", 1)[1].strip()
        elif line.startswith("Merchant_description:"):
            result["Merchant_description"] = line.split(":", 1)[1].strip()
        elif line.startswith("Memory_keywords:"):
            result["Memory_keywords"] = line.split(":", 1)[1].strip()
    if not result["Merchant"] or not result["Merchant_description"] or not result["Memory_keywords"]:
        raise ValueError("Model response must contain Merchant, Merchant_description, and Memory_keywords.")
    result["Memory_keywords"] = "; ".join(normalize_memory_keywords(result["Memory_keywords"]))
    if CJK_PATTERN.search(result["Merchant"]) or CJK_PATTERN.search(result["Merchant_description"]):
        raise ValueError("Model response contains Chinese characters.")
    return result


def checkpoint_concept_memory(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    memory: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                merchant = str(record["Merchant"]).strip()
                keywords = normalize_memory_keywords(record["Memory_keywords"])
                if merchant and not CJK_PATTERN.search(merchant):
                    memory[str(record["Target_id"])] = merchant + ": " + "; ".join(keywords)
            except (KeyError, ValueError):
                continue
    return memory


def build_prompt_with_memory(row: Mapping[str, object], memory: list[str]) -> str:
    enriched_row = dict(row)
    if memory:
        enriched_row["existing_merchant_concepts"] = "\n".join("- " + concept for concept in memory)
    return build_target_description_prompt(enriched_row)


def generate_target_record(
    client: Any,
    row: Mapping[str, object],
    memory: list[str],
    args: argparse.Namespace,
) -> Dict[str, str]:
    target_id = str(row["Target_id"])
    prompt = build_prompt_with_memory(row, memory)
    last_error: Optional[Exception] = None
    last_response = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = generate_english_model_output(
                client=client,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                presence_penalty=args.presence_penalty,
                max_tokens=args.max_tokens,
                max_retries=args.max_retries,
            )
            last_response = response
            return parse_model_output(target_id, response)
        except ValueError as exc:
            last_error = exc
            if attempt < args.max_retries:
                prompt += "\n\nYour previous response missed or malformed the required Memory_keywords line. Regenerate all three lines exactly as specified."
    debug_path = Path(args.checkpoint).parent / "target_long_text_invalid_response.txt"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(last_response, encoding="utf-8")
    raise RuntimeError(f"Target {target_id} generation failed after {args.max_retries + 1} attempts: {last_error}")


def validate_input(df: pd.DataFrame, input_path: Path) -> None:
    required_columns = {"Target_id", "behavior_level1", "behavior_level2", "basic_profile"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {', '.join(sorted(missing))}")


def word_count(text: str) -> int:
    """Count English word tokens, treating hyphenated words as one token."""
    return len(WORD_PATTERN.findall(text))


def target_id_sort_key(target_id: object) -> tuple[int, str]:
    """Sort conventional IDs such as T1039 by their numeric suffix."""
    value = str(target_id)
    suffix = value[1:] if value.startswith("T") else value
    return (int(suffix), value) if suffix.isdigit() else (float("inf"), value)


def replace_target_records(output_path: Path, records: list[Mapping[str, str]]) -> None:
    """Replace only regenerated merchant records and preserve every other output row."""
    if not records:
        return
    existing = read_csv(output_path)
    required_columns = ["Target_id", "Merchant", "Merchant_description"]
    missing = set(required_columns) - set(existing.columns)
    if missing:
        raise ValueError(f"{output_path} is missing required columns: {', '.join(sorted(missing))}")
    existing = existing[required_columns]
    existing["Target_id"] = existing["Target_id"].astype(str)

    replacements = pd.DataFrame(records)[required_columns]
    replacements["Target_id"] = replacements["Target_id"].astype(str)
    if replacements["Target_id"].duplicated().any():
        raise ValueError("Regeneration produced duplicate Target_id values.")
    missing_ids = sorted(set(replacements["Target_id"]) - set(existing["Target_id"]))
    if missing_ids:
        raise ValueError(
            "Cannot replace records because Target_id values are missing from the output: " + ", ".join(missing_ids)
        )

    replacements = replacements.set_index("Target_id")
    for column in ["Merchant", "Merchant_description"]:
        existing[column] = existing["Target_id"].map(replacements[column]).fillna(existing[column])
    existing["_target_id_sort_key"] = existing["Target_id"].map(target_id_sort_key)
    existing = existing.sort_values("_target_id_sort_key", kind="stable").drop(columns="_target_id_sort_key")
    existing.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Replaced {len(replacements)} merchant records in {output_path}; total rows: {len(existing)}", flush=True)


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

    df = read_profile_input(args.input)
    validate_input(df, args.input)
    df["Target_id"] = df["Target_id"].astype(str)

    targeted_regeneration = args.regenerate_under_min_words is not None
    if targeted_regeneration:
        if args.regenerate_under_min_words < 1:
            raise ValueError("--regenerate-under-min-words must be at least 1.")
        if args.limit is not None:
            raise ValueError("--limit cannot be combined with --regenerate-under-min-words.")
        if args.resume:
            raise ValueError("--resume cannot be combined with --regenerate-under-min-words.")
        if not args.output.exists():
            raise FileNotFoundError(f"Merchant long-text file does not exist: {args.output}")
        existing = read_csv(args.output)
        required_columns = {"Target_id", "Merchant_description"}
        if missing := required_columns - set(existing.columns):
            raise ValueError(f"{args.output} is missing required columns: {', '.join(sorted(missing))}")
        target_ids = set(
            existing.loc[
                existing["Merchant_description"].fillna("").astype(str).map(word_count)
                < args.regenerate_under_min_words,
                "Target_id",
            ].astype(str)
        )
        unknown_ids = sorted(target_ids - set(df["Target_id"]))
        if unknown_ids:
            raise ValueError(f"Target_id values not found in {args.input}: {', '.join(unknown_ids)}")
        df = df[df["Target_id"].isin(target_ids)].copy()

    if not args.resume and not targeted_regeneration:
        print(f"Resetting output files: {args.output} and {args.checkpoint}", flush=True)
        reset_output_files(args.output, args.checkpoint)
    elif args.resume:
        restore_output_from_checkpoint(args.checkpoint, args.output)

    done = load_existing_ids(args.output, "Target_id") if args.resume else set()
    concept_memory = checkpoint_concept_memory(args.checkpoint) if (args.resume or targeted_regeneration) else {}
    ordered = df.sort_values(["behavior_level1", "behavior_level2", "Target_id"], kind="stable")
    pending_count = int((~ordered["Target_id"].astype(str).isin(done)).sum())

    print(f"Loaded {len(df)} merchant base profiles from {args.input}", flush=True)
    if targeted_regeneration:
        print(
            f"Regenerating merchant descriptions with fewer than {args.regenerate_under_min_words} English words.",
            flush=True,
        )
    print(f"Pending merchant descriptions: {pending_count}", flush=True)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    generated = 0
    regenerated_records: list[Mapping[str, str]] = []
    group_count = ordered.groupby(["behavior_level1", "behavior_level2"], sort=True).ngroups
    for group_index, ((level1, level2), group) in enumerate(
        ordered.groupby(["behavior_level1", "behavior_level2"], sort=True), 1
    ):
        rows = group.to_dict(orient="records")
        memory = [concept_memory[str(row["Target_id"])] for row in rows if str(row["Target_id"]) in concept_memory]
        pending_rows = [row for row in rows if str(row["Target_id"]) not in done]
        if not pending_rows:
            continue
        print(f"[MCC level 2] {group_index}/{group_count}: {level1} / {level2}", flush=True)
        for row in pending_rows:
            if args.limit is not None and generated >= args.limit:
                return
            target_id = str(row["Target_id"])
            generated += 1
            print(f"[Target] {generated}/{min(pending_count, args.limit or pending_count)}: generating {target_id}", flush=True)
            record = generate_target_record(client, row, memory, args)
            keywords = normalize_memory_keywords(record["Memory_keywords"])
            concept = record["Merchant"] + ": " + "; ".join(keywords)
            memory.append(concept)
            concept_memory[target_id] = concept
            if targeted_regeneration:
                regenerated_records.append(record)
            else:
                append_jsonl(args.checkpoint, record)
                append_csv(args.output, record)
            if args.sleep_seconds:
                time.sleep(args.sleep_seconds)

    if targeted_regeneration:
        replace_target_records(args.output, regenerated_records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate merchant descriptions from target_basic_profile.jsonl.")
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
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of Target descriptions to generate.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output instead of regenerating from the first record.")
    parser.add_argument(
        "--regenerate-under-min-words",
        type=int,
        metavar="WORD_COUNT",
        help=(
            "Regenerate every merchant description in target_long_text.csv with fewer than WORD_COUNT English words. "
            "Only selected merchant records are replaced; the file remains sorted by Target_id."
        ),
    )
    return parser.parse_args()


def main() -> None:
    generate_descriptions(parse_args())


if __name__ == "__main__":
    main()
