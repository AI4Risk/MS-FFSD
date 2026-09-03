#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create template-based long-text descriptions for single-transaction users."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
DEFAULT_INPUT = WORK_DIR / "single_transaction_source.jsonl"
DEFAULT_OUTPUT = WORK_DIR / "source_single_transaction_long_text.csv"

PERIOD_DESCRIPTIONS = {
    "late_night": "late at night",
    "commute_morning": "during the morning commute",
    "workday_morning": "during the morning",
    "lunch": "around lunchtime",
    "afternoon": "during the afternoon",
    "commute_evening": "during the evening commute",
    "night": "at night",
}
DAY_NAMES = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}
AGE_DESCRIPTIONS = {
    "Young Adults": "young adult",
    "Middle-aged Adults": "middle-aged",
    "Older Adults": "older adult",
}
EDUCATION_DESCRIPTIONS = {
    "Primary School": "a primary school education",
    "Junior Secondary School": "a junior secondary school education",
    "Senior Secondary School": "a senior secondary school education",
    "Junior College and Above": "a junior college education or higher",
}


def text_value(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if value is None:
        return "Unknown"
    value = str(value).strip()
    return value if value and value.lower() != "nan" else "Unknown"


def as_bool(value: object) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def locality_description(transaction: Mapping[str, object]) -> str:
    return "out-of-province" if as_bool(transaction.get("is_out_province_transaction")) else "local"


def indefinite_article(phrase: str) -> str:
    return "an" if phrase.lower().startswith(("a", "e", "i", "o", "u")) else "a"


def period_description(transaction: Mapping[str, object]) -> str:
    """Return a natural time phrase, or no phrase when no valid tag is available."""
    return PERIOD_DESCRIPTIONS.get(text_value(transaction, "period_tag"), "")


def normalized_age_gender(source: Mapping[str, object]) -> str:
    age_group = AGE_DESCRIPTIONS.get(text_value(source, "age_group"), text_value(source, "age_group").lower())
    gender = text_value(source, "gender").lower()
    phrase = f"{age_group} {gender}"
    return f"{indefinite_article(phrase)} {phrase}"


def normalized_education(source: Mapping[str, object]) -> str:
    education = text_value(source, "education")
    return EDUCATION_DESCRIPTIONS.get(education, f"an education in {education}")


def calendar_context(transaction: Mapping[str, object]) -> str:
    day_of_week = transaction.get("dayofweek")
    try:
        day = DAY_NAMES.get(int(day_of_week))
    except (TypeError, ValueError):
        day = None
    parts = [f"on {day}" if day else ("on a weekend" if as_bool(transaction.get("is_weekend")) else "on a weekday")]
    if as_bool(transaction.get("is_month_start")):
        parts.append("at the start of the month")
    elif as_bool(transaction.get("is_month_middle")):
        parts.append("in the middle of the month")
    elif as_bool(transaction.get("is_month_end")):
        parts.append("at the end of the month")
    if as_bool(transaction.get("is_common_bill_day")):
        parts.append("on a common bill-payment day")
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return " and ".join(parts)
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def build_description(record: Mapping[str, object]) -> str:
    source = record["raw_source_fields"]
    transaction = record["raw_transaction_fields"]
    target = record["raw_target_fields"]
    if not all(isinstance(value, Mapping) for value in (source, transaction, target)):
        raise ValueError("Each record must contain raw_source_fields, raw_transaction_fields, and raw_target_fields objects.")
    period = period_description(transaction)
    transaction_time = f" {period}" if period else ""
    locality = locality_description(transaction)
    return (
        f"The user is {normalized_age_gender(source)} from "
        f"{text_value(source, 'province')}, has {normalized_education(source)}, and works in "
        f"{text_value(source, 'occupation_industry')}. "
        f"The user made {indefinite_article(locality)} {locality} transaction{transaction_time} "
        f"with a merchant associated with {text_value(target, 'behavior_level2')}. "
        f"The transaction occurred {calendar_context(transaction)}."
    )


def generate_descriptions(input_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8") as source_file, output_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["Source_id", "Source_description"])
        writer.writeheader()
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_id = text_value(record, "Source_id")
            if source_id == "Unknown":
                raise ValueError(f"Line {line_number} has no Source_id.")
            writer.writerow({"Source_id": source_id, "Source_description": build_description(record)})
            count += 1
    print(f"Wrote {count} template-based single-transaction user descriptions to {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate template-based descriptions for single-transaction users.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_descriptions(args.input, args.output)
