#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module 4: rule-based base profiles and LLM-enriched long-text profiles."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from initialization.merchant_initialization import BEHAVIOR_HIERARCHY

WORK_DIR = REPO_ROOT / "work"
DEFAULT_SOURCE_INPUT = WORK_DIR / "source_optimized.csv"
DEFAULT_TARGET_INPUT = WORK_DIR / "target_final.csv"
DEFAULT_TRANSACTION_INPUT = WORK_DIR / "transaction_optimized.csv"
DEFAULT_TARGET_BASIC_JSON_OUTPUT = WORK_DIR / "target_basic_profile.jsonl"
DEFAULT_MULTI_TRANSACTION_SOURCE_JSON_OUTPUT = WORK_DIR / "multi_transaction_source.jsonl"
DEFAULT_SINGLE_TRANSACTION_SOURCE_JSON_OUTPUT = WORK_DIR / "single_transaction_source.jsonl"

# A transaction outside the user's province is classified as online only when
# its merchant category has one of these explicit online/digital semantics.
ONLINE_SEMANTIC_KEYWORDS = (
    "online", "internet", "software", "gaming", "data services", "information search",
    "live streaming", "social networking", "telecom payment", "digital",
)

CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

CONSUMPTION_CATEGORY_BY_BEHAVIOR_LEVEL1 = {
    level1: category
    for level1, category in {
        "Food, Tobacco and Liquor Retail": "Food, Tobacco and Liquor", "Food and Beverage Services": "Food, Tobacco and Liquor", "Apparel and General Shopping": "Clothing and Footwear",
        "Housing Payments and Home Improvement Services": "Housing", "Household and Personal Goods Retail": "Household Equipment, Furnishings and Services",
        "Local Lifestyle and Self-Service Sharing": "Household Equipment, Furnishings and Services", "Local Commuting and On-Demand Travel": "Transport and Communications",
        "Long-Distance Travel Transport": "Transport and Communications", "Vehicle Energy and Automotive Services": "Transport and Communications",
        "Telecommunications, Logistics and Digital Infrastructure Services": "Transport and Communications", "Education Payments and Training Services": "Education, Culture and Recreation",
        "Offline Culture, Sports and Entertainment": "Education, Culture and Recreation", "Online Entertainment and Hobby Spending": "Education, Culture and Recreation",
        "Hotels, Scenic Attractions and Tourism Services": "Education, Culture and Recreation", "Pharmaceutical, Medical Device and Health Retail": "Health Care and Medical Services",
        "General Clinical Care and Health Management": "Health Care and Medical Services", "Specialized Consumer Medical and Care Services": "Health Care and Medical Services",
        "High-Value Goods and Professional Services": "Miscellaneous Goods and Services",
    }.items()
}


def fine_grained_candidates(behavior_level1: str, behavior_level2: str) -> List[str]:
    for level1_map in BEHAVIOR_HIERARCHY.values():
        level2_map = level1_map.get(behavior_level1)
        if level2_map and behavior_level2 in level2_map:
            return list(level2_map[behavior_level2])
    return [behavior_level2]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]
    return df


def read_csv(path: Path) -> pd.DataFrame:
    return clean_columns(pd.read_csv(path))


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes"})


def pct(value: float) -> str:
    return f"{value:.1%}"


def top_distribution(series: pd.Series, limit: int = 3) -> str:
    values = series.dropna().astype(str)
    values = values[~values.isin(["", "nan", "None"])]
    if values.empty:
        return "No valid observations"
    names = values.value_counts().head(limit).index.tolist()
    return "; ".join(str(name) for name in names)


def label_amount(mean: float, low: float, high: float) -> str:
    if mean <= low:
        return "low amount"
    if mean >= high:
        return "high amount"
    return "medium amount"


def label_volatility(cv: float) -> str:
    return "stable amount" if cv <= 0.50 else "variable amount"


def label_frequency(count: int, date_count: int) -> str:
    daily_average = count / max(date_count, 1)
    if daily_average >= 1:
        return "high daily frequency"
    if count >= 4:
        return "high weekly frequency"
    return "low frequency"


def label_intraday(group: pd.DataFrame) -> str:
    shares = group["period_tag"].astype(str).value_counts(normalize=True)
    def share(*names: str) -> float:
        return float(sum(shares.get(name, 0.0) for name in names))

    patterns = {
        "concentrated around commuting periods": share("commute_morning", "commute_evening"),
        "concentrated around lunch and evening": share("lunch", "night"),
        "concentrated at night": share("night", "late_night"),
        "concentrated on workday daytime": share("workday_morning", "afternoon"),
    }
    best_name, best_share = max(patterns.items(), key=lambda item: item[1])
    if best_share >= 0.35:
        return best_name
    if not shares.empty and float(shares.iloc[0]) <= 0.35:
        return "distributed across daily periods"
    return "no clear intraday pattern"


def label_week(group: pd.DataFrame) -> str:
    weekend_ratio = as_bool(group["is_weekend"]).mean()
    if weekend_ratio >= 0.40:
        return "weekend-oriented"
    if weekend_ratio <= 0.20:
        return "weekday-oriented"
    return "balanced across the week"


def label_month(group: pd.DataFrame) -> str:
    start_ratio = as_bool(group["is_month_start"]).mean()
    end_ratio = as_bool(group["is_month_end"]).mean()
    bill_ratio = as_bool(group["is_common_bill_day"]).mean()
    labels: List[str] = []
    if start_ratio >= 0.18:
        labels.append("early-month pattern")
    if end_ratio >= 0.18:
        labels.append("late-month pattern")
    if bill_ratio >= 0.18 and not labels:
        labels.append("monthly-cycle pattern")
    return "; ".join(labels) if labels else "weak monthly pattern"


def enrich_transactions(source_df: pd.DataFrame, target_df: pd.DataFrame, txn_df: pd.DataFrame) -> pd.DataFrame:
    txn = txn_df.copy()
    target_columns = ["Target", "behavior_level1", "behavior_level2"]
    txn = txn.merge(target_df[target_columns], on="Target", how="left")
    txn = txn.merge(source_df[["Source", "province"]], on="Source", how="left")
    category_text = (
        txn["behavior_level1"].fillna("").astype(str)
        + " "
        + txn["behavior_level2"].fillna("").astype(str)
    ).str.casefold()
    txn["online_semantic"] = category_text.apply(
        lambda text: any(keyword.casefold() in text for keyword in ONLINE_SEMANTIC_KEYWORDS)
    )
    txn["out_province"] = as_bool(txn["is_out_province_transaction"])
    txn["online_tendency_transaction"] = txn["out_province"] & txn["online_semantic"]
    txn["remote_tendency_transaction"] = txn["out_province"] & ~txn["online_semantic"]
    txn["Amount"] = pd.to_numeric(txn["Amount"], errors="coerce").fillna(0.0)
    return txn


def transaction_stats(group: pd.DataFrame, amount_low: float, amount_high: float) -> Dict[str, object]:
    amount_mean = float(group["Amount"].mean())
    amount_std = float(group["Amount"].std(ddof=0))
    amount_cv = amount_std / amount_mean if amount_mean > 0 else 0.0
    return {
        "transaction_count": int(len(group)),
        "avg_amount": round(amount_mean, 2),
        "amount_cv": round(amount_cv, 4),
        "amount_preference": label_amount(amount_mean, amount_low, amount_high),
        "amount_volatility": label_volatility(amount_cv),
        "frequency_label": label_frequency(len(group), group["date"].nunique()),
        "intraday_pattern": label_intraday(group),
        "week_pattern": label_week(group),
        "month_pattern": label_month(group),
        "online_tendency_ratio": float(group["online_tendency_transaction"].mean()),
        "remote_tendency_ratio": float(group["remote_tendency_transaction"].mean()),
    }


def tendency_text(online_ratio: float, remote_ratio: float) -> str:
    if online_ratio >= 0.20 and online_ratio >= remote_ratio:
        return "online-oriented"
    if remote_ratio >= 0.20:
        return "remote-oriented"
    return "primarily local and offline"


def category_metrics(group: pd.DataFrame) -> Dict[str, object]:
    distribution = group["behavior_level2"].dropna().astype(str).value_counts(normalize=True)
    hhi = float((distribution ** 2).sum()) if not distribution.empty else 0.0
    if hhi >= 0.50:
        concentration = "high category concentration"
    elif hhi >= 0.25:
        concentration = "moderate category concentration"
    else:
        concentration = "dispersed categories"
    diversity = "high category diversity" if len(distribution) >= 5 else "moderate category diversity" if len(distribution) >= 3 else "low category diversity"
    return {
        "preferred_categories": top_distribution(group["behavior_level2"]),
        "category_concentration": concentration,
        "category_diversity": diversity,
        "category_hhi": round(hhi, 4),
    }


def print_profile_progress(entity_name: str, current: int, total: int) -> None:
    step = max(total // 20, 1)
    if current == 1 or current == total or current % step == 0:
        print(f"[{entity_name}] {current}/{total}", flush=True)


def build_user_profiles(source_df: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    transaction_counts = txn.groupby("Source", sort=False).size()
    multi_transaction_sources = transaction_counts[transaction_counts.gt(1)].index
    profile_txn = txn[txn["Source"].isin(multi_transaction_sources)].copy()
    profile_source_df = source_df[source_df["Source"].isin(multi_transaction_sources)].copy()
    print(f"Building user base profiles for {len(profile_source_df)} multi-transaction users...", flush=True)
    if profile_txn.empty:
        return profile_source_df.assign(transaction_count=pd.Series(dtype=int))

    amount_low, amount_high = txn["Amount"].quantile([0.33, 0.67]).tolist()
    group = profile_txn.groupby("Source", sort=False)
    print("[users] aggregating transaction, amount, and channel statistics...", flush=True)
    stats = group.agg(
        transaction_count=("Amount", "size"),
        avg_amount=("Amount", "mean"),
        active_date_count=("date", "nunique"),
        online_tendency_ratio=("online_tendency_transaction", "mean"),
        remote_tendency_ratio=("remote_tendency_transaction", "mean"),
    )
    stats["amount_cv"] = group["Amount"].std(ddof=0).div(stats["avg_amount"].replace(0, np.nan)).fillna(0.0)
    stats["amount_preference"] = np.select(
        [stats["avg_amount"] <= amount_low, stats["avg_amount"] >= amount_high],
        ["low amount", "high amount"],
        default="medium amount",
    )
    stats["amount_volatility"] = np.where(stats["amount_cv"] <= 0.50, "stable amount", "variable amount")
    daily_average = stats["transaction_count"].div(stats["active_date_count"].clip(lower=1))
    stats["frequency_label"] = np.select(
        [daily_average >= 1, stats["transaction_count"] >= 4],
        ["high daily frequency", "high weekly frequency"],
        default="low frequency",
    )

    period_tags = ["late_night", "commute_morning", "workday_morning", "lunch", "afternoon", "commute_evening", "night"]
    print("[users] aggregating intraday, weekly, and monthly patterns...", flush=True)
    period = pd.crosstab(profile_txn["Source"], profile_txn["period_tag"], normalize="index").reindex(columns=period_tags, fill_value=0.0)
    period_scores = pd.DataFrame(
        {
            "concentrated around commuting periods": period["commute_morning"] + period["commute_evening"],
            "concentrated around lunch and evening": period["lunch"] + period["night"],
            "concentrated at night": period["night"] + period["late_night"],
            "concentrated on workday daytime": period["workday_morning"] + period["afternoon"],
        }
    )
    best_period = period_scores.idxmax(axis=1)
    best_score = period_scores.max(axis=1)
    stats["intraday_pattern"] = np.where(
        best_score >= 0.35,
        best_period,
        np.where(period.max(axis=1) <= 0.35, "distributed across daily periods", "no clear intraday pattern"),
    )

    weekend_ratio = as_bool(profile_txn["is_weekend"]).groupby(profile_txn["Source"]).mean()
    stats["week_pattern"] = np.select(
        [weekend_ratio >= 0.40, weekend_ratio <= 0.20],
        ["weekend-oriented", "weekday-oriented"],
        default="balanced across the week",
    )
    start_ratio = as_bool(profile_txn["is_month_start"]).groupby(profile_txn["Source"]).mean()
    end_ratio = as_bool(profile_txn["is_month_end"]).groupby(profile_txn["Source"]).mean()
    bill_ratio = as_bool(profile_txn["is_common_bill_day"]).groupby(profile_txn["Source"]).mean()
    stats["month_pattern"] = np.select(
        [start_ratio.ge(0.18) & end_ratio.ge(0.18), start_ratio.ge(0.18), end_ratio.ge(0.18), bill_ratio.ge(0.18)],
        ["early-month pattern; late-month pattern", "early-month pattern", "late-month pattern", "monthly-cycle pattern"],
        default="weak monthly pattern",
    )

    print("[users] aggregating category preference and concentration...", flush=True)
    category_count = profile_txn.dropna(subset=["behavior_level2"]).groupby(["Source", "behavior_level2"], sort=False).size().rename("count").reset_index()
    category_count["ratio"] = category_count["count"].div(category_count.groupby("Source")["count"].transform("sum"))
    category_hhi = category_count.assign(square=lambda frame: frame["ratio"] ** 2).groupby("Source")["square"].sum()
    category_size = category_count.groupby("Source").size()
    top_category = category_count.sort_values(["Source", "count", "behavior_level2"], ascending=[True, False, True]).groupby("Source", sort=False).head(3).copy()
    top_category["label"] = top_category["behavior_level2"].astype(str)
    category_text = top_category.groupby("Source", sort=False)["label"].agg("; ".join)
    stats["preferred_categories"] = category_text
    stats["category_hhi"] = category_hhi
    stats["category_concentration"] = np.select(
        [stats["category_hhi"] >= 0.50, stats["category_hhi"] >= 0.25],
        ["high category concentration", "moderate category concentration"],
        default="dispersed categories",
    )
    stats["category_diversity"] = np.select(
        [category_size >= 5, category_size >= 3],
        ["high category diversity", "moderate category diversity"],
        default="low category diversity",
    )

    print("[users] constructing base-profile text...", flush=True)
    profiles = profile_source_df.merge(stats.reset_index(), on="Source", how="left")
    profiles["transaction_count"] = profiles["transaction_count"].fillna(0).astype(int)
    for column in ["avg_amount", "amount_cv", "online_tendency_ratio", "remote_tendency_ratio", "category_hhi"]:
        profiles[column] = profiles[column].fillna(0.0)
    profiles["preferred_categories"] = profiles["preferred_categories"].fillna("No valid observations")
    for column in ["amount_preference", "amount_volatility", "frequency_label", "intraday_pattern", "week_pattern", "month_pattern", "category_concentration", "category_diversity"]:
        profiles[column] = profiles[column].fillna("No transactions" if column not in {"frequency_label", "category_concentration", "category_diversity"} else "No valid observations")
    tendency = np.select(
        [profiles["online_tendency_ratio"].ge(0.20) & profiles["online_tendency_ratio"].ge(profiles["remote_tendency_ratio"]), profiles["remote_tendency_ratio"].ge(0.20)],
        ["online-oriented", "remote-oriented"],
        default="primarily local and offline",
    )
    profiles["basic_profile"] = (
        "Province: " + profiles["province"].fillna("Unknown").astype(str)
        + "; Gender: " + profiles["gender"].fillna("Unknown").astype(str)
        + "; Age group: " + profiles["age_group"].fillna("Unknown").astype(str)
        + "; Education: " + profiles["education"].fillna("Unknown").astype(str)
        + "; Occupation: " + profiles["occupation_industry"].fillna("Unknown").astype(str)
        + "; Spending level: " + profiles["amount_preference"].astype(str)
        + "; Amount pattern: " + profiles["amount_volatility"].astype(str)
        + "; Frequency: " + profiles["frequency_label"].astype(str)
        + "; Intraday pattern: " + profiles["intraday_pattern"].astype(str)
        + "; Weekly pattern: " + profiles["week_pattern"].astype(str)
        + "; Monthly pattern: " + profiles["month_pattern"].astype(str)
        + "; Channel tendency: " + pd.Series(tendency, index=profiles.index).astype(str)
        + "; Preferred categories: " + profiles["preferred_categories"].astype(str)
        + "."
    )
    print(f"[users] {len(profiles)}/{len(profiles)}", flush=True)
    return profiles.drop(columns=["active_date_count"])


def user_profile_record(profile: Mapping[str, object]) -> Dict[str, object]:
    preferred_categories = [
        category
        for category in str(profile.get("preferred_categories", "")).split(";")
        if category and category != "No valid observations"
    ]
    return {
        "Source_id": str(profile["Source"]),
        "user_attributes": {
            "Province": str(profile.get("province", "Unknown")),
            "Gender": str(profile.get("gender", "Unknown")),
            "Age": str(profile.get("age_group", "Unknown")),
            "Education": str(profile.get("education", "Unknown")),
            "Occupation": str(profile.get("occupation_industry", "Unknown")),
        },
        "transaction_profile": {
            "amount_level": str(profile.get("amount_preference", "No transactions")),
            "amount_volatility": str(profile.get("amount_volatility", "No transactions")),
            "transaction_frequency": str(profile.get("frequency_label", "No valid observations")),
            "intraday_pattern": str(profile.get("intraday_pattern", "No transactions")),
            "weekly_pattern": str(profile.get("week_pattern", "No transactions")),
            "monthly_pattern": str(profile.get("month_pattern", "No transactions")),
            "channel_tendency": tendency_text(
                float(profile.get("online_tendency_ratio", 0.0)),
                float(profile.get("remote_tendency_ratio", 0.0)),
            ),
        },
        "spending_preferences": {
            "preferred_categories": preferred_categories,
        },
    }


def json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_single_transaction_user_context(
    path: Path,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
) -> int:
    transaction_counts = transaction_df.groupby("Source", sort=False).size()
    single_transaction_sources = set(transaction_counts[transaction_counts.eq(1)].index.astype(str))
    single_transactions = transaction_df[transaction_df["Source"].astype(str).isin(single_transaction_sources)]
    source_lookup = source_df.set_index("Source").to_dict(orient="index")
    target_lookup = target_df.set_index("Target").to_dict(orient="index")

    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(single_transactions)
    print(f"Writing raw context for {total} single-transaction users...", flush=True)
    with path.open("w", encoding="utf-8") as file:
        columns = single_transactions.columns.tolist()
        for index, values in enumerate(single_transactions.itertuples(index=False, name=None), start=1):
            transaction = dict(zip(columns, values))
            source_id = str(transaction["Source"])
            target_id = str(transaction["Target"])
            record = {
                "Source_id": source_id,
                "raw_source_fields": json_safe(source_lookup.get(source_id, {})),
                "raw_transaction_fields": json_safe(transaction),
                "raw_target_fields": json_safe(target_lookup.get(target_id, {})),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print_profile_progress("single-transaction users", index, total)
    return total


def build_merchant_profiles(target_df: pd.DataFrame, source_df: pd.DataFrame, txn: pd.DataFrame) -> pd.DataFrame:
    amount_low, amount_high = txn["Amount"].quantile([0.33, 0.67]).tolist()
    source_attributes = source_df.set_index("Source")
    rows: List[Dict[str, object]] = []
    merchant_rows = target_df.to_dict(orient="records")
    print(f"Building merchant base profiles for {len(merchant_rows)} merchants...", flush=True)
    groups = {key: group for key, group in txn.groupby("Target", sort=False)}
    for index, merchant in enumerate(merchant_rows, start=1):
        target_id = merchant["Target"]
        group = groups.get(target_id, txn.iloc[0:0])
        behavior_level1 = str(merchant.get("behavior_level1", "Unknown"))
        behavior_level2 = str(merchant.get("behavior_level2", "Unknown"))
        consumption_category = CONSUMPTION_CATEGORY_BY_BEHAVIOR_LEVEL1.get(
            behavior_level1, "Miscellaneous Goods and Services"
        )
        candidates = fine_grained_candidates(behavior_level1, behavior_level2)
        if group.empty:
            stats = {"transaction_count": 0, "amount_preference": "No transactions", "amount_volatility": "No transactions", "frequency_label": "low frequency", "intraday_pattern": "No transactions", "week_pattern": "No transactions", "month_pattern": "No transactions", "online_tendency_ratio": 0.0, "remote_tendency_ratio": 0.0, "avg_amount": 0.0, "amount_cv": 0.0}
            customer = {key: "No valid observations" for key in ["typical_gender", "typical_age_group", "typical_education", "typical_occupation"]}
        else:
            stats = transaction_stats(group, amount_low, amount_high)
            linked_sources = source_attributes.reindex(group["Source"].unique())
            customer = {
                "typical_gender": top_distribution(linked_sources["gender"], limit=1),
                "typical_age_group": top_distribution(linked_sources["age_group"], limit=1),
                "typical_education": top_distribution(linked_sources["education"], limit=1),
                "typical_occupation": top_distribution(linked_sources["occupation_industry"]),
            }
        profile = (
            f"Consumption category: {consumption_category}; MCC level 1: {behavior_level1}; MCC level 2: {behavior_level2}; "
            f"Amount pattern: {stats['amount_preference']} and {stats['amount_volatility']}; "
            f"Transaction frequency: {stats['frequency_label']}; intraday pattern: {stats['intraday_pattern']}; "
            f"weekly pattern: {stats['week_pattern']}; monthly pattern: {stats['month_pattern']}; "
            f"channel tendency: {tendency_text(float(stats['online_tendency_ratio']), float(stats['remote_tendency_ratio']))}."
        )
        profile_json = json.dumps(
            {
                "Target_id": target_id,
                "merchant_category": {
                    "Consumption_category": consumption_category,
                    "MCC_level1": behavior_level1,
                    "MCC_level2": behavior_level2,
                    "fine_grained_candidates": candidates,
                },
                "transaction_profile": {
                    "amount_level": stats["amount_preference"],
                    "amount_volatility": stats["amount_volatility"],
                    "transaction_frequency": stats["frequency_label"],
                    "intraday_pattern": stats["intraday_pattern"],
                    "weekly_pattern": stats["week_pattern"],
                    "monthly_pattern": stats["month_pattern"],
                    "channel_tendency": tendency_text(float(stats["online_tendency_ratio"]), float(stats["remote_tendency_ratio"])),
                },
                "associated_user_profile": {
                    "dominant_gender": customer["typical_gender"],
                    "dominant_age_group": customer["typical_age_group"],
                    "dominant_education": customer["typical_education"],
                    "occupation_top3": customer["typical_occupation"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        rows.append({**merchant, **stats, **customer, "consumption_category": consumption_category, "basic_profile": profile, "basic_profile_json": profile_json})
        print_profile_progress("merchants", index, len(merchant_rows))
    return pd.DataFrame(rows)


def append_jsonl(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate merchant and user JSONL inputs for downstream text generation.")
    parser.add_argument("--source-input", type=Path, default=DEFAULT_SOURCE_INPUT)
    parser.add_argument("--target-input", type=Path, default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--transaction-input", type=Path, default=DEFAULT_TRANSACTION_INPUT)
    parser.add_argument("--target-basic-json-output", type=Path, default=DEFAULT_TARGET_BASIC_JSON_OUTPUT)
    parser.add_argument(
        "--source-multi-transaction-basic-json-output",
        type=Path,
        default=DEFAULT_MULTI_TRANSACTION_SOURCE_JSON_OUTPUT,
    )
    parser.add_argument(
        "--source-single-transaction-context-json-output",
        type=Path,
        default=DEFAULT_SINGLE_TRANSACTION_SOURCE_JSON_OUTPUT,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading source, final target, and optimized transaction tables...", flush=True)
    source_df, target_df, txn_df = read_csv(args.source_input), read_csv(args.target_input), read_csv(args.transaction_input)
    print(f"Loaded {len(source_df)} users, {len(target_df)} merchants, and {len(txn_df)} transactions.", flush=True)
    print("Joining user, merchant, and transaction attributes...", flush=True)
    txn = enrich_transactions(source_df, target_df, txn_df)
    user_profiles = build_user_profiles(source_df, txn)
    multi_transaction_user_profiles = user_profiles[user_profiles["transaction_count"].gt(1)]
    print("Writing multi-transaction user base-profile JSONL...", flush=True)
    write_jsonl(
        args.source_multi_transaction_basic_json_output,
        (
            user_profile_record(row)
            for row in multi_transaction_user_profiles.to_dict(orient="records")
        ),
    )
    print(
        "Multi-transaction user base profiles written: "
        f"{args.source_multi_transaction_basic_json_output} ({len(multi_transaction_user_profiles)} users)",
        flush=True,
    )
    single_transaction_count = write_single_transaction_user_context(
        args.source_single_transaction_context_json_output,
        source_df,
        target_df,
        txn_df,
    )
    print(
        "Single-transaction user raw contexts written: "
        f"{args.source_single_transaction_context_json_output} ({single_transaction_count} users)",
        flush=True,
    )
    merchant_profiles = build_merchant_profiles(target_df, source_df, txn)
    print("Writing merchant base-profile JSONL...", flush=True)
    write_jsonl(
        args.target_basic_json_output,
        [
            json.loads(row["basic_profile_json"])
            for row in merchant_profiles[["basic_profile_json"]].to_dict(orient="records")
        ],
    )
    print(f"Merchant base profiles written: {args.target_basic_json_output}", flush=True)


if __name__ == "__main__":
    main()
