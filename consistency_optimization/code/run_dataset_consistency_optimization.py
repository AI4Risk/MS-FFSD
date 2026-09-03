#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run alternating consistency optimization on work/source.csv,
work/target.csv, and work/transaction.csv.

The optimizer itself works on an enriched transaction-level table. This runner
keeps the public input/output contract as three tables:
- source.csv: user semantic attributes
- target.csv: merchant semantic attributes
- transaction.csv: immutable transactions
"""

from __future__ import print_function

import argparse
import os

import numpy as np
import pandas as pd

from analyze_consistency_outputs import build_merchant_profile_after
from consistency_alternating_optimizer import (
    ConsistencyOptimizerConfig,
    explain_stop_reason,
    optimize_consistency,
    quantile_normalize,
    safe_div,
    write_json,
)
from manual_merchant_priors import build_manual_prior_dict, build_manual_prior_table


SOURCE_COL = "Source"
TARGET_COL = "Target"
AMOUNT_COL = "Amount"
SOURCE_PROVINCE_COL = "province"
TRANSACTION_PROVINCE_COL = "transaction_province"
OUT_PROVINCE_COL = "is_out_province_transaction"
CATEGORY_COL = "behavior_level1"
MCC_COL = "behavior_level2"
GENDER_COL = "gender"
AGE_COL = "age_group"
EDUCATION_COL = "education"
OCCUPATION_COL = "occupation_industry"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(values):
    values = {k: float(v) for k, v in values.items()}
    total = sum(values.values())
    if total <= 0:
        return values
    return {k: v / total for k, v in values.items()}


def parse_percent(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
        return float(text) / 100.0
    return float(text)


def read_wide_distribution_csv(path, value_map=None):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out = {}
    value_map = value_map or {}
    for key, value in row.items():
        mapped = value_map.get(str(key), str(key))
        out[mapped] = out.get(mapped, 0.0) + parse_percent(value)
    return normalize(out)


def read_age_nation_csv(path, source):
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if "Age Category" not in df.columns:
        return read_wide_distribution_csv(path)
    gender_share = source[GENDER_COL].astype(str).value_counts(normalize=True).to_dict()
    male_share = float(gender_share.get("Male", 0.5))
    female_share = float(gender_share.get("Female", 0.5))
    if male_share + female_share <= 0:
        male_share = female_share = 0.5
    else:
        total = male_share + female_share
        male_share /= total
        female_share /= total
    out = {}
    for row in df.to_dict("records"):
        age = str(row["Age Category"])
        out[age] = parse_percent(row.get("Male", 0.0)) * male_share + parse_percent(row.get("Female", 0.0)) * female_share
    return normalize(out)


def build_amount_preference_bins(target, transaction):
    tx_target = transaction[[TARGET_COL, AMOUNT_COL]].merge(
        target[[TARGET_COL, CATEGORY_COL]], on=TARGET_COL, how="left"
    )
    target_amount = (
        tx_target.groupby([CATEGORY_COL, TARGET_COL])[AMOUNT_COL]
        .mean()
        .rename("target_amount_mean")
        .reset_index()
    )
    bins = {}
    for category, part in target_amount.groupby(CATEGORY_COL):
        values = part["target_amount_mean"].dropna()
        if len(values) < 4:
            continue
        q25, q50, q75 = values.quantile([0.25, 0.50, 0.75]).tolist()
        bins[str(category)] = {
            "q25": float(q25),
            "q50": float(q50),
            "q75": float(q75),
            "basis": "target_mean_amount_quantiles_within_behavior_level1",
        }
    return bins


def load_behavior_level2_prior_table(path, target):
    expected_pairs = target[[CATEGORY_COL, MCC_COL]].drop_duplicates()
    fallback = build_manual_prior_table(expected_pairs, category_col=CATEGORY_COL, mcc_col=MCC_COL)
    if not path or not os.path.exists(path):
        return fallback, "manual_2024_business_priors"

    table = pd.read_csv(path)
    required = {CATEGORY_COL, MCC_COL}
    missing_cols = sorted(required - set(table.columns))
    if missing_cols:
        raise ValueError("Behavior level2 prior table missing columns: %s" % ",".join(missing_cols))

    table[CATEGORY_COL] = table[CATEGORY_COL].astype(str)
    table[MCC_COL] = table[MCC_COL].astype(str)
    expected_pairs = expected_pairs.astype({CATEGORY_COL: str, MCC_COL: str})
    table = expected_pairs.merge(table, on=[CATEGORY_COL, MCC_COL], how="left")
    fallback = fallback.astype({CATEGORY_COL: str, MCC_COL: str})
    fallback_cols = [col for col in fallback.columns if col not in [CATEGORY_COL, MCC_COL]]
    fill = fallback[[CATEGORY_COL, MCC_COL] + fallback_cols].rename(
        columns={col: "%s_fallback" % col for col in fallback_cols}
    )
    table = table.merge(fill, on=[CATEGORY_COL, MCC_COL], how="left")
    for col in fallback_cols:
        fallback_col = "%s_fallback" % col
        if col not in table.columns:
            table[col] = table[fallback_col]
        else:
            table[col] = table[col].where(table[col].notna(), table[fallback_col])
        if fallback_col in table.columns:
            table = table.drop(columns=[fallback_col])

    output_cols = [col for col in table.columns if not col.endswith("_fallback")]
    return table[output_cols], path


def active_hours_for_category(category):
    if category in ["Food and Beverage Services"]:
        return [7, 8, 11, 12, 13, 17, 18, 19, 20, 21]
    if category in ["Local Commuting and On-Demand Travel", "Long-Distance Travel Transport", "Hotels, Scenic Attractions and Tourism Services"]:
        return list(range(6, 23))
    if category in ["General Clinical Care and Health Management", "Pharmaceutical, Medical Device and Health Retail", "Specialized Consumer Medical and Care Services"]:
        return list(range(8, 19))
    if category in ["Education Payments and Training Services"]:
        return list(range(8, 22))
    if category in ["Online Entertainment and Hobby Spending", "Telecommunications, Logistics and Digital Infrastructure Services"]:
        return list(range(9, 24))
    if category in ["Offline Culture, Sports and Entertainment"]:
        return list(range(9, 23))
    return list(range(8, 23))


def category_online_prior(category):
    base = {
        "Food and Beverage Services": 0.18,
        "Food, Tobacco and Liquor Retail": 0.26,
        "Apparel and General Shopping": 0.60,
        "Housing Payments and Home Improvement Services": 0.32,
        "Telecommunications, Logistics and Digital Infrastructure Services": 0.68,
        "Offline Culture, Sports and Entertainment": 0.22,
        "Online Entertainment and Hobby Spending": 0.88,
        "Local Commuting and On-Demand Travel": 0.22,
        "Education Payments and Training Services": 0.42,
        "Hotels, Scenic Attractions and Tourism Services": 0.58,
        "High-Value Goods and Professional Services": 0.50,
        "Local Lifestyle and Self-Service Sharing": 0.20,
        "Household and Personal Goods Retail": 0.55,
        "General Clinical Care and Health Management": 0.28,
        "Pharmaceutical, Medical Device and Health Retail": 0.42,
        "Vehicle Energy and Automotive Services": 0.16,
        "Long-Distance Travel Transport": 0.62,
        "Specialized Consumer Medical and Care Services": 0.22,
    }
    return float(base.get(category, 0.50))


def refine_online_prior(category, mcc):
    text = "%s %s" % (category, mcc)
    mcc_text = "%s" % mcc
    score = category_online_prior(category)
    high_online = [
        "Online", "Internet", "Live Streaming", "Information Search", "Forums", "Gaming", "Social Networking", "Video, Audio",
        "Online Medical", "Express Delivery and Logistics", "Postal", "Air Travel", "Rail", "Travel Agency", "Direct Tourism",
    ]
    offline = [
        "Full-Service Restaurant", "Restaurant", "Fast Food", "Cafeteria", "Bar", "Coffee", "Convenience", "Supermarket",
        "Fresh Produce", "Tobacco, Liquor", "Property Management", "Parking", "Fuel", "Charging and Battery Swap", "Public Transport",
        "Scenic Attractions", "Clinic", "Independent Practitioners", "Dental", "Hair", "Beauty", "Optical Store",
        "Repair", "Housekeeping", "Cleaning", "Designated Driving",
    ]
    high_value_remote = ["Hotel", "Inn", "Homestay", "Tourism", "Air Travel", "Rail", "Water Transport"]
    if any(k in text for k in high_online):
        score = max(score, 0.78)
    if any(k in text for k in high_value_remote):
        score = max(score, 0.60)
    if any(k in mcc_text for k in offline):
        score = min(score, 0.30)
    if "Online Entertainment" in category:
        score = max(score, 0.84)
    if "Offline" in category:
        score = min(score, 0.26)
    return float(np.clip(score, 0.05, 0.95))


def online_bucket(score):
    if score >= 0.75:
        return "strong_online"
    if score >= 0.55:
        return "mixed_online"
    if score >= 0.35:
        return "mixed_offline"
    return "strong_offline"


def build_dataset_priors(
    source,
    target,
    transaction,
    behavior_level2_prior_csv=None,
    age_nation_csv=None,
    education_nation_csv=None,
    occupation_nation_csv=None,
):
    enriched_target = transaction[[TARGET_COL, AMOUNT_COL]].merge(target, on=TARGET_COL, how="left")
    amount_by_cat = enriched_target.groupby(CATEGORY_COL)[AMOUNT_COL].sum()
    observed_amount_share = normalize(amount_by_cat.to_dict())

    # 2021 household consumption expenditure priors, mapped to the actual
    # behavior_level1 taxonomy. The split inside large official buckets follows
    # the dataset taxonomy and is intentionally conservative because this layer
    # is only a final no-worse-than-initial validation while behavior_level1 is
    # frozen by the optimizer.
    category_amount_prior = normalize(
        {
            "Food and Beverage Services": 0.075,
            "Food, Tobacco and Liquor Retail": 0.205,
            "Apparel and General Shopping": 0.070,
            "Housing Payments and Home Improvement Services": 0.185,
            "Telecommunications, Logistics and Digital Infrastructure Services": 0.045,
            "Offline Culture, Sports and Entertainment": 0.035,
            "Online Entertainment and Hobby Spending": 0.020,
            "Local Commuting and On-Demand Travel": 0.035,
            "Education Payments and Training Services": 0.055,
            "Hotels, Scenic Attractions and Tourism Services": 0.020,
            "High-Value Goods and Professional Services": 0.030,
            "Local Lifestyle and Self-Service Sharing": 0.020,
            "Household and Personal Goods Retail": 0.055,
            "General Clinical Care and Health Management": 0.045,
            "Pharmaceutical, Medical Device and Health Retail": 0.025,
            "Vehicle Energy and Automotive Services": 0.030,
            "Long-Distance Travel Transport": 0.030,
            "Specialized Consumer Medical and Care Services": 0.025,
        }
    )

    default_education_prior = normalize(
        {
            "Primary School": 0.277,
            "Junior Secondary School": 0.384,
            "Senior Secondary School": 0.168,
            "Junior College and Above": 0.171,
        }
    )
    education_name_map = {
        "Senior Secondary School (including Secondary Technical School)": "Senior Secondary School",
    }
    age_prior = read_age_nation_csv(age_nation_csv, source) or normalize(source[AGE_COL].value_counts(normalize=True).to_dict())
    education_prior = read_wide_distribution_csv(education_nation_csv, education_name_map) or default_education_prior
    occupation_prior = read_wide_distribution_csv(occupation_nation_csv) or normalize(
        source[OCCUPATION_COL].value_counts(normalize=True).to_dict()
    )

    manual_mcc_prior_table, behavior_level2_prior_source = load_behavior_level2_prior_table(
        behavior_level2_prior_csv, target
    )
    mcc_business_priors = build_manual_prior_dict(
        manual_mcc_prior_table,
        category_col=CATEGORY_COL,
        mcc_col=MCC_COL,
    )
    amount_bins = build_amount_preference_bins(target, transaction)

    category_prior_rows = []
    for category in sorted(target[CATEGORY_COL].unique()):
        category_prior_rows.append(
            {
                CATEGORY_COL: category,
                "online_score": category_online_prior(category),
                "online_bucket": online_bucket(category_online_prior(category)),
                "official_amount_prior": category_amount_prior.get(category, 0.0),
                "observed_amount_share_before": observed_amount_share.get(category, 0.0),
            }
        )

    priors = {
        "metadata": {
            "description": "Dataset-specific priors for source.csv/target.csv/transaction.csv.",
            "behavior_level1_is_frozen": True,
            "behavior_level2_is_optimizable": True,
            "behavior_level2_prior_source": behavior_level2_prior_source,
            "online_prior_anchor": "NBS 2024: physical-goods online retail share 26.5% of total retail sales",
            "online_score_note": "Compared with quantile-normalized merchant cross-province ratio, not raw out-province share.",
            "amount_preference_note": "Merchant amount labels are bucketed by target mean amount quantiles within each behavior_level1.",
        },
        "amount_preference_bins_by_category": amount_bins,
        "consumption_category_amount": category_amount_prior,
        "user_attribute_national": {
            "age": age_prior,
            "education": education_prior,
            "occupation": occupation_prior,
        },
        "user_attribute_by_province": {
            "__default__": {
                "age": age_prior,
                "education": education_prior,
                "occupation": occupation_prior,
            }
        },
        "consumption_category_business_priors": {
            row[CATEGORY_COL]: {
                "online_score": row["online_score"],
                "amount_preference": "medium",
                "active_hours": active_hours_for_category(row[CATEGORY_COL]),
            }
            for row in category_prior_rows
        },
        "mcc_business_priors": mcc_business_priors,
    }
    return priors, pd.DataFrame(category_prior_rows), manual_mcc_prior_table


def merge_three_tables(source, target, transaction):
    enriched = transaction.merge(source, on=SOURCE_COL, how="left", validate="many_to_one")
    enriched = enriched.merge(target, on=TARGET_COL, how="left", validate="many_to_one")
    missing_source = int(enriched[SOURCE_PROVINCE_COL].isna().sum())
    missing_target = int(enriched[CATEGORY_COL].isna().sum())
    if missing_source or missing_target:
        raise ValueError("Merge produced missing source=%s target=%s" % (missing_source, missing_target))
    return enriched


def write_distribution_table(before, after, col, out_path, group_col=None):
    if group_col is None:
        before_counts = before[col].value_counts().rename("before_count")
        after_counts = after[col].value_counts().rename("after_count")
        result = pd.concat([before_counts, after_counts], axis=1).fillna(0)
        result.index.name = col
        group_total_before = float(result["before_count"].sum())
        group_total_after = float(result["after_count"].sum())
        result["before_share"] = result["before_count"] / max(group_total_before, 1.0)
        result["after_share"] = result["after_count"] / max(group_total_after, 1.0)
        result["share_delta"] = result["after_share"] - result["before_share"]
        result.reset_index().to_csv(out_path, index=False, encoding="utf-8-sig")
        return result.reset_index()

    before_counts = before.groupby([group_col, col]).size().rename("before_count").reset_index()
    after_counts = after.groupby([group_col, col]).size().rename("after_count").reset_index()
    result = before_counts.merge(after_counts, on=[group_col, col], how="outer").fillna(0)
    before_totals = result.groupby(group_col)["before_count"].transform("sum")
    after_totals = result.groupby(group_col)["after_count"].transform("sum")
    result["before_share"] = result["before_count"] / before_totals.replace(0, np.nan)
    result["after_share"] = result["after_count"] / after_totals.replace(0, np.nan)
    result["share_delta"] = result["after_share"] - result["before_share"]
    result = result.fillna(0)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def target_distribution_table(transaction, target_before, target_after, col, out_path):
    include_category = col == MCC_COL and CATEGORY_COL in target_before.columns and CATEGORY_COL in target_after.columns
    target_cols = [TARGET_COL, col]
    if include_category:
        target_cols = [TARGET_COL, CATEGORY_COL, col]
    before = transaction[[TARGET_COL, AMOUNT_COL]].merge(target_before[target_cols], on=TARGET_COL, how="left")
    after = transaction[[TARGET_COL, AMOUNT_COL]].merge(target_after[target_cols], on=TARGET_COL, how="left")
    rows = []
    if include_category:
        values = sorted(
            set(zip(before[CATEGORY_COL].dropna().astype(str), before[col].dropna().astype(str)))
            | set(zip(after[CATEGORY_COL].dropna().astype(str), after[col].dropna().astype(str)))
        )
    else:
        values = [(None, value) for value in sorted(set(before[col].dropna().astype(str)) | set(after[col].dropna().astype(str)))]
    for category, value in values:
        if include_category:
            b = before[(before[CATEGORY_COL].astype(str) == category) & (before[col].astype(str) == value)]
            a = after[(after[CATEGORY_COL].astype(str) == category) & (after[col].astype(str) == value)]
            before_target_mask = (target_before[CATEGORY_COL].astype(str) == category) & (target_before[col].astype(str) == value)
            after_target_mask = (target_after[CATEGORY_COL].astype(str) == category) & (target_after[col].astype(str) == value)
        else:
            b = before[before[col].astype(str) == value]
            a = after[after[col].astype(str) == value]
            before_target_mask = target_before[col].astype(str) == value
            after_target_mask = target_after[col].astype(str) == value
        row = {}
        if include_category:
            row[CATEGORY_COL] = category
        row.update(
            {
                col: value,
                "before_target_count": int(before_target_mask.sum()),
                "after_target_count": int(after_target_mask.sum()),
                "before_tx_count": int(len(b)),
                "after_tx_count": int(len(a)),
                "before_amount_sum": float(b[AMOUNT_COL].sum()),
                "after_amount_sum": float(a[AMOUNT_COL].sum()),
            }
        )
        rows.append(
            row
        )
    result = pd.DataFrame(rows)
    for prefix in ["before", "after"]:
        result["%s_target_share" % prefix] = result["%s_target_count" % prefix] / max(
            float(result["%s_target_count" % prefix].sum()), 1.0
        )
        result["%s_tx_share" % prefix] = result["%s_tx_count" % prefix] / max(
            float(result["%s_tx_count" % prefix].sum()), 1.0
        )
        result["%s_amount_share" % prefix] = result["%s_amount_sum" % prefix] / max(
            float(result["%s_amount_sum" % prefix].sum()), 1.0
        )
    result["target_share_delta"] = result["after_target_share"] - result["before_target_share"]
    result["tx_share_delta"] = result["after_tx_share"] - result["before_tx_share"]
    result["amount_share_delta"] = result["after_amount_share"] - result["before_amount_share"]
    sort_cols = [CATEGORY_COL, col] if include_category else [col]
    result = result.sort_values(sort_cols).reset_index(drop=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    return result


def online_alignment_table(transaction, target_before, target_after, priors, out_path_target, out_path_category):
    tx = transaction[[TARGET_COL, OUT_PROVINCE_COL, AMOUNT_COL]].copy()
    target_stats = tx.groupby(TARGET_COL).agg(
        tx_count=(AMOUNT_COL, "size"),
        amount_sum=(AMOUNT_COL, "sum"),
        cross_ratio=(OUT_PROVINCE_COL, "mean"),
    )
    vals = target_stats["cross_ratio"].values.astype(float)
    low = float(np.quantile(vals, 0.10))
    high = float(np.quantile(vals, 0.95))
    target_stats["cross_score"] = [quantile_normalize(v, low, high) for v in target_stats["cross_ratio"]]

    before = target_stats.reset_index().merge(target_before, on=TARGET_COL, how="left")
    after = target_stats.reset_index().merge(target_after, on=TARGET_COL, how="left")
    rows = []
    prior_map = priors.get("mcc_business_priors", {})
    for b, a in zip(before.to_dict("records"), after.to_dict("records")):
        old_mcc = b[MCC_COL]
        new_mcc = a[MCC_COL]
        old_score = float(prior_map.get(old_mcc, {}).get("online_score", 0.50))
        new_score = float(prior_map.get(new_mcc, {}).get("online_score", 0.50))
        rows.append(
            {
                TARGET_COL: b[TARGET_COL],
                "tx_count": b["tx_count"],
                "amount_sum": b["amount_sum"],
                "cross_ratio": b["cross_ratio"],
                "cross_score": b["cross_score"],
                "behavior_level1": b[CATEGORY_COL],
                "mcc_before": old_mcc,
                "mcc_after": new_mcc,
                "online_prior_before": old_score,
                "online_prior_after": new_score,
                "alignment_error_before": abs(float(b["cross_score"]) - old_score),
                "alignment_error_after": abs(float(b["cross_score"]) - new_score),
                "alignment_error_delta": abs(float(b["cross_score"]) - new_score)
                - abs(float(b["cross_score"]) - old_score),
                "changed": old_mcc != new_mcc,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out_path_target, index=False, encoding="utf-8-sig")

    category = (
        result.groupby("behavior_level1")
        .agg(
            target_count=(TARGET_COL, "size"),
            changed_count=("changed", "sum"),
            mean_cross_ratio=("cross_ratio", "mean"),
            mean_cross_score=("cross_score", "mean"),
            mean_error_before=("alignment_error_before", "mean"),
            mean_error_after=("alignment_error_after", "mean"),
            mean_error_delta=("alignment_error_delta", "mean"),
        )
        .reset_index()
    )
    category["changed_rate"] = category["changed_count"] / category["target_count"]
    category.to_csv(out_path_category, index=False, encoding="utf-8-sig")
    return result, category


def write_change_tables(source_before, source_after, target_before, target_after, output_dir):
    source_cmp = source_before.merge(source_after, on=SOURCE_COL, suffixes=("_before", "_after"))
    source_changed_mask = False
    for col in [AGE_COL, EDUCATION_COL, OCCUPATION_COL]:
        if col not in source_before.columns or col not in source_after.columns:
            continue
        source_cmp["%s_changed" % col] = source_cmp["%s_before" % col] != source_cmp["%s_after" % col]
        source_changed_mask = source_changed_mask | source_cmp["%s_changed" % col]
    source_changes = source_cmp[source_changed_mask].copy()
    source_changes.to_csv(os.path.join(output_dir, "source_changes_final.csv"), index=False, encoding="utf-8-sig")

    target_cmp = target_before.merge(target_after, on=TARGET_COL, suffixes=("_before", "_after"))
    target_cmp["behavior_level2_changed"] = target_cmp["%s_before" % MCC_COL] != target_cmp["%s_after" % MCC_COL]
    target_changes = target_cmp[target_cmp["behavior_level2_changed"]].copy()
    target_changes.to_csv(os.path.join(output_dir, "target_changes_final.csv"), index=False, encoding="utf-8-sig")
    return source_changes, target_changes


def write_behavior_level2_change_flow_tables(target_before, target_after, output_dir):
    cmp_df = target_before.merge(target_after, on=TARGET_COL, suffixes=("_before", "_after"))
    before_cat = "%s_before" % CATEGORY_COL
    before_mcc = "%s_before" % MCC_COL
    after_mcc = "%s_after" % MCC_COL
    cmp_df["behavior_level2_changed"] = cmp_df[before_mcc] != cmp_df[after_mcc]
    changed = cmp_df[cmp_df["behavior_level2_changed"]].copy()

    if changed.empty:
        transitions = pd.DataFrame(
            columns=[CATEGORY_COL, "behavior_level2_before", "behavior_level2_after", "changed_target_count"]
        )
    else:
        transitions = (
            changed.groupby([before_cat, before_mcc, after_mcc])
            .size()
            .rename("changed_target_count")
            .reset_index()
            .rename(
                columns={
                    before_cat: CATEGORY_COL,
                    before_mcc: "behavior_level2_before",
                    after_mcc: "behavior_level2_after",
                }
            )
            .sort_values([CATEGORY_COL, "changed_target_count"], ascending=[True, False])
        )
    transitions.to_csv(
        os.path.join(output_dir, "target_behavior_level2_transitions_by_behavior_level1.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    flow_rows = []
    for category, part in cmp_df.groupby(before_cat):
        changed_part = part[part["behavior_level2_changed"]]
        before_counts = part[before_mcc].value_counts().to_dict()
        after_counts = part[after_mcc].value_counts().to_dict()
        out_counts = changed_part[before_mcc].value_counts().to_dict()
        in_counts = changed_part[after_mcc].value_counts().to_dict()
        values = sorted(set(before_counts) | set(after_counts) | set(out_counts) | set(in_counts))
        for value in values:
            out_count = int(out_counts.get(value, 0))
            in_count = int(in_counts.get(value, 0))
            flow_rows.append(
                {
                    CATEGORY_COL: category,
                    MCC_COL: value,
                    "before_target_count": int(before_counts.get(value, 0)),
                    "after_target_count": int(after_counts.get(value, 0)),
                    "out_count": out_count,
                    "in_count": in_count,
                    "gross_flow_count": out_count + in_count,
                    "net_flow_count": in_count - out_count,
                    "has_internal_change_even_if_net_zero": (out_count + in_count) > 0 and (in_count - out_count) == 0,
                }
            )

        swap_pair_count = 0
        swap_target_lower_bound = 0
        if not changed_part.empty:
            pair_counts = changed_part.groupby([before_mcc, after_mcc]).size().to_dict()
            seen = set()
            for (old, new), count in pair_counts.items():
                if old == new:
                    continue
                unordered = tuple(sorted([old, new]))
                if unordered in seen:
                    continue
                reverse = pair_counts.get((new, old), 0)
                if reverse:
                    swap_pair_count += 1
                    swap_target_lower_bound += 2 * min(int(count), int(reverse))
                seen.add(unordered)

        involved_values = set(changed_part[before_mcc].dropna().astype(str)) | set(changed_part[after_mcc].dropna().astype(str))
        summary_rows.append(
            {
                CATEGORY_COL: category,
                "target_count": int(len(part)),
                "changed_target_count": int(len(changed_part)),
                "changed_rate": safe_div(len(changed_part), len(part), 0.0),
                "changed_from_level2_count": int(changed_part[before_mcc].nunique()),
                "changed_to_level2_count": int(changed_part[after_mcc].nunique()),
                "involved_level2_count": int(len(involved_values)),
                "transition_pair_count": int(len(changed_part.groupby([before_mcc, after_mcc])) if not changed_part.empty else 0),
                "swap_pair_count": int(swap_pair_count),
                "swap_target_lower_bound": int(swap_target_lower_bound),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("changed_target_count", ascending=False)
    flow = pd.DataFrame(flow_rows).sort_values([CATEGORY_COL, "gross_flow_count"], ascending=[True, False])
    summary.to_csv(
        os.path.join(output_dir, "target_behavior_level2_change_summary_by_behavior_level1.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    flow.to_csv(
        os.path.join(output_dir, "target_behavior_level2_flow_by_behavior_level1.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    return summary, transitions, flow


def summarize_metrics(source_before, source_after, target_before, target_after, transaction, report, alignment, output_dir):
    rows = []
    rows.append({"metric": "source_count", "value": len(source_before)})
    rows.append({"metric": "target_count", "value": len(target_before)})
    rows.append({"metric": "transaction_count", "value": len(transaction)})
    rows.append({"metric": "source_age_changed_count", "value": int((source_before[AGE_COL] != source_after[AGE_COL]).sum())})
    rows.append(
        {
            "metric": "source_education_changed_count",
            "value": int((source_before[EDUCATION_COL] != source_after[EDUCATION_COL]).sum()),
        }
    )
    if OCCUPATION_COL in source_before.columns and OCCUPATION_COL in source_after.columns:
        rows.append(
            {
                "metric": "source_occupation_changed_count",
                "value": int((source_before[OCCUPATION_COL] != source_after[OCCUPATION_COL]).sum()),
            }
        )
    rows.append({"metric": "target_behavior_level2_changed_count", "value": int((target_before[MCC_COL] != target_after[MCC_COL]).sum())})
    rows.append({"metric": "mean_online_alignment_error_before", "value": float(alignment["alignment_error_before"].mean())})
    rows.append({"metric": "mean_online_alignment_error_after", "value": float(alignment["alignment_error_after"].mean())})
    rows.append({"metric": "median_online_alignment_error_before", "value": float(alignment["alignment_error_before"].median())})
    rows.append({"metric": "median_online_alignment_error_after", "value": float(alignment["alignment_error_after"].median())})
    rows.append({"metric": "optimizer_stop_reason", "value": report.get("stop_reason", "")})
    rows.append({"metric": "final_constraint_passed", "value": report["summary"]["final_constraint_passed"]})
    rows.append({"metric": "frozen_fields_unchanged", "value": report["summary"]["frozen_fields_unchanged"]})
    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
    return result


def write_stop_condition(report, output_dir):
    reason = report.get("stop_reason", "")
    rounds = report.get("rounds", [])
    row = {
        "stop_reason": reason,
        "explanation": explain_stop_reason(reason),
        "rounds_completed": len(rounds),
        "max_iterations": report.get("config", {}).get("max_iterations", ""),
        "last_round_user_accepted": rounds[-1]["user"]["accepted"] if rounds else 0,
        "last_round_merchant_accepted": rounds[-1]["merchant"]["accepted"] if rounds else 0,
        "final_constraint_passed": report.get("summary", {}).get("final_constraint_passed", ""),
        "frozen_fields_unchanged": report.get("summary", {}).get("frozen_fields_unchanged", ""),
    }
    pd.DataFrame([row]).to_csv(os.path.join(output_dir, "optimizer_stop_condition.csv"), index=False, encoding="utf-8-sig")
    write_json(os.path.join(output_dir, "optimizer_stop_condition.json"), row)
    return row


def build_parser():
    parser = argparse.ArgumentParser(description="Run dataset three-table consistency optimization.")
    parser.add_argument("--source_csv", default="work/source.csv")
    parser.add_argument("--target_csv", default="work/target.csv")
    parser.add_argument("--transaction_csv", default="work/transaction.csv")
    parser.add_argument(
        "--dataset_dir",
        default=None,
        help="Directory for optimized source/target/transaction CSV outputs.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for analysis report outputs.",
    )
    parser.add_argument(
        "--behavior_level2_prior_csv",
        default="consistency_optimization/priors/prior_behavior_level2_table.csv",
    )
    parser.add_argument("--age_nation_csv", default="consistency_optimization/priors/age_nation.csv")
    parser.add_argument("--education_nation_csv", default="consistency_optimization/priors/education_nation.csv")
    parser.add_argument("--occupation_nation_csv", default="consistency_optimization/priors/occupation_nation.csv")
    parser.add_argument(
        "--skip_analysis",
        action="store_true",
        help="Only write optimized source/target/transaction tables to work/.",
    )
    parser.add_argument("--max_iterations", type=int, default=12)
    parser.add_argument("--merchant_round_budget", type=float, default=0.20)
    parser.add_argument("--merchant_total_budget", type=float, default=1.00)
    parser.add_argument(
        "--user_round_budget",
        type=float,
        default=0.20,
        help="Per-round budget as a share of evidence-qualified user candidates.",
    )
    parser.add_argument("--user_total_budget", type=float, default=1.00)
    parser.add_argument("--user_field_round_budget", type=float, default=0.20)
    parser.add_argument("--mcc_net_flow_cap_rate", type=float, default=0.12)
    parser.add_argument("--merchant_category_min_changes_per_round", type=int, default=3)
    parser.add_argument("--merchant_min_gain", type=float, default=0.015)
    parser.add_argument("--merchant_coverage_min_gain", type=float, default=-10.0)
    parser.add_argument("--user_min_gain", type=float, default=-0.10)
    parser.add_argument(
        "--allow_nonpositive_gain",
        action="store_true",
        help="Use the retained legacy gain thresholds, including nonpositive proposals.",
    )
    parser.add_argument("--epsilon_mcc1", type=float, default=0.06)
    parser.add_argument("--mcc_max_share_per_category", type=float, default=0.65)
    parser.add_argument("--mcc_max_share_slack", type=float, default=0.20)
    parser.add_argument("--mcc_min_active_level2_ratio", type=float, default=0.60)
    parser.add_argument("--mcc_category_change_rate_cap", type=float, default=0.65)
    parser.add_argument("--merchant_change_penalty", type=float, default=0.03)
    parser.add_argument("--user_change_penalty", type=float, default=0.0)
    parser.add_argument("--user_population_score_weight", type=float, default=3.00)
    parser.add_argument("--user_mcc_score_weight", type=float, default=1.20)
    parser.add_argument("--user_amount_score_weight", type=float, default=0.20)
    parser.add_argument("--user_time_score_weight", type=float, default=0.15)
    parser.add_argument("--user_constraint_score_weight", type=float, default=0.50)
    parser.add_argument("--epsilon_age", type=float, default=0.08)
    parser.add_argument("--epsilon_education", type=float, default=0.08)
    parser.add_argument("--epsilon_occupation", type=float, default=0.10)
    parser.add_argument("--gain_threshold", type=float, default=0.05)
    parser.add_argument("--gain_stop_patience", type=int, default=2)
    parser.add_argument(
        "--min_user_tx_for_behavior",
        type=int,
        default=5,
        help="Minimum transactions required for a user's amount and time signals.",
    )
    parser.add_argument(
        "--min_user_merchants_for_category",
        type=int,
        default=3,
        help="Minimum unique merchants required for a user's merchant-group signals.",
    )
    parser.add_argument(
        "--min_merchant_tx_for_behavior",
        type=int,
        default=5,
        help="Minimum transactions required for a merchant's operating-behavior signals.",
    )
    parser.add_argument(
        "--min_merchant_users_for_customer",
        type=int,
        default=3,
        help="Minimum unique users required for a merchant's customer-profile signals.",
    )
    parser.add_argument("--max_user_candidates_per_round", type=int, default=60000)
    parser.add_argument(
        "--disable_response_budget",
        action="store_true",
        help="Disable the strict-positive response budget after a side exhausts its first-change budget.",
    )
    parser.add_argument(
        "--response_affected_entity_rate",
        type=float,
        default=0.20,
        help=(
            "Share of strict-positive candidates selected within the entity set "
            "directly affected by the opposite side's latest primary changes."
        ),
    )
    parser.add_argument("--random_seed", type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()
    result_root = os.path.join("work", "consistency_optimization")
    if args.dataset_dir is None:
        args.dataset_dir = "work"
    if args.output_dir is None:
        args.output_dir = os.path.join(result_root, "analysis")
    ensure_dir(args.dataset_dir)
    ensure_dir(args.output_dir)
    analysis_dir = os.path.join(args.output_dir, "analysis_outputs")
    if not args.skip_analysis:
        ensure_dir(analysis_dir)

    print("Reading source table:", args.source_csv)
    source = pd.read_csv(args.source_csv)
    print("Reading target table:", args.target_csv)
    target = pd.read_csv(args.target_csv)
    print("Reading transaction table:", args.transaction_csv)
    transaction = pd.read_csv(args.transaction_csv)

    priors, category_prior_table, mcc_prior_table = build_dataset_priors(
        source,
        target,
        transaction,
        behavior_level2_prior_csv=args.behavior_level2_prior_csv,
        age_nation_csv=args.age_nation_csv,
        education_nation_csv=args.education_nation_csv,
        occupation_nation_csv=args.occupation_nation_csv,
    )
    if not args.skip_analysis:
        write_json(os.path.join(analysis_dir, "dataset_consistency_priors_2024.json"), priors)
        mcc_prior_table.to_csv(
            os.path.join(analysis_dir, "prior_behavior_level2_table.csv"), index=False, encoding="utf-8-sig"
        )
        mcc_prior_table.to_csv(
            os.path.join(analysis_dir, "manual_merchant_behavior_level2_priors.csv"), index=False, encoding="utf-8-sig"
        )

    enriched_before = merge_three_tables(source, target, transaction)
    config = ConsistencyOptimizerConfig(
        source_col=SOURCE_COL,
        target_col=TARGET_COL,
        amount_col=AMOUNT_COL,
        timestamp_col="datetime",
        hour_col="hour",
        source_province_col=SOURCE_PROVINCE_COL,
        transaction_province_col=TRANSACTION_PROVINCE_COL,
        consumption_category_col=CATEGORY_COL,
        mcc_level1_col=MCC_COL,
        gender_col=GENDER_COL,
        age_col=AGE_COL,
        education_col=EDUCATION_COL,
        occupation_col=OCCUPATION_COL,
        max_iterations=args.max_iterations,
        merchant_round_budget=args.merchant_round_budget,
        merchant_total_budget=args.merchant_total_budget,
        user_round_budget=args.user_round_budget,
        user_total_budget=args.user_total_budget,
        user_field_round_budget=args.user_field_round_budget,
        mcc_net_flow_cap_rate=args.mcc_net_flow_cap_rate,
        merchant_category_min_changes_per_round=args.merchant_category_min_changes_per_round,
        merchant_min_gain=args.merchant_min_gain,
        merchant_coverage_min_gain=args.merchant_coverage_min_gain,
        user_min_gain=args.user_min_gain,
        require_positive_gain=not args.allow_nonpositive_gain,
        epsilon_mcc1=args.epsilon_mcc1,
        epsilon_age=args.epsilon_age,
        epsilon_education=args.epsilon_education,
        epsilon_occupation=args.epsilon_occupation,
        mcc_max_share_per_category=args.mcc_max_share_per_category,
        mcc_max_share_slack=args.mcc_max_share_slack,
        mcc_min_active_level2_ratio=args.mcc_min_active_level2_ratio,
        mcc_category_change_rate_cap=args.mcc_category_change_rate_cap,
        merchant_change_penalty=args.merchant_change_penalty,
        user_change_penalty=args.user_change_penalty,
        user_population_score_weight=args.user_population_score_weight,
        user_mcc_score_weight=args.user_mcc_score_weight,
        user_amount_score_weight=args.user_amount_score_weight,
        user_time_score_weight=args.user_time_score_weight,
        user_constraint_score_weight=args.user_constraint_score_weight,
        gain_threshold=args.gain_threshold,
        gain_stop_patience=args.gain_stop_patience,
        min_user_tx_for_behavior=args.min_user_tx_for_behavior,
        min_user_merchants_for_category=args.min_user_merchants_for_category,
        min_merchant_tx_for_behavior=args.min_merchant_tx_for_behavior,
        min_merchant_users_for_customer=args.min_merchant_users_for_customer,
        max_user_candidates_per_round=args.max_user_candidates_per_round,
        enable_response_budget=not args.disable_response_budget,
        response_affected_entity_rate=args.response_affected_entity_rate,
        random_seed=args.random_seed,
    )

    print("Running consistency optimizer ...")
    enriched_after, report = optimize_consistency(enriched_before, config=config, external_priors=priors)
    write_json(os.path.join(args.output_dir, "consistency_optimization_report.json"), report)
    if not args.skip_analysis:
        write_json(os.path.join(analysis_dir, "consistency_optimization_report.json"), report)

    target_after = target.copy()
    target_update = enriched_after[[TARGET_COL, MCC_COL]].drop_duplicates(TARGET_COL)
    target_after = target_after.drop(columns=[MCC_COL]).merge(target_update, on=TARGET_COL, how="left")
    target_after = target_after[target.columns]

    source_after = source.copy()
    source_update_cols = [SOURCE_COL, AGE_COL, EDUCATION_COL]
    if OCCUPATION_COL in source.columns:
        source_update_cols.append(OCCUPATION_COL)
    source_update = enriched_after[source_update_cols].drop_duplicates(SOURCE_COL)
    source_after = source_after.drop(columns=[col for col in source_update_cols if col != SOURCE_COL]).merge(
        source_update, on=SOURCE_COL, how="left"
    )
    source_after = source_after[source.columns]

    print("Writing optimized three tables ...")
    source_after.to_csv(os.path.join(args.dataset_dir, "source_optimized.csv"), index=False, encoding="utf-8-sig")
    target_after.to_csv(os.path.join(args.dataset_dir, "target_optimized.csv"), index=False, encoding="utf-8-sig")
    transaction.to_csv(os.path.join(args.dataset_dir, "transaction_optimized.csv"), index=False, encoding="utf-8-sig")

    if args.skip_analysis:
        print("Done.")
        print("Optimized tables dir:", args.dataset_dir)
        return

    print("Writing detailed analysis tables ...")
    pd.DataFrame(report.get("change_history", [])).to_csv(
        os.path.join(analysis_dir, "optimizer_change_history.csv"), index=False, encoding="utf-8-sig"
    )
    source_changes, target_changes = write_change_tables(source, source_after, target, target_after, analysis_dir)

    write_distribution_table(source, source_after, AGE_COL, os.path.join(analysis_dir, "source_age_distribution_before_after.csv"))
    write_distribution_table(
        source,
        source_after,
        EDUCATION_COL,
        os.path.join(analysis_dir, "source_education_distribution_before_after.csv"),
    )
    write_distribution_table(
        source,
        source_after,
        AGE_COL,
        os.path.join(analysis_dir, "source_age_by_province_before_after.csv"),
        group_col=SOURCE_PROVINCE_COL,
    )
    write_distribution_table(
        source,
        source_after,
        EDUCATION_COL,
        os.path.join(analysis_dir, "source_education_by_province_before_after.csv"),
        group_col=SOURCE_PROVINCE_COL,
    )
    if OCCUPATION_COL in source.columns and OCCUPATION_COL in source_after.columns:
        write_distribution_table(
            source,
            source_after,
            OCCUPATION_COL,
            os.path.join(analysis_dir, "source_occupation_distribution_before_after.csv"),
        )
        write_distribution_table(
            source,
            source_after,
            OCCUPATION_COL,
            os.path.join(analysis_dir, "source_occupation_by_province_before_after.csv"),
            group_col=SOURCE_PROVINCE_COL,
        )
    target_distribution_table(
        transaction,
        target,
        target_after,
        CATEGORY_COL,
        os.path.join(analysis_dir, "target_behavior_level1_distribution_before_after.csv"),
    )
    target_distribution_table(
        transaction,
        target,
        target_after,
        MCC_COL,
        os.path.join(analysis_dir, "target_behavior_level2_distribution_before_after.csv"),
    )
    write_behavior_level2_change_flow_tables(target, target_after, analysis_dir)
    alignment, _ = online_alignment_table(
        transaction,
        target,
        target_after,
        priors,
        os.path.join(analysis_dir, "merchant_online_alignment_before_after.csv"),
        os.path.join(analysis_dir, "merchant_online_alignment_by_behavior_level1.csv"),
    )
    merchant_profile = build_merchant_profile_after(
        args.dataset_dir,
        analysis_dir,
        os.path.join(args.dataset_dir, "transaction_optimized.csv"),
        occupation_col=OCCUPATION_COL if OCCUPATION_COL in source.columns else "occupation_industry",
    )
    merchant_profile.to_csv(
        os.path.join(analysis_dir, "merchant_profile_after.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    summary = summarize_metrics(source, source_after, target, target_after, transaction, report, alignment, analysis_dir)
    stop_condition = write_stop_condition(report, analysis_dir)

    print("Done.")
    print(summary.to_string(index=False))
    print("Stop condition:", stop_condition["stop_reason"], "-", stop_condition["explanation"])
    print("Final target changes:", len(target_changes))
    print("Final source changes:", len(source_changes))
    print("Optimized tables dir:", args.dataset_dir)
    print("Analysis outputs dir:", analysis_dir)


if __name__ == "__main__":
    main()
