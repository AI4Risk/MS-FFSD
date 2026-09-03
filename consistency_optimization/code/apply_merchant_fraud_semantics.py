#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Apply fraud-aware merchant semantics to existing merchant category fields.

The script does not add columns to the final target table. It rewrites a small
set of merchant behavior categories when both conditions hold:

1. The merchant has transaction-label evidence of fraud tendency.
2. The merchant still has a material post-consistency residual.

For selected merchants, the script searches for a target category that is both
more compatible with the merchant profile and more fraud-risk expressive. It
prefers a behavior_level2 change within the original behavior_level1, and only
allows a behavior_level1 change when no same-level candidate is good enough.
"""

from __future__ import print_function

import argparse
import json
import os

import numpy as np
import pandas as pd


LABEL_RANK = {"low": 0, "low_to_medium": 1, "medium": 2, "high": 3}

RISK_KEYWORDS = [
    "Information Search and Online Forums",
    "Online Social Networking",
    "Online Video, Audio and Reading",
    "Internet Data Services",
    "Gaming",
    "Gaming Transactions and Related Services",
    "Live Streaming",
    "Nutrition and Health Products",
    "Hair, Beauty and Nail Services",
    "Medical Aesthetics",
    "High-Value Jewelry and Accessories",
    "Online Tools",
    "Travel Agencies and Tourism Services",
    "Direct Tourism-Related Services",
    "Board Games, Table Games and Gaming Cafes",
    "Telecom Payment and Operator Services",
    "Digital Electronics and Home Appliances",
    "Digital and Entertainment Equipment Rental",
    "Appliance, Furniture and Other Repair Services",
    "Nightlife Bar and Dining",
    "KTV and Leisure Clubs",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rewrite existing merchant category fields with fraud-aware semantics."
    )
    parser.add_argument(
        "--analysis_dir",
        default="work/consistency_optimization/analysis/analysis_outputs",
        help="Analysis output directory from the selected consistency optimization run.",
    )
    parser.add_argument(
        "--transaction_csv",
        default="work/transaction_optimized.csv",
        help="Optimized transaction table containing Target and Labels.",
    )
    parser.add_argument(
        "--output_dir",
        default="work/consistency_optimization/fraud_semantics",
        help="Output directory for rewritten target table and audit files.",
    )
    parser.add_argument(
        "--final_target_output",
        default="work/target_final.csv",
        help="Full final Target table for downstream modules after fraud-semantic updates.",
    )
    parser.add_argument("--target_col", default="Target")
    parser.add_argument("--label_col", default="Labels")
    parser.add_argument("--min_labeled_tx", type=int, default=20)
    parser.add_argument("--min_fraud_tx", type=int, default=3)
    parser.add_argument("--min_fraud_rate_labeled", type=float, default=0.15)
    parser.add_argument("--high_fraud_tx", type=int, default=50)
    parser.add_argument("--high_fraud_rate_labeled", type=float, default=0.12)
    parser.add_argument("--min_residual_dims", type=int, default=1)
    parser.add_argument("--overall_score_threshold", type=float, default=0.65)
    parser.add_argument("--same_l1_fit_tolerance", type=float, default=0.08)
    parser.add_argument("--cross_l1_min_fit_gain", type=float, default=0.05)
    parser.add_argument("--min_risk_gain", type=float, default=0.12)
    parser.add_argument("--cross_l1_min_risk", type=float, default=0.65)
    parser.add_argument("--max_target_changes", type=int, default=120)
    parser.add_argument("--max_l1_change_rate", type=float, default=0.08)
    parser.add_argument("--max_level2_share_per_l1", type=float, default=0.65)
    parser.add_argument("--risk_weight", type=float, default=0.35)
    parser.add_argument("--random_seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def as_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def parse_hours(value):
    if pd.isna(value):
        return set()
    return {int(v) for v in str(value).split(",") if str(v).strip() != ""}


def label_distance(observed, prior):
    if observed not in LABEL_RANK or prior not in LABEL_RANK:
        return 0.5
    return abs(LABEL_RANK[observed] - LABEL_RANK[prior]) / 3.0


def parse_distribution(value):
    dist = {}
    if pd.isna(value) or str(value).strip() == "":
        return dist
    for item in str(value).split("|"):
        if ":" not in item:
            continue
        key, raw_val = item.rsplit(":", 1)
        try:
            dist[key] = float(raw_val)
        except ValueError:
            continue
    return dist


def tvd(left, right):
    keys = set(left) | set(right)
    return 0.5 * sum(abs(float(left.get(k, 0.0)) - float(right.get(k, 0.0))) for k in keys)


def prior_occupation_distribution(row):
    return parse_distribution(row.get("occupation_industry_distribution_prior", ""))


def prior_age_distribution(row):
    return {
        "Young Adults": as_float(row.get("age_young_adults_prior"), 0.0),
        "Middle-aged Adults": as_float(row.get("age_middle_aged_adults_prior"), 0.0),
        "Older Adults": as_float(row.get("age_older_adults_prior"), 0.0),
    }


def prior_education_distribution(row):
    return {
        "Primary School": as_float(row.get("education_primary_school_prior"), 0.0),
        "Junior Secondary School": as_float(row.get("education_junior_secondary_prior"), 0.0),
        "Senior Secondary School": as_float(row.get("education_senior_secondary_prior"), 0.0),
        "Junior College and Above": as_float(row.get("education_junior_college_above_prior"), 0.0),
    }


def fraud_risk_score(prior_row):
    name = str(prior_row["behavior_level2"])
    score = 0.18
    score += 0.22 * as_float(prior_row.get("online_score"), 0.0)
    if str(prior_row.get("amount_preference")) == "high":
        score += 0.13
    elif str(prior_row.get("amount_preference")) == "medium":
        score += 0.06
    if str(prior_row.get("amount_volatility")) == "high":
        score += 0.10
    elif str(prior_row.get("amount_volatility")) == "medium":
        score += 0.04
    if any(token in name for token in RISK_KEYWORDS):
        score += 0.32
    return float(max(0.0, min(1.0, score)))


def profile_fit_score(profile_row, prior_row):
    cross_score = as_float(profile_row.get("cross_score"), np.nan)
    online_prior = as_float(prior_row.get("online_score"), np.nan)
    online_error = abs(cross_score - online_prior) if not pd.isna(cross_score) else 0.5

    amount_error = label_distance(str(profile_row.get("amount_label")), str(prior_row.get("amount_preference")))
    volatility_error = label_distance(
        str(profile_row.get("amount_cv_label")), str(prior_row.get("amount_volatility"))
    )

    top_hours = parse_hours(profile_row.get("top_hours"))
    active_hours = parse_hours(prior_row.get("active_hours"))
    if top_hours:
        hour_hit = sum(1 for hour in top_hours if hour in active_hours) / float(len(top_hours))
    else:
        hour_hit = 0.0

    female_ratio = as_float(profile_row.get("female_ratio"), 0.5)
    female_prior = as_float(prior_row.get("female_ratio_prior"), 0.5)
    gender_error = min(1.0, abs(female_ratio - female_prior))

    age_error = tvd(parse_distribution(profile_row.get("age_distribution")), prior_age_distribution(prior_row))
    education_error = tvd(
        parse_distribution(profile_row.get("education_distribution")), prior_education_distribution(prior_row)
    )
    occupation_error = tvd(
        parse_distribution(profile_row.get("occupation_distribution")), prior_occupation_distribution(prior_row)
    )

    error = (
        0.18 * min(1.0, online_error)
        + 0.18 * amount_error
        + 0.14 * volatility_error
        + 0.14 * (1.0 - hour_hit)
        + 0.08 * gender_error
        + 0.09 * age_error
        + 0.09 * education_error
        + 0.10 * occupation_error
    )
    return float(max(0.0, min(1.0, 1.0 - error)))


def build_label_summary(transaction, target_col, label_col):
    grouped = transaction.groupby(target_col)[label_col]
    summary = grouped.agg(label_tx_count="size").reset_index()
    summary["normal_count"] = grouped.apply(lambda s: int((s == 0).sum())).values
    summary["fraud_count"] = grouped.apply(lambda s: int((s == 1).sum())).values
    summary["unlabeled_count"] = grouped.apply(lambda s: int((s == 2).sum())).values
    summary["labeled_count"] = summary["normal_count"] + summary["fraud_count"]
    summary["fraud_rate_labeled"] = np.where(
        summary["labeled_count"] > 0, summary["fraud_count"] / summary["labeled_count"], np.nan
    )
    summary["fraud_rate_all"] = summary["fraud_count"] / summary["label_tx_count"]
    summary["unlabeled_rate"] = summary["unlabeled_count"] / summary["label_tx_count"]
    return summary


def add_residual_features(frame):
    out = frame.copy()
    out["amount_gap"] = out["amount_label"].map(LABEL_RANK) - out["amount_prior"].map(LABEL_RANK)
    out["amount_abs_gap"] = out["amount_gap"].abs()
    out["volatility_gap"] = out["amount_cv_label"].map(LABEL_RANK) - out["volatility_prior"].map(LABEL_RANK)
    out["volatility_abs_gap"] = out["volatility_gap"].abs()
    out["residual_dim_count"] = (
        (pd.to_numeric(out["online_error"], errors="coerce") >= 0.45).astype(int)
        + (out["amount_abs_gap"] >= 2).astype(int)
        + (out["volatility_abs_gap"] >= 2).astype(int)
        + (pd.to_numeric(out["top_hour_prior_hit_rate"], errors="coerce") <= 0.333).astype(int)
        + (pd.to_numeric(out["demographic_tvd"], errors="coerce") >= 0.70).astype(int)
        + (pd.to_numeric(out["occupation_tvd"], errors="coerce") >= 0.90).astype(int)
    )
    return out


def fraud_evidence_mask(frame, args):
    standard = (
        (frame["labeled_count"] >= args.min_labeled_tx)
        & (frame["fraud_count"] >= args.min_fraud_tx)
        & (frame["fraud_rate_labeled"] >= args.min_fraud_rate_labeled)
    )
    high_count = (
        (frame["fraud_count"] >= args.high_fraud_tx)
        & (frame["fraud_rate_labeled"] >= args.high_fraud_rate_labeled)
    )
    return standard | high_count


def residual_mask(frame, args):
    overall = pd.to_numeric(frame["overall_prior_alignment_score"], errors="coerce")
    return (frame["residual_dim_count"] >= args.min_residual_dims) | (overall <= args.overall_score_threshold)


def l1_change_count_limit(target, args):
    by_l1 = target.groupby("behavior_level1")["Target"].nunique().to_dict()
    return {k: max(1, int(np.floor(v * args.max_l1_change_rate))) for k, v in by_l1.items()}


def current_l2_share(target, behavior_level1, behavior_level2):
    group = target[target["behavior_level1"] == behavior_level1]
    if len(group) == 0:
        return 0.0
    return float((group["behavior_level2"] == behavior_level2).sum()) / float(len(group))


def candidate_rows(prior, behavior_level1=None):
    if behavior_level1 is None:
        return prior
    return prior[prior["behavior_level1"] == behavior_level1]


def choose_candidate(profile_row, prior, target, args, allow_cross_l1):
    current_l1 = str(profile_row["behavior_level1"])
    current_l2 = str(profile_row["behavior_level2"])
    current_prior = prior[
        (prior["behavior_level1"] == current_l1) & (prior["behavior_level2"] == current_l2)
    ]
    if current_prior.empty:
        return None
    current_prior = current_prior.iloc[0]
    current_fit = profile_fit_score(profile_row, current_prior)
    current_risk = fraud_risk_score(current_prior)

    rows = candidate_rows(prior, None if allow_cross_l1 else current_l1)
    best = None
    for _, cand in rows.iterrows():
        cand_l1 = str(cand["behavior_level1"])
        cand_l2 = str(cand["behavior_level2"])
        if cand_l1 == current_l1 and cand_l2 == current_l2:
            continue
        if current_l2_share(target, cand_l1, cand_l2) >= args.max_level2_share_per_l1:
            continue

        fit = profile_fit_score(profile_row, cand)
        risk = fraud_risk_score(cand)
        if risk - current_risk < args.min_risk_gain:
            continue
        if cand_l1 == current_l1:
            if fit < current_fit - args.same_l1_fit_tolerance:
                continue
        else:
            if fit < current_fit + args.cross_l1_min_fit_gain or risk < args.cross_l1_min_risk:
                continue

        objective = fit + args.risk_weight * risk
        if best is None or objective > best["objective"]:
            best = {
                "new_behavior_level1": cand_l1,
                "new_behavior_level2": cand_l2,
                "new_fit_score": fit,
                "new_risk_score": risk,
                "objective": objective,
                "old_fit_score": current_fit,
                "old_risk_score": current_risk,
            }
    return best


def explain_change(row, chosen):
    parts = []
    if row["fraud_count"] > 0:
        parts.append(
            "fraud_rate_labeled=%.3f, fraud_count=%d"
            % (as_float(row["fraud_rate_labeled"], 0.0), int(row["fraud_count"]))
        )
    if int(row["residual_dim_count"]) > 0:
        parts.append("residual_dims=%d" % int(row["residual_dim_count"]))
    parts.append(
        "fit %.3f->%.3f, risk %.3f->%.3f"
        % (
            chosen["old_fit_score"],
            chosen["new_fit_score"],
            chosen["old_risk_score"],
            chosen["new_risk_score"],
        )
    )
    return "; ".join(parts)


def main():
    args = parse_args()
    np.random.seed(args.random_seed)
    ensure_dir(args.output_dir)

    merchant_profile = read_csv(os.path.join(args.analysis_dir, "merchant_profile_after.csv"))
    prior = read_csv(os.path.join(args.analysis_dir, "prior_behavior_level2_table.csv"))
    transaction = read_csv(args.transaction_csv)
    label_summary = build_label_summary(transaction, args.target_col, args.label_col)

    profile = merchant_profile.merge(label_summary, on=args.target_col, how="left")
    profile[["normal_count", "fraud_count", "unlabeled_count", "labeled_count"]] = profile[
        ["normal_count", "fraud_count", "unlabeled_count", "labeled_count"]
    ].fillna(0).astype(int)
    for col in ["fraud_rate_labeled", "fraud_rate_all", "unlabeled_rate"]:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").fillna(0.0)
    profile = add_residual_features(profile)

    target = merchant_profile[["Target", "behavior_level1", "behavior_level2"]].copy()
    target_before = target.copy()

    profile["fraud_evidence"] = fraud_evidence_mask(profile, args)
    profile["residual_evidence"] = residual_mask(profile, args)
    candidates = profile[profile["fraud_evidence"] & profile["residual_evidence"]].copy()
    candidates = candidates.sort_values(
        ["fraud_rate_labeled", "fraud_count", "residual_dim_count", "tx_count"],
        ascending=[False, False, False, False],
    )

    l1_limits = l1_change_count_limit(target, args)
    l1_changes = {key: 0 for key in l1_limits}
    audit_rows = []
    changed_targets = set()

    for _, row in candidates.iterrows():
        if len(changed_targets) >= args.max_target_changes:
            break
        target_id = row["Target"]
        old_l1 = str(row["behavior_level1"])
        if l1_changes.get(old_l1, 0) >= l1_limits.get(old_l1, 1):
            continue

        chosen = choose_candidate(row, prior, target, args, allow_cross_l1=False)
        change_scope = "same_level1"
        if chosen is None:
            chosen = choose_candidate(row, prior, target, args, allow_cross_l1=True)
            change_scope = "cross_level1"
        if chosen is None:
            continue

        mask = target["Target"] == target_id
        old_l2 = str(target.loc[mask, "behavior_level2"].iloc[0])
        target.loc[mask, "behavior_level1"] = chosen["new_behavior_level1"]
        target.loc[mask, "behavior_level2"] = chosen["new_behavior_level2"]
        changed_targets.add(target_id)
        l1_changes[old_l1] = l1_changes.get(old_l1, 0) + 1

        audit_rows.append(
            {
                "Target": target_id,
                "change_scope": change_scope,
                "old_behavior_level1": old_l1,
                "old_behavior_level2": old_l2,
                "new_behavior_level1": chosen["new_behavior_level1"],
                "new_behavior_level2": chosen["new_behavior_level2"],
                "tx_count": int(row["tx_count"]),
                "labeled_count": int(row["labeled_count"]),
                "fraud_count": int(row["fraud_count"]),
                "fraud_rate_labeled": round(as_float(row["fraud_rate_labeled"], 0.0), 6),
                "fraud_rate_all": round(as_float(row["fraud_rate_all"], 0.0), 6),
                "residual_dim_count": int(row["residual_dim_count"]),
                "overall_prior_alignment_score": row["overall_prior_alignment_score"],
                "old_fit_score": round(chosen["old_fit_score"], 6),
                "new_fit_score": round(chosen["new_fit_score"], 6),
                "old_risk_score": round(chosen["old_risk_score"], 6),
                "new_risk_score": round(chosen["new_risk_score"], 6),
                "reason": explain_change(row, chosen),
            }
        )

    audit = pd.DataFrame(audit_rows)
    unchanged_candidates = candidates[~candidates["Target"].isin(changed_targets)].copy()

    target_path = os.path.join(args.output_dir, "target_fraud_semantic.csv")
    final_target_path = args.final_target_output
    audit_path = os.path.join(args.output_dir, "merchant_fraud_semantic_changes.csv")
    candidate_path = os.path.join(args.output_dir, "merchant_fraud_semantic_candidates.csv")
    summary_path = os.path.join(args.output_dir, "merchant_fraud_semantic_summary.json")

    target.to_csv(target_path, index=False, encoding="utf-8-sig")
    final_target_parent = os.path.dirname(final_target_path)
    if final_target_parent:
        ensure_dir(final_target_parent)
    target.to_csv(final_target_path, index=False, encoding="utf-8-sig")
    audit.to_csv(audit_path, index=False, encoding="utf-8-sig")
    unchanged_candidates.to_csv(candidate_path, index=False, encoding="utf-8-sig")

    summary = {
        "input_analysis_dir": args.analysis_dir,
        "transaction_csv": args.transaction_csv,
        "target_count": int(len(target)),
        "candidate_count": int(len(candidates)),
        "changed_count": int(len(audit)),
        "same_level1_changes": int((audit["change_scope"] == "same_level1").sum()) if not audit.empty else 0,
        "cross_level1_changes": int((audit["change_scope"] == "cross_level1").sum()) if not audit.empty else 0,
        "max_target_changes": int(args.max_target_changes),
        "max_l1_change_rate": float(args.max_l1_change_rate),
        "changed_level1_counts": l1_changes,
        "original_level1_distribution": target_before["behavior_level1"].value_counts().to_dict(),
        "final_level1_distribution": target["behavior_level1"].value_counts().to_dict(),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    print("Wrote:", target_path)
    print("Wrote final Target table:", final_target_path)
    print("Wrote:", audit_path)
    print("Wrote:", candidate_path)
    print("Wrote:", summary_path)
    print("candidate_count:", summary["candidate_count"])
    print("changed_count:", summary["changed_count"])
    print("same_level1_changes:", summary["same_level1_changes"])
    print("cross_level1_changes:", summary["cross_level1_changes"])


if __name__ == "__main__":
    main()
