#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build post-optimization analysis tables for dataset consistency outputs."""

from __future__ import print_function

import argparse
import json
import os

import numpy as np
import pandas as pd

from consistency_alternating_optimizer import (
    amount_to_label,
    cv_to_label,
    group_distribution_dict,
    label_distance,
    top_hours,
    tvd,
)
from manual_merchant_priors import manual_prior_to_business_prior


AGE_PRIOR_COLS = [
    ("Young Adults", "age_young_adults_prior"),
    ("Middle-aged Adults", "age_middle_aged_adults_prior"),
    ("Older Adults", "age_older_adults_prior"),
]

EDUCATION_PRIOR_COLS = [
    ("Primary School", "education_primary_school_prior"),
    ("Junior Secondary School", "education_junior_secondary_prior"),
    ("Senior Secondary School", "education_senior_secondary_prior"),
    ("Junior College and Above", "education_junior_college_above_prior"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="work", help="Directory containing optimized source/target tables.")
    parser.add_argument(
        "--output_dir",
        default="work/consistency_optimization/analysis/analysis_outputs",
        help="Directory containing analysis outputs from run_dataset_consistency_optimization.py.",
    )
    parser.add_argument("--transaction_csv", default="work/transaction_optimized.csv")
    parser.add_argument("--occupation_col", default="occupation_industry")
    return parser.parse_args()


def dist_dict(df, col):
    if len(df) == 0 or col not in df.columns:
        return {}
    vc = df[col].dropna().astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in vc.items()}


def dominant_from_prior(row, items):
    values = [(name, float(row[col])) for name, col in items]
    return max(values, key=lambda x: x[1])


def parse_hours(value):
    if pd.isna(value):
        return set()
    return {int(v) for v in str(value).split(",") if str(v).strip()}


def pct_label(item):
    if item[0] == "NA":
        return ""
    return "%s(%.1f%%)" % (item[0], item[1] * 100.0)


def dist_to_str(dist):
    if not dist:
        return ""
    items = sorted(dist.items(), key=lambda x: (-x[1], str(x[0])))
    return "|".join("%s:%.4f" % (k, v) for k, v in items)


def read_outputs(dataset_dir, analysis_dir, transaction_csv):
    source = pd.read_csv(os.path.join(dataset_dir, "source_optimized.csv"))
    target = pd.read_csv(os.path.join(dataset_dir, "target_optimized.csv"))
    transaction = pd.read_csv(transaction_csv)
    prior = pd.read_csv(os.path.join(analysis_dir, "prior_behavior_level2_table.csv"))
    alignment = pd.read_csv(os.path.join(analysis_dir, "merchant_online_alignment_before_after.csv"))
    prior_json_path = os.path.join(analysis_dir, "dataset_consistency_priors_2024.json")
    prior_json = {}
    if os.path.exists(prior_json_path):
        with open(prior_json_path, "r", encoding="utf-8") as f:
            prior_json = json.load(f)
    return source, target, transaction, prior, alignment, prior_json


def build_behavior_level2_profile(dataset_dir, analysis_dir, transaction_csv, occupation_col="occupation_industry"):
    source, target, transaction, prior, alignment, prior_json = read_outputs(dataset_dir, analysis_dir, transaction_csv)
    amount_bins = prior_json.get("amount_preference_bins_by_category", {})
    enriched = transaction.merge(source, on="Source", how="left", validate="many_to_one")
    enriched = enriched.merge(target, on="Target", how="left", validate="many_to_one")

    rows = []
    for _, prior_row in prior.sort_values(["behavior_level1", "behavior_level2"]).iterrows():
        category = prior_row["behavior_level1"]
        mcc = prior_row["behavior_level2"]
        tx_group = enriched[(enriched["behavior_level1"] == category) & (enriched["behavior_level2"] == mcc)]
        target_group = target[(target["behavior_level1"] == category) & (target["behavior_level2"] == mcc)]
        merchant_group = alignment[(alignment["behavior_level1"] == category) & (alignment["mcc_after"] == mcc)]

        tx_count = int(len(tx_group))
        merchant_count = int(target_group["Target"].nunique())
        amount_mean = float(tx_group["Amount"].mean()) if tx_count else np.nan
        amount_std = float(tx_group["Amount"].std()) if tx_count > 1 else 0.0
        amount_cv = 0.0 if pd.isna(amount_mean) or amount_mean == 0 else amount_std / amount_mean
        amount_observed = amount_to_label(amount_mean, amount_bins.get(str(category))) if tx_count else "NA"
        volatility_observed = cv_to_label(amount_cv) if tx_count else "NA"

        cross_ratio = float(merchant_group["cross_ratio"].mean()) if not merchant_group.empty else np.nan
        cross_score = float(merchant_group["cross_score"].mean()) if not merchant_group.empty else np.nan
        online_prior = float(prior_row["online_score"])
        online_error = abs(cross_score - online_prior) if not pd.isna(cross_score) else np.nan

        prior_hours = parse_hours(prior_row["active_hours"])
        if tx_count:
            top_hours = tx_group["hour"].dropna().astype(int).value_counts().head(3).index.tolist()
        else:
            top_hours = []
        hour_hit = sum(1 for hour in top_hours if hour in prior_hours) / float(len(top_hours) or 1)

        gender_dist = dist_dict(tx_group, "gender")
        female_ratio = gender_dist.get("Female", 0.0)
        gender_prior = {"Female": float(prior_row["female_ratio_prior"]), "Male": 1.0 - float(prior_row["female_ratio_prior"])}

        age_dist = dist_dict(tx_group, "age_group")
        age_prior = {name: float(prior_row[col]) for name, col in AGE_PRIOR_COLS}
        age_dom_after = max(age_dist.items(), key=lambda x: x[1]) if age_dist else ("NA", np.nan)
        age_dom_prior = dominant_from_prior(prior_row, AGE_PRIOR_COLS)

        edu_dist = dist_dict(tx_group, "education")
        edu_prior = {name: float(prior_row[col]) for name, col in EDUCATION_PRIOR_COLS}
        edu_dom_after = max(edu_dist.items(), key=lambda x: x[1]) if edu_dist else ("NA", np.nan)
        edu_dom_prior = dominant_from_prior(prior_row, EDUCATION_PRIOR_COLS)

        business_prior = manual_prior_to_business_prior(prior_row)
        occupation_dist = dist_dict(tx_group, occupation_col)
        occupation_prior = business_prior.get("occupation_distribution", {})
        occ_dom_after = max(occupation_dist.items(), key=lambda x: x[1]) if occupation_dist else ("NA", np.nan)
        occ_dom_prior = max(occupation_prior.items(), key=lambda x: x[1]) if occupation_prior else ("NA", np.nan)

        demo_parts = [tvd(gender_dist, gender_prior), tvd(age_dist, age_prior), tvd(edu_dist, edu_prior)]
        if occupation_dist and occupation_prior:
            demo_parts.append(tvd(occupation_dist, occupation_prior))
        demographic_tvd = (
            float(np.mean(demo_parts))
            if tx_count
            else np.nan
        )
        amount_error = label_distance(amount_observed, str(prior_row["amount_preference"])) if tx_count else np.nan
        volatility_error = label_distance(volatility_observed, str(prior_row["amount_volatility"])) if tx_count else np.nan
        overall_score = np.nanmean(
            [
                1.0 - min(1.0, online_error) if not pd.isna(online_error) else np.nan,
                1.0 - amount_error if not pd.isna(amount_error) else np.nan,
                1.0 - volatility_error if not pd.isna(volatility_error) else np.nan,
                hour_hit,
                1.0 - demographic_tvd if not pd.isna(demographic_tvd) else np.nan,
            ]
        )

        rows.append(
            {
                "behavior_level1": category,
                "behavior_level2": mcc,
                "merchant_count_after": merchant_count,
                "tx_count_after": tx_count,
                "cross_ratio_after": round(cross_ratio, 4) if not pd.isna(cross_ratio) else "",
                "cross_score_after": round(cross_score, 4) if not pd.isna(cross_score) else "",
                "online_score_prior": round(online_prior, 3),
                "online_error": round(online_error, 4) if not pd.isna(online_error) else "",
                "amount_observed": amount_observed,
                "amount_prior": prior_row["amount_preference"],
                "amount_mean_after": round(amount_mean, 2) if not pd.isna(amount_mean) else "",
                "volatility_observed": volatility_observed,
                "volatility_prior": prior_row["amount_volatility"],
                "top_hours_after": ",".join(str(x) for x in top_hours),
                "active_hours_prior": prior_row["active_hours"],
                "top_hour_prior_hit_rate": round(hour_hit, 3),
                "female_ratio_after": round(female_ratio, 3),
                "female_ratio_prior": round(float(prior_row["female_ratio_prior"]), 3),
                "female_ratio_diff": round(female_ratio - float(prior_row["female_ratio_prior"]), 3),
                "age_dominant_after": pct_label(age_dom_after),
                "age_dominant_prior": pct_label(age_dom_prior),
                "education_dominant_after": pct_label(edu_dom_after),
                "education_dominant_prior": pct_label(edu_dom_prior),
                "occupation_dominant_after": pct_label(occ_dom_after),
                "occupation_dominant_prior": pct_label(occ_dom_prior),
                "occupation_tvd": round(tvd(occupation_dist, occupation_prior), 4) if occupation_dist and occupation_prior else "",
                "demographic_tvd": round(demographic_tvd, 4) if not pd.isna(demographic_tvd) else "",
                "overall_prior_alignment_score": round(float(overall_score), 4) if not pd.isna(overall_score) else "",
                "prior_basis": prior_row["prior_basis"],
            }
        )

    return pd.DataFrame(rows)


def build_merchant_profile_after(dataset_dir, analysis_dir, transaction_csv, occupation_col="occupation_industry"):
    source, target, transaction, prior, alignment, prior_json = read_outputs(
        dataset_dir, analysis_dir, transaction_csv
    )
    amount_bins = prior_json.get("amount_preference_bins_by_category", {})
    prior_lookup = {
        (str(row["behavior_level1"]), str(row["behavior_level2"])): row for _, row in prior.iterrows()
    }
    align_by_target = alignment.set_index("Target", drop=False)

    enriched = transaction.merge(source, on="Source", how="left", validate="many_to_one")
    enriched = enriched.merge(target, on="Target", how="left", validate="many_to_one")
    tx_groups = {target_id: group for target_id, group in enriched.groupby("Target", sort=False)}

    gender_map = group_distribution_dict(enriched, "Target", "gender")
    age_map = group_distribution_dict(enriched, "Target", "age_group")
    edu_map = group_distribution_dict(enriched, "Target", "education")
    occ_map = group_distribution_dict(enriched, "Target", occupation_col)
    hour_map = {
        target_id: top_hours(group["hour"], topn=3)
        for target_id, group in enriched.groupby("Target", sort=False)
        if "hour" in group.columns
    }

    rows = []
    for _, merchant in target.sort_values("Target").iterrows():
        target_id = merchant["Target"]
        category = merchant["behavior_level1"]
        mcc = merchant["behavior_level2"]
        tx_group = tx_groups.get(target_id, enriched.iloc[0:0])
        prior_row = prior_lookup.get((str(category), str(mcc)))

        tx_count = int(len(tx_group))
        amount_sum = float(tx_group["Amount"].sum()) if tx_count else 0.0
        amount_mean = float(tx_group["Amount"].mean()) if tx_count else np.nan
        amount_std = float(tx_group["Amount"].std()) if tx_count > 1 else 0.0
        amount_cv = 0.0 if pd.isna(amount_mean) or amount_mean == 0 else amount_std / amount_mean
        amount_observed = amount_to_label(amount_mean, amount_bins.get(str(category))) if tx_count else "NA"
        volatility_observed = cv_to_label(amount_cv) if tx_count else "NA"

        if target_id in align_by_target.index:
            align_row = align_by_target.loc[target_id]
            cross_ratio = float(align_row["cross_ratio"])
            cross_score = float(align_row["cross_score"])
        else:
            cross_ratio = np.nan
            cross_score = np.nan

        top_hour_list = hour_map.get(target_id, [])
        gender_dist = gender_map.get(str(target_id), {})
        age_dist = age_map.get(str(target_id), {})
        edu_dist = edu_map.get(str(target_id), {})
        occupation_dist = occ_map.get(str(target_id), {})

        if prior_row is None:
            rows.append(
                {
                    "Target": target_id,
                    "behavior_level1": category,
                    "behavior_level2": mcc,
                    "tx_count": tx_count,
                    "amount_sum": round(amount_sum, 2),
                    "amount_mean": round(amount_mean, 2) if not pd.isna(amount_mean) else "",
                    "amount_cv": round(amount_cv, 4) if tx_count else "",
                    "amount_label": amount_observed,
                    "amount_cv_label": volatility_observed,
                    "cross_ratio": round(cross_ratio, 4) if not pd.isna(cross_ratio) else "",
                    "cross_score": round(cross_score, 4) if not pd.isna(cross_score) else "",
                    "top_hours": ",".join(str(x) for x in top_hour_list),
                    "gender_distribution": dist_to_str(gender_dist),
                    "age_distribution": dist_to_str(age_dist),
                    "education_distribution": dist_to_str(edu_dist),
                    "occupation_distribution": dist_to_str(occupation_dist),
                }
            )
            continue

        online_prior = float(prior_row["online_score"])
        online_error = abs(cross_score - online_prior) if not pd.isna(cross_score) else np.nan

        prior_hours = parse_hours(prior_row["active_hours"])
        hour_hit = sum(1 for hour in top_hour_list if hour in prior_hours) / float(len(top_hour_list) or 1)

        female_ratio = gender_dist.get("Female", 0.0)
        gender_prior = {
            "Female": float(prior_row["female_ratio_prior"]),
            "Male": 1.0 - float(prior_row["female_ratio_prior"]),
        }

        age_prior = {name: float(prior_row[col]) for name, col in AGE_PRIOR_COLS}
        age_dom_after = max(age_dist.items(), key=lambda x: x[1]) if age_dist else ("NA", np.nan)
        age_dom_prior = dominant_from_prior(prior_row, AGE_PRIOR_COLS)

        edu_prior = {name: float(prior_row[col]) for name, col in EDUCATION_PRIOR_COLS}
        edu_dom_after = max(edu_dist.items(), key=lambda x: x[1]) if edu_dist else ("NA", np.nan)
        edu_dom_prior = dominant_from_prior(prior_row, EDUCATION_PRIOR_COLS)

        business_prior = manual_prior_to_business_prior(prior_row)
        occupation_prior = business_prior.get("occupation_distribution", {})
        occ_dom_after = max(occupation_dist.items(), key=lambda x: x[1]) if occupation_dist else ("NA", np.nan)
        occ_dom_prior = max(occupation_prior.items(), key=lambda x: x[1]) if occupation_prior else ("NA", np.nan)

        demo_parts = [tvd(gender_dist, gender_prior), tvd(age_dist, age_prior), tvd(edu_dist, edu_prior)]
        if occupation_dist and occupation_prior:
            demo_parts.append(tvd(occupation_dist, occupation_prior))
        demographic_tvd = float(np.mean(demo_parts)) if tx_count else np.nan

        amount_error = label_distance(amount_observed, str(prior_row["amount_preference"])) if tx_count else np.nan
        volatility_error = label_distance(volatility_observed, str(prior_row["amount_volatility"])) if tx_count else np.nan
        overall_score = np.nanmean(
            [
                1.0 - min(1.0, online_error) if not pd.isna(online_error) else np.nan,
                1.0 - amount_error if not pd.isna(amount_error) else np.nan,
                1.0 - volatility_error if not pd.isna(volatility_error) else np.nan,
                hour_hit,
                1.0 - demographic_tvd if not pd.isna(demographic_tvd) else np.nan,
            ]
        )

        rows.append(
            {
                "Target": target_id,
                "behavior_level1": category,
                "behavior_level2": mcc,
                "tx_count": tx_count,
                "amount_sum": round(amount_sum, 2),
                "amount_mean": round(amount_mean, 2) if not pd.isna(amount_mean) else "",
                "amount_cv": round(amount_cv, 4) if tx_count else "",
                "amount_label": amount_observed,
                "amount_cv_label": volatility_observed,
                "cross_ratio": round(cross_ratio, 4) if not pd.isna(cross_ratio) else "",
                "cross_score": round(cross_score, 4) if not pd.isna(cross_score) else "",
                "online_score_prior": round(online_prior, 3),
                "online_error": round(online_error, 4) if not pd.isna(online_error) else "",
                "amount_prior": prior_row["amount_preference"],
                "volatility_prior": prior_row["amount_volatility"],
                "top_hours": ",".join(str(x) for x in top_hour_list),
                "active_hours_prior": prior_row["active_hours"],
                "top_hour_prior_hit_rate": round(hour_hit, 3),
                "female_ratio": round(female_ratio, 3),
                "female_ratio_prior": round(float(prior_row["female_ratio_prior"]), 3),
                "female_ratio_diff": round(female_ratio - float(prior_row["female_ratio_prior"]), 3),
                "age_dominant": pct_label(age_dom_after),
                "age_dominant_prior": pct_label(age_dom_prior),
                "education_dominant": pct_label(edu_dom_after),
                "education_dominant_prior": pct_label(edu_dom_prior),
                "occupation_dominant": pct_label(occ_dom_after),
                "occupation_dominant_prior": pct_label(occ_dom_prior),
                "gender_distribution": dist_to_str(gender_dist),
                "age_distribution": dist_to_str(age_dist),
                "education_distribution": dist_to_str(edu_dist),
                "occupation_distribution": dist_to_str(occupation_dist),
                "occupation_tvd": round(tvd(occupation_dist, occupation_prior), 4)
                if occupation_dist and occupation_prior
                else "",
                "demographic_tvd": round(demographic_tvd, 4) if not pd.isna(demographic_tvd) else "",
                "overall_prior_alignment_score": round(float(overall_score), 4) if not pd.isna(overall_score) else "",
                "prior_basis": prior_row["prior_basis"],
            }
        )

    return pd.DataFrame(rows)


def write_selected_profile(profile, output_dir):
    selected_labels = [
        "Medical Aesthetics",
        "Online Medical Services",
        "Fuel Supply Services",
        "Full-Service Restaurant",
        "Fast Food and Snacks",
        "Hotels, Inns and Homestays",
        "Direct Tourism-Related Services",
        "Air Travel Services",
        "Expressway Toll and Service Areas",
        "Gaming",
        "Apparel, Footwear and Bags",
        "Primary and Secondary Education",
        "Software Development Services",
        "Hair, Beauty and Nail Services",
        "Pet Hospital and Other Pet Services",
    ]
    selected = profile[profile["behavior_level2"].isin(selected_labels)].copy()
    selected.to_csv(
        os.path.join(output_dir, "behavior_level2_profile_after_vs_prior_selected.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def main():
    args = parse_args()
    profile = build_behavior_level2_profile(
        args.dataset_dir, args.output_dir, args.transaction_csv, args.occupation_col
    )
    out_path = os.path.join(args.output_dir, "behavior_level2_profile_after_vs_prior.csv")
    profile.to_csv(out_path, index=False, encoding="utf-8-sig")
    write_selected_profile(profile, args.output_dir)

    merchant_profile = build_merchant_profile_after(
        args.dataset_dir, args.output_dir, args.transaction_csv, args.occupation_col
    )
    merchant_out_path = os.path.join(args.output_dir, "merchant_profile_after.csv")
    merchant_profile.to_csv(merchant_out_path, index=False, encoding="utf-8-sig")

    non_empty = profile[profile["tx_count_after"] > 0].copy()
    print("Wrote:", out_path)
    print("Wrote:", merchant_out_path, "shape:", merchant_profile.shape)
    print("shape:", profile.shape)
    print("non_empty_behavior_level2:", len(non_empty))
    if not non_empty.empty:
        print("amount_match_rate:", round(float((non_empty["amount_observed"] == non_empty["amount_prior"]).mean()), 4))
        print("volatility_match_rate:", round(float((non_empty["volatility_observed"] == non_empty["volatility_prior"]).mean()), 4))
        print("top_hour_hit_rate_mean:", round(float(non_empty["top_hour_prior_hit_rate"].mean()), 4))
        print("online_error_mean:", round(float(non_empty["online_error"].mean()), 4))
        print("demographic_tvd_mean:", round(float(non_empty["demographic_tvd"].mean()), 4))
        if "occupation_tvd" in non_empty.columns:
            occupation_tvd = pd.to_numeric(non_empty["occupation_tvd"], errors="coerce").dropna()
            if not occupation_tvd.empty:
                print("occupation_tvd_mean:", round(float(occupation_tvd.mean()), 4))
        print("overall_prior_alignment_score_mean:", round(float(non_empty["overall_prior_alignment_score"].mean()), 4))


if __name__ == "__main__":
    main()
