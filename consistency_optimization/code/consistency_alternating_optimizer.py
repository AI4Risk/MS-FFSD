#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alternating consistency optimizer for FFSD user and merchant semantics.

This module is intentionally written as a post-initialization optimizer. It
expects the module-3 initialization result to be an enriched transaction table:
each row contains Source/Target ids, amount/time/province fields, user semantic
attributes, and merchant semantic attributes.
"""

from __future__ import print_function

import json
import math
import os
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


UNKNOWN = "__UNKNOWN__"
ALL_PROVINCES = "__ALL_PROVINCES__"


AMOUNT_LEVELS = ["low", "low_to_medium", "medium", "high"]


MCC_KEYWORD_PRIORS_2021 = [
    {
        "name": "online_platform_or_virtual",
        "keywords": [
            "Information Search and Online Forums", "Internet Data Services",
            "Software Development Services", "Online Medical Services",
        ],
        "online_score": 0.92,
        "amount_preference": "medium",
        "active_hours": list(range(9, 24)),
    },
    {
        "name": "apparel_fashion_jewelry",
        "keywords": [
            "Apparel, Footwear and Bags",
        ],
        "online_score": 0.82,
        "amount_preference": "medium",
        "active_hours": list(range(10, 24)),
    },
    {
        "name": "consumer_electronics",
        "keywords": [
            "Appliance, Furniture and Other Repair Services", "Digital Electronics and Home Appliances",
            "Digital and Entertainment Equipment Rental", "Books, Media, Arts and Musical Instruments",
        ],
        "online_score": 0.78,
        "amount_preference": "high",
        "active_hours": list(range(9, 23)),
    },
    {
        "name": "daily_health_beauty",
        "keywords": [
            "Department Store", "Hair, Beauty and Nail Services", "Bathing, Wellness and Health Services",
            "Pharmaceutical Sales", "Nutrition and Health Products", "Health and Assistive Therapy Equipment",
            "Health Examinations and Consultations", "Medical Aesthetics",
        ],
        "online_score": 0.68,
        "amount_preference": "low_to_medium",
        "active_hours": list(range(8, 23)),
    },
    {
        "name": "home_furniture_decoration",
        "keywords": [
            "Hardware and Building Materials", "Home Furnishings and Textiles",
            "Building Decoration and Renovation Services", "Appliance, Furniture and Other Repair Services",
            "Digital Electronics and Home Appliances",
        ],
        "online_score": 0.62,
        "amount_preference": "high",
        "active_hours": list(range(9, 22)),
    },
    {
        "name": "restaurant_food_delivery",
        "keywords": [
            "Specialty Food Retail", "Nightlife Bar and Dining", "Banquet and Large-Scale Catering",
        ],
        "online_score": 0.45,
        "amount_preference": "low_to_medium",
        "active_hours": [7, 8, 11, 12, 13, 17, 18, 19, 20, 21],
    },
    {
        "name": "travel_hotel_transport",
        "keywords": [
            "Urban Public Transport", "Long-Distance Ground Travel", "Air Travel Services",
            "Water Sightseeing Transport", "Hotels, Inns and Homestays",
            "Travel Agencies and Tourism Services", "Direct Tourism-Related Services",
        ],
        "online_score": 0.60,
        "amount_preference": "high",
        "active_hours": list(range(6, 23)),
    },
    {
        "name": "local_offline_service",
        "keywords": [
            "Convenience Store", "Supermarket and Hypermarket", "Appliance, Furniture and Other Repair Services",
            "Designated Driving and Parking Services", "Vehicle Supplies, Repair and Maintenance",
        ],
        "online_score": 0.18,
        "amount_preference": "low_to_medium",
        "active_hours": list(range(6, 23)),
    },
]


@dataclass
class ConsistencyOptimizerConfig:
    source_col: str = "Source"
    target_col: str = "Target"
    amount_col: str = "Amount"
    timestamp_col: str = "datetime"
    hour_col: str = "hour"
    source_province_col: str = "source_province"
    transaction_province_col: str = "transaction_province"
    consumption_category_col: str = "consumption_category"
    mcc_level1_col: str = "MCC_level1"
    gender_col: str = "gender"
    age_col: str = "age_group"
    education_col: str = "education"
    occupation_col: str = "occupation"

    max_iterations: int = 8
    random_seed: int = 42

    merchant_round_budget: float = 0.20
    merchant_total_budget: float = 1.00
    user_round_budget: float = 0.20
    user_total_budget: float = 1.00
    user_field_round_budget: float = 0.20
    mcc_net_flow_cap_rate: float = 0.12
    merchant_category_min_changes_per_round: int = 3

    merchant_min_gain: float = 0.015
    merchant_coverage_min_gain: float = -10.0
    user_min_gain: float = -0.10
    require_positive_gain: bool = True
    merchant_change_penalty: float = 0.03
    user_change_penalty: float = 0.0
    user_population_score_weight: float = 3.00
    user_mcc_score_weight: float = 1.20
    user_amount_score_weight: float = 0.20
    user_time_score_weight: float = 0.15
    user_constraint_score_weight: float = 0.50

    soft_constraint_strength: float = 0.6
    soft_weight_min: float = 0.25
    soft_weight_max: float = 4.0

    epsilon_count: float = 0.01
    epsilon_amount: float = 0.015
    epsilon_mcc1: float = 0.06
    epsilon_age: float = 0.08
    epsilon_education: float = 0.08
    epsilon_occupation: float = 0.10
    mcc_max_share_per_category: float = 0.65
    mcc_max_share_slack: float = 0.20
    mcc_min_active_level2_ratio: float = 0.60
    mcc_category_change_rate_cap: float = 0.65

    merchant_stop_threshold: float = 0.002
    user_stop_threshold: float = 0.001
    min_accepted_changes: int = 3
    gain_stop_patience: int = 2
    gain_threshold: float = 0.05
    rejection_stop_threshold: float = 0.95

    max_candidate_mcc_per_merchant: int = 18
    max_candidate_values_per_user_field: int = 8
    min_user_tx_for_behavior: int = 5
    min_user_merchants_for_category: int = 3
    min_merchant_tx_for_behavior: int = 5
    min_merchant_users_for_customer: int = 3
    max_user_candidates_per_round: int = 60000
    enable_response_budget: bool = True
    response_affected_entity_rate: float = 0.20


@dataclass
class ChangeRecord:
    kind: str
    entity_id: str
    field: str
    old_value: str
    new_value: str
    gain: float
    iteration: int
    mode: str = "single"
    province: str = ALL_PROVINCES


def read_json(path):
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def safe_str(value):
    if pd.isna(value):
        return UNKNOWN
    return str(value)


def safe_div(num, den, default=0.0):
    den = float(den)
    if den == 0.0:
        return default
    return float(num) / den


STOP_REASON_EXPLANATIONS = {
    "max_iterations": "Reached max_iterations.",
    "change_rate_below_threshold": "Both merchant and user change rates fell below their thresholds.",
    "too_few_accepted_changes": "The round accepted too few changes.",
    "accepted_gain_decay": "Mean accepted gain stayed below the threshold for the configured patience.",
}


def explain_stop_reason(reason):
    return STOP_REASON_EXPLANATIONS.get(reason, "Another configured stop condition was reached.")


def print_round_progress(round_report, accepted_total, accepted_gain_mean):
    iteration = round_report["iteration"]
    user = round_report["user"]
    merchant = round_report["merchant"]
    print(
        "Round %d: user %d/%d accepted, merchant %d/%d accepted, total %d, mean gain %.4f"
        % (
            iteration,
            user["accepted"],
            user["proposed"],
            merchant["accepted"],
            merchant["proposed"],
            accepted_total,
            accepted_gain_mean,
        )
    )


def print_optimization_stop_summary(report):
    reason = report.get("stop_reason", "")
    summary = report.get("summary", {})
    rounds = report.get("rounds", [])
    print("")
    print("Optimization stopped.")
    print("Stop reason: %s - %s" % (reason, explain_stop_reason(reason)))
    print("Rounds completed: %d" % len(rounds))
    print(
        "Cumulative changes: users=%d, merchants=%d, records=%d"
        % (
            summary.get("user_changed_count", 0),
            summary.get("merchant_changed_count", 0),
            summary.get("total_change_records", 0),
        )
    )


def amount_to_label(value, bins=None):
    try:
        x = float(value)
    except Exception:
        return "medium"
    if bins:
        try:
            q25 = float(bins["q25"])
            q50 = float(bins["q50"])
            q75 = float(bins["q75"])
            if x < q25:
                return "low"
            if x < q50:
                return "low_to_medium"
            if x < q75:
                return "medium"
            return "high"
        except Exception:
            pass
    if x < 20:
        return "low"
    if x < 100:
        return "low_to_medium"
    if x < 500:
        return "medium"
    return "high"


def cv_to_label(value):
    try:
        x = float(value)
    except Exception:
        return "medium"
    if x < 0.5:
        return "low"
    if x < 1.0:
        return "low_to_medium"
    if x < 2.0:
        return "medium"
    return "high"


def label_distance(a, b):
    if a not in AMOUNT_LEVELS or b not in AMOUNT_LEVELS:
        return 0.5
    return abs(AMOUNT_LEVELS.index(a) - AMOUNT_LEVELS.index(b)) / 3.0


def value_distribution(series):
    s = series.dropna().astype(str)
    if len(s) == 0:
        return {}
    counts = s.value_counts()
    total = float(counts.sum())
    return {str(k): float(v) / total for k, v in counts.items()}


def normalize_dict(values):
    clean = {safe_str(k): max(float(v), 0.0) for k, v in values.items()}
    total = sum(clean.values())
    if total <= 0:
        n = max(len(clean), 1)
        return {k: 1.0 / n for k in clean}
    return {k: v / total for k, v in clean.items()}


def tvd(p, q):
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(float(p.get(k, 0.0)) - float(q.get(k, 0.0))) for k in keys)


def distribution_from_counts(counts):
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {safe_str(k): float(v) / total for k, v in counts.items()}


def top_hours(series, topn=3):
    s = series.dropna()
    if len(s) == 0:
        return []
    values = s.astype(int).value_counts().head(topn).index.tolist()
    return [int(v) for v in values]


def jaccard(a, b):
    set_a = set(a or [])
    set_b = set(b or [])
    if not set_a or not set_b:
        return 0.5
    return float(len(set_a & set_b)) / float(len(set_a | set_b))


def dist_to_long(df, group_col, value_col):
    if value_col not in df.columns:
        return pd.DataFrame(columns=[group_col, value_col, "prob"])
    tmp = df[[group_col, value_col]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=[group_col, value_col, "prob"])
    tmp[value_col] = tmp[value_col].astype(str)
    counts = tmp.groupby([group_col, value_col]).size().rename("count").reset_index()
    totals = counts.groupby(group_col)["count"].transform("sum")
    counts["prob"] = counts["count"] / totals
    return counts[[group_col, value_col, "prob"]]


def group_distribution_dict(df, group_col, value_col):
    long_df = dist_to_long(df, group_col, value_col)
    if long_df.empty:
        return {}
    out = {}
    for group, part in long_df.groupby(group_col, sort=False):
        out[safe_str(group)] = dict(zip(part[value_col].astype(str), part["prob"].astype(float)))
    return out


def infer_business_prior(mcc_value, category_value, external_priors=None):
    external_priors = external_priors or {}
    custom = external_priors.get("mcc_business_priors", {})
    category_custom = external_priors.get("consumption_category_business_priors", {})
    direct_keys = [
        safe_str(mcc_value),
        "%s||%s" % (safe_str(category_value), safe_str(mcc_value)),
    ]
    for key in direct_keys:
        if key in custom:
            prior = dict(custom[key])
            prior.setdefault("name", "external_%s" % key)
            prior.setdefault("keywords", [])
            prior.setdefault("online_score", 0.50)
            prior.setdefault("amount_preference", "medium")
            prior.setdefault("amount_volatility", "medium")
            prior.setdefault("active_hours", list(range(8, 23)))
            return prior
    if safe_str(category_value) in category_custom:
        prior = dict(category_custom[safe_str(category_value)])
        prior.setdefault("name", "external_%s" % safe_str(category_value))
        prior.setdefault("keywords", [])
        prior.setdefault("online_score", 0.50)
        prior.setdefault("amount_preference", "medium")
        prior.setdefault("amount_volatility", "medium")
        prior.setdefault("active_hours", list(range(8, 23)))
        return prior

    text = ("%s %s" % (safe_str(category_value), safe_str(mcc_value))).lower()
    best = None
    for item in MCC_KEYWORD_PRIORS_2021:
        for kw in item["keywords"]:
            if kw.lower() in text:
                best = item
                break
        if best is not None:
            break
    if best is None:
        return {
            "name": "generic_mixed",
            "online_score": 0.50,
            "amount_preference": "medium",
            "amount_volatility": "medium",
            "active_hours": list(range(8, 23)),
        }
    prior = dict(best)
    prior.setdefault("amount_volatility", "medium")
    return prior


def quantile_normalize(value, low, high):
    if value is None or pd.isna(value):
        return None
    if high <= low:
        return 0.0 if float(value) <= low else 1.0
    return float(np.clip((float(value) - float(low)) / (float(high) - float(low)), 0.0, 1.0))


class ConstraintController(object):
    """Soft during rounds, hard only at the final validation/repair stage."""

    def __init__(self, initial_df, config, external_priors=None, merchant_table=None, user_table=None):
        self.config = config
        self.external_priors = external_priors or {}
        self.merchant_table = merchant_table if merchant_table is not None else self._merchant_entity_frame(initial_df)
        self.user_table = user_table if user_table is not None else self._user_entity_frame(initial_df)
        c = self.config
        if not self.merchant_table.empty and c.consumption_category_col in self.merchant_table.columns:
            self._cached_consumption_count = value_distribution(self.merchant_table[c.consumption_category_col])
        else:
            self._cached_consumption_count = {}
        if c.consumption_category_col in initial_df.columns and c.amount_col in initial_df.columns:
            amount = initial_df.groupby(c.consumption_category_col, sort=False)[c.amount_col].sum()
            self._cached_consumption_amount = normalize_dict(amount.to_dict())
        else:
            self._cached_consumption_amount = {}
        self.initial = self._snapshot(initial_df)
        self.initial_target_to_mcc = {}
        self.initial_target_to_category = {}
        self.initial_mcc_values_by_category = defaultdict(set)
        merchants = self.merchant_table
        c = self.config
        if not merchants.empty and c.mcc_level1_col in merchants.columns and c.consumption_category_col in merchants.columns:
            tmp = merchants[[c.target_col, c.consumption_category_col, c.mcc_level1_col]].copy()
            tmp[c.target_col] = tmp[c.target_col].astype(str)
            tmp[c.consumption_category_col] = tmp[c.consumption_category_col].astype(str)
            tmp[c.mcc_level1_col] = tmp[c.mcc_level1_col].astype(str)
            self.initial_target_to_mcc = dict(zip(tmp[c.target_col], tmp[c.mcc_level1_col]))
            self.initial_target_to_category = dict(zip(tmp[c.target_col], tmp[c.consumption_category_col]))
            for category, part in tmp.groupby(c.consumption_category_col):
                self.initial_mcc_values_by_category[safe_str(category)] = set(part[c.mcc_level1_col].astype(str))

    def _merchant_entity_frame(self, df):
        c = self.config
        cols = [x for x in [c.target_col, c.consumption_category_col, c.mcc_level1_col] if x in df.columns]
        if c.target_col not in cols:
            return pd.DataFrame()
        return df[cols].drop_duplicates(c.target_col)

    def _user_entity_frame(self, df):
        c = self.config
        cols = [c.source_col]
        if c.source_province_col in df.columns:
            cols.append(c.source_province_col)
        for col in [c.age_col, c.education_col, c.occupation_col]:
            if col in df.columns:
                cols.append(col)
        return df[cols].drop_duplicates(c.source_col)

    def _snapshot(self, df):
        c = self.config
        snap = {
            "consumption_count": dict(self._cached_consumption_count),
            "consumption_amount": dict(self._cached_consumption_amount),
            "mcc": {},
            "user": {},
        }

        merchants = self.merchant_table
        users = self.user_table
        if not merchants.empty and c.mcc_level1_col in merchants.columns:
            snap["mcc"] = value_distribution(merchants[c.mcc_level1_col])

        if not users.empty:
            for field_name, col in self.user_fields():
                if col not in users.columns:
                    continue
                snap["user"][field_name] = {}
                snap["user"][field_name][ALL_PROVINCES] = value_distribution(users[col])
        return snap

    def user_fields(self):
        c = self.config
        return [("age", c.age_col), ("education", c.education_col), ("occupation", c.occupation_col)]

    def target_distribution(self, kind, province=None, field=None):
        priors = self.external_priors
        if kind == "consumption_count":
            return normalize_dict(priors.get("consumption_category_count", self.initial["consumption_count"]))
        if kind == "consumption_amount":
            return normalize_dict(priors.get("consumption_category_amount", self.initial["consumption_amount"]))
        if kind == "mcc":
            return normalize_dict(priors.get("mcc_level1", self.initial["mcc"]))
        if kind == "user":
            national = priors.get("user_attribute_national", {})
            if field in national:
                return normalize_dict(national[field])
            by_province = priors.get("user_attribute_by_province", {})
            if province in by_province and field in by_province[province]:
                return normalize_dict(by_province[province][field])
            if "__default__" in by_province and field in by_province["__default__"]:
                return normalize_dict(by_province["__default__"][field])
            return self.initial["user"].get(field, {}).get(province, {})
        return {}

    def _weights_from_current(self, current, target):
        c = self.config
        keys = set(current.keys()) | set(target.keys())
        if not keys:
            return {}
        n = float(len(keys))
        out = {}
        for key in keys:
            cur = float(current.get(key, 0.0))
            tgt = float(target.get(key, 0.0))
            base = max(tgt, 1.0 / max(n, 1.0))
            delta = (tgt - cur) / base
            weight = 1.0 + c.soft_constraint_strength * delta
            out[key] = float(np.clip(weight, c.soft_weight_min, c.soft_weight_max))
        return out

    def _mcc_structure_state(self, df):
        c = self.config
        merchants = self.merchant_table
        empty = {
            "total_changed_rate": 0.0,
            "total_changed_count": 0,
            "total_target_count": 0,
            "category": {},
        }
        required = [c.target_col, c.consumption_category_col, c.mcc_level1_col]
        if merchants.empty or any(col not in merchants.columns for col in required):
            return empty

        tmp = merchants[required].copy()
        tmp[c.target_col] = tmp[c.target_col].astype(str)
        tmp[c.consumption_category_col] = tmp[c.consumption_category_col].astype(str)
        tmp[c.mcc_level1_col] = tmp[c.mcc_level1_col].astype(str)
        tmp["_initial_mcc"] = tmp[c.target_col].map(self.initial_target_to_mcc)
        tmp["_changed"] = tmp["_initial_mcc"].notna() & (tmp[c.mcc_level1_col] != tmp["_initial_mcc"])

        total_count = int(len(tmp))
        total_changed = int(tmp["_changed"].sum())
        state = {
            "total_changed_rate": safe_div(total_changed, total_count, 0.0),
            "total_changed_count": total_changed,
            "total_target_count": total_count,
            "category": {},
        }

        for category, part in tmp.groupby(c.consumption_category_col):
            category = safe_str(category)
            counts = part[c.mcc_level1_col].value_counts()
            target_count = int(counts.sum())
            if target_count <= 0:
                continue
            current_values = set(counts.index.astype(str))
            initial_values = set(self.initial_mcc_values_by_category.get(category, set()))
            reference_values = current_values | initial_values
            initial_unique = max(len(initial_values), len(reference_values), 1)
            active_count = int(len(current_values))
            min_active = int(math.ceil(initial_unique * c.mcc_min_active_level2_ratio))
            min_active = max(1, min(min_active, initial_unique))
            max_share = float(counts.max()) / float(target_count)
            uniform_share = 1.0 / float(initial_unique)
            max_share_limit = max(c.mcc_max_share_per_category, uniform_share + c.mcc_max_share_slack)
            max_share_limit = min(max_share_limit, 1.0)
            changed_count = int(part["_changed"].sum())
            state["category"][category] = {
                "target_count": target_count,
                "changed_count": changed_count,
                "change_rate": safe_div(changed_count, target_count, 0.0),
                "active_level2_count": active_count,
                "initial_level2_count": initial_unique,
                "min_active_level2_count": min_active,
                "max_level2_share": max_share,
                "max_level2_share_limit": max_share_limit,
                "dominant_level2": safe_str(counts.idxmax()),
            }
        return state

    def _mcc_structural_weights(self, df):
        c = self.config
        merchants = self.merchant_table
        required = [c.consumption_category_col, c.mcc_level1_col]
        if merchants.empty or any(col not in merchants.columns for col in required):
            return {}
        tmp = merchants[required].copy()
        tmp[c.consumption_category_col] = tmp[c.consumption_category_col].astype(str)
        tmp[c.mcc_level1_col] = tmp[c.mcc_level1_col].astype(str)
        weights = defaultdict(list)
        for category, part in tmp.groupby(c.consumption_category_col):
            category = safe_str(category)
            total = float(len(part))
            if total <= 0:
                continue
            counts = part[c.mcc_level1_col].value_counts().to_dict()
            values = set(counts.keys()) | set(self.initial_mcc_values_by_category.get(category, set()))
            if not values:
                continue
            expected = 1.0 / float(len(values))
            for value in values:
                cur = float(counts.get(value, 0.0)) / total
                delta = (expected - cur) / max(expected, 1e-6)
                weight = 1.0 + c.soft_constraint_strength * delta
                weights[safe_str(value)].append(float(np.clip(weight, c.soft_weight_min, c.soft_weight_max)))
        return {key: float(np.mean(vals)) for key, vals in weights.items() if vals}

    def evaluate(self, df):
        current = self._snapshot(df)
        c = self.config
        errors = {}
        weights = {"mcc": {}, "user": defaultdict(dict), "consumption": {}}

        target_count = self.target_distribution("consumption_count")
        target_amount = self.target_distribution("consumption_amount")
        errors["consumption_count_tvd"] = tvd(current["consumption_count"], target_count)
        errors["consumption_amount_tvd"] = tvd(current["consumption_amount"], target_amount)
        weights["consumption"] = self._weights_from_current(current["consumption_count"], target_count)

        target_mcc = self.target_distribution("mcc")
        errors["mcc_tvd"] = tvd(current["mcc"], target_mcc)
        errors["mcc_structure"] = self._mcc_structure_state(df)
        weights["mcc"] = self._mcc_structural_weights(df)

        user_errors = {}
        for field_name, _ in self.user_fields():
            user_errors[field_name] = {}
            for province, cur_dist in current["user"].get(field_name, {}).items():
                target = self.target_distribution("user", ALL_PROVINCES, field_name)
                user_errors[field_name][province] = tvd(cur_dist, target)
                weights["user"][(field_name, province)] = self._weights_from_current(cur_dist, target)
        errors["user_tvd"] = user_errors

        return {"current": current, "errors": errors, "weights": weights}

    def final_validate(self, df):
        state = self.evaluate(df)
        e = state["errors"]
        c = self.config
        violations = []
        if e["consumption_count_tvd"] > tvd(self.initial["consumption_count"], self.target_distribution("consumption_count")) + c.epsilon_count:
            violations.append({"kind": "consumption_count", "value": e["consumption_count_tvd"]})
        if e["consumption_amount_tvd"] > tvd(self.initial["consumption_amount"], self.target_distribution("consumption_amount")) + c.epsilon_amount:
            violations.append({"kind": "consumption_amount", "value": e["consumption_amount_tvd"]})
        mcc_structure = e.get("mcc_structure", {})
        total_change_rate = float(mcc_structure.get("total_changed_rate", 0.0))
        if total_change_rate > c.merchant_total_budget:
            violations.append(
                {
                    "kind": "mcc",
                    "metric": "total_change_rate",
                    "value": total_change_rate,
                    "limit": c.merchant_total_budget,
                }
            )
        for category, item in mcc_structure.get("category", {}).items():
            if float(item.get("change_rate", 0.0)) > c.mcc_category_change_rate_cap:
                violations.append(
                    {
                        "kind": "mcc",
                        "metric": "category_change_rate",
                        "category": category,
                        "value": item.get("change_rate", 0.0),
                        "limit": c.mcc_category_change_rate_cap,
                    }
                )
            if int(item.get("active_level2_count", 0)) < int(item.get("min_active_level2_count", 1)):
                violations.append(
                    {
                        "kind": "mcc",
                        "metric": "active_level2_count",
                        "category": category,
                        "value": item.get("active_level2_count", 0),
                        "limit": item.get("min_active_level2_count", 1),
                    }
                )
            if float(item.get("max_level2_share", 0.0)) > float(item.get("max_level2_share_limit", 1.0)):
                violations.append(
                    {
                        "kind": "mcc",
                        "metric": "max_level2_share",
                        "category": category,
                        "value": item.get("max_level2_share", 0.0),
                        "limit": item.get("max_level2_share_limit", 1.0),
                        "dominant_level2": item.get("dominant_level2", UNKNOWN),
                    }
                )
        eps = {"age": c.epsilon_age, "education": c.epsilon_education, "occupation": c.epsilon_occupation}
        for field_name, by_province in e["user_tvd"].items():
            for province, value in by_province.items():
                if value > eps[field_name]:
                    violations.append(
                        {
                            "kind": "user",
                            "field": field_name,
                            "scope": ALL_PROVINCES,
                            "value": value,
                            "limit": eps[field_name],
                        }
                    )
        return {"passed": len(violations) == 0, "violations": violations, "state": state}


class AlternatingConsistencyOptimizer(object):
    def __init__(self, config=None, external_priors=None):
        self.config = config or ConsistencyOptimizerConfig()
        self.external_priors = external_priors or {}
        self.rng = np.random.RandomState(self.config.random_seed)
        self.merchant_last_change = {}
        self.user_last_change = {}
        self.changed_merchants = set()
        self.changed_users = set()
        self.first_changed_merchants = set()
        self.first_changed_users = set()
        # Primary-stage budgets count field/category change events. The unique
        # entity sets above remain for coverage reporting.
        self.primary_merchant_change_records = 0
        self.primary_user_change_records = 0
        self.response_merchant_records = 0
        self.response_user_records = 0
        self.change_history = []
        self.warnings = []

    def _existing_cols(self, df, cols):
        return [c for c in cols if c in df.columns]

    def _build_entity_tables(self, df):
        c = self.config
        merchant_cols = [x for x in [c.target_col, c.consumption_category_col, c.mcc_level1_col] if x in df.columns]
        merchant_table = df[merchant_cols].drop_duplicates(c.target_col, keep="first").copy()
        merchant_table[c.target_col] = merchant_table[c.target_col].astype(str)
        if c.consumption_category_col in merchant_table.columns:
            merchant_table[c.consumption_category_col] = merchant_table[c.consumption_category_col].astype(str)
        if c.mcc_level1_col in merchant_table.columns:
            merchant_table[c.mcc_level1_col] = merchant_table[c.mcc_level1_col].astype(str)

        user_cols = [c.source_col]
        if c.source_province_col in df.columns:
            user_cols.append(c.source_province_col)
        for _, col in [("age", c.age_col), ("education", c.education_col), ("occupation", c.occupation_col)]:
            if col in df.columns:
                user_cols.append(col)
        user_table = df[user_cols].drop_duplicates(c.source_col, keep="first").copy()
        user_table[c.source_col] = user_table[c.source_col].astype(str)
        return merchant_table, user_table

    def _prepare_working_frame(self, df):
        c = self.config
        df = df.copy()
        df[c.source_col] = df[c.source_col].astype(str)
        df[c.target_col] = df[c.target_col].astype(str)
        return df

    def _component_availability(self, df):
        c = self.config
        merchant = all(col in df.columns for col in [c.target_col, c.consumption_category_col, c.mcc_level1_col])
        user_fields = []
        for name, col in [("age", c.age_col), ("education", c.education_col), ("occupation", c.occupation_col)]:
            if col in df.columns:
                user_fields.append((name, col))
        if c.source_province_col not in df.columns:
            self.warnings.append("source_province_col is missing; user constraints fall back to global distribution.")
        if c.transaction_province_col not in df.columns:
            self.warnings.append("transaction_province_col is missing; cross-province score uses a neutral value.")
        if c.hour_col not in df.columns and c.timestamp_col not in df.columns:
            self.warnings.append("hour/timestamp columns are missing; time consistency score uses a neutral value.")
        return merchant, user_fields

    def _merchant_demographic_fields(self, df, user_fields):
        """Fields used to score merchant-user population consistency.

        These are not all optimizable user fields. For example, gender can be a
        merchant-profile signal while remaining frozen on the user side.
        """
        c = self.config
        fields = []
        if c.gender_col in df.columns:
            fields.append(("gender", c.gender_col))
        seen = {col for _, col in fields}
        for field_name, col in user_fields:
            if col not in seen:
                fields.append((field_name, col))
                seen.add(col)
        return fields

    def _ensure_hour_col(self, df):
        c = self.config
        if c.hour_col in df.columns:
            return df, c.hour_col
        if c.timestamp_col in df.columns:
            tmp_col = "_optimizer_hour"
            parsed = pd.to_datetime(df[c.timestamp_col], errors="coerce")
            df[tmp_col] = parsed.dt.hour
            return df, tmp_col
        return df, None

    def _frozen_snapshot(self, df):
        c = self.config
        cols = self._existing_cols(
            df,
            [c.amount_col, c.timestamp_col, c.hour_col, c.source_province_col, c.transaction_province_col],
        )
        return cols, df[cols].copy() if cols else pd.DataFrame(index=df.index)

    def _frozen_validation(self, df, cols, snapshot):
        problems = []
        for col in cols:
            if col not in df.columns:
                problems.append("%s was removed" % col)
            elif not df[col].equals(snapshot[col]):
                problems.append("%s changed" % col)
        return problems

    def _cross_ratio_by_group(self, df, group_col):
        c = self.config
        if c.source_province_col not in df.columns or c.transaction_province_col not in df.columns:
            return {}
        cross = df[c.source_province_col].astype(str) != df[c.transaction_province_col].astype(str)
        tmp = pd.DataFrame({group_col: df[group_col], "_cross": cross.astype(float)})
        return tmp.groupby(group_col)["_cross"].mean().to_dict()

    def _build_mcc_priors(self, df, hour_col, demographic_fields):
        c = self.config
        priors = {
            "amount_label": {},
            "cross_ratio": {},
            "cross_score": {},
            "top_hours": {},
            "attr_dist": defaultdict(dict),
            "global_attr_dist": {},
            "mcc_dist": value_distribution(df[c.mcc_level1_col]) if c.mcc_level1_col in df.columns else {},
        }
        if c.mcc_level1_col not in df.columns:
            return priors
        amount_mean = df.groupby(c.mcc_level1_col, sort=False)[c.amount_col].mean()
        priors["amount_label"] = {safe_str(k): amount_to_label(v) for k, v in amount_mean.items()}
        if c.consumption_category_col in df.columns:
            amount_by_cat_mcc = df.groupby([c.consumption_category_col, c.mcc_level1_col], sort=False)[c.amount_col].mean()
            for (category, mcc), value in amount_by_cat_mcc.items():
                category_key = safe_str(category)
                mcc_key = safe_str(mcc)
                priors["amount_label"]["%s||%s" % (category_key, mcc_key)] = self._amount_label(
                    value, category_key
                )
        priors["cross_ratio"] = self._cross_ratio_by_group(df, c.mcc_level1_col)
        if priors["cross_ratio"]:
            vals = np.asarray(list(priors["cross_ratio"].values()), dtype=float)
            low = float(np.quantile(vals, 0.10))
            high = float(np.quantile(vals, 0.95))
            priors["cross_score"] = {
                safe_str(k): quantile_normalize(v, low, high) for k, v in priors["cross_ratio"].items()
            }
        if hour_col is not None:
            priors["top_hours"] = {
                safe_str(k): top_hours(v, topn=5) for k, v in df.groupby(c.mcc_level1_col, sort=False)[hour_col]
            }
        for field_name, col in demographic_fields:
            priors["attr_dist"][field_name] = group_distribution_dict(df, c.mcc_level1_col, col)
            priors["global_attr_dist"][field_name] = value_distribution(df[col])
        return priors

    def _amount_bins_for_category(self, category):
        bins = self.external_priors.get("amount_preference_bins_by_category", {})
        return bins.get(safe_str(category))

    def _amount_label(self, value, category=None):
        return amount_to_label(value, self._amount_bins_for_category(category))

    def _build_merchant_profiles(self, df, hour_col, demographic_fields):
        c = self.config
        profiles = {}
        base = df.groupby(c.target_col, sort=False).agg(
            tx_count=(c.amount_col, "size"),
            user_count=(c.source_col, "nunique"),
            amount_mean=(c.amount_col, "mean"),
            amount_std=(c.amount_col, "std"),
            category=(c.consumption_category_col, "first"),
        )
        cross = self._cross_ratio_by_group(df, c.target_col)
        if cross:
            cross_vals = np.asarray(list(cross.values()), dtype=float)
            cross_low = float(np.quantile(cross_vals, 0.10))
            cross_high = float(np.quantile(cross_vals, 0.95))
        else:
            cross_low = 0.0
            cross_high = 1.0
        hour_map = {}
        if hour_col is not None:
            hour_map = {safe_str(k): top_hours(v, topn=3) for k, v in df.groupby(c.target_col, sort=False)[hour_col]}
        attr_maps = {}
        for field_name, col in demographic_fields:
            attr_maps[field_name] = group_distribution_dict(df, c.target_col, col)
        for row in base.itertuples():
            target_key = safe_str(row.Index)
            mean = row.amount_mean
            std = 0.0 if pd.isna(row.amount_std) else float(row.amount_std)
            cv = safe_div(std, mean, 0.0)
            category = safe_str(row.category)
            profiles[target_key] = {
                "tx_count": int(row.tx_count),
                "user_count": int(row.user_count),
                "amount_label": self._amount_label(mean, category),
                "amount_cv_label": cv_to_label(cv),
                "cross_ratio": cross.get(target_key, None),
                "cross_score": quantile_normalize(cross.get(target_key, None), cross_low, cross_high),
                "top_hours": hour_map.get(target_key, []),
                "attr_dist": {field_name: attr_maps[field_name].get(target_key, {}) for field_name, _ in demographic_fields},
            }
        return profiles

    def _merchant_score(self, profile, mcc_value, category_value, mcc_priors, weights):
        c = self.config
        business = infer_business_prior(mcc_value, category_value, self.external_priors)
        weight = max(float(weights.get(mcc_value, 1.0)), 1e-6)
        soft_bonus = math.log(weight)
        score = 0.35 * soft_bonus

        if profile["tx_count"] >= c.min_merchant_tx_for_behavior:
            cross_score = profile.get("cross_score")
            if cross_score is None:
                online_match = 0.5
            else:
                online_match = 1.0 - abs(float(cross_score) - float(business["online_score"]))
            amount_ref = business.get("amount_preference") or mcc_priors["amount_label"].get(
                "%s||%s" % (safe_str(category_value), safe_str(mcc_value)),
                mcc_priors["amount_label"].get(mcc_value, "medium"),
            )
            amount_match = 1.0 - label_distance(profile["amount_label"], amount_ref)
            volatility_ref = business.get("amount_volatility", "medium")
            volatility_match = 1.0 - label_distance(profile.get("amount_cv_label", "medium"), volatility_ref)
            candidate_hours = business.get("active_hours") or mcc_priors["top_hours"].get(mcc_value, list(range(8, 23)))
            time_match = jaccard(profile.get("top_hours", []), candidate_hours)
            score += 2.00 * online_match + 0.80 * amount_match + 0.30 * volatility_match + 0.35 * time_match

        if profile["user_count"] >= c.min_merchant_users_for_customer:
            demo_scores = []
            for field_name, dist in profile.get("attr_dist", {}).items():
                ref = business.get("%s_distribution" % field_name)
                if not ref:
                    ref = mcc_priors["attr_dist"].get(field_name, {}).get(mcc_value)
                if not ref:
                    ref = mcc_priors["global_attr_dist"].get(field_name, {})
                if ref:
                    demo_scores.append(1.0 - tvd(dist, ref))
            if demo_scores:
                score += 1.00 * float(np.mean(demo_scores))

        return score

    def _propose_merchant_changes(
        self,
        merchant_table,
        profiles,
        mcc_priors,
        constraint_weights,
        iteration,
        fresh_only=False,
        strict_positive=False,
        selection_limit_override=None,
        candidate_targets=None,
    ):
        c = self.config
        merchants = merchant_table
        category_to_mcc = (
            merchants.groupby(c.consumption_category_col, sort=False)[c.mcc_level1_col]
            .apply(lambda s: sorted(set(s.dropna().astype(str))))
            .to_dict()
        )
        proposals = []
        eligible_merchant_targets = {
            target
            for target, profile in profiles.items()
            if (
                profile["tx_count"] >= c.min_merchant_tx_for_behavior
                or profile["user_count"] >= c.min_merchant_users_for_customer
            )
        }
        merchant_candidate_count = len(eligible_merchant_targets)
        if candidate_targets is not None:
            candidate_targets = {safe_str(x) for x in candidate_targets}
        for row in merchants.itertuples(index=False):
            target = safe_str(getattr(row, c.target_col))
            if candidate_targets is not None and target not in candidate_targets:
                continue
            if fresh_only and target in self.changed_merchants:
                continue
            category = safe_str(getattr(row, c.consumption_category_col))
            old_mcc = safe_str(getattr(row, c.mcc_level1_col))
            profile = profiles.get(target)
            if profile is None:
                continue
            if (
                profile["tx_count"] < c.min_merchant_tx_for_behavior
                and profile["user_count"] < c.min_merchant_users_for_customer
            ):
                continue
            candidates = [m for m in category_to_mcc.get(category, []) if m != old_mcc]
            if len(candidates) > c.max_candidate_mcc_per_merchant:
                candidates = sorted(
                    candidates,
                    key=lambda x: constraint_weights.get(x, 1.0),
                    reverse=True,
                )[: c.max_candidate_mcc_per_merchant]
            if not candidates:
                continue
            old_score = self._merchant_score(profile, old_mcc, category, mcc_priors, constraint_weights)
            best_mcc = None
            best_score = None
            for cand in candidates:
                last = self.merchant_last_change.get(target)
                if last and cand == last["old_value"] and old_mcc == last["new_value"]:
                    continue
                score = self._merchant_score(profile, cand, category, mcc_priors, constraint_weights)
                score -= c.merchant_change_penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_mcc = cand
            if best_mcc is None:
                continue
            gain = float(best_score - old_score)
            if strict_positive:
                minimum_gain = 0.0
                accepted = gain > minimum_gain
            elif c.require_positive_gain:
                # Keep normal runs interpretable: every accepted merchant
                # change must improve its current local consistency score.
                minimum_gain = 0.0
                accepted = gain > minimum_gain
            else:
                minimum_gain = min(c.merchant_min_gain, c.merchant_coverage_min_gain)
                accepted = gain >= minimum_gain
            if accepted:
                proposals.append(
                    {
                        "kind": "merchant",
                        "entity_id": target,
                        "field": c.mcc_level1_col,
                        "category": category,
                        "old_value": old_mcc,
                        "new_value": best_mcc,
                        "gain": gain,
                        "coverage_only": gain < c.merchant_min_gain,
                    }
                )

        proposals.sort(key=lambda x: x["gain"], reverse=True)
        round_limit = max(1, int(math.ceil(merchant_candidate_count * c.merchant_round_budget)))
        total_limit = int(math.floor(merchant_candidate_count * c.merchant_total_budget))
        remaining_total_slots = max(0, total_limit - self.primary_merchant_change_records)
        limit = selection_limit_override if selection_limit_override is not None else (
            round_limit if strict_positive else min(round_limit, remaining_total_slots)
        )
        if limit <= 0:
            return proposals, []
        # Flow caps remain anchored to the full merchant table because they are
        # structural distribution constraints, not optimization budgets.
        flow_cap = max(1, int(math.ceil(len(merchants) * c.mcc_net_flow_cap_rate)))
        selected = []
        inflow = defaultdict(int)
        outflow = defaultdict(int)
        selected_ids = set()

        if c.merchant_category_min_changes_per_round > 0:
            by_category = defaultdict(list)
            for prop in proposals:
                by_category[prop.get("category", UNKNOWN)].append(prop)
            for category in sorted(by_category.keys()):
                picked = 0
                for prop in by_category[category]:
                    if len(selected) >= limit or picked >= c.merchant_category_min_changes_per_round:
                        break
                    if prop["entity_id"] in selected_ids:
                        continue
                    if inflow[prop["new_value"]] >= flow_cap or outflow[prop["old_value"]] >= flow_cap:
                        continue
                    selected.append(prop)
                    selected_ids.add(prop["entity_id"])
                    inflow[prop["new_value"]] += 1
                    outflow[prop["old_value"]] += 1
                    picked += 1

        for prop in proposals:
            if len(selected) >= limit:
                break
            if prop["entity_id"] in selected_ids:
                continue
            if prop.get("coverage_only", False):
                continue
            if inflow[prop["new_value"]] >= flow_cap or outflow[prop["old_value"]] >= flow_cap:
                continue
            selected.append(prop)
            selected_ids.add(prop["entity_id"])
            inflow[prop["new_value"]] += 1
            outflow[prop["old_value"]] += 1
        return proposals, selected

    def _apply_merchant_changes(self, df, merchant_table, selected, iteration, budget_phase="first"):
        c = self.config
        if not selected:
            return df
        mapping = {p["entity_id"]: p["new_value"] for p in selected}
        merchant_mask = merchant_table[c.target_col].isin(mapping)
        merchant_table.loc[merchant_mask, c.mcc_level1_col] = merchant_table.loc[merchant_mask, c.target_col].map(mapping)
        mask = df[c.target_col].isin(mapping)
        df.loc[mask, c.mcc_level1_col] = df.loc[mask, c.target_col].map(mapping)
        for p in selected:
            self.changed_merchants.add(p["entity_id"])
            if budget_phase == "first":
                self.first_changed_merchants.add(p["entity_id"])
                self.primary_merchant_change_records += 1
            self.merchant_last_change[p["entity_id"]] = {
                "iteration": iteration,
                "old_value": p["old_value"],
                "new_value": p["new_value"],
            }
            record = {k: v for k, v in p.items() if k not in ("category", "coverage_only")}
            self.change_history.append(ChangeRecord(iteration=iteration, mode="single", **record))
        return df

    def _build_user_profiles(self, df, hour_col):
        c = self.config
        profiles = {}
        base = df.groupby(c.source_col, sort=False).agg(
            tx_count=(c.amount_col, "size"),
            merchant_count=(c.target_col, "nunique"),
            amount_mean=(c.amount_col, "mean"),
            amount_std=(c.amount_col, "std"),
        )
        eligible = base[
            (base["tx_count"] >= c.min_user_tx_for_behavior)
            | (base["merchant_count"] >= c.min_user_merchants_for_category)
        ].sort_values(["tx_count", "merchant_count"], ascending=False)
        if c.max_user_candidates_per_round and len(eligible) > c.max_user_candidates_per_round:
            eligible = eligible.head(c.max_user_candidates_per_round)
        eligible_sources = set(eligible.index.astype(str))
        profile_df = df[df[c.source_col].astype(str).isin(eligible_sources)]
        mcc_dist = (
            group_distribution_dict(profile_df, c.source_col, c.mcc_level1_col)
            if c.mcc_level1_col in profile_df.columns
            else {}
        )
        hour_map = {}
        if hour_col is not None:
            hour_map = {
                safe_str(k): top_hours(v, topn=3)
                for k, v in profile_df.groupby(c.source_col, sort=False)[hour_col]
            }
        for row in eligible.itertuples():
            source_key = safe_str(row.Index)
            mean = row.amount_mean
            std = 0.0 if pd.isna(row.amount_std) else float(row.amount_std)
            profiles[source_key] = {
                "tx_count": int(row.tx_count),
                "merchant_count": int(row.merchant_count),
                "amount_label": amount_to_label(mean),
                "amount_cv_label": cv_to_label(safe_div(std, mean, 0.0)),
                "mcc_dist": mcc_dist.get(source_key, {}),
                "top_hours": hour_map.get(source_key, []),
            }
        return profiles

    def _build_user_attr_priors(self, df, hour_col, user_fields):
        c = self.config
        priors = {}
        for field_name, col in user_fields:
            item = {"mcc_dist": {}, "amount_label": {}, "top_hours": {}}
            if c.mcc_level1_col in df.columns:
                item["mcc_dist"] = group_distribution_dict(df, col, c.mcc_level1_col)
            amount_mean = df.groupby(col, sort=False)[c.amount_col].mean()
            item["amount_label"] = {safe_str(k): amount_to_label(v) for k, v in amount_mean.items()}
            if hour_col is not None:
                item["top_hours"] = {safe_str(k): top_hours(v, topn=5) for k, v in df.groupby(col, sort=False)[hour_col]}
            priors[field_name] = item
        return priors

    def _source_candidate_probs(self, df, user_fields, category_sources=None):
        c = self.config
        source_df = df
        if category_sources is not None:
            source_df = df[df[c.source_col].astype(str).isin(category_sources)]
        source_target = source_df.groupby([c.source_col, c.target_col]).size().rename("weight").reset_index()
        out = {}
        for field_name, col in user_fields:
            target_dist = dist_to_long(df, c.target_col, col)
            if target_dist.empty:
                out[field_name] = {}
                continue
            merged = source_target.merge(target_dist, on=c.target_col, how="left")
            merged["score"] = merged["weight"] * merged["prob"]
            agg = merged.groupby([c.source_col, col])["score"].sum().reset_index()
            totals = agg.groupby(c.source_col)["score"].transform("sum")
            agg["prob"] = agg["score"] / totals
            probs = {}
            for src, part in agg.groupby(c.source_col, sort=False):
                probs[safe_str(src)] = dict(zip(part[col].astype(str), part["prob"].astype(float)))
            out[field_name] = probs
        return out

    def _user_score(self, profile, field_name, value, candidate_probs, attr_priors, constraint_weight):
        priors = attr_priors.get(field_name, {})
        weight = max(float(constraint_weight), 1e-6)
        c = self.config
        score = c.user_constraint_score_weight * math.log(weight)

        if profile["merchant_count"] >= c.min_user_merchants_for_category:
            merchant_population = float(candidate_probs.get(value, 0.0))
            mcc_ref = priors.get("mcc_dist", {}).get(value, {})
            mcc_match = 1.0 - tvd(profile.get("mcc_dist", {}), mcc_ref) if mcc_ref else 0.5
            score += c.user_population_score_weight * merchant_population
            score += c.user_mcc_score_weight * mcc_match

        if profile["tx_count"] >= c.min_user_tx_for_behavior:
            amount_ref = priors.get("amount_label", {}).get(value, "medium")
            amount_match = 1.0 - label_distance(profile.get("amount_label", "medium"), amount_ref)
            time_match = jaccard(profile.get("top_hours", []), priors.get("top_hours", {}).get(value, []))
            score += c.user_amount_score_weight * amount_match
            score += c.user_time_score_weight * time_match

        return score

    def _propose_user_changes(
        self,
        user_table,
        user_fields,
        user_profiles,
        attr_priors,
        source_probs,
        constraint_weights,
        iteration,
        fresh_only=False,
        strict_positive=False,
        candidate_sources=None,
    ):
        c = self.config
        users = user_table.copy()

        n_total_users = len(users)
        if user_profiles:
            users = users[users[c.source_col].astype(str).isin(user_profiles)].copy()
            users["_optimizer_tx_count"] = users[c.source_col].map(
                lambda x: user_profiles.get(safe_str(x), {}).get("tx_count", 0)
            )
            users["_optimizer_merchant_count"] = users[c.source_col].map(
                lambda x: user_profiles.get(safe_str(x), {}).get("merchant_count", 0)
            )
            users = users.sort_values(
                ["_optimizer_tx_count", "_optimizer_merchant_count"], ascending=False
            )
        if candidate_sources is not None:
            candidate_sources = {safe_str(x) for x in candidate_sources}
            users = users[users[c.source_col].astype(str).isin(candidate_sources)].copy()

        proposals_by_field = defaultdict(list)
        global_values = {name: sorted(user_table[col].dropna().astype(str).unique().tolist()) for name, col in user_fields}
        candidate_user_count = len(users)

        for row in users.itertuples(index=False):
            source = safe_str(getattr(row, c.source_col))
            province = ALL_PROVINCES
            if fresh_only and source in self.changed_users:
                continue
            profile = user_profiles.get(source)
            if profile is None:
                continue
            for field_name, col in user_fields:
                change_key = (source, col)
                if change_key in self.user_last_change:
                    continue
                old_value = safe_str(getattr(row, col))
                probs = source_probs.get(field_name, {}).get(source, {})
                candidates = sorted(probs.keys(), key=lambda x: probs.get(x, 0.0), reverse=True)
                for extra in global_values[field_name]:
                    if extra not in candidates:
                        candidates.append(extra)
                candidates = [v for v in candidates if v != old_value][: c.max_candidate_values_per_user_field]
                if not candidates:
                    continue
                field_weights = constraint_weights.get((field_name, province), {})
                old_score = self._user_score(
                    profile,
                    field_name,
                    old_value,
                    probs,
                    attr_priors,
                    field_weights.get(old_value, 1.0),
                )
                best_value = None
                best_score = None
                for cand in candidates:
                    last = self.user_last_change.get(change_key)
                    if last and cand == last["old_value"] and old_value == last["new_value"]:
                        continue
                    score = self._user_score(
                        profile,
                        field_name,
                        cand,
                        probs,
                        attr_priors,
                        field_weights.get(cand, 1.0),
                    )
                    score -= c.user_change_penalty
                    if best_score is None or score > best_score:
                        best_score = score
                        best_value = cand
                if best_value is None:
                    continue
                gain = float(best_score - old_score)
                if strict_positive:
                    minimum_gain = 0.0
                    accepted = gain > minimum_gain
                elif c.require_positive_gain:
                    # The legacy negative threshold remains configurable, but
                    # is only used when explicitly opting out of this mode.
                    minimum_gain = 0.0
                    accepted = gain > minimum_gain
                else:
                    minimum_gain = c.user_min_gain
                    accepted = gain >= minimum_gain
                if accepted:
                    proposals_by_field[field_name].append(
                        {
                            "kind": "user",
                            "entity_id": source,
                            "field": col,
                            "field_name": field_name,
                            "old_value": old_value,
                            "new_value": best_value,
                            "gain": gain,
                            "province": province,
                        }
                    )

        # User budgets are defined on the evidence-qualified candidate pool.
        # Using the full user population here would let one round consume most
        # of the candidate-pool total budget before merchants can respond.
        field_limit = max(1, int(math.ceil(candidate_user_count * c.user_field_round_budget)))
        limited = {}
        for field_name, proposals in proposals_by_field.items():
            proposals.sort(key=lambda x: x["gain"], reverse=True)
            limited[field_name] = proposals[:field_limit]
        return proposals_by_field, limited, n_total_users, candidate_user_count

    def _select_user_changes(self, limited_by_field, candidate_user_count, strict_positive=False, selection_limit_override=None):
        c = self.config
        round_user_limit = max(1, int(math.ceil(candidate_user_count * c.user_round_budget)))
        total_user_limit = int(math.floor(candidate_user_count * c.user_total_budget))
        remaining_total_slots = max(0, total_user_limit - self.primary_user_change_records)
        if selection_limit_override is not None:
            round_user_limit = selection_limit_override
        elif not strict_positive:
            round_user_limit = min(round_user_limit, remaining_total_slots)
        if round_user_limit <= 0:
            return []
        selected = []
        touched = set()
        field_counts = defaultdict(int)
        field_limit = max(1, int(math.ceil(candidate_user_count * c.user_field_round_budget)))

        # Priority 1: same-province pair swaps preserve provincial field marginals.
        for field_name, proposals in limited_by_field.items():
            buckets = defaultdict(list)
            for prop in proposals:
                key = (prop["province"], prop["old_value"], prop["new_value"])
                buckets[key].append(prop)
            for key in sorted(buckets.keys()):
                province, old_value, new_value = key
                rev = (province, new_value, old_value)
                if rev not in buckets or old_value == new_value:
                    continue
                left = deque(sorted(buckets[key], key=lambda x: x["gain"], reverse=True))
                right = deque(sorted(buckets[rev], key=lambda x: x["gain"], reverse=True))
                while left and right:
                    a = left.popleft()
                    b = right.popleft()
                    if a["entity_id"] in touched or b["entity_id"] in touched:
                        continue
                    if len(touched) + 2 > round_user_limit:
                        break
                    if field_counts[field_name] + 2 > field_limit:
                        break
                    a = dict(a)
                    b = dict(b)
                    a["mode"] = "swap"
                    b["mode"] = "swap"
                    selected.extend([a, b])
                    touched.add(a["entity_id"])
                    touched.add(b["entity_id"])
                    field_counts[field_name] += 2

        # Priority 2: single-point resampling, controlled by soft constraints and final repair.
        leftovers = []
        selected_keys = {(x["entity_id"], x["field"]) for x in selected}
        for proposals in limited_by_field.values():
            for prop in proposals:
                if (prop["entity_id"], prop["field"]) not in selected_keys:
                    leftovers.append(prop)
        leftovers.sort(key=lambda x: x["gain"], reverse=True)
        for prop in leftovers:
            field_name = prop["field_name"]
            if prop["entity_id"] in touched:
                continue
            if len(touched) >= round_user_limit:
                break
            if field_counts[field_name] >= field_limit:
                continue
            prop = dict(prop)
            prop["mode"] = "single"
            selected.append(prop)
            touched.add(prop["entity_id"])
            field_counts[field_name] += 1
        return selected

    def _apply_user_changes(self, df, user_table, selected, iteration, budget_phase="first"):
        c = self.config
        if not selected:
            return df
        by_field = defaultdict(dict)
        for p in selected:
            by_field[p["field"]][p["entity_id"]] = p["new_value"]
        for field, mapping in by_field.items():
            user_mask = user_table[c.source_col].isin(mapping)
            user_table.loc[user_mask, field] = user_table.loc[user_mask, c.source_col].map(mapping)
            mask = df[c.source_col].isin(mapping)
            df.loc[mask, field] = df.loc[mask, c.source_col].map(mapping)
        for p in selected:
            source = p["entity_id"]
            self.changed_users.add(source)
            if budget_phase == "first":
                self.first_changed_users.add(source)
                self.primary_user_change_records += 1
            self.user_last_change[(source, p["field"])] = {
                "iteration": iteration,
                "field_name": p["field_name"],
                "old_value": p["old_value"],
                "new_value": p["new_value"],
            }
            record_kwargs = {k: p[k] for k in ["kind", "entity_id", "field", "old_value", "new_value", "gain", "province"]}
            self.change_history.append(ChangeRecord(iteration=iteration, mode=p.get("mode", "single"), **record_kwargs))
        return df

    def _change_records_as_dict(self):
        return [asdict(x) for x in self.change_history]

    def _revert_changes(self, df, merchant_table, user_table, records):
        c = self.config
        for rec in records:
            if rec.kind == "merchant":
                merchant_mask = merchant_table[c.target_col] == rec.entity_id
                merchant_table.loc[merchant_mask, rec.field] = rec.old_value
                mask = df[c.target_col] == rec.entity_id
                df.loc[mask, rec.field] = rec.old_value
            elif rec.kind == "user":
                user_mask = user_table[c.source_col] == rec.entity_id
                user_table.loc[user_mask, rec.field] = rec.old_value
                mask = df[c.source_col] == rec.entity_id
                df.loc[mask, rec.field] = rec.old_value
        return df

    def _final_repair(self, df, merchant_table, user_table, controller):
        repair_report = {"reverted": [], "initial_validation": None, "final_validation": None}
        validation = controller.final_validate(df)
        repair_report["initial_validation"] = {
            "passed": validation["passed"],
            "violations": validation["violations"],
        }
        if validation["passed"]:
            repair_report["final_validation"] = repair_report["initial_validation"]
            return df, repair_report

        # Revert lowest-gain changes that can move the violated marginals back.
        # For MCC repairs, preserve the small per-behavior-level coverage floor
        # when there are other reversible changes available.
        target_to_category = {}
        active_merchant_changes_by_category = Counter()
        c = self.config
        if not merchant_table.empty and c.consumption_category_col in merchant_table.columns:
            target_to_category = dict(
                zip(
                    merchant_table[c.target_col].astype(str),
                    merchant_table[c.consumption_category_col].astype(str),
                )
            )
            for rec in self.change_history:
                if rec.kind == "merchant":
                    active_merchant_changes_by_category[target_to_category.get(rec.entity_id, UNKNOWN)] += 1

        def repair_sort_key(rec):
            if rec.kind != "merchant":
                return (0, rec.gain)
            category = target_to_category.get(rec.entity_id, UNKNOWN)
            floor = max(1, int(self.config.merchant_category_min_changes_per_round))
            protects_category_floor = active_merchant_changes_by_category.get(category, 0) <= floor
            return (1 if protects_category_floor else 0, rec.gain)

        def record_key(rec):
            return (rec.kind, rec.entity_id, rec.field, rec.old_value, rec.new_value, rec.iteration, rec.mode)

        reverted_record_keys = set()
        reverted_merchants_to_initial = set()

        def current_changed_merchants():
            merchants = controller.merchant_table
            required = [c.target_col, c.consumption_category_col, c.mcc_level1_col]
            if merchants.empty or any(col not in merchants.columns for col in required):
                return pd.DataFrame(columns=required + ["_initial_mcc", "_changed"])
            tmp = merchants[required].copy()
            tmp[c.target_col] = tmp[c.target_col].astype(str)
            tmp[c.consumption_category_col] = tmp[c.consumption_category_col].astype(str)
            tmp[c.mcc_level1_col] = tmp[c.mcc_level1_col].astype(str)
            tmp["_initial_mcc"] = tmp[c.target_col].map(controller.initial_target_to_mcc)
            tmp["_changed"] = tmp["_initial_mcc"].notna() & (tmp[c.mcc_level1_col] != tmp["_initial_mcc"])
            return tmp[tmp["_changed"]].copy()

        def merchant_records_by_entity():
            grouped = defaultdict(list)
            for rec in self.change_history:
                if rec.kind != "merchant":
                    continue
                if record_key(rec) in reverted_record_keys:
                    continue
                if rec.entity_id in reverted_merchants_to_initial:
                    continue
                grouped[rec.entity_id].append(rec)
            return grouped

        def batch_revert_merchants_to_initial(limit, category=None):
            if limit <= 0:
                return 0
            changed = current_changed_merchants()
            if category is not None:
                changed = changed[changed[c.consumption_category_col] == safe_str(category)]
            changed_ids = set(changed[c.target_col].astype(str).tolist())
            if not changed_ids:
                return 0

            grouped = merchant_records_by_entity()
            candidates = []
            for merchant_id in changed_ids:
                records = grouped.get(merchant_id, [])
                if not records:
                    continue
                candidates.append(sorted(records, key=repair_sort_key)[0])
            candidates = sorted(candidates, key=repair_sort_key)[:limit]
            if not candidates:
                return 0

            initial_mapping = {}
            for rec in candidates:
                initial_value = controller.initial_target_to_mcc.get(rec.entity_id)
                if initial_value is None:
                    continue
                initial_mapping[rec.entity_id] = safe_str(initial_value)
            if not initial_mapping:
                return 0

            mask = df[c.target_col].isin(initial_mapping)
            df.loc[mask, c.mcc_level1_col] = df.loc[mask, c.target_col].map(initial_mapping)
            merchant_mask = merchant_table[c.target_col].isin(initial_mapping)
            merchant_table.loc[merchant_mask, c.mcc_level1_col] = merchant_table.loc[merchant_mask, c.target_col].map(initial_mapping)

            for rec in candidates:
                if rec.entity_id not in initial_mapping:
                    continue
                reverted_merchants_to_initial.add(rec.entity_id)
                category_value = target_to_category.get(rec.entity_id, UNKNOWN)
                active_merchant_changes_by_category[category_value] -= 1
                for old_rec in grouped.get(rec.entity_id, []):
                    reverted_record_keys.add(record_key(old_rec))
                item = asdict(rec)
                item["repair_mode"] = "batch_to_initial"
                item["repaired_to"] = initial_mapping[rec.entity_id]
                repair_report["reverted"].append(item)
            return len(initial_mapping)

        max_steps = len(self.change_history)
        print(
            "Final repair: %d violations, reverting up to %d changes one-by-one ..."
            % (len(validation["violations"]), max_steps)
        )

        # Common large violations are handled in batches. Without this, a long
        # run can exceed a total/category budget and then spend minutes or hours
        # doing "revert one change + recompute all constraints" repeatedly.
        for _ in range(4):
            violations = validation["violations"]
            mcc_structure = validation.get("state", {}).get("errors", {}).get("mcc_structure", {})
            total_violation = next(
                (
                    v for v in violations
                    if v.get("kind") == "mcc" and v.get("metric") == "total_change_rate"
                ),
                None,
            )
            if total_violation:
                total_changed = int(mcc_structure.get("total_changed_count", 0))
                total_count = int(mcc_structure.get("total_target_count", 0))
                limit_count = int(math.floor(total_count * c.merchant_total_budget))
                reverted = batch_revert_merchants_to_initial(max(0, total_changed - limit_count))
                if reverted > 0:
                    validation = controller.final_validate(df)
                    print(
                        "  repair batch: reverted %d merchants to initial, passed=%s, violations=%d"
                        % (reverted, validation["passed"], len(validation["violations"]))
                    )
                    continue

            category_violation = next(
                (
                    v for v in violations
                    if v.get("kind") == "mcc" and v.get("metric") == "category_change_rate"
                ),
                None,
            )
            if category_violation:
                category = safe_str(category_violation.get("category", UNKNOWN))
                item = mcc_structure.get("category", {}).get(category, {})
                changed_count = int(item.get("changed_count", 0))
                target_count = int(item.get("target_count", 0))
                limit_count = int(math.floor(target_count * c.mcc_category_change_rate_cap))
                reverted = batch_revert_merchants_to_initial(max(0, changed_count - limit_count), category=category)
                if reverted > 0:
                    validation = controller.final_validate(df)
                    print(
                        "  repair batch: reverted %d merchants in category %s, passed=%s, violations=%d"
                        % (reverted, category, validation["passed"], len(validation["violations"]))
                    )
                    continue
            break

        step = 0
        last_violation_count = len(validation["violations"])
        while not validation["passed"] and step < max_steps:
            violations = validation["violations"]
            candidates = []
            for rec in self.change_history:
                if record_key(rec) in reverted_record_keys:
                    continue
                if rec.kind == "merchant" and rec.entity_id in reverted_merchants_to_initial:
                    continue
                for v in violations:
                    if v["kind"] == "mcc" and rec.kind == "merchant":
                        violation_category = v.get("category")
                        if violation_category and target_to_category.get(rec.entity_id, UNKNOWN) != violation_category:
                            continue
                        candidates.append(rec)
                    elif v["kind"] == "user" and rec.kind == "user" and rec.mode == "single":
                        if rec.province == v["province"] and rec.field.endswith(v["field"]):
                            candidates.append(rec)
                        elif v["field"] in rec.field:
                            candidates.append(rec)
                    elif v["kind"].startswith("consumption") and rec.kind == "merchant":
                        candidates.append(rec)
            if not candidates:
                break
            rec = sorted(candidates, key=repair_sort_key)[0]
            self._revert_changes(df, merchant_table, user_table, [rec])
            repair_report["reverted"].append(asdict(rec))
            reverted_record_keys.add(record_key(rec))
            if rec.kind == "merchant":
                category = target_to_category.get(rec.entity_id, UNKNOWN)
                active_merchant_changes_by_category[category] -= 1
            validation = controller.final_validate(df)
            step += 1
            violation_count = len(validation["violations"])
            if validation["passed"] or violation_count != last_violation_count or step == 1 or step % 100 == 0:
                print(
                    "  repair step %d: reverted 1 change, passed=%s, violations=%d"
                    % (step, validation["passed"], violation_count)
                )
                last_violation_count = violation_count

        repair_report["final_validation"] = {
            "passed": validation["passed"],
            "violations": validation["violations"],
        }
        return df, repair_report

    def optimize(self, input_df):
        c = self.config
        df = self._prepare_working_frame(input_df)
        merchant_available, user_fields = self._component_availability(df)
        demographic_fields = self._merchant_demographic_fields(df, user_fields)
        if not merchant_available and not user_fields:
            raise ValueError(
                "No optimizable fields found. Need merchant fields "
                "(consumption_category + MCC_level1) or user fields "
                "(age_group / education / occupation)."
            )

        df, hour_col = self._ensure_hour_col(df)
        frozen_cols, frozen_snapshot = self._frozen_snapshot(df)
        merchant_table, user_table = self._build_entity_tables(df)
        controller = ConstraintController(
            df, self.config, self.external_priors, merchant_table=merchant_table, user_table=user_table
        )
        report = {
            "method": "ffsd_alternating_consistency_optimizer_v1_user_first",
            "config": asdict(self.config),
            "optimization_order": ["user", "merchant"],
            "warnings": self.warnings,
            "rounds": [],
            "business_prior_notes": {
                "online_retail_ratio_2024": "NBS 2024 communique: physical-goods online retail was 26.5% of total retail sales.",
                "category_online_prior": "Keyword priors use a 2024 post-pandemic online/offline retail anchor, not the earlier 2021 pandemic-period uplift.",
            },
        }

        low_gain_rounds = 0
        user_first_exhaustion_round = None
        merchant_first_exhaustion_round = None
        merchant_primary_ids_previous_round = set()
        for iteration in range(1, self.config.max_iterations + 1):
            user_primary_ids = set()
            merchant_primary_ids = set()
            before_state = controller.evaluate(df)
            weights = before_state["weights"]
            round_report = {
                "iteration": iteration,
                "constraints_before": before_state["errors"],
                "optimization_order": ["user", "merchant"],
                "merchant": {"proposed": 0, "accepted": 0, "first_accepted": 0, "response_accepted": 0, "accepted_gain_mean": 0.0},
                "user": {"proposed": 0, "accepted": 0, "first_accepted": 0, "response_accepted": 0, "accepted_gain_mean": 0.0},
            }

            user_selected = []
            if user_fields:
                # User attributes are initialized mostly from global constraints,
                # while merchant categories already have transaction-image support.
                # Calibrate users first, then let merchant scoring see the updated
                # user-side population profile.
                user_profiles = self._build_user_profiles(df, hour_col)
                attr_priors = self._build_user_attr_priors(df, hour_col, user_fields)
                category_sources = {
                    source
                    for source, profile in user_profiles.items()
                    if profile["merchant_count"] >= self.config.min_user_merchants_for_category
                }
                source_probs = self._source_candidate_probs(df, user_fields, category_sources)
                user_proposals, limited, n_users, n_user_candidates = self._propose_user_changes(
                    user_table,
                    user_fields,
                    user_profiles,
                    attr_priors,
                    source_probs,
                    weights["user"],
                    iteration,
                    fresh_only=False,
                )
                user_selected = self._select_user_changes(limited, n_user_candidates)
                round_report["user"]["proposed"] = sum(len(v) for v in user_proposals.values())
                round_report["user"]["candidate_users_considered"] = n_user_candidates
                round_report["user"]["first_accepted"] = len(user_selected)
                if user_selected:
                    df = self._apply_user_changes(df, user_table, user_selected, iteration, budget_phase="first")
                user_primary_ids = {p["entity_id"] for p in user_selected}

                user_budget_exhausted = (
                    n_user_candidates > 0
                    and self.primary_user_change_records >= int(
                        math.floor(n_user_candidates * self.config.user_total_budget)
                    )
                )
                if user_budget_exhausted and user_first_exhaustion_round is None:
                    user_first_exhaustion_round = iteration
                round_report["user"]["budget_exhausted"] = user_budget_exhausted

                user_response_selected = []
                if (
                    self.config.enable_response_budget
                    and user_first_exhaustion_round is not None
                    and iteration - 1 > user_first_exhaustion_round
                    and merchant_primary_ids_previous_round
                ):
                    affected_sources = set(
                        df.loc[
                            df[c.target_col].astype(str).isin(merchant_primary_ids_previous_round),
                            c.source_col,
                        ].astype(str)
                    )
                    if affected_sources:
                        response_proposals, response_limited, _, _ = self._propose_user_changes(
                            user_table, user_fields, user_profiles, attr_priors, source_probs,
                            weights["user"], iteration, strict_positive=True,
                            candidate_sources=affected_sources,
                        )
                        response_candidate_count = len({
                            p["entity_id"] for proposals in response_limited.values() for p in proposals
                        })
                        round_report["user"]["response_candidate_users"] = response_candidate_count
                        response_limit = max(
                            1,
                            int(math.ceil(response_candidate_count * self.config.response_affected_entity_rate)),
                        ) if response_candidate_count else 0
                        if response_limit > 0:
                            user_response_selected = self._select_user_changes(
                                response_limited, response_candidate_count, strict_positive=True,
                                selection_limit_override=response_limit,
                            )
                            if user_response_selected:
                                df = self._apply_user_changes(
                                    df, user_table, user_response_selected, iteration, budget_phase="response"
                                )
                                self.response_user_records += len(user_response_selected)
                        round_report["user"]["response_proposed"] = sum(len(v) for v in response_proposals.values())
                user_selected.extend(user_response_selected)
                round_report["user"]["response_accepted"] = len(user_response_selected)
                round_report["user"]["accepted"] = len(user_selected)
                if user_selected:
                    round_report["user"]["accepted_gain_mean"] = float(np.mean([x["gain"] for x in user_selected]))

            merchant_selected = []
            if merchant_available:
                # Recalculate after user changes because merchant demographic
                # profiles depend on the current user attributes.
                mcc_priors = self._build_mcc_priors(df, hour_col, demographic_fields)
                merchant_profiles = self._build_merchant_profiles(df, hour_col, demographic_fields)
                merchant_candidate_count = sum(
                    1
                    for profile in merchant_profiles.values()
                    if profile["tx_count"] >= self.config.min_merchant_tx_for_behavior
                    or profile["user_count"] >= self.config.min_merchant_users_for_customer
                )
                round_report["merchant"]["candidate_merchants_considered"] = merchant_candidate_count
                proposals, merchant_selected = self._propose_merchant_changes(
                    merchant_table,
                    merchant_profiles,
                    mcc_priors,
                    weights["mcc"],
                    iteration,
                    fresh_only=True,
                )
                round_report["merchant"]["proposed"] = len(proposals)
                round_report["merchant"]["first_accepted"] = len(merchant_selected)
                if merchant_selected:
                    df = self._apply_merchant_changes(
                        df, merchant_table, merchant_selected, iteration, budget_phase="first"
                    )
                merchant_primary_ids = {p["entity_id"] for p in merchant_selected}

                merchant_budget_exhausted = self.primary_merchant_change_records >= int(
                    math.floor(merchant_candidate_count * self.config.merchant_total_budget)
                )
                if merchant_budget_exhausted and merchant_first_exhaustion_round is None:
                    merchant_first_exhaustion_round = iteration
                round_report["merchant"]["budget_exhausted"] = merchant_budget_exhausted

                merchant_response_selected = []
                if (
                    self.config.enable_response_budget
                    and merchant_first_exhaustion_round is not None
                    and merchant_first_exhaustion_round < iteration
                    and user_primary_ids
                ):
                    affected_targets = set(
                        df.loc[
                            df[c.source_col].astype(str).isin(user_primary_ids),
                            c.target_col,
                        ].astype(str)
                    )
                    if affected_targets:
                        response_proposals, merchant_response_selected = self._propose_merchant_changes(
                            merchant_table, merchant_profiles, mcc_priors, weights["mcc"], iteration,
                            strict_positive=True,
                            candidate_targets=affected_targets,
                        )
                        response_candidate_count = len({p["entity_id"] for p in response_proposals})
                        round_report["merchant"]["response_candidate_merchants"] = response_candidate_count
                        response_limit = max(
                            1,
                            int(math.ceil(response_candidate_count * self.config.response_affected_entity_rate)),
                        ) if response_candidate_count else 0
                        if response_limit > 0:
                            _, merchant_response_selected = self._propose_merchant_changes(
                                merchant_table, merchant_profiles, mcc_priors, weights["mcc"], iteration,
                                strict_positive=True,
                                selection_limit_override=response_limit,
                                candidate_targets=affected_targets,
                            )
                            if merchant_response_selected:
                                df = self._apply_merchant_changes(
                                    df, merchant_table, merchant_response_selected, iteration, budget_phase="response"
                                )
                                self.response_merchant_records += len(merchant_response_selected)
                        round_report["merchant"]["response_proposed"] = len(response_proposals)
                merchant_selected.extend(merchant_response_selected)
                round_report["merchant"]["response_accepted"] = len(merchant_response_selected)
                round_report["merchant"]["accepted"] = len(merchant_selected)
                if merchant_selected:
                    round_report["merchant"]["accepted_gain_mean"] = float(
                        np.mean([x["gain"] for x in merchant_selected])
                    )

            after_state = controller.evaluate(df)
            round_report["constraints_after"] = after_state["errors"]
            round_report["next_round_weight_summary"] = self._summarize_weights(after_state["weights"])
            report["rounds"].append(round_report)
            merchant_primary_ids_previous_round = merchant_primary_ids

            accepted_total = round_report["merchant"]["accepted"] + round_report["user"]["accepted"]
            gain_values = []
            gain_values.extend([x["gain"] for x in merchant_selected])
            gain_values.extend([x["gain"] for x in user_selected])
            accepted_gain_mean = float(np.mean(gain_values)) if gain_values else 0.0
            print_round_progress(round_report, accepted_total, accepted_gain_mean)
            if accepted_gain_mean < self.config.gain_threshold:
                low_gain_rounds += 1
            else:
                low_gain_rounds = 0

            merchant_rate = safe_div(round_report["merchant"]["accepted"], max(1, len(merchant_table)))
            user_rate = safe_div(round_report["user"]["accepted"], max(1, len(user_table)))
            if merchant_rate < self.config.merchant_stop_threshold and user_rate < self.config.user_stop_threshold:
                report["stop_reason"] = "change_rate_below_threshold"
                break
            if accepted_total < self.config.min_accepted_changes:
                report["stop_reason"] = "too_few_accepted_changes"
                break
            if low_gain_rounds >= self.config.gain_stop_patience:
                report["stop_reason"] = "accepted_gain_decay"
                break
        else:
            report["stop_reason"] = "max_iterations"

        df, repair_report = self._final_repair(df, merchant_table, user_table, controller)
        report["final_repair"] = repair_report
        final_validation = controller.final_validate(df)
        report["final_constraints"] = {
            "passed": final_validation["passed"],
            "violations": final_validation["violations"],
            "errors": final_validation["state"]["errors"],
        }
        report["frozen_field_problems"] = self._frozen_validation(df, frozen_cols, frozen_snapshot)
        report["change_history"] = self._change_records_as_dict()
        report["summary"] = {
            "merchant_changed_count": len(self.changed_merchants),
            "user_changed_count": len(self.changed_users),
            "merchant_first_changed_count": len(self.first_changed_merchants),
            "user_first_changed_count": len(self.first_changed_users),
            "merchant_primary_change_records": self.primary_merchant_change_records,
            "user_primary_change_records": self.primary_user_change_records,
            "merchant_response_change_records": self.response_merchant_records,
            "user_response_change_records": self.response_user_records,
            "total_change_records": len(self.change_history),
            "final_constraint_passed": report["final_constraints"]["passed"],
            "frozen_fields_unchanged": len(report["frozen_field_problems"]) == 0,
        }
        print_optimization_stop_summary(report)
        if "_optimizer_hour" in df.columns:
            df = df.drop(columns=["_optimizer_hour"])
        return df, report

    def _n_merchants(self, df):
        c = self.config
        return df[c.target_col].nunique() if c.target_col in df.columns else 0

    def _n_users(self, df):
        c = self.config
        return df[c.source_col].nunique() if c.source_col in df.columns else 0

    def _summarize_weights(self, weights):
        mcc = weights.get("mcc", {})
        top_raise = sorted(mcc.items(), key=lambda x: x[1], reverse=True)[:10]
        top_lower = sorted(mcc.items(), key=lambda x: x[1])[:10]
        user_summary = {}
        for key, values in weights.get("user", {}).items():
            field_name, province = key
            user_summary["%s|%s" % (field_name, province)] = {
                "raise": sorted(values.items(), key=lambda x: x[1], reverse=True)[:5],
                "lower": sorted(values.items(), key=lambda x: x[1])[:5],
            }
        return {"mcc_raise": top_raise, "mcc_lower": top_lower, "user": user_summary}


def optimize_consistency(df, config=None, external_priors=None):
    optimizer = AlternatingConsistencyOptimizer(config=config, external_priors=external_priors)
    return optimizer.optimize(df)
