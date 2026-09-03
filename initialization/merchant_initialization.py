#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize FFSD merchant semantic categories with adaptive level-1 scoring and
automatic soft-balance calibration.

The script maps each Target to a consumption category, MCC level 1, and an
optimizable MCC level 2. It learns soft-balance hyperparameters from the current
data instead of storing weights from a previous run. The default input is
work/transaction.csv and the output is work/target.csv.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
DEFAULT_TRANSACTION_INPUT = WORK_DIR / "transaction.csv"
DEFAULT_TARGET_OUTPUT = WORK_DIR / "target.csv"


# ============================================================
# ============================================================

OFFICIAL_EXPENSE = {
    "Food, Tobacco and Liquor": 6397.3,
    "Clothing and Footwear": 1238.4,
    "Housing": 5215.3,
    "Household Equipment, Furnishings and Services": 1259.5,
    "Transport and Communications": 2761.8,
    "Education, Culture and Recreation": 2032.2,
    "Health Care and Medical Services": 1843.1,
    "Miscellaneous Goods and Services": 462.2,
}

MERCHANT_AMOUNT_PRIOR = {
    "Food, Tobacco and Liquor": 0.30,
    "Housing": 0.14,
    "Transport and Communications": 0.16,
    "Education, Culture and Recreation": 0.13,
    "Health Care and Medical Services": 0.11,
    "Household Equipment, Furnishings and Services": 0.04,
    "Clothing and Footwear": 0.09,
    "Miscellaneous Goods and Services": 0.03,
}

MERCHANT_COUNT_PRIOR = {
    "Food, Tobacco and Liquor": 0.27,
    "Clothing and Footwear": 0.10,
    "Housing": 0.07,
    "Household Equipment, Furnishings and Services": 0.08,
    "Transport and Communications": 0.16,
    "Education, Culture and Recreation": 0.17,
    "Health Care and Medical Services": 0.11,
    "Miscellaneous Goods and Services": 0.04,
}

JOINT_GREEDY_ALPHA = 0.35
JOINT_GREEDY_BETA = 0.65


# ============================================================
# ============================================================

BEHAVIOR_HIERARCHY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "Food, Tobacco and Liquor": {
        "Food, Tobacco and Liquor Retail": {
            "Convenience Store": ["Grocery Store", "Chain Convenience Store"],
            "Supermarket and Hypermarket": ["Supermarket", "Hypermarket"],
            "Fresh Produce Retail": ["Fruit Store", "Fresh Produce, Meat and Seafood"],
            "Specialty Food Retail": ["Specialty Food Retail"],
            "Tobacco, Liquor and Tea Retail": ["Tea Retail", "Liquor Retail", "Tobacco and Cigars"],
        },
        "Food and Beverage Services": {
            "Fast Food and Snacks": ["Chinese Fast Food", "Western Fast Food", "Snacks and Prepared Foods"],
            "Beverages, Bakery and Coffee": ["Beverages and Desserts", "Bakery and Pastries", "Coffee and Tea Shops"],
            "Cafeteria and Group Dining": ["Campus Group Dining", "Institutional Group Dining"],
            "Full-Service Restaurant": ["Chinese Restaurant", "Western Restaurant", "Asian Restaurant", "Hot Pot Restaurant"],
            "Nightlife Bar and Dining": ["Bars and Pubs"],
            "Banquet and Large-Scale Catering": ["Banquet Catering"],
        },
    },
    "Clothing and Footwear": {
        "Apparel and General Shopping": {
            "Apparel, Footwear and Bags": ["Apparel, Footwear and Bags"],
            "Department Store": ["Department Store"],
            "Shopping Mall": ["Shopping Mall"],
            "Outlet Shopping": ["Outlet Shopping"],
            "Commercial Street Retail": ["Commercial Street Retail"],
        },
    },
    "Housing": {
        "Housing Payments and Home Improvement Services": {
            "Utility Bill Payments": [
                "Electricity Bill Payment", "Water Bill Payment",
                "Sanitation Bill Payment", "Gas Bill Payment",
            ],
            "Property Management": ["Property Management"],
            "Real Estate and Housing Agency": ["Real Estate and Housing Agency"],
            "Hardware and Building Materials": ["Hardware and Building Materials"],
            "Home Furnishings and Textiles": ["Home Furnishings and Textiles"],
            "Building Decoration and Renovation Services": ["Building Decoration and Renovation Services"],
            "Appliance, Furniture and Other Repair Services": ["Appliance, Furniture and Other Repair Services"],
        },
    },
    "Household Equipment, Furnishings and Services": {
        "Household and Personal Goods Retail": {
            "Beauty and Personal Care": ["Beauty and Personal Care"],
            "Mother and Baby Products and Children's Toys": ["Mother and Baby Products and Children's Toys"],
            "Office Supplies": ["Office Supplies"],
            "Flowers and Green Plants": ["Flowers and Green Plants"],
            "Pets and Pet Supplies": ["Pets and Pet Supplies"],
            "Digital Electronics and Home Appliances": ["Digital Electronics and Home Appliances"],
        },
        "Local Lifestyle and Self-Service Sharing": {
            "Hair, Beauty and Nail Services": ["Hair, Beauty and Nail Services"],
            "Bathing, Wellness and Health Services": ["Bathing, Wellness and Health Services"],
            "Housekeeping and Cleaning Services": ["Housekeeping and Cleaning Services"],
            "Pet Hospital and Other Pet Services": ["Pet Hospital and Other Pet Services"],
            "Digital and Entertainment Equipment Rental": ["Digital and Entertainment Equipment Rental"],
            "Self-Service Sharing Services": [
                "Unattended Self-Service", "Unattended Self-Service Retail", "Unattended Self-Service Entertainment",
                "Shared Power Banks and Other Shared Rentals",
            ],
        },
    },
    "Transport and Communications": {
        "Local Commuting and On-Demand Travel": {
            "Urban Public Transport": ["Metro", "Public Transport"],
            "Shared Two-Wheel Mobility Services": ["Shared Two-Wheel Mobility Services"],
            "On-Demand Ride-Hailing": ["Taxi Services", "Ride-Hailing Services"],
            "Designated Driving and Parking Services": ["Designated Driving", "Parking Services"],
        },
        "Long-Distance Travel Transport": {
            "Long-Distance Ground Travel": ["Intercity Coach Travel", "Rail Passenger Transport"],
            "Air Travel Services": ["Airlines", "Airports", "Air Ticket Agents"],
            "Water Sightseeing Transport": ["Cruise and Sightseeing Water Transport"],
            "Car Rental": ["Car Rental"],
            "Expressway Toll and Service Areas": ["Electronic Toll Collection", "Manual Toll Collection", "Expressway Service Areas"],
        },
        "Vehicle Energy and Automotive Services": {
            "Fuel Supply Services": ["Fuel Cards and Fuel Services", "Fuel and Gas Stations"],
            "New Energy Charging and Battery Swap Services": ["Electric Vehicle Charging and Battery Swapping", "Two-Wheel Electric Vehicle Charging and Battery Swapping"],
            "Vehicle Supplies, Repair and Maintenance": ["Vehicle Parts and Accessories", "Vehicle Cleaning, Repair and Maintenance"],
        },
        "Telecommunications, Logistics and Digital Infrastructure Services": {
            "Telecom Payment and Operator Services": ["Telecom Operators", "Internet Telephony", "Pay Television", "Mobile Top-Ups and Bill Payments"],
            "Information Search and Online Forums": ["Information Search Services and Online Forums"],
            "Internet Data Services": ["Internet Data Services"],
            "Software Development Services": ["Software Development Services"],
            "Express Delivery and Logistics Services": ["Logistics and Courier Companies", "Individual Courier Services"],
            "Basic Postal Services": ["Basic Postal Services"],
        },
    },
    "Education, Culture and Recreation": {
        "Education Payments and Training Services": {
            "Preschool Care Services": ["Public Preschool Care", "Private Preschool Care"],
            "Primary and Secondary Education": ["Public Primary and Secondary Schools", "Private Primary and Secondary Schools"],
            "Higher Education": ["Public Universities and Colleges", "Private Universities and Colleges"],
            "Youth Palace and Development Center": ["Youth Palace and Development Center"],
            "Other Education and Training": ["Other Education and Training"],
        },
        "Offline Culture, Sports and Entertainment": {
            "KTV and Leisure Clubs": ["KTV, Dance Halls and Leisure Clubs"],
            "Board Games, Table Games and Gaming Cafes": ["Board Games, Table Games and Gaming Cafes"],
            "Fitness, Yoga and Dance": ["Fitness, Yoga and Dance"],
            "Cinema, Performances and Events": ["Cinemas, Performances and Events"],
            "Amusement Parks and Carnivals": ["Amusement Parks and Carnivals"],
            "Cultural and Sports Venues": ["Cultural Venues", "Sports Venues"],
            "Books, Media, Arts and Musical Instruments": ["Books, Media, Arts and Musical Instruments"],
            "Outdoor Sports Equipment": ["Outdoor Sports Equipment"],
        },
        "Online Entertainment and Hobby Spending": {
            "Online Social Networking": ["Online Social Networking"],
            "Online Video, Audio and Reading": ["Online Books, Video and Music"],
            "Gaming": ["Gaming"],
            "Live Streaming": ["Live Streaming"],
            "Gaming Transactions and Related Services": ["Gaming-Related Services and Trading Platforms"],
        },
        "Hotels, Scenic Attractions and Tourism Services": {
            "Hotels, Inns and Homestays": ["Hotels, Inns and Homestays"],
            "Scenic Attractions": ["Scenic Attractions"],
            "Travel Agencies and Tourism Services": ["Travel Agencies and Tourism Services"],
            "Direct Tourism-Related Services": ["Direct Tourism-Related Services"],
            "Scenic Attraction Touring Services": ["Sightseeing Vehicles", "Tour Boat Piers"],
        },
    },
    "Health Care and Medical Services": {
        "Pharmaceutical, Medical Device and Health Retail": {
            "Pharmaceutical Sales": ["Pharmaceutical Sales"],
            "Nutrition and Health Products": ["Nutrition and Health Products"],
            "Medical Device Sales": ["Medical Device Sales"],
            "Health and Assistive Therapy Equipment": ["Health and Assistive Therapy Equipment"],
            "Optical Store": ["Optical Store"],
        },
        "General Clinical Care and Health Management": {
            "Public Primary Medical Care": ["Public Hospitals", "Community Health Centers", "Specialized Public Health Institutions"],
            "Private Hospital": ["Private Hospital"],
            "Clinics and Independent Practitioners": ["Clinics", "Independent Medical Practitioners"],
            "Medical Laboratories and Diagnostic Centers": ["Medical Laboratories and Diagnostic Centers"],
            "Health Examinations and Consultations": ["Health Examinations and Consultations"],
        },
        "Specialized Consumer Medical and Care Services": {
            "Medical Aesthetics": ["Medical Aesthetics"],
            "Ophthalmic Medical Services": ["Ophthalmic Medical Services"],
            "Dental Medical Services": ["Dental Medical Services"],
            "Online Medical Services": ["Online Medical Services"],
            "Care Institution Services": ["Care Institution Services"],
        },
    },
    "Miscellaneous Goods and Services": {
        "High-Value Goods and Professional Services": {
            "High-Value Jewelry and Accessories": ["Jewelry, Watches and Accessories"],
            "Wedding and Photography Services": ["Matchmaking, Wedding and Photography Services"],
            "Advertising, Exhibition and Printing Services": ["Advertising, Exhibitions and Printing Services"],
            "Legal Services": ["Legal Consultation and Law Firms"],
            "Accounting and Financial Consulting Services": ["Accounting and Financial Consulting Services"],
            "Recruitment Services": ["Recruitment Services"],
            "Online Tools": ["Online Tools"],
        },
    },
}

PERIOD_TAGS = [
    "late_night", "early_morning", "commute_morning", "workday_morning",
    "lunch", "afternoon", "commute_evening", "night",
]

STRONG_ONLINE_THRESHOLD = 0.45
MIXED_ONLINE_THRESHOLD = 0.15
WEEKEND_ENHANCED_THRESHOLD = 0.40
WORKDAY_OBVIOUS_THRESHOLD = 0.25
MONTH_OBVIOUS_THRESHOLD = 0.45
PERIOD_OBVIOUS_THRESHOLD = 0.35
DISTRIBUTED_MAX_PERIOD_THRESHOLD = 0.45
DISTRIBUTED_MIN_ACTIVE_PERIODS = 3

# ============================================================
# ============================================================

EFFECTIVE_AMOUNT_CAP_QUANTILE = 0.995
EXTREME_TOTAL_QUANTILE = 0.99
HIGH_COUNT_QUANTILE = 0.95
FEW_LARGE_INFLATION_THRESHOLD = 20.0
FEW_LARGE_CV_THRESHOLD = 3.0
HIGH_FREQ_REGULAR_INFLATION_THRESHOLD = 3.0
HIGH_FREQ_REGULAR_CV_THRESHOLD = 1.0
TYPICAL_HIGH_MEDIAN_QUANTILE = 0.99


# ============================================================
# ============================================================

def parse_amount_value(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("$", "").replace("￥", "").replace("¥", "")
    if s == "":
        return np.nan
    return float(s)


def parse_bool_value(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x) != 0.0
    s = str(x).strip().lower()
    return s in {"true", "1", "yes", "y", "t"}


def level_by_quantiles(series: pd.Series, high_quantile: float = 0.70, low_quantile: float = 0.30) -> pd.Series:
    q_low = series.quantile(low_quantile)
    q_high = series.quantile(high_quantile)

    def label(v: float) -> str:
        if pd.isna(v):
            return "medium"
        if v >= q_high:
            return "high"
        if v <= q_low:
            return "low"
        return "medium"

    return series.apply(label)


def normalize_transaction_columns(df: pd.DataFrame, target_col: str, amount_col: str) -> pd.DataFrame:
    df = df.copy()
    if target_col not in df.columns:
        raise ValueError(f"Input table is missing the Target column: {target_col}")
    if amount_col not in df.columns:
        raise ValueError(f"Input table is missing the Amount column: {amount_col}")

    df[amount_col] = df[amount_col].apply(parse_amount_value)
    df = df.dropna(subset=[target_col, amount_col])

    if "date" in df.columns:
        parsed_date = pd.to_datetime(df["date"], errors="coerce")
    elif "datetime" in df.columns:
        parsed_date = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        parsed_date = pd.Series(pd.NaT, index=df.index)
    df["_parsed_date"] = parsed_date

    if "period_tag" not in df.columns:
        if "hour" in df.columns:
            hour = pd.to_numeric(df["hour"], errors="coerce")
        elif "datetime" in df.columns:
            hour = pd.to_datetime(df["datetime"], errors="coerce").dt.hour
        else:
            hour = pd.Series(np.nan, index=df.index)
        df["period_tag"] = hour.apply(hour_to_period_tag)
    df["period_tag"] = df["period_tag"].where(df["period_tag"].isin(PERIOD_TAGS), "unknown")

    if "is_weekend" not in df.columns:
        if "dayofweek" in df.columns:
            dow = pd.to_numeric(df["dayofweek"], errors="coerce")
            df["is_weekend"] = dow >= 5
        else:
            df["is_weekend"] = df["_parsed_date"].dt.dayofweek >= 5
    df["is_weekend"] = df["is_weekend"].apply(parse_bool_value)

    if "is_month_start" not in df.columns:
        day = df["_parsed_date"].dt.day
        df["is_month_start"] = day.between(1, 10, inclusive="both")
    if "is_month_middle" not in df.columns:
        day = df["_parsed_date"].dt.day
        df["is_month_middle"] = day.between(11, 20, inclusive="both")
    if "is_month_end" not in df.columns:
        day = df["_parsed_date"].dt.day
        df["is_month_end"] = day >= 21

    for col in ["is_month_start", "is_month_middle", "is_month_end"]:
        df[col] = df[col].apply(parse_bool_value)

    if "is_out_province_transaction" not in df.columns:
        df["is_out_province_transaction"] = 0
    df["is_out_province_transaction"] = pd.to_numeric(
        df["is_out_province_transaction"].apply(lambda x: 1 if parse_bool_value(x) else 0),
        errors="coerce",
    ).fillna(0).astype(int)

    return df


def hour_to_period_tag(hour: Any) -> str:
    if pd.isna(hour):
        return "unknown"
    h = int(hour)
    if 0 <= h <= 5:
        return "late_night"
    if 6 <= h <= 7:
        return "early_morning"
    if 8 <= h <= 9:
        return "commute_morning"
    if 10 <= h <= 11:
        return "workday_morning"
    if 12 <= h <= 13:
        return "lunch"
    if 14 <= h <= 16:
        return "afternoon"
    if 17 <= h <= 18:
        return "commute_evening"
    return "night"


def build_target_profiles(df: pd.DataFrame, target_col: str, amount_col: str) -> pd.DataFrame:
    df = normalize_transaction_columns(df, target_col, amount_col)

    grouped = df.groupby(target_col)[amount_col]
    profiles = grouped.agg(
        transaction_count="count",
        total_amount="sum",
        avg_amount="mean",
        median_amount="median",
        max_amount="max",
        p95_amount=lambda x: x.quantile(0.95),
        p99_amount=lambda x: x.quantile(0.99),
        amount_std="std",
    ).reset_index().rename(columns={target_col: "target_id"})

    profiles["amount_std"] = profiles["amount_std"].fillna(0.0)
    profiles["amount_cv"] = profiles["amount_std"] / profiles["avg_amount"].replace(0, np.nan)
    profiles["amount_cv"] = profiles["amount_cv"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    profiles["amount_level"] = level_by_quantiles(profiles["median_amount"])

    profiles["frequency_level"] = level_by_quantiles(profiles["transaction_count"])
    profiles["time_frequency"] = profiles["frequency_level"].map({
        "high": "daily_high_frequency",
        "medium": "weekly_high_frequency",
        "low": "low_frequency",
    })
    profiles["volatility_level"] = level_by_quantiles(profiles["amount_cv"])

    period_dist = pd.crosstab(df[target_col], df["period_tag"], normalize="index")
    for tag in PERIOD_TAGS:
        if tag not in period_dist.columns:
            period_dist[tag] = 0.0
    period_dist = period_dist[PERIOD_TAGS].reset_index().rename(columns={target_col: "target_id"})
    period_dist = period_dist.rename(columns={tag: f"period_ratio_{tag}" for tag in PERIOD_TAGS})
    profiles = profiles.merge(period_dist, on="target_id", how="left")
    for tag in PERIOD_TAGS:
        profiles[f"period_ratio_{tag}"] = profiles[f"period_ratio_{tag}"].fillna(0.0)

    bool_profile = df.groupby(target_col).agg(
        weekend_ratio=("is_weekend", "mean"),
        month_start_ratio=("is_month_start", "mean"),
        month_middle_ratio=("is_month_middle", "mean"),
        month_end_ratio=("is_month_end", "mean"),
        out_province_ratio=("is_out_province_transaction", "mean"),
    ).reset_index().rename(columns={target_col: "target_id"})
    profiles = profiles.merge(bool_profile, on="target_id", how="left")

    profiles = attach_rule_features(profiles)

    profiles = attach_effective_amount_features(profiles)
    return profiles


def attach_effective_amount_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Build effective_total_amount for consumption-category allocation.

    The original total_amount remains unchanged. Regular high-frequency Targets
    keep their totals, while sparse large-value and extreme Targets are capped
    using transaction statistics or global quantiles. The extreme_amount_type
    field records which adjustment was applied.
    """
    profiles = profiles.copy()

    profiles["median_amount_safe"] = profiles["median_amount"].replace(0, np.nan)
    profiles["median_based_total"] = (
        profiles["median_amount_safe"] * profiles["transaction_count"]
    ).fillna(profiles["total_amount"])

    profiles["avg_median_ratio"] = (
        profiles["avg_amount"] / profiles["median_amount_safe"]
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    profiles["total_median_inflation_ratio"] = (
        profiles["total_amount"] / profiles["median_based_total"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    profiles["max_amount_share"] = (
        profiles["max_amount"] / profiles["total_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    total_cap = float(profiles["total_amount"].quantile(EFFECTIVE_AMOUNT_CAP_QUANTILE))
    extreme_total_threshold = float(profiles["total_amount"].quantile(EXTREME_TOTAL_QUANTILE))
    high_count_threshold = float(profiles["transaction_count"].quantile(HIGH_COUNT_QUANTILE))
    typical_high_median_threshold = float(profiles["median_amount"].quantile(TYPICAL_HIGH_MEDIAN_QUANTILE))

    profiles["effective_amount_cap"] = total_cap

    few_large_mask = (
        (profiles["total_median_inflation_ratio"] >= FEW_LARGE_INFLATION_THRESHOLD)
        & (profiles["amount_cv"] >= FEW_LARGE_CV_THRESHOLD)
    )
    high_freq_regular_mask = (
        (profiles["total_amount"] >= extreme_total_threshold)
        & (profiles["transaction_count"] >= high_count_threshold)
        & (profiles["total_median_inflation_ratio"] <= HIGH_FREQ_REGULAR_INFLATION_THRESHOLD)
        & (profiles["amount_cv"] <= HIGH_FREQ_REGULAR_CV_THRESHOLD)
    )
    typical_high_amount_mask = (
        (profiles["total_amount"] >= extreme_total_threshold)
        & (profiles["median_amount"] >= typical_high_median_threshold)
        & (profiles["total_median_inflation_ratio"] <= HIGH_FREQ_REGULAR_INFLATION_THRESHOLD)
    )
    extreme_total_mask = profiles["total_amount"] >= extreme_total_threshold

    profiles["extreme_amount_type"] = "regular"
    profiles.loc[extreme_total_mask, "extreme_amount_type"] = "mixed_extreme"
    profiles.loc[typical_high_amount_mask, "extreme_amount_type"] = "typical_high_amount"
    profiles.loc[high_freq_regular_mask, "extreme_amount_type"] = "high_frequency_regular"
    profiles.loc[few_large_mask, "extreme_amount_type"] = "few_large_transactions"

    effective = profiles["total_amount"].copy()

    few_large_cap = profiles["median_based_total"] * FEW_LARGE_INFLATION_THRESHOLD
    effective.loc[few_large_mask] = np.minimum(
        profiles.loc[few_large_mask, "total_amount"],
        few_large_cap.loc[few_large_mask],
    )

    global_cap_mask = (typical_high_amount_mask | extreme_total_mask) & (~high_freq_regular_mask) & (~few_large_mask)
    effective.loc[global_cap_mask] = np.minimum(
        profiles.loc[global_cap_mask, "total_amount"],
        total_cap,
    )

    effective.loc[high_freq_regular_mask] = profiles.loc[high_freq_regular_mask, "total_amount"]

    profiles["effective_total_amount"] = effective.clip(lower=0.0)
    profiles["effective_amount_ratio_to_raw"] = (
        profiles["effective_total_amount"] / profiles["total_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    return profiles.drop(columns=["median_amount_safe"])


def attach_rule_features(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles = profiles.copy()
    period_cols = [f"period_ratio_{tag}" for tag in PERIOD_TAGS]

    profiles["commute_ratio"] = profiles["period_ratio_commute_morning"] + profiles["period_ratio_commute_evening"]
    profiles["dining_ratio"] = (
        profiles["period_ratio_lunch"]
        + profiles["period_ratio_night"]
        + profiles["period_ratio_late_night"]
    )
    profiles["night_ratio"] = profiles["period_ratio_night"] + profiles["period_ratio_late_night"]
    profiles["work_service_ratio"] = profiles["period_ratio_workday_morning"] + profiles["period_ratio_afternoon"]
    profiles["max_period_ratio"] = profiles[period_cols].max(axis=1)
    profiles["active_period_count"] = (profiles[period_cols] >= 0.10).sum(axis=1)
    profiles["is_time_distributed"] = (
        (profiles["max_period_ratio"] <= DISTRIBUTED_MAX_PERIOD_THRESHOLD)
        & (profiles["active_period_count"] >= DISTRIBUTED_MIN_ACTIVE_PERIODS)
    )

    profiles["is_commute_obvious"] = profiles["commute_ratio"] >= PERIOD_OBVIOUS_THRESHOLD
    profiles["is_dining_obvious"] = profiles["dining_ratio"] >= PERIOD_OBVIOUS_THRESHOLD
    profiles["is_night_obvious"] = profiles["night_ratio"] >= PERIOD_OBVIOUS_THRESHOLD
    profiles["is_work_service_obvious"] = profiles["work_service_ratio"] >= PERIOD_OBVIOUS_THRESHOLD

    profiles["is_weekend_enhanced"] = profiles["weekend_ratio"] >= WEEKEND_ENHANCED_THRESHOLD
    profiles["is_workday_obvious"] = profiles["weekend_ratio"] <= WORKDAY_OBVIOUS_THRESHOLD

    profiles["is_month_start_obvious"] = profiles["month_start_ratio"] >= MONTH_OBVIOUS_THRESHOLD
    profiles["is_month_middle_obvious"] = profiles["month_middle_ratio"] >= MONTH_OBVIOUS_THRESHOLD
    profiles["is_month_end_obvious"] = profiles["month_end_ratio"] >= MONTH_OBVIOUS_THRESHOLD
    profiles["is_month_cycle_obvious"] = profiles[[
        "month_start_ratio", "month_middle_ratio", "month_end_ratio"
    ]].max(axis=1) >= MONTH_OBVIOUS_THRESHOLD

    profiles["online_tendency"] = profiles["out_province_ratio"].apply(classify_online_tendency)
    profiles["is_strong_offline"] = profiles["online_tendency"] == "strong_offline"
    profiles["is_online_mixed"] = profiles["online_tendency"] == "mixed_online_offline"
    profiles["is_strong_online"] = profiles["online_tendency"] == "strong_online"
    profiles["is_online_or_mixed"] = profiles["online_tendency"].isin(["strong_online", "mixed_online_offline"])

    return profiles


def classify_online_tendency(out_ratio: float) -> str:
    if pd.isna(out_ratio):
        return "strong_offline"
    if out_ratio >= STRONG_ONLINE_THRESHOLD:
        return "strong_online"
    if out_ratio >= MIXED_ONLINE_THRESHOLD:
        return "mixed_online_offline"
    return "strong_offline"


# ============================================================
# ============================================================

def attach_candidate_categories(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Return all consumption categories as candidates for every Target.

    Consumption categories form the macro statistical layer. Transaction
    behavior is used later to select MCC level 1 within the assigned category.
    """
    profiles = profiles.copy()
    all_categories = list(MERCHANT_AMOUNT_PRIOR.keys())
    profiles["profile_type"] = "all_categories_statistical_greedy"
    profiles["candidate_categories"] = [all_categories.copy() for _ in range(len(profiles))]
    return profiles

def _assignment_order(
    profiles: pd.DataFrame,
    sort_columns: List[str],
) -> pd.Index:
    """Return a deterministic processing order for stateful greedy assignments."""
    # Legacy historical order (not used):
    # return profiles.sort_values(sort_columns, ascending=[False] * len(sort_columns)).index
    # Target ID resolves value ties consistently across platforms and pandas versions.
    return profiles.sort_values(
        [*sort_columns, "target_id"],
        ascending=[False] * len(sort_columns) + [True],
        kind="mergesort",
    ).index


def greedy_assign_consumption_category(profiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profiles = profiles.copy()

    if "effective_total_amount" not in profiles.columns:
        profiles = attach_effective_amount_features(profiles)

    official_total = sum(OFFICIAL_EXPENSE.values())
    official_amount_ratio = {c: v / official_total for c, v in OFFICIAL_EXPENSE.items()}
    categories = list(MERCHANT_AMOUNT_PRIOR.keys())

    missing_count_prior = [c for c in categories if c not in MERCHANT_COUNT_PRIOR]
    if missing_count_prior:
        raise ValueError(f"MERCHANT_COUNT_PRIOR is missing categories: {missing_count_prior}")

    missing_amount_prior = [c for c in categories if c not in MERCHANT_AMOUNT_PRIOR]
    if missing_amount_prior:
        raise ValueError(f"MERCHANT_AMOUNT_PRIOR is missing categories: {missing_amount_prior}")

    count_prior_sum = sum(MERCHANT_COUNT_PRIOR[c] for c in categories)
    if abs(count_prior_sum - 1.0) > 1e-6:
        raise ValueError(f"MERCHANT_COUNT_PRIOR must sum to 1; got {count_prior_sum}")

    amount_prior_sum = sum(MERCHANT_AMOUNT_PRIOR[c] for c in categories)
    if abs(amount_prior_sum - 1.0) > 1e-6:
        raise ValueError(f"MERCHANT_AMOUNT_PRIOR must sum to 1; got {amount_prior_sum}")

    raw_total_amount_all_targets = float(profiles["total_amount"].sum())
    effective_total_amount_all_targets = float(profiles["effective_total_amount"].sum())
    total_target_count = int(len(profiles))

    amount_quota = {
        c: effective_total_amount_all_targets * MERCHANT_AMOUNT_PRIOR[c]
        for c in categories
    }
    count_quota = {c: total_target_count * MERCHANT_COUNT_PRIOR[c] for c in categories}

    assigned_effective_amount = {c: 0.0 for c in categories}
    assigned_raw_amount = {c: 0.0 for c in categories}
    assigned_count = {c: 0 for c in categories}
    assignment = {}

    order = _assignment_order(profiles, ["effective_total_amount"])
    for idx in order:
        row = profiles.loc[idx]
        target_id = row["target_id"]
        effective_amount = float(row["effective_total_amount"])
        raw_amount = float(row["total_amount"])
        candidates = row["candidate_categories"]

        best_category = None
        best_score = -float("inf")
        for category in candidates:
            amount_quota_c = amount_quota[category]
            count_quota_c = count_quota[category]
            amount_gap_score = (
                1.0 - assigned_effective_amount[category] / amount_quota_c
                if amount_quota_c > 0 else -float("inf")
            )
            count_gap_score = (
                1.0 - assigned_count[category] / count_quota_c
                if count_quota_c > 0 else -float("inf")
            )
            score = JOINT_GREEDY_ALPHA * amount_gap_score + JOINT_GREEDY_BETA * count_gap_score
            if score > best_score:
                best_score = score
                best_category = category

        if best_category is None:
            best_category = candidates[0]

        assignment[target_id] = best_category
        assigned_effective_amount[best_category] += effective_amount
        assigned_raw_amount[best_category] += raw_amount
        assigned_count[best_category] += 1

    profiles["consumption_category"] = profiles["target_id"].map(assignment)

    assigned_effective_amount_ratio = {
        c: assigned_effective_amount[c] / effective_total_amount_all_targets
        if effective_total_amount_all_targets > 0 else 0.0
        for c in categories
    }
    assigned_raw_amount_ratio = {
        c: assigned_raw_amount[c] / raw_total_amount_all_targets
        if raw_total_amount_all_targets > 0 else 0.0
        for c in categories
    }
    assigned_count_ratio = {
        c: assigned_count[c] / total_target_count if total_target_count > 0 else 0.0
        for c in categories
    }

    report = pd.DataFrame([
        {
            "consumption_category": c,
            "official_amount_ratio": official_amount_ratio[c],
            "merchant_amount_prior": MERCHANT_AMOUNT_PRIOR[c],

            "assigned_amount_ratio": assigned_effective_amount_ratio[c],
            "amount_ratio_diff": assigned_effective_amount_ratio[c] - MERCHANT_AMOUNT_PRIOR[c],
            "amount_quota": amount_quota[c],
            "assigned_amount": assigned_effective_amount[c],

            "assigned_effective_amount_ratio": assigned_effective_amount_ratio[c],
            "effective_amount_ratio_diff": assigned_effective_amount_ratio[c] - MERCHANT_AMOUNT_PRIOR[c],
            "effective_amount_quota": amount_quota[c],
            "assigned_effective_amount": assigned_effective_amount[c],
            "assigned_raw_amount_ratio": assigned_raw_amount_ratio[c],
            "raw_amount_ratio_diff": assigned_raw_amount_ratio[c] - MERCHANT_AMOUNT_PRIOR[c],
            "assigned_raw_amount": assigned_raw_amount[c],

            "merchant_count_prior": MERCHANT_COUNT_PRIOR[c],
            "assigned_count_ratio": assigned_count_ratio[c],
            "count_ratio_diff": assigned_count_ratio[c] - MERCHANT_COUNT_PRIOR[c],
            "count_quota": count_quota[c],
            "assigned_count": assigned_count[c],
            "alpha_amount_weight": JOINT_GREEDY_ALPHA,
            "beta_count_weight": JOINT_GREEDY_BETA,
        }
        for c in categories
    ])
    return profiles, report

# ============================================================
# ============================================================


SINGLE_LEVEL1_CATEGORIES = {
    "Clothing and Footwear": "Apparel and General Shopping",
    "Housing": "Housing Payments and Home Improvement Services",
    "Miscellaneous Goods and Services": "High-Value Goods and Professional Services",
}


def rank_to_level(rank_value: float) -> str:
    if pd.isna(rank_value):
        return "medium"
    if rank_value >= 0.67:
        return "high"
    if rank_value <= 0.33:
        return "low"
    return "medium"


def attach_category_relative_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Compute category-relative merchant behavior after assigning consumption_category.

    Amount, frequency, temporal concentration, and online tendency signals used
    by MCC level-1 assignment are evaluated within each consumption category.
    """
    profiles = profiles.copy()
    if "consumption_category" not in profiles.columns:
        raise ValueError("Assign consumption_category before computing category-relative profiles.")

    profiles["month_cycle_strength"] = profiles[[
        "month_start_ratio", "month_middle_ratio", "month_end_ratio"
    ]].max(axis=1)

    relative_cols = {
        "median_amount": "category_amount",
        "transaction_count": "category_frequency",
        "effective_total_amount": "category_effective_amount",

        "commute_ratio": "category_commute",
        "dining_ratio": "category_dining",
        "night_ratio": "category_night",
        "work_service_ratio": "category_work_service",
        "active_period_count": "category_active_period",

        "weekend_ratio": "category_weekend",
        "month_start_ratio": "category_month_start",
        "month_middle_ratio": "category_month_middle",
        "month_end_ratio": "category_month_end",
        "month_cycle_strength": "category_month_cycle",
        "out_province_ratio": "category_out_province",
    }

    def pct_rank(s: pd.Series) -> pd.Series:
        if len(s) <= 1:
            return pd.Series([0.5] * len(s), index=s.index)
        return s.rank(method="average", pct=True)

    for raw_col, prefix in relative_cols.items():
        if raw_col not in profiles.columns:
            profiles[raw_col] = 0.0
        rank_col = f"{prefix}_rank"
        level_col = f"{prefix}_level"
        profiles[rank_col] = profiles.groupby("consumption_category")[raw_col].transform(pct_rank)
        profiles[level_col] = profiles[rank_col].apply(rank_to_level)

    profiles["category_amount_rank"] = profiles["category_amount_rank"]
    profiles["category_frequency_rank"] = profiles["category_frequency_rank"]
    profiles["category_effective_amount_rank"] = profiles["category_effective_amount_rank"]
    profiles["category_amount_level"] = profiles["category_amount_level"]
    profiles["category_frequency_level"] = profiles["category_frequency_level"]
    profiles["category_effective_amount_level"] = profiles["category_effective_amount_level"]

    profiles["category_is_amount_high"] = profiles["category_amount_level"] == "high"
    profiles["category_is_amount_medium_or_high"] = profiles["category_amount_level"].isin(["medium", "high"])
    profiles["category_is_amount_low_or_medium"] = profiles["category_amount_level"].isin(["low", "medium"])

    profiles["category_is_frequency_high_or_medium"] = profiles["category_frequency_level"].isin(["medium", "high"])
    profiles["category_is_frequency_low"] = profiles["category_frequency_level"] == "low"
    profiles["category_is_frequency_low_or_medium"] = profiles["category_frequency_level"].isin(["low", "medium"])

    profiles["category_is_effective_amount_high"] = profiles["category_effective_amount_level"] == "high"

    profiles["category_is_commute_obvious"] = profiles["category_commute_level"] == "high"
    profiles["category_is_dining_obvious"] = profiles["category_dining_level"] == "high"
    profiles["category_is_dining_not_low"] = profiles["category_dining_level"].isin(["medium", "high"])
    profiles["category_is_night_obvious"] = profiles["category_night_level"] == "high"
    profiles["category_is_work_service_obvious"] = profiles["category_work_service_level"] == "high"
    profiles["category_is_time_distributed"] = profiles["category_active_period_level"] == "high"

    profiles["category_is_weekend_enhanced"] = profiles["category_weekend_level"] == "high"
    profiles["category_is_workday_obvious"] = profiles["category_weekend_level"] == "low"

    profiles["category_is_month_start_obvious"] = profiles["category_month_start_level"] == "high"
    profiles["category_is_month_middle_obvious"] = profiles["category_month_middle_level"] == "high"
    profiles["category_is_month_end_obvious"] = profiles["category_month_end_level"] == "high"
    profiles["category_is_month_cycle_obvious"] = profiles["category_month_cycle_level"] == "high"
    profiles["category_is_month_edge_obvious"] = (
        profiles["category_is_month_start_obvious"] | profiles["category_is_month_end_obvious"]
    )

    def category_online_label(row: pd.Series) -> str:
        r = float(row.get("out_province_ratio", 0.0) or 0.0)
        rel = row.get("category_out_province_level", "medium")

        if r >= 0.60:
            return "strong_online"
        if r >= 0.20:
            return "mixed_online_offline"
        if rel == "high" and r >= 0.08:
            return "mixed_online_offline"
        return "strong_offline"

    profiles["category_online_tendency"] = profiles.apply(category_online_label, axis=1)
    profiles["category_is_strong_offline"] = profiles["category_online_tendency"] == "strong_offline"
    profiles["category_is_online_mixed"] = profiles["category_online_tendency"] == "mixed_online_offline"
    profiles["category_is_strong_online"] = profiles["category_online_tendency"] == "strong_online"
    profiles["category_is_online_or_mixed"] = profiles["category_online_tendency"].isin(["strong_online", "mixed_online_offline"])

    return profiles


def cat(row: pd.Series, name: str, default: bool = False) -> bool:
    return bool(row.get(name, default))


def cat_amount_high(row: pd.Series) -> bool:
    return cat(row, "category_is_amount_high")


def cat_amount_medium_or_high(row: pd.Series) -> bool:
    return cat(row, "category_is_amount_medium_or_high")


def cat_amount_low_or_medium(row: pd.Series) -> bool:
    return cat(row, "category_is_amount_low_or_medium")


def cat_frequency_high_or_medium(row: pd.Series) -> bool:
    return cat(row, "category_is_frequency_high_or_medium")


def cat_frequency_low(row: pd.Series) -> bool:
    return cat(row, "category_is_frequency_low")


def cat_frequency_low_or_medium(row: pd.Series) -> bool:
    return cat(row, "category_is_frequency_low_or_medium")


def assign_behavior_level1_by_rules(row: pd.Series) -> str:
    category = row["consumption_category"]

    if category == "Food, Tobacco and Liquor":
        return assign_food_level1(row)
    if category in SINGLE_LEVEL1_CATEGORIES:
        return SINGLE_LEVEL1_CATEGORIES[category]
    if category == "Household Equipment, Furnishings and Services":
        return assign_life_goods_service_level1(row)
    if category == "Transport and Communications":
        return assign_transport_communication_level1(row)
    if category == "Education, Culture and Recreation":
        return assign_education_entertainment_level1(row)
    if category == "Health Care and Medical Services":
        return assign_healthcare_level1(row)

    return next(iter(BEHAVIOR_HIERARCHY[category].keys()))


def assign_food_level1(row: pd.Series) -> str:
    """
    Split Food, Tobacco and Liquor using within-category amount, frequency, and meal-time patterns.
    """
    retail_like = (
        cat_frequency_high_or_medium(row)
        and cat(row, "category_is_time_distributed")
        and cat_amount_low_or_medium(row)
    )

    if cat(row, "category_is_dining_obvious"):
        return "Food and Beverage Services"

    if cat_amount_medium_or_high(row) and cat(row, "category_is_dining_not_low") and not retail_like:
        return "Food and Beverage Services"

    if cat_amount_high(row) and cat(row, "category_is_night_obvious") and not retail_like:
        return "Food and Beverage Services"

    return "Food, Tobacco and Liquor Retail"


def assign_life_goods_service_level1(row: pd.Series) -> str:
    """
    Split Household Equipment, Furnishings and Services into local services and distributed retail.
    """
    if cat(row, "category_is_weekend_enhanced") and cat(row, "category_is_strong_offline"):
        return "Local Lifestyle and Self-Service Sharing"

    if (
        cat(row, "category_is_night_obvious")
        and cat(row, "category_is_strong_offline")
        and not cat(row, "category_is_effective_amount_high")
    ):
        return "Local Lifestyle and Self-Service Sharing"

    return "Household and Personal Goods Retail"


def assign_transport_communication_level1(row: pd.Series) -> str:
    """
    Split Transport and Communications using category-relative amount, frequency, and timing signals.
    """
    if cat(row, "category_is_strong_offline") and cat_frequency_high_or_medium(row):
        if cat(row, "category_is_commute_obvious"):
            return "Local Commuting and On-Demand Travel"
        if cat_amount_low_or_medium(row) and cat(row, "category_is_time_distributed") and not cat(row, "category_is_effective_amount_high"):
            return "Local Commuting and On-Demand Travel"

    if cat_amount_high(row) and cat_frequency_low(row):
        if (
            cat(row, "category_is_weekend_enhanced")
            or cat(row, "category_is_month_end_obvious")
            or cat(row, "category_is_online_or_mixed")
        ):
            return "Long-Distance Travel Transport"

    if cat(row, "category_is_strong_online"):
        return "Telecommunications, Logistics and Digital Infrastructure Services"
    if cat(row, "category_is_online_mixed") and (
        cat(row, "category_is_month_cycle_obvious")
        or cat(row, "category_is_time_distributed")
        or cat_amount_low_or_medium(row)
    ):
        return "Telecommunications, Logistics and Digital Infrastructure Services"
    if cat(row, "category_is_month_cycle_obvious") and cat_amount_low_or_medium(row):
        return "Telecommunications, Logistics and Digital Infrastructure Services"

    if cat(row, "category_is_strong_offline") and (
        cat_amount_medium_or_high(row) or cat(row, "category_is_effective_amount_high")
    ):
        return "Vehicle Energy and Automotive Services"

    if cat(row, "category_is_online_or_mixed") or cat(row, "category_is_month_cycle_obvious"):
        return "Telecommunications, Logistics and Digital Infrastructure Services"
    if cat_frequency_low(row) and cat_amount_high(row):
        return "Long-Distance Travel Transport"
    if cat(row, "category_is_strong_offline"):
        return "Local Commuting and On-Demand Travel" if cat_amount_low_or_medium(row) else "Vehicle Energy and Automotive Services"
    return "Telecommunications, Logistics and Digital Infrastructure Services"


def assign_education_entertainment_level1(row: pd.Series) -> str:
    """
    Split Education, Culture and Recreation using category-relative amount and temporal signals.
    """
    if cat_amount_high(row) and cat_frequency_low(row) and (
        cat(row, "category_is_weekend_enhanced")
        or cat(row, "category_is_month_end_obvious")
        or cat(row, "category_is_online_or_mixed")
    ):
        return "Hotels, Scenic Attractions and Tourism Services"

    if cat_amount_high(row) and (
        cat(row, "category_is_month_cycle_obvious")
        or cat(row, "category_is_month_edge_obvious")
        or cat(row, "category_is_workday_obvious")
        or cat(row, "category_is_work_service_obvious")
    ):
        return "Education Payments and Training Services"

    if cat(row, "category_is_online_or_mixed") and (
        cat_frequency_high_or_medium(row)
        or cat(row, "category_is_night_obvious")
        or cat(row, "category_is_time_distributed")
    ) and not cat_amount_high(row):
        return "Online Entertainment and Hobby Spending"

    if (not cat_amount_high(row)) and cat(row, "category_is_strong_offline") and (
        cat(row, "category_is_weekend_enhanced")
        or cat(row, "category_is_night_obvious")
        or cat(row, "category_is_time_distributed")
    ):
        return "Offline Culture, Sports and Entertainment"

    if cat_amount_high(row):
        if cat_frequency_low(row) or cat(row, "category_is_weekend_enhanced") or cat(row, "category_is_online_or_mixed"):
            return "Hotels, Scenic Attractions and Tourism Services"
        return "Education Payments and Training Services"

    if cat(row, "category_is_online_or_mixed"):
        return "Online Entertainment and Hobby Spending"
    return "Offline Culture, Sports and Entertainment"


def assign_healthcare_level1(row: pd.Series) -> str:
    """
    Split Health Care and Medical Services using category-relative behavioral signals.
    """
    if cat(row, "category_is_strong_offline") and (
        cat(row, "category_is_workday_obvious") or cat(row, "category_is_work_service_obvious")
    ) and (cat_amount_medium_or_high(row) or cat_frequency_low_or_medium(row)):
        return "General Clinical Care and Health Management"

    if cat_amount_high(row) and (
        cat_frequency_low_or_medium(row) or cat(row, "category_is_month_cycle_obvious")
    ):
        return "Specialized Consumer Medical and Care Services"

    if cat_amount_low_or_medium(row) and (
        cat_frequency_high_or_medium(row)
        or cat(row, "category_is_month_cycle_obvious")
        or cat(row, "category_is_time_distributed")
    ):
        return "Pharmaceutical, Medical Device and Health Retail"

    if cat(row, "category_is_strong_offline"):
        return "General Clinical Care and Health Management"
    if cat_amount_high(row):
        return "Specialized Consumer Medical and Care Services"
    return "Pharmaceutical, Medical Device and Health Retail"



# ============================================================
# ============================================================

TRANSPORT_LEVEL1_CLASSES = [
    "Local Commuting and On-Demand Travel",
    "Long-Distance Travel Transport",
    "Vehicle Energy and Automotive Services",
    "Telecommunications, Logistics and Digital Infrastructure Services",
]

TRANSPORT_LEVEL1_COUNT_PRIOR = {c: 0.25 for c in TRANSPORT_LEVEL1_CLASSES}
TRANSPORT_LEVEL1_EFFECTIVE_AMOUNT_PRIOR = {c: 0.25 for c in TRANSPORT_LEVEL1_CLASSES}
TRANSPORT_LEVEL1_RAW_AMOUNT_PRIOR = {c: 0.25 for c in TRANSPORT_LEVEL1_CLASSES}

TRANSPORT_BALANCE_LAMBDA_EFFECTIVE = 5.0
TRANSPORT_BALANCE_LAMBDA_RAW = 0.5
TRANSPORT_BALANCE_LAMBDA_COUNT = 4.0
TRANSPORT_BALANCE_SCORE_WEIGHT = 1.0


def _rank_value(row: pd.Series, name: str, default: float = 0.5) -> float:
    try:
        value = row.get(name, default)
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def transport_level1_semantic_scores(row: pd.Series) -> Dict[str, float]:
    """
    Score four MCC level-1 categories from Transport and Communications profiles.
    Semantic scores are combined with count and amount soft-balancing terms.
    """
    amount_rank = _rank_value(row, "category_amount_rank")
    frequency_rank = _rank_value(row, "category_frequency_rank")
    effective_rank = _rank_value(row, "category_effective_amount_rank")
    commute_rank = _rank_value(row, "category_commute_rank")
    weekend_rank = _rank_value(row, "category_weekend_rank")
    month_end_rank = _rank_value(row, "category_month_end_rank")
    month_cycle_rank = _rank_value(row, "category_month_cycle_rank")
    out_province_rank = _rank_value(row, "category_out_province_rank")
    active_period_rank = _rank_value(row, "category_active_period_rank")

    strong_offline_bonus = 0.6 if cat(row, "category_is_strong_offline") else 0.0
    strong_online_bonus = 1.5 if cat(row, "category_is_strong_online") else 0.0
    online_mixed_bonus = 1.0 if cat(row, "category_is_online_mixed") else 0.0

    scores = {
        "Local Commuting and On-Demand Travel": (
            1.8 * commute_rank
            + 0.8 * frequency_rank
            + 0.5 * (1.0 - amount_rank)
            + 0.4 * active_period_rank
            + strong_offline_bonus
            - 0.5 * month_cycle_rank
        ),
        "Long-Distance Travel Transport": (
            1.2 * amount_rank
            + 0.8 * effective_rank
            + 0.7 * weekend_rank
            + 0.6 * month_end_rank
            + 0.7 * out_province_rank
            + 0.5 * (1.0 - frequency_rank)
            - 0.5 * commute_rank
        ),
        "Vehicle Energy and Automotive Services": (
            1.1 * amount_rank
            + 0.8 * effective_rank
            + (0.8 if cat(row, "category_is_strong_offline") else 0.0)
            + 0.3 * weekend_rank
            - 0.7 * commute_rank
            - 0.7 * out_province_rank
            - 0.5 * month_cycle_rank
        ),
        "Telecommunications, Logistics and Digital Infrastructure Services": (
            1.0 * out_province_rank
            + 1.0 * month_cycle_rank
            + 0.6 * active_period_rank
            + 0.5 * (1.0 - amount_rank)
            + strong_online_bonus
            + online_mixed_bonus
            - 0.4 * weekend_rank
        ),
    }
    return scores


def rebalance_transport_communication_level1(profiles: pd.DataFrame) -> pd.DataFrame:
    profiles = profiles.copy()
    mask = profiles["consumption_category"].eq("Transport and Communications")
    if not mask.any():
        return profiles

    transport = profiles.loc[mask].copy()
    total_n = max(len(transport), 1)
    total_effective = max(float(transport["effective_total_amount"].sum()), 1e-9)
    total_raw = max(float(transport["total_amount"].sum()), 1e-9)

    assigned_n = {c: 0 for c in TRANSPORT_LEVEL1_CLASSES}
    assigned_effective = {c: 0.0 for c in TRANSPORT_LEVEL1_CLASSES}
    assigned_raw = {c: 0.0 for c in TRANSPORT_LEVEL1_CLASSES}
    assignments: Dict[int, str] = {}

    order = _assignment_order(
        transport,
        ["effective_total_amount", "total_amount", "transaction_count"],
    )

    for idx in order:
        row = profiles.loc[idx]
        scores = transport_level1_semantic_scores(row)
        best_level1, best_value = None, -1e18

        for level1 in TRANSPORT_LEVEL1_CLASSES:
            effective_den = total_effective * TRANSPORT_LEVEL1_EFFECTIVE_AMOUNT_PRIOR[level1] + 1e-9
            raw_den = total_raw * TRANSPORT_LEVEL1_RAW_AMOUNT_PRIOR[level1] + 1e-9
            count_den = total_n * TRANSPORT_LEVEL1_COUNT_PRIOR[level1] + 1e-9

            projected_effective_ratio = (assigned_effective[level1] + float(row["effective_total_amount"])) / effective_den
            projected_raw_ratio = (assigned_raw[level1] + float(row["total_amount"])) / raw_den
            projected_count_ratio = (assigned_n[level1] + 1) / count_den

            balance_penalty = (
                TRANSPORT_BALANCE_LAMBDA_EFFECTIVE * projected_effective_ratio ** 2
                + TRANSPORT_BALANCE_LAMBDA_RAW * projected_raw_ratio ** 2
                + TRANSPORT_BALANCE_LAMBDA_COUNT * projected_count_ratio ** 2
            )
            value = TRANSPORT_BALANCE_SCORE_WEIGHT * scores[level1] - balance_penalty

            if value > best_value:
                best_value = value
                best_level1 = level1

        assert best_level1 is not None
        assignments[idx] = best_level1
        assigned_n[best_level1] += 1
        assigned_effective[best_level1] += float(row["effective_total_amount"])
        assigned_raw[best_level1] += float(row["total_amount"])

    profiles.loc[list(assignments.keys()), "behavior_level1"] = [assignments[i] for i in assignments.keys()]
    return profiles



# ============================================================
# ============================================================

BALANCED_LEVEL1_CATEGORIES = [
    "Food, Tobacco and Liquor",
    "Household Equipment, Furnishings and Services",
    "Transport and Communications",
    "Education, Culture and Recreation",
    "Health Care and Medical Services",
]

LEVEL1_BALANCE_LAMBDA_EFFECTIVE = 5.0
LEVEL1_BALANCE_LAMBDA_RAW = 0.5
LEVEL1_BALANCE_LAMBDA_COUNT = 4.0
LEVEL1_BALANCE_SCORE_WEIGHT = 1.0

DEFAULT_LEVEL1_BALANCE_CONFIG = {
    "Food, Tobacco and Liquor": {"effective": 1.5, "raw": 0.5, "count": 5.0, "score": 1.1},
    "Household Equipment, Furnishings and Services": {"effective": 4.0, "raw": 1.5, "count": 2.5, "score": 1.1},
    "Transport and Communications": {"effective": 4.0, "raw": 1.5, "count": 2.5, "score": 1.1},
    "Education, Culture and Recreation": {"effective": 4.0, "raw": 1.5, "count": 2.5, "score": 1.1},
    "Health Care and Medical Services": {"effective": 4.0, "raw": 1.5, "count": 2.5, "score": 1.1},
}

AUTO_TUNE_GRID = {
    "effective": [1.5, 3.0, 5.0],
    "raw": [0.5, 1.2, 2.0],
    "count": [1.0, 2.5, 4.5],
    "score": [0.9, 1.2],
}

AUTO_TUNE_OBJECTIVE_WEIGHTS = {
    "count": 0.25,
    "effective": 0.30,
    "raw": 0.25,
    "semantic": 0.20,
}

AUTO_TUNE_IMPOSSIBLE_BUFFER = 0.06


def _rv(row: pd.Series, name: str, default: float = 0.5) -> float:
    return _rank_value(row, name, default)


def _b(row: pd.Series, name: str) -> float:
    return 1.0 if cat(row, name) else 0.0


def _common_rank_features(row: pd.Series) -> Dict[str, float]:
    return {
        "amount": _rv(row, "category_amount_rank"),
        "frequency": _rv(row, "category_frequency_rank"),
        "effective": _rv(row, "category_effective_amount_rank"),
        "weekend": _rv(row, "category_weekend_rank"),
        "month_start": _rv(row, "category_month_start_rank"),
        "month_middle": _rv(row, "category_month_middle_rank"),
        "month_end": _rv(row, "category_month_end_rank"),
        "month_cycle": _rv(row, "category_month_cycle_rank"),
        "out_province": _rv(row, "category_out_province_rank"),
        "commute": _rv(row, "category_commute_rank"),
        "dining": _rv(row, "category_dining_rank"),
        "night": _rv(row, "category_night_rank"),
        "work_service": _rv(row, "category_work_service_rank"),
        "active_period": _rv(row, "category_active_period_rank"),
        "offline": _b(row, "category_is_strong_offline"),
        "online": _b(row, "category_is_strong_online"),
        "mixed": _b(row, "category_is_online_mixed"),
        "online_or_mixed": _b(row, "category_is_online_or_mixed"),
    }


def food_level1_semantic_scores(row: pd.Series) -> Dict[str, float]:
    f = _common_rank_features(row)
    return {
        "Food, Tobacco and Liquor Retail": (
            1.1 * f["frequency"]
            + 1.0 * (1.0 - f["amount"])
            + 0.8 * f["active_period"]
            + 0.8 * (1.0 - f["dining"])
            + 0.3 * (1.0 - f["night"])
        ),
        "Food and Beverage Services": (
            1.6 * f["dining"]
            + 0.9 * f["night"]
            + 0.8 * f["amount"]
            + 0.3 * f["weekend"]
            - 0.4 * f["active_period"]
        ),
    }


def life_goods_service_level1_semantic_scores(row: pd.Series) -> Dict[str, float]:
    f = _common_rank_features(row)
    return {
        "Household and Personal Goods Retail": (
            1.0 * f["active_period"]
            + 0.8 * f["online_or_mixed"]
            + 0.8 * f["effective"]
            + 0.5 * f["frequency"]
            + 0.3 * f["amount"]
            - 0.4 * f["weekend"]
        ),
        "Local Lifestyle and Self-Service Sharing": (
            1.2 * f["offline"]
            + 1.1 * f["weekend"]
            + 0.8 * f["night"]
            + 0.4 * f["amount"]
            + 0.3 * (1.0 - f["frequency"])
            - 0.3 * f["online"]
        ),
    }


def education_entertainment_level1_semantic_scores(row: pd.Series) -> Dict[str, float]:
    f = _common_rank_features(row)
    month_edge = max(f["month_start"], f["month_end"])
    return {
        "Education Payments and Training Services": (
            1.2 * f["amount"]
            + 0.9 * f["effective"]
            + 1.0 * f["month_cycle"]
            + 0.8 * month_edge
            + 0.7 * f["work_service"]
            + 0.3 * (1.0 - f["frequency"])
        ),
        "Offline Culture, Sports and Entertainment": (
            1.2 * f["offline"]
            + 1.0 * f["weekend"]
            + 0.9 * f["night"]
            + 0.6 * f["active_period"]
            + 0.5 * (1.0 - f["amount"])
            - 0.3 * f["online"]
        ),
        "Online Entertainment and Hobby Spending": (
            1.3 * f["online_or_mixed"]
            + 1.0 * f["night"]
            + 0.8 * f["active_period"]
            + 0.6 * f["frequency"]
            + 0.5 * (1.0 - f["amount"])
            - 0.3 * f["weekend"]
        ),
        "Hotels, Scenic Attractions and Tourism Services": (
            1.2 * f["amount"]
            + 0.9 * f["effective"]
            + 1.0 * f["weekend"]
            + 0.8 * f["month_end"]
            + 0.9 * f["out_province"]
            + 0.5 * (1.0 - f["frequency"])
            - 0.3 * f["work_service"]
        ),
    }


def healthcare_level1_semantic_scores(row: pd.Series) -> Dict[str, float]:
    f = _common_rank_features(row)
    workday_signal = max(f["work_service"], 1.0 - f["weekend"])
    return {
        "Pharmaceutical, Medical Device and Health Retail": (
            1.0 * (1.0 - f["amount"])
            + 0.9 * f["frequency"]
            + 0.8 * f["active_period"]
            + 0.8 * f["month_cycle"]
            + 0.5 * f["online_or_mixed"]
        ),
        "General Clinical Care and Health Management": (
            1.2 * f["offline"]
            + 1.0 * workday_signal
            + 0.8 * f["amount"]
            + 0.6 * f["effective"]
            + 0.3 * (1.0 - f["frequency"])
            - 0.4 * f["online"]
        ),
        "Specialized Consumer Medical and Care Services": (
            1.4 * f["amount"]
            + 0.8 * f["effective"]
            + 0.7 * (1.0 - f["frequency"])
            + 0.8 * f["month_cycle"]
            + 0.5 * f["online_or_mixed"]
            + 0.3 * f["weekend"]
        ),
    }


def level1_semantic_scores_for_category(row: pd.Series, category: str) -> Dict[str, float]:
    if category == "Food, Tobacco and Liquor":
        return food_level1_semantic_scores(row)
    if category == "Household Equipment, Furnishings and Services":
        return life_goods_service_level1_semantic_scores(row)
    if category == "Transport and Communications":
        return transport_level1_semantic_scores(row)
    if category == "Education, Culture and Recreation":
        return education_entertainment_level1_semantic_scores(row)
    if category == "Health Care and Medical Services":
        return healthcare_level1_semantic_scores(row)
    return {level1: 1.0 for level1 in BEHAVIOR_HIERARCHY[category].keys()}


def rebalance_category_level1(
    profiles: pd.DataFrame,
    category: str,
    count_prior: Dict[str, float] | None = None,
    effective_prior: Dict[str, float] | None = None,
    raw_prior: Dict[str, float] | None = None,
    balance_config: Dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Assign MCC level 1 within one consumption category using semantic scores and soft balance.
    """
    profiles = profiles.copy()
    mask = profiles["consumption_category"].eq(category)
    if not mask.any():
        return profiles

    classes = list(BEHAVIOR_HIERARCHY[category].keys())
    if len(classes) <= 1:
        return profiles

    if count_prior is None:
        count_prior = {c: 1.0 / len(classes) for c in classes}
    if effective_prior is None:
        effective_prior = {c: 1.0 / len(classes) for c in classes}
    if raw_prior is None:
        raw_prior = {c: 1.0 / len(classes) for c in classes}

    subset = profiles.loc[mask].copy()
    total_n = max(len(subset), 1)
    total_effective = max(float(subset["effective_total_amount"].sum()), 1e-9)
    total_raw = max(float(subset["total_amount"].sum()), 1e-9)

    assigned_n = {c: 0 for c in classes}
    assigned_effective = {c: 0.0 for c in classes}
    assigned_raw = {c: 0.0 for c in classes}
    assignments: Dict[int, str] = {}

    order = _assignment_order(
        subset,
        ["effective_total_amount", "total_amount", "transaction_count"],
    )

    for idx in order:
        row = profiles.loc[idx]
        scores = level1_semantic_scores_for_category(row, category)
        best_level1, best_value = None, -1e18

        for level1 in classes:
            effective_den = total_effective * effective_prior[level1] + 1e-9
            raw_den = total_raw * raw_prior[level1] + 1e-9
            count_den = total_n * count_prior[level1] + 1e-9

            projected_effective_ratio = (assigned_effective[level1] + float(row["effective_total_amount"])) / effective_den
            projected_raw_ratio = (assigned_raw[level1] + float(row["total_amount"])) / raw_den
            projected_count_ratio = (assigned_n[level1] + 1) / count_den

            cfg = balance_config or DEFAULT_LEVEL1_BALANCE_CONFIG.get(category, {})
            lambda_effective = float(cfg.get("effective", LEVEL1_BALANCE_LAMBDA_EFFECTIVE))
            lambda_raw = float(cfg.get("raw", LEVEL1_BALANCE_LAMBDA_RAW))
            lambda_count = float(cfg.get("count", LEVEL1_BALANCE_LAMBDA_COUNT))
            score_weight = float(cfg.get("score", LEVEL1_BALANCE_SCORE_WEIGHT))

            balance_penalty = (
                lambda_effective * projected_effective_ratio ** 2
                + lambda_raw * projected_raw_ratio ** 2
                + lambda_count * projected_count_ratio ** 2
            )
            value = score_weight * float(scores.get(level1, 0.0)) - balance_penalty

            if value > best_value:
                best_value = value
                best_level1 = level1

        assert best_level1 is not None
        assignments[idx] = best_level1
        assigned_n[best_level1] += 1
        assigned_effective[best_level1] += float(row["effective_total_amount"])
        assigned_raw[best_level1] += float(row["total_amount"])

    profiles.loc[list(assignments.keys()), "behavior_level1"] = [assignments[i] for i in assignments.keys()]
    return profiles



def _safe_share_array(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    total = float(arr.sum())
    if total <= 0:
        return np.ones_like(arr) / max(len(arr), 1)
    return arr / total


def _level1_summary_arrays(profiles: pd.DataFrame, category: str) -> Dict[str, Any]:
    classes = list(BEHAVIOR_HIERARCHY[category].keys())
    subset = profiles[profiles["consumption_category"].eq(category)].copy()
    if subset.empty:
        k = len(classes)
        zero = np.zeros(k, dtype=float)
        return {"classes": classes, "count": zero, "effective": zero, "raw": zero,
                "count_share": zero, "effective_share": zero, "raw_share": zero}

    grouped = subset.groupby("behavior_level1").agg(
        count=("target_id", "count"),
        effective=("effective_total_amount", "sum"),
        raw=("total_amount", "sum"),
    )
    count = np.array([float(grouped.loc[c, "count"]) if c in grouped.index else 0.0 for c in classes], dtype=float)
    effective = np.array([float(grouped.loc[c, "effective"]) if c in grouped.index else 0.0 for c in classes], dtype=float)
    raw = np.array([float(grouped.loc[c, "raw"]) if c in grouped.index else 0.0 for c in classes], dtype=float)
    return {
        "classes": classes,
        "count": count,
        "effective": effective,
        "raw": raw,
        "count_share": _safe_share_array(count.tolist()),
        "effective_share": _safe_share_array(effective.tolist()),
        "raw_share": _safe_share_array(raw.tolist()),
    }


def _semantic_fidelity_loss(profiles: pd.DataFrame, category: str) -> Tuple[float, float]:
    subset = profiles[profiles["consumption_category"].eq(category)]
    if subset.empty:
        return 0.0, 0.0
    losses, not_top = [], []
    for _, row in subset.iterrows():
        scores = level1_semantic_scores_for_category(row, category)
        if not scores:
            continue
        assigned = row.get("behavior_level1")
        max_score = max(scores.values())
        assigned_score = float(scores.get(assigned, max_score))
        denom = max(abs(max_score), 1.0)
        losses.append(max(0.0, max_score - assigned_score) / denom)
        not_top.append(0.0 if assigned_score >= max_score - 1e-9 else 1.0)
    if not losses:
        return 0.0, 0.0
    return float(np.mean(losses)), float(np.mean(not_top))


def _metric_balance_loss(shares: np.ndarray, target: float, single_max_share: float, buffer: float) -> float:
    """
    Combine overall deviation from the balance target with an excess-concentration penalty.
    """
    if len(shares) == 0:
        return 0.0
    rmse = float(np.sqrt(np.mean((shares - target) ** 2))) / max(target, 1e-9)
    peak_floor = max(target, float(single_max_share)) + buffer
    peak_excess = max(0.0, float(np.max(shares)) - peak_floor) / max(target, 1e-9)
    return 0.45 * rmse + 0.55 * peak_excess


def evaluate_level1_assignment(profiles: pd.DataFrame, category: str) -> Dict[str, float]:
    subset = profiles[profiles["consumption_category"].eq(category)]
    classes = list(BEHAVIOR_HIERARCHY[category].keys())
    k = max(len(classes), 1)
    target = 1.0 / k
    arrays = _level1_summary_arrays(profiles, category)

    total_effective = max(float(subset["effective_total_amount"].sum()), 1e-9)
    total_raw = max(float(subset["total_amount"].sum()), 1e-9)
    single_effective_share = float(subset["effective_total_amount"].max() / total_effective) if len(subset) else 0.0
    single_raw_share = float(subset["total_amount"].max() / total_raw) if len(subset) else 0.0

    count_loss = _metric_balance_loss(arrays["count_share"], target, 1.0 / max(len(subset), 1), AUTO_TUNE_IMPOSSIBLE_BUFFER)
    effective_loss = _metric_balance_loss(arrays["effective_share"], target, single_effective_share, AUTO_TUNE_IMPOSSIBLE_BUFFER)
    raw_loss = _metric_balance_loss(arrays["raw_share"], target, single_raw_share, AUTO_TUNE_IMPOSSIBLE_BUFFER)
    semantic_loss, not_top_ratio = _semantic_fidelity_loss(profiles, category)

    w = AUTO_TUNE_OBJECTIVE_WEIGHTS
    objective = (
        w["count"] * count_loss
        + w["effective"] * effective_loss
        + w["raw"] * raw_loss
        + w["semantic"] * semantic_loss
    )
    return {
        "objective": float(objective),
        "count_loss": float(count_loss),
        "effective_loss": float(effective_loss),
        "raw_loss": float(raw_loss),
        "semantic_loss": float(semantic_loss),
        "not_top_ratio": float(not_top_ratio),
        "max_count_share": float(np.max(arrays["count_share"])) if len(arrays["count_share"]) else 0.0,
        "max_effective_share": float(np.max(arrays["effective_share"])) if len(arrays["effective_share"]) else 0.0,
        "max_raw_share": float(np.max(arrays["raw_share"])) if len(arrays["raw_share"]) else 0.0,
        "single_effective_share": single_effective_share,
        "single_raw_share": single_raw_share,
    }


def _iter_level1_balance_candidates() -> List[Dict[str, float]]:
    candidates = []
    for effective in AUTO_TUNE_GRID["effective"]:
        for raw in AUTO_TUNE_GRID["raw"]:
            for count in AUTO_TUNE_GRID["count"]:
                for score in AUTO_TUNE_GRID["score"]:
                    candidates.append({
                        "effective": float(effective),
                        "raw": float(raw),
                        "count": float(count),
                        "score": float(score),
                    })
    return candidates


def autotune_level1_balance_configs(profiles: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """
    Search soft-balance weights while keeping semantic scoring functions fixed.
    """
    tuned: Dict[str, Dict[str, float]] = {}
    records: List[Dict[str, Any]] = []
    candidates = _iter_level1_balance_candidates()

    for category in BALANCED_LEVEL1_CATEGORIES:
        if profiles["consumption_category"].eq(category).sum() == 0:
            continue
        best_cfg = DEFAULT_LEVEL1_BALANCE_CONFIG.get(category, {
            "effective": LEVEL1_BALANCE_LAMBDA_EFFECTIVE,
            "raw": LEVEL1_BALANCE_LAMBDA_RAW,
            "count": LEVEL1_BALANCE_LAMBDA_COUNT,
            "score": LEVEL1_BALANCE_SCORE_WEIGHT,
        })
        best_metrics = None
        best_obj = float("inf")

        for cfg in candidates:
            trial = rebalance_category_level1(
                profiles,
                category,
                balance_config=cfg,
            )
            metrics = evaluate_level1_assignment(trial, category)
            obj = metrics["objective"]
            if obj < best_obj:
                best_obj = obj
                best_cfg = cfg
                best_metrics = metrics

        tuned[category] = dict(best_cfg)
        rec = {"consumption_category": category, **best_cfg}
        if best_metrics:
            rec.update(best_metrics)
        records.append(rec)

    report = pd.DataFrame(records).sort_values("consumption_category") if records else pd.DataFrame()
    return tuned, report


def rebalance_all_multi_level1_categories(
    profiles: pd.DataFrame,
    auto_tune_level1: bool = True,
    tuning_report_path: str | None = None,
) -> pd.DataFrame:
    profiles = profiles.copy()
    if auto_tune_level1:
        tuned_configs, tuning_report = autotune_level1_balance_configs(profiles)
        profiles.attrs["level1_balance_config"] = tuned_configs
        if tuning_report_path is not None and not tuning_report.empty:
            Path(tuning_report_path).parent.mkdir(parents=True, exist_ok=True)
            tuning_report.to_csv(tuning_report_path, index=False, encoding="utf-8-sig")
    else:
        tuned_configs = DEFAULT_LEVEL1_BALANCE_CONFIG

    for category in BALANCED_LEVEL1_CATEGORIES:
        profiles = rebalance_category_level1(
            profiles,
            category,
            balance_config=tuned_configs.get(category, DEFAULT_LEVEL1_BALANCE_CONFIG.get(category, {})),
        )
    return profiles


def assign_behavior_level1(
    profiles: pd.DataFrame,
    auto_tune_level1: bool = True,
    tuning_report_path: str | None = None,
) -> pd.DataFrame:
    profiles = profiles.copy()
    if "category_amount_level" not in profiles.columns:
        profiles = attach_category_relative_features(profiles)

    profiles["behavior_level1"] = profiles.apply(assign_behavior_level1_by_rules, axis=1)
    profiles = rebalance_all_multi_level1_categories(
        profiles,
        auto_tune_level1=auto_tune_level1,
        tuning_report_path=tuning_report_path,
    )
    return profiles


# ============================================================
# ============================================================

def get_behavior_level2_weight(level2: str, row: pd.Series) -> float:
    return 1.0


def weighted_choice(items: List[Tuple[str, float]], rng: random.Random) -> str:
    total_weight = sum(w for _, w in items)
    if total_weight <= 0:
        return rng.choice(items)[0]
    r = rng.uniform(0, total_weight)
    acc = 0.0
    for item, weight in items:
        acc += weight
        if acc >= r:
            return item
    return items[-1][0]


def assign_behavior_level2(profiles: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    profiles = profiles.copy()
    rng = random.Random(seed)

    level2_values, detail_candidates = [], []
    # Legacy historical order (not used): processing_profiles = profiles
    processing_profiles = profiles.sort_values("target_id", kind="mergesort")

    assignment: Dict[Any, Tuple[str, str]] = {}
    for idx, row in processing_profiles.iterrows():
        category = row["consumption_category"]
        level1 = row["behavior_level1"]
        level2_dict = BEHAVIOR_HIERARCHY[category][level1]
        weighted_items = [
            (level2, get_behavior_level2_weight(level2, row))
            for level2 in level2_dict.keys()
        ]
        level2 = weighted_choice(weighted_items, rng)
        assignment[idx] = (level2, json.dumps(level2_dict[level2], ensure_ascii=False))

    for idx in profiles.index:
        level2, detail_candidates_json = assignment[idx]
        level2_values.append(level2)
        detail_candidates.append(detail_candidates_json)

    profiles["behavior_level2"] = level2_values
    profiles["level2_detail_candidates"] = detail_candidates

    profiles["mcc_level1"] = profiles["behavior_level1"]
    profiles["mcc_level2"] = profiles["behavior_level2"]
    return profiles


# ============================================================
# ============================================================

def build_behavior_count_reports(profiles: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_targets = len(profiles)
    if total_targets == 0:
        raise ValueError("profiles is empty; statistics cannot be generated.")

    level1_report = (
        profiles
        .groupby(["consumption_category", "behavior_level1"])
        .agg(
            target_count=("target_id", "count"),
            total_amount=("total_amount", "sum"),
            avg_target_amount=("total_amount", "mean"),
            avg_out_province_ratio=("out_province_ratio", "mean"),
            avg_weekend_ratio=("weekend_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["consumption_category", "target_count", "total_amount"], ascending=[True, False, False])
    )
    level1_report["target_ratio"] = level1_report["target_count"] / total_targets

    level2_report = (
        profiles
        .groupby(["consumption_category", "behavior_level1", "behavior_level2"])
        .agg(
            target_count=("target_id", "count"),
            total_amount=("total_amount", "sum"),
            avg_target_amount=("total_amount", "mean"),
        )
        .reset_index()
        .sort_values(["consumption_category", "behavior_level1", "target_count", "total_amount"], ascending=[True, True, False, False])
    )
    level2_report["target_ratio"] = level2_report["target_count"] / total_targets

    cross_report = level2_report.copy()
    return level1_report, level2_report, cross_report


def run_initialization(
    df: pd.DataFrame,
    target_col: str,
    amount_col: str,
    seed: int,
    auto_tune_level1: bool = True,
    tuning_report_path: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profiles = build_target_profiles(df, target_col, amount_col)
    profiles = attach_candidate_categories(profiles)
    profiles, report = greedy_assign_consumption_category(profiles)
    profiles = assign_behavior_level1(
        profiles,
        auto_tune_level1=auto_tune_level1,
        tuning_report_path=tuning_report_path,
    )
    profiles = assign_behavior_level2(profiles, seed=seed)
    return profiles, report


def build_target_table(merchant_df: pd.DataFrame) -> pd.DataFrame:
    df = merchant_df.copy()
    if "target_id" in df.columns and "Target" not in df.columns:
        df = df.rename(columns={"target_id": "Target"})

    required = ["Target", "behavior_level1", "behavior_level2"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Merchant table missing columns for target.csv: {missing}")

    return df[required].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize FFSD merchant semantic categories.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_TRANSACTION_INPUT),
        help="Transaction table path. Defaults to work/transaction.csv.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_TARGET_OUTPUT),
        help="Merchant table output path. Defaults to work/target.csv.",
    )
    parser.add_argument("--target_col", default="Target", help="Target identifier column.")
    parser.add_argument("--amount_col", default="Amount", help="Transaction amount column.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--disable_auto_tune_level1",
        action="store_true",
        help="Disable automatic MCC level-1 soft-balance calibration.",
    )
    parser.add_argument(
        "--level1_tuning_output",
        default=None,
        help="Optional MCC level-1 calibration report path.",
    )
    parser.add_argument(
        "--write_reports",
        action="store_true",
        help="Write optional category allocation reports.",
    )
    parser.add_argument("--report_output", default=None, help="Optional consumption-category amount report.")
    parser.add_argument("--behavior_count_output", default=None, help="Optional MCC count report prefix.")
    args = parser.parse_args()

    output_path = Path(args.output)

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise ValueError("Input must be a CSV or Excel file.")

    missing = [c for c in [args.target_col, args.amount_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns {missing}; got {list(df.columns)}")

    auto_tune_level1 = not args.disable_auto_tune_level1
    tuning_report_path = args.level1_tuning_output
    if tuning_report_path is None and args.write_reports:
        out_path_tmp = output_path
        tuning_report_path = str(out_path_tmp.with_name(out_path_tmp.stem + "_level1_tuning_report.csv"))

    profiles, report = run_initialization(
        df,
        args.target_col,
        args.amount_col,
        args.seed,
        auto_tune_level1=auto_tune_level1,
        tuning_report_path=tuning_report_path,
    )
    target_df = build_target_table(profiles)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Wrote merchant table: {output_path}")

    if not args.write_reports:
        return

    profiles = profiles.copy()
    profiles["candidate_categories"] = profiles["candidate_categories"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    report_output = args.report_output
    if report_output is None:
        report_output = str(output_path.with_name(output_path.stem + "_amount_report.csv"))
    report.to_csv(report_output, index=False, encoding="utf-8-sig")

    level1_report, level2_report, cross_report = build_behavior_count_reports(profiles)
    if args.behavior_count_output is None:
        behavior_count_prefix = str(output_path.with_name(output_path.stem + "_behavior_count"))
    else:
        behavior_count_prefix = args.behavior_count_output

    level1_output = behavior_count_prefix + "_level1.csv"
    level2_output = behavior_count_prefix + "_level2.csv"
    cross_output = behavior_count_prefix + "_cross.csv"

    level1_report.to_csv(level1_output, index=False, encoding="utf-8-sig")
    level2_report.to_csv(level2_output, index=False, encoding="utf-8-sig")
    cross_report.to_csv(cross_output, index=False, encoding="utf-8-sig")

    print(f"[OK] Wrote consumption-category amount report: {report_output}")
    if tuning_report_path is not None and Path(tuning_report_path).exists():
        print(f"[OK] Wrote MCC level-1 calibration report: {tuning_report_path}")
    print(f"[OK] Wrote MCC level-1 count report: {level1_output}")
    print(f"[OK] Wrote MCC level-2 count report: {level2_output}")
    print(f"[OK] Wrote category cross-tabulation: {cross_output}")

    print("\nConsumption-category amount and merchant-count check:")
    print(report[[
        "consumption_category",
        "official_amount_ratio",
        "merchant_amount_prior",
        "assigned_amount_ratio",
        "amount_ratio_diff",
        "merchant_count_prior",
        "assigned_count_ratio",
        "count_ratio_diff",
        "assigned_count",
    ]].to_string(index=False))

    print("\nMCC level-1 merchant counts:")
    print(level1_report[["consumption_category", "behavior_level1", "target_count", "target_ratio", "total_amount"]].to_string(index=False))


if __name__ == "__main__":
    main()
