#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Synthetic timestamp generator for FFSD.

This script assigns synthetic, physically meaningful timestamps to FFSD rows.
It uses IEEE-CIS as a rhythm reference, while preserving FFSD's original id
ordering. The generated timestamps are synthetic context, not recovered truth.

The implementation intentionally keeps the main pieces readable:
1. Learn global/user/fraud rhythms from IEEE-CIS.
2. Build IEEE pseudo users from transaction + identity fields.
3. Match FFSD source users to similar IEEE pseudo users.
4. Detect local FFSD blocks, especially fraud/anomaly blocks.
5. Generate ordered timestamps within 2021-01 to 2021-10.
"""

from __future__ import print_function

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd


SECONDS_PER_BIN = 30 * 60
HOUR_OF_WEEK_BINS = 7 * 48


@dataclass
class GeneratorConfig:
    ffsd_order_col: str = "id"
    ffsd_source_col: str = "source"
    ffsd_target_col: str = "target"
    ffsd_amount_col: str = "amount"
    ffsd_label_col: str = "label"
    ffsd_fraud_positive_label: int = 1

    ieee_transaction_id_col: str = "TransactionID"
    ieee_time_col: str = "TransactionDT"
    ieee_amount_col: str = "TransactionAmt"
    ieee_label_col: str = "isFraud"

    synthetic_start: str = "2021-01-01 00:00:00"
    synthetic_end: str = "2021-10-31 23:59:59"
    time_bin_minutes: int = 30
    smooth_alpha: float = 1.0
    random_seed: int = 42

    calibrate_daily_phase: bool = True
    low_activity_target_start_hour: int = 2
    low_activity_window_hours: int = 3

    pseudo_user_match_top_k: int = 5
    min_ieee_user_tx_for_matching: int = 3
    min_ffsd_source_tx_for_user_matching: int = 3
    max_ffsd_sources_for_matching: int = 5000
    max_ieee_users_for_matching: int = 5000

    fraud_controller_mode: str = "label_aware"  # label_aware or label_blind
    fraud_burst_min_count: int = 2
    max_block_size: int = 20
    equal_amount_tolerance: float = 0.01

    allow_same_second_transactions: bool = True


def ensure_columns(df, cols, df_name):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("%s is missing required columns: %s" % (df_name, missing))


def safe_read_csv(path):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def stable_hash_frame(df, prefix):
    """Create stable string ids from the available columns of a small frame."""
    if df.shape[1] == 0:
        return pd.Series([prefix + "_missing"] * len(df), index=df.index)
    values = df.fillna("__NA__").astype(str)
    hashed = pd.util.hash_pandas_object(values, index=False).astype(str)
    return prefix + "_" + hashed


def circular_distance(a, b, mod):
    d = abs(a - b) % mod
    return min(d, mod - d)


def choose_label_column(df, preferred):
    if preferred in df.columns:
        return preferred
    for candidate in ["isFraud", "label", "fraud", "target"]:
        if candidate in df.columns:
            return candidate
    return None


def merge_ieee_tables(transaction_df, identity_df, label_df, config):
    """Merge IEEE transaction, identity, and label tables when they are present."""
    ensure_columns(transaction_df, [config.ieee_transaction_id_col, config.ieee_time_col], "IEEE transaction")
    ieee = transaction_df.copy()

    if identity_df is not None and config.ieee_transaction_id_col in identity_df.columns:
        ieee = ieee.merge(identity_df, on=config.ieee_transaction_id_col, how="left", suffixes=("", "_identity"))

    if label_df is not None:
        ensure_columns(label_df, [config.ieee_transaction_id_col], "IEEE label")
        label_col = choose_label_column(label_df, config.ieee_label_col)
        if label_col is None:
            # If the file only has TransactionID and one other column, treat the other column as label.
            other_cols = [c for c in label_df.columns if c != config.ieee_transaction_id_col]
            if len(other_cols) == 1:
                label_col = other_cols[0]
        if label_col is None:
            raise ValueError("Cannot identify label column in IEEE label file.")
        labels = label_df[[config.ieee_transaction_id_col, label_col]].copy()
        labels = labels.rename(columns={label_col: config.ieee_label_col})
        if config.ieee_label_col in ieee.columns:
            ieee = ieee.drop(columns=[config.ieee_label_col])
        ieee = ieee.merge(labels, on=config.ieee_transaction_id_col, how="left")

    if config.ieee_label_col not in ieee.columns:
        ieee[config.ieee_label_col] = 0
    ieee[config.ieee_label_col] = ieee[config.ieee_label_col].fillna(0).astype(int)

    if config.ieee_amount_col not in ieee.columns:
        # Keep the pipeline runnable even for reduced IEEE samples.
        ieee[config.ieee_amount_col] = 0.0

    return ieee


def add_ieee_pseudo_ids(ieee):
    """Construct pseudo users/sessions because IEEE-CIS has no explicit user id."""
    high_user_cols = [
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain",
    ]
    mid_user_cols = [
        "card1", "card2", "card3", "card5", "card6", "addr1", "P_emaildomain",
    ]
    session_cols = ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33", "id_38"]
    merchant_cols = ["ProductCD", "R_emaildomain"]

    available_high = [c for c in high_user_cols if c in ieee.columns]
    available_mid = [c for c in mid_user_cols if c in ieee.columns]
    available_session = [c for c in session_cols if c in ieee.columns]
    available_merchant = [c for c in merchant_cols if c in ieee.columns]

    ieee = ieee.copy()
    if len(available_high) >= 4:
        high_ids = stable_hash_frame(ieee[available_high], "user_high")
        high_unique_ratio = float(high_ids.nunique()) / max(float(len(high_ids)), 1.0)
        if high_unique_ratio <= 0.95:
            ieee["ieee_pseudo_user_id"] = high_ids
            user_confidence = "high"
        elif len(available_mid) >= 3:
            # A very sparse high-confidence key cannot support user-level rhythm
            # learning, so fall back to a broader payment-subject key.
            ieee["ieee_pseudo_user_id"] = stable_hash_frame(ieee[available_mid], "user_mid")
            user_confidence = "medium_high_key_too_sparse"
        else:
            ieee["ieee_pseudo_user_id"] = high_ids
            user_confidence = "high_sparse"
    elif len(available_mid) >= 3:
        ieee["ieee_pseudo_user_id"] = stable_hash_frame(ieee[available_mid], "user_mid")
        user_confidence = "medium"
    else:
        # Fallback: group by the strongest available transaction-side fields.
        fallback_cols = [c for c in ["card1", "addr1", "P_emaildomain", "ProductCD"] if c in ieee.columns]
        ieee["ieee_pseudo_user_id"] = stable_hash_frame(ieee[fallback_cols], "user_fallback")
        user_confidence = "low"

    session_base = ["ieee_pseudo_user_id"] + available_session
    ieee["ieee_pseudo_session_id"] = stable_hash_frame(ieee[session_base], "session")
    ieee["ieee_pseudo_merchant_context"] = stable_hash_frame(ieee[available_merchant], "merchant")

    return ieee, {
        "pseudo_user_confidence": user_confidence,
        "pseudo_user_fields_high_available": available_high,
        "pseudo_user_fields_mid_available": available_mid,
        "pseudo_session_fields_available": available_session,
        "pseudo_merchant_fields_available": available_merchant,
    }


def calibrate_ieee_phase(ieee, config):
    """Shift TransactionDT hours so low activity aligns with 02:00-05:00."""
    dt = ieee[config.ieee_time_col].astype(float).values
    best_shift = 0
    best_score = None
    target_start = config.low_activity_target_start_hour
    window = config.low_activity_window_hours

    for shift in range(24):
        hours = ((dt + shift * 3600) // 3600).astype(int) % 24
        counts = np.bincount(hours, minlength=24).astype(float)
        window_counts = []
        for start in range(24):
            total = sum(counts[(start + j) % 24] for j in range(window))
            window_counts.append(total)
        min_start = int(np.argmin(window_counts))
        score = (window_counts[min_start], circular_distance(min_start, target_start, 24))
        if best_score is None or score[1] < best_score[1] or (score[1] == best_score[1] and score[0] < best_score[0]):
            best_score = score
            best_shift = shift

    return best_shift


def add_time_features_from_transaction_dt(ieee, config, shift_hours):
    reference = pd.Timestamp(config.synthetic_start)
    shifted_seconds = ieee[config.ieee_time_col].astype(float) + shift_hours * 3600
    dt = reference + pd.to_timedelta(shifted_seconds, unit="s")

    ieee = ieee.copy()
    ieee["_ieee_synthetic_clock"] = dt
    ieee["_dow"] = dt.dt.dayofweek.astype(int)
    ieee["_minute_of_day"] = dt.dt.hour.astype(int) * 60 + dt.dt.minute.astype(int)
    ieee["_half_hour_bin"] = (ieee["_minute_of_day"] // 30).astype(int)
    ieee["_hour_of_week_bin"] = ieee["_dow"] * 48 + ieee["_half_hour_bin"]
    ieee["_day_of_month"] = dt.dt.day.astype(int)
    return ieee


def smoothed_prob_from_counts(counts, alpha):
    values = np.asarray(counts, dtype=float) + float(alpha)
    total = values.sum()
    if total <= 0:
        return np.ones(len(values), dtype=float) / float(len(values))
    return values / total


def learn_global_rhythms(ieee, config):
    how_counts = np.bincount(ieee["_hour_of_week_bin"].astype(int), minlength=HOUR_OF_WEEK_BINS)
    dom_counts = np.bincount(ieee["_day_of_month"].astype(int), minlength=32)[1:32]

    fraud = ieee[ieee[config.ieee_label_col].astype(int) == 1]
    if len(fraud) > 0:
        fraud_how_counts = np.bincount(fraud["_hour_of_week_bin"].astype(int), minlength=HOUR_OF_WEEK_BINS)
    else:
        fraud_how_counts = how_counts.copy()

    return {
        "global_hour_of_week": smoothed_prob_from_counts(how_counts, config.smooth_alpha),
        "day_of_month": smoothed_prob_from_counts(dom_counts, config.smooth_alpha),
        "fraud_hour_of_week": smoothed_prob_from_counts(fraud_how_counts, config.smooth_alpha),
    }


def profile_group(group, amount_col, label_col, counterparty_col):
    amounts = pd.to_numeric(group[amount_col], errors="coerce").fillna(0.0)
    rounded = (amounts.round(0) == amounts).mean() if len(amounts) else 0.0
    amount_mean = float(amounts.mean()) if len(amounts) else 0.0
    amount_std = float(amounts.std(ddof=0)) if len(amounts) else 0.0
    amount_cv = amount_std / (abs(amount_mean) + 1e-6)
    same_amount_repeat = 1.0 - float(amounts.round(2).nunique()) / max(float(len(amounts)), 1.0)
    unique_counterparty_ratio = float(group[counterparty_col].nunique()) / max(float(len(group)), 1.0)
    if label_col in group.columns:
        fraud_ratio = float(pd.to_numeric(group[label_col], errors="coerce").fillna(0).mean())
    else:
        fraud_ratio = 0.0
    return {
        "tx_count_log": math.log1p(len(group)),
        "fraud_ratio": fraud_ratio,
        "amount_mean_log": math.log1p(abs(amount_mean)),
        "amount_std_log": math.log1p(abs(amount_std)),
        "amount_cv": min(amount_cv, 10.0),
        "round_amount_ratio": float(rounded),
        "unique_counterparty_ratio": unique_counterparty_ratio,
        "same_amount_repeat_ratio": same_amount_repeat,
    }


PROFILE_COLUMNS = [
    "tx_count_log",
    "fraud_ratio",
    "amount_mean_log",
    "amount_std_log",
    "amount_cv",
    "round_amount_ratio",
    "unique_counterparty_ratio",
    "same_amount_repeat_ratio",
]


def build_profiles(df, group_col, amount_col, label_col, counterparty_col, min_count):
    rows = []
    for key, group in df.groupby(group_col):
        if len(group) < min_count:
            continue
        profile = profile_group(group, amount_col, label_col, counterparty_col)
        profile[group_col] = key
        rows.append(profile)
    if not rows:
        return pd.DataFrame(columns=[group_col] + PROFILE_COLUMNS)
    return pd.DataFrame(rows)


def zscore_pairwise(left, right, columns):
    combined = pd.concat([left[columns], right[columns]], axis=0, ignore_index=True)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0).replace(0, 1.0)
    return ((left[columns] - mean) / std).values, ((right[columns] - mean) / std).values


def learn_ieee_user_distributions(ieee, config):
    distributions = {}
    for user_id, group in ieee.groupby("ieee_pseudo_user_id"):
        if len(group) < config.min_ieee_user_tx_for_matching:
            continue
        counts = np.bincount(group["_hour_of_week_bin"].astype(int), minlength=HOUR_OF_WEEK_BINS)
        distributions[user_id] = smoothed_prob_from_counts(counts, config.smooth_alpha)
    return distributions


def match_ffsd_sources_to_ieee(ffsd, ieee, user_distributions, config):
    """Match FFSD source users to similar IEEE pseudo users using profile vectors."""
    ffsd_profiles = build_profiles(
        ffsd,
        config.ffsd_source_col,
        config.ffsd_amount_col,
        config.ffsd_label_col,
        config.ffsd_target_col,
        config.min_ffsd_source_tx_for_user_matching,
    )
    ieee_profiles = build_profiles(
        ieee,
        "ieee_pseudo_user_id",
        config.ieee_amount_col,
        config.ieee_label_col,
        "ieee_pseudo_merchant_context",
        config.min_ieee_user_tx_for_matching,
    )

    if len(ffsd_profiles) == 0 or len(ieee_profiles) == 0:
        return {}, {"matched_source_count": 0, "matchable_ieee_user_count": len(ieee_profiles)}

    # Only keep IEEE users for which we learned an hour-of-week rhythm.
    ieee_profiles = ieee_profiles[ieee_profiles["ieee_pseudo_user_id"].isin(user_distributions.keys())].reset_index(drop=True)
    if len(ieee_profiles) == 0:
        return {}, {"matched_source_count": 0, "matchable_ieee_user_count": 0}

    # Full pairwise matching is too expensive for FFSD-scale data. Match the
    # most behaviorally informative high-volume users and let low-frequency
    # users shrink back to global rhythm.
    ffsd_profiles = ffsd_profiles.sort_values("tx_count_log", ascending=False).head(
        int(config.max_ffsd_sources_for_matching)
    ).reset_index(drop=True)
    ieee_profiles = ieee_profiles.sort_values("tx_count_log", ascending=False).head(
        int(config.max_ieee_users_for_matching)
    ).reset_index(drop=True)

    left_x, right_x = zscore_pairwise(ffsd_profiles, ieee_profiles, PROFILE_COLUMNS)
    matches = {}
    top_k = max(1, int(config.pseudo_user_match_top_k))
    ieee_ids = ieee_profiles["ieee_pseudo_user_id"].values

    for i, source in enumerate(ffsd_profiles[config.ffsd_source_col].values):
        distances = np.sqrt(((right_x - left_x[i]) ** 2).sum(axis=1))
        order = np.argsort(distances)[:top_k]
        selected_distances = distances[order]
        # Convert smaller distances into larger softmax weights.
        logits = -selected_distances
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        matches[source] = [
            {"ieee_pseudo_user_id": str(ieee_ids[j]), "weight": float(w), "distance": float(distances[j])}
            for j, w in zip(order, weights)
        ]

    return matches, {
        "matched_source_count": len(matches),
        "matchable_ieee_user_count": len(ieee_profiles),
    }


def mixed_source_distribution(source, source_matches, user_distributions, global_distribution):
    if source not in source_matches:
        return global_distribution
    mixed = np.zeros_like(global_distribution, dtype=float)
    total_weight = 0.0
    for item in source_matches[source]:
        user_id = item["ieee_pseudo_user_id"]
        if user_id in user_distributions:
            w = float(item["weight"])
            mixed += w * user_distributions[user_id]
            total_weight += w
    if total_weight <= 0:
        return global_distribution
    mixed = mixed / mixed.sum()
    # Shrink toward global rhythm so small or imperfect matches do not dominate.
    out = 0.6 * mixed + 0.4 * global_distribution
    return out / out.sum()


def extract_ieee_fraud_templates(ieee, config, max_templates=500):
    """Extract relative-time templates from IEEE fraud bursts."""
    templates = []
    if config.ieee_label_col not in ieee.columns:
        return templates

    fraud_window_seconds = 10 * 60
    fraud = ieee[ieee[config.ieee_label_col].astype(int) == 1].copy()
    if len(fraud) == 0:
        return templates

    for user_id, group in fraud.groupby("ieee_pseudo_user_id"):
        group = group.sort_values(config.ieee_time_col)
        times = group[config.ieee_time_col].astype(float).values
        amounts = pd.to_numeric(group[config.ieee_amount_col], errors="coerce").fillna(0.0).values
        if len(times) < config.fraud_burst_min_count:
            continue

        start = 0
        for end in range(1, len(times) + 1):
            is_break = end == len(times) or (times[end] - times[end - 1] > fraud_window_seconds)
            if not is_break:
                continue
            if end - start >= config.fraud_burst_min_count:
                block_times = times[start:end]
                block_amounts = amounts[start:end]
                deltas = block_times - block_times[0]
                mean_amount = abs(block_amounts.mean()) + 1e-6
                cv = float(block_amounts.std()) / mean_amount
                template_type = "equal_amount_fraud_burst" if cv <= config.equal_amount_tolerance else "fraud_burst"
                first_row = group.iloc[start]
                templates.append({
                    "template_id": "ieee_template_%05d" % len(templates),
                    "template_type": template_type,
                    "n_transactions": int(end - start),
                    "interarrival_seconds": [float(x) for x in deltas],
                    "amount_cv": cv,
                    "hour_of_week_bin": int(first_row["_hour_of_week_bin"]),
                    "pseudo_user_id": str(user_id),
                })
                if len(templates) >= max_templates:
                    return templates
            start = end

    return templates


def nearly_equal(a, b, tolerance):
    denom = max(abs(float(a)), abs(float(b)), 1e-6)
    return abs(float(a) - float(b)) / denom <= tolerance


def classify_block(rows, config):
    """Classify a contiguous FFSD block candidate."""
    labels = rows[config.ffsd_label_col].astype(int).values if config.ffsd_label_col in rows.columns else np.zeros(len(rows))
    sources = rows[config.ffsd_source_col].astype(str).values
    targets = rows[config.ffsd_target_col].astype(str).values
    amounts = pd.to_numeric(rows[config.ffsd_amount_col], errors="coerce").fillna(0.0).values

    all_fraud = len(labels) > 0 and labels.sum() == len(labels)
    same_source = len(set(sources)) == 1
    same_target = len(set(targets)) == 1
    same_pair = same_source and same_target
    equal_amount = all(nearly_equal(amounts[0], x, config.equal_amount_tolerance) for x in amounts)

    if all_fraud and len(rows) >= config.fraud_burst_min_count and equal_amount:
        return "equal_amount_fraud_burst"
    if all_fraud and len(rows) >= config.fraud_burst_min_count and same_pair:
        return "same_source_target_fraud_repeat"
    if all_fraud and len(rows) >= config.fraud_burst_min_count and same_target:
        return "multi_source_same_target_fraud"
    if all_fraud and len(rows) >= config.fraud_burst_min_count and same_source:
        return "source_fraud_burst"
    if all_fraud:
        return "isolated_fraud"
    if same_pair and len(rows) > 1:
        return "same_source_target_normal_repeat"
    return "normal"


def make_ffsd_blocks(ffsd_sorted, config):
    """Create contiguous blocks without changing FFSD id order."""
    blocks = []
    n = len(ffsd_sorted)
    i = 0
    block_id = 0

    label_aware = config.fraud_controller_mode == "label_aware" and config.ffsd_label_col in ffsd_sorted.columns

    while i < n:
        row = ffsd_sorted.iloc[i]
        max_end = min(n, i + config.max_block_size)
        best_end = i + 1
        best_type = "isolated_fraud" if label_aware and int(row[config.ffsd_label_col]) == 1 else "normal"

        # Fraud/anomaly blocks are detected only over contiguous id order ranges.
        if label_aware and int(row[config.ffsd_label_col]) == 1:
            for end in range(i + 2, max_end + 1):
                candidate = ffsd_sorted.iloc[i:end]
                if candidate[config.ffsd_label_col].astype(int).sum() != len(candidate):
                    break
                block_type = classify_block(candidate, config)
                if block_type != "isolated_fraud":
                    best_end = end
                    best_type = block_type
        else:
            # On FFSD-scale data, making one block per normal transaction is
            # too slow. Group consecutive non-fraud rows into small chunks while
            # preserving the global id order. Fraud rows remain isolated for the
            # controller above.
            best_end = i + 1
            best_type = "normal"
            for end in range(i + 1, max_end + 1):
                candidate = ffsd_sorted.iloc[i:end]
                if label_aware and candidate[config.ffsd_label_col].astype(int).sum() > 0:
                    break
                best_end = end

        block_rows = ffsd_sorted.iloc[i:best_end]
        blocks.append({
            "block_id": block_id,
            "row_positions": list(range(i, best_end)),
            "block_type": best_type,
            "source": str(block_rows.iloc[0][config.ffsd_source_col]),
            "target": str(block_rows.iloc[0][config.ffsd_target_col]),
            "n": int(best_end - i),
        })
        block_id += 1
        i = best_end

    return blocks


def build_calendar_bins(config, rhythms):
    start = pd.Timestamp(config.synthetic_start)
    end = pd.Timestamp(config.synthetic_end)
    freq = "%dmin" % config.time_bin_minutes
    starts = pd.date_range(start=start, end=end, freq=freq)
    bins = pd.DataFrame({"bin_start": starts})
    bins["bin_end"] = bins["bin_start"] + pd.to_timedelta(config.time_bin_minutes * 60 - 1, unit="s")
    bins.loc[bins["bin_end"] > end, "bin_end"] = end
    bins["dow"] = bins["bin_start"].dt.dayofweek.astype(int)
    bins["half_hour_bin"] = ((bins["bin_start"].dt.hour * 60 + bins["bin_start"].dt.minute) // 30).astype(int)
    bins["hour_of_week_bin"] = bins["dow"] * 48 + bins["half_hour_bin"]
    bins["day_of_month"] = bins["bin_start"].dt.day.astype(int)

    how = rhythms["global_hour_of_week"]
    dom = rhythms["day_of_month"]
    weights = []
    for _, row in bins.iterrows():
        weights.append(float(how[int(row["hour_of_week_bin"])]) * float(dom[int(row["day_of_month"]) - 1]))
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(bins), dtype=float)
    bins["global_weight"] = weights / weights.sum()
    bins["global_cdf"] = bins["global_weight"].cumsum()
    return bins


def quantile_to_bin_index(bins, q):
    q = min(max(float(q), 0.0), 1.0)
    idx = int(np.searchsorted(bins["global_cdf"].values, q, side="left"))
    return min(max(idx, 0), len(bins) - 1)


def choose_template(block_type, block_n, templates, rng):
    if not templates:
        return None
    if "equal_amount" in block_type:
        candidates = [t for t in templates if t["template_type"] == "equal_amount_fraud_burst"]
    else:
        candidates = templates
    if not candidates:
        candidates = templates
    distances = np.asarray([abs(t["n_transactions"] - block_n) for t in candidates], dtype=float)
    best = np.where(distances == distances.min())[0]
    return candidates[int(rng.choice(best))]


def block_hour_distribution(block, rhythms, source_matches, user_distributions, templates, config):
    global_dist = rhythms["global_hour_of_week"]
    source_dist = mixed_source_distribution(block["source"], source_matches, user_distributions, global_dist)

    if "fraud" in block["block_type"]:
        fraud_dist = rhythms["fraud_hour_of_week"]
        dist = 0.5 * fraud_dist + 0.3 * source_dist + 0.2 * global_dist
    elif block["source"] in source_matches:
        dist = 0.6 * source_dist + 0.4 * global_dist
    else:
        dist = global_dist
    dist = dist / dist.sum()
    return dist


def sample_block_start(bins, left_idx, right_idx, min_time, block_dist, rng):
    left_idx = max(0, int(left_idx))
    right_idx = min(len(bins) - 1, int(right_idx))
    if right_idx < left_idx:
        right_idx = left_idx

    candidates = bins.iloc[left_idx:right_idx + 1].copy()
    candidates = candidates[candidates["bin_end"] >= min_time]
    if len(candidates) == 0:
        return min_time

    how_weights = np.asarray([block_dist[int(x)] for x in candidates["hour_of_week_bin"].values], dtype=float)
    base = candidates["global_weight"].values.astype(float)
    weights = 0.5 * base + 0.5 * how_weights
    if weights.sum() <= 0:
        weights = np.ones(len(candidates), dtype=float)
    weights = weights / weights.sum()

    chosen_pos = int(rng.choice(np.arange(len(candidates)), p=weights))
    chosen = candidates.iloc[chosen_pos]
    start = max(pd.Timestamp(chosen["bin_start"]), pd.Timestamp(min_time))
    end = pd.Timestamp(chosen["bin_end"])
    if end < start:
        return start
    seconds = int((end - start).total_seconds())
    offset = int(rng.randint(0, seconds + 1)) if seconds > 0 else 0
    return start + pd.to_timedelta(offset, unit="s")


def generate_default_deltas(n, block_type, rng):
    if n <= 1:
        return [0]
    if "fraud" in block_type:
        # Fraud bursts should be compact: seconds to minutes.
        gaps = rng.randint(1, 90, size=n - 1)
    else:
        # Normal rows are already grouped into small order-preserving blocks.
        # Keep the within-block span short to avoid repeatedly squeezing a
        # block to the right edge of a 30-minute calendar bin.
        gaps = rng.randint(1, 45, size=n - 1)
    return [0] + list(np.cumsum(gaps).astype(int))


def fit_deltas_to_window(deltas, max_seconds, rng):
    if len(deltas) <= 1:
        return [0]
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas - deltas.min()
    if deltas.max() <= 0:
        deltas = np.arange(len(deltas), dtype=float)
    if max_seconds <= 0:
        return [0] * len(deltas)
    if deltas.max() > max_seconds:
        # Compress into the available segment, but do not force the last
        # transaction onto the bin boundary such as xx:29:59 or xx:59:59.
        min_span = min(max_seconds, max(len(deltas) - 1, 1))
        if max_seconds > min_span:
            reserve = int(rng.randint(1, min(300, max_seconds - min_span) + 1))
            target_span = max(min_span, max_seconds - reserve)
        else:
            target_span = max_seconds
        deltas = deltas / deltas.max() * float(target_span)
    out = np.maximum.accumulate(np.floor(deltas).astype(int))
    return [int(x) for x in out]


def is_half_hour_right_boundary(ts):
    return ts.second == 59 and ts.minute in (29, 59)


def de_boundary_times(times, config, rng):
    """Move exact half-hour boundary timestamps slightly earlier.

    The block scheduler uses 30-minute calendar bins. When many consecutive
    blocks compete for the same bin, some timestamps can pile up at xx:29:59 or
    xx:59:59. This pass removes that mechanical artifact while preserving the
    non-decreasing FFSD order.
    """
    if not times:
        return times

    start_time = pd.Timestamp(config.synthetic_start)
    end_time = pd.Timestamp(config.synthetic_end)
    adjusted = list(pd.to_datetime(times))
    n = len(adjusted)
    i = 0
    while i < n:
        current = adjusted[i]
        if not is_half_hour_right_boundary(current):
            i += 1
            continue

        j = i + 1
        while j < n and adjusted[j] == current and is_half_hour_right_boundary(adjusted[j]):
            j += 1

        lower_bound = adjusted[i - 1] if i > 0 else start_time
        upper_bound = adjusted[j] if j < n else end_time
        available_back = int((current - lower_bound).total_seconds())
        available_forward = int((upper_bound - current).total_seconds())
        run_len = j - i
        if available_back > 1:
            span = min(available_back - 1, max(30, min(300, run_len * 3)))
            # Deterministic spacing with a tiny random phase keeps the run
            # ordered and avoids moving every group into the same pattern.
            phase = int(rng.randint(0, max(1, min(10, span))))
            for k, pos in enumerate(range(i, j)):
                offset = int(math.floor((k + 1) * float(span) / float(run_len + 1)))
                offset = min(span, max(1, offset + phase))
                adjusted[pos] = current - pd.to_timedelta(span - offset + 1, unit="s")
        elif available_forward > 1:
            span = min(available_forward - 1, max(30, min(300, run_len * 3)))
            phase = int(rng.randint(0, max(1, min(10, span))))
            for k, pos in enumerate(range(i, j)):
                offset = int(math.floor((k + 1) * float(span) / float(run_len + 1)))
                offset = min(span, max(1, offset + phase))
                adjusted[pos] = current + pd.to_timedelta(offset, unit="s")

        i = j

    return adjusted


def generate_ordered_timestamps(ffsd_sorted, blocks, bins, rhythms, source_matches, user_distributions, templates, config):
    rng = np.random.RandomState(config.random_seed)
    synthetic_end = pd.Timestamp(config.synthetic_end)
    current_time = pd.Timestamp(config.synthetic_start)
    result_times = [None] * len(ffsd_sorted)
    metadata = {}

    b_count = max(len(blocks), 1)
    for b_idx, block in enumerate(blocks):
        q0 = float(b_idx) / float(b_count)
        q1 = float(b_idx + 1) / float(b_count)
        left_idx = quantile_to_bin_index(bins, q0)
        right_idx = quantile_to_bin_index(bins, q1)

        block_dist = block_hour_distribution(block, rhythms, source_matches, user_distributions, templates, config)
        block_start = sample_block_start(bins, left_idx, right_idx, current_time, block_dist, rng)

        # Never move backwards. Same-second transactions are allowed when needed.
        if block_start < current_time:
            block_start = current_time

        block_n = block["n"]
        template = None
        if "fraud" in block["block_type"]:
            template = choose_template(block["block_type"], block_n, templates, rng)

        if template is not None:
            template_deltas = template["interarrival_seconds"]
            if len(template_deltas) >= block_n:
                deltas = template_deltas[:block_n]
            else:
                extra = generate_default_deltas(block_n - len(template_deltas) + 1, block["block_type"], rng)
                deltas = template_deltas + [template_deltas[-1] + x for x in extra[1:]]
            generation_mode = template["template_type"]
            template_id = template["template_id"]
        else:
            deltas = generate_default_deltas(block_n, block["block_type"], rng)
            generation_mode = block["block_type"] if "fraud" in block["block_type"] else "source_matched_rhythm"
            template_id = None

        # The block should stay within its quantile segment where possible.
        segment_right_time = pd.Timestamp(bins.iloc[right_idx]["bin_end"])
        max_seconds = int(max((segment_right_time - block_start).total_seconds(), 0))
        deltas = fit_deltas_to_window(deltas, max_seconds, rng)

        for local_i, row_pos in enumerate(block["row_positions"]):
            t = block_start + pd.to_timedelta(int(deltas[local_i]), unit="s")
            if t > synthetic_end:
                t = synthetic_end
            if not config.allow_same_second_transactions and row_pos > 0 and result_times[row_pos - 1] is not None and t <= result_times[row_pos - 1]:
                t = result_times[row_pos - 1] + pd.to_timedelta(1, unit="s")
                if t > synthetic_end:
                    t = synthetic_end
            result_times[row_pos] = t
            metadata[row_pos] = {
                "block_id": block["block_id"],
                "block_type": block["block_type"],
                "generation_mode": generation_mode,
                "matched_ieee_template_id": template_id,
                "fraud_controller_used": bool("fraud" in block["block_type"]),
                "time_source": "ieee_fraud_template" if template_id else ("ieee_pseudo_user" if block["source"] in source_matches else "ieee_global"),
                "matched_ieee_user_count": len(source_matches.get(block["source"], [])),
            }

        current_time = result_times[block["row_positions"][-1]]

    result_times = de_boundary_times(result_times, config, rng)
    return result_times, metadata


def add_synthetic_features(ffsd_sorted, times, metadata, config):
    out = ffsd_sorted.copy()
    out["datetime"] = pd.to_datetime(times)
    start = pd.Timestamp(config.synthetic_start)
    out["timestamp_seconds_from_start"] = (out["datetime"] - start).dt.total_seconds().astype("int64")
    out["date"] = out["datetime"].dt.date.astype(str)
    out["year"] = out["datetime"].dt.year.astype(int)
    out["month"] = out["datetime"].dt.month.astype(int)
    out["day"] = out["datetime"].dt.day.astype(int)
    out["hour"] = out["datetime"].dt.hour.astype(int)
    out["minute"] = out["datetime"].dt.minute.astype(int)
    out["second"] = out["datetime"].dt.second.astype(int)
    out["dayofweek"] = out["datetime"].dt.dayofweek.astype(int)
    out["is_weekend"] = out["dayofweek"] >= 5
    out["is_month_start"] = out["day"] <= 3
    out["is_month_middle"] = out["day"].isin([14, 15, 16])
    out["is_month_end"] = out["day"] >= 28
    out["is_common_bill_day"] = out["day"].isin([1, 5, 10, 15, 20, 25, 28])

    def period_tag(hour):
        if 0 <= hour < 5:
            return "late_night"
        if 5 <= hour < 8:
            return "early_morning"
        if 8 <= hour < 10:
            return "commute_morning"
        if 10 <= hour < 12:
            return "workday_morning"
        if 12 <= hour < 14:
            return "lunch"
        if 14 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 20:
            return "commute_evening"
        return "night"

    out["period_tag"] = out["hour"].map(period_tag)

    meta_df = pd.DataFrame.from_dict(metadata, orient="index")
    meta_df.index.name = "_sorted_pos"
    out["_sorted_pos"] = np.arange(len(out))
    out = out.merge(meta_df.reset_index(), on="_sorted_pos", how="left")
    out = out.drop(columns=["_sorted_pos"])
    return out


def validate_output(out, config):
    start = pd.Timestamp(config.synthetic_start)
    end = pd.Timestamp(config.synthetic_end)
    ordered = out.sort_values(config.ffsd_order_col)
    times = pd.to_datetime(ordered["datetime"])
    problems = []
    if times.isna().any():
        problems.append("datetime contains null values")
    if len(times) and times.min() < start:
        problems.append("datetime is earlier than synthetic_start")
    if len(times) and times.max() > end:
        problems.append("datetime is later than synthetic_end")
    if len(times) > 1 and (times.diff().dropna() < pd.Timedelta(seconds=0)).any():
        problems.append("datetime is not non-decreasing after sorting by FFSD id")
    return problems


def generate_synthetic_timestamps(ffsd_df, ieee_transaction_df, ieee_identity_df, ieee_label_df, config):
    ensure_columns(
        ffsd_df,
        [config.ffsd_order_col, config.ffsd_source_col, config.ffsd_target_col, config.ffsd_amount_col],
        "FFSD",
    )
    ffsd_df = ffsd_df.copy()
    if config.ffsd_label_col not in ffsd_df.columns:
        ffsd_df[config.ffsd_label_col] = 0

    # FFSD uses a multi-class Labels field in this project. Keep the original
    # Labels column untouched, and create a binary internal fraud flag for
    # label-aware timestamp synthesis.
    internal_ffsd_label_col = "_ffsd_is_fraud_for_synthesis"
    raw_labels = pd.to_numeric(ffsd_df[config.ffsd_label_col], errors="coerce")
    ffsd_df[internal_ffsd_label_col] = (raw_labels == config.ffsd_fraud_positive_label).astype(int)
    run_config = replace(config, ffsd_label_col=internal_ffsd_label_col)

    ieee = merge_ieee_tables(ieee_transaction_df, ieee_identity_df, ieee_label_df, run_config)
    ieee, pseudo_report = add_ieee_pseudo_ids(ieee)
    shift_hours = calibrate_ieee_phase(ieee, run_config) if run_config.calibrate_daily_phase else 0
    ieee = add_time_features_from_transaction_dt(ieee, run_config, shift_hours)
    rhythms = learn_global_rhythms(ieee, run_config)
    user_distributions = learn_ieee_user_distributions(ieee, run_config)
    source_matches, match_report = match_ffsd_sources_to_ieee(ffsd_df, ieee, user_distributions, run_config)
    templates = extract_ieee_fraud_templates(ieee, run_config)
    bins = build_calendar_bins(run_config, rhythms)

    ffsd_sorted = ffsd_df.copy()
    ffsd_sorted["_original_row_order"] = np.arange(len(ffsd_sorted))
    ffsd_sorted = ffsd_sorted.sort_values(run_config.ffsd_order_col).reset_index(drop=True)
    blocks = make_ffsd_blocks(ffsd_sorted, run_config)
    times, metadata = generate_ordered_timestamps(
        ffsd_sorted, blocks, bins, rhythms, source_matches, user_distributions, templates, run_config
    )
    out_sorted = add_synthetic_features(ffsd_sorted, times, metadata, run_config)
    problems = validate_output(out_sorted, run_config)
    out = out_sorted.sort_values("_original_row_order").drop(columns=["_original_row_order"])
    if internal_ffsd_label_col in out.columns:
        out = out.drop(columns=[internal_ffsd_label_col])

    report = {
        "method": "ffsd_synthetic_timestamp_ieee_conditioned_v1",
        "not_real_timestamp": True,
        "synthetic_start": run_config.synthetic_start,
        "synthetic_end": run_config.synthetic_end,
        "preserve_ffsd_id_order": True,
        "config": asdict(config),
        "ffsd_original_label_col": config.ffsd_label_col,
        "ffsd_internal_fraud_positive_label": int(config.ffsd_fraud_positive_label),
        "ffsd_label_distribution": {
            str(k): int(v) for k, v in ffsd_df[config.ffsd_label_col].value_counts(dropna=False).items()
        },
        "n_ffsd": int(len(ffsd_df)),
        "n_ieee": int(len(ieee)),
        "best_shift_hour": int(shift_hours),
        "pseudo_id_report": pseudo_report,
        "user_match_report": match_report,
        "n_ieee_user_distributions": int(len(user_distributions)),
        "n_ieee_fraud_templates": int(len(templates)),
        "n_ffsd_blocks": int(len(blocks)),
        "ffsd_block_type_counts": {
            str(k): int(v)
            for k, v in pd.Series([b["block_type"] for b in blocks]).value_counts().items()
        },
        "validation_problems": problems,
    }
    return out, report


def make_demo_data(config):
    """Create small demo data so the full pipeline is runnable without private files."""
    rng = np.random.RandomState(config.random_seed)

    # FFSD demo: id-ordered transactions with a few fraud bursts.
    n_ffsd = 240
    sources = np.array(["U%02d" % i for i in range(1, 16)])
    targets = np.array(["M%02d" % i for i in range(1, 10)])
    ffsd = pd.DataFrame({
        "id": np.arange(1, n_ffsd + 1),
        "source": rng.choice(sources, size=n_ffsd),
        "target": rng.choice(targets, size=n_ffsd),
        "amount": np.round(rng.lognormal(mean=3.2, sigma=0.7, size=n_ffsd), 2),
        "label": np.zeros(n_ffsd, dtype=int),
    })
    for start in [45, 120, 190]:
        ffsd.loc[start:start + 3, "source"] = "U_fraud"
        ffsd.loc[start:start + 3, "target"] = "M_risky"
        ffsd.loc[start:start + 3, "amount"] = 88.88
        ffsd.loc[start:start + 3, "label"] = 1

    # IEEE demo: transaction stream over roughly two months, including fraud bursts.
    n_ieee = 900
    increments = rng.exponential(scale=3600, size=n_ieee).astype(int) + 1
    transaction_dt = np.cumsum(increments)
    ieee = pd.DataFrame({
        "TransactionID": np.arange(100000, 100000 + n_ieee),
        "TransactionDT": transaction_dt,
        "TransactionAmt": np.round(rng.lognormal(mean=3.1, sigma=0.8, size=n_ieee), 2),
        "ProductCD": rng.choice(["W", "C", "R", "H"], size=n_ieee),
        "card1": rng.choice(np.arange(1000, 1030), size=n_ieee),
        "card2": rng.choice(np.arange(200, 230), size=n_ieee),
        "card3": rng.choice([150, 185], size=n_ieee),
        "card4": rng.choice(["visa", "mastercard"], size=n_ieee),
        "card5": rng.choice(np.arange(100, 110), size=n_ieee),
        "card6": rng.choice(["debit", "credit"], size=n_ieee),
        "addr1": rng.choice(np.arange(10, 20), size=n_ieee),
        "addr2": rng.choice([87, 96], size=n_ieee),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "outlook.com"], size=n_ieee),
        "R_emaildomain": rng.choice(["merchant.com", "shop.com", "pay.com"], size=n_ieee),
    })
    labels = pd.DataFrame({"TransactionID": ieee["TransactionID"], "isFraud": np.zeros(n_ieee, dtype=int)})
    for start in [150, 460, 720]:
        burst_ids = ieee.loc[start:start + 4, "TransactionID"].values
        labels.loc[labels["TransactionID"].isin(burst_ids), "isFraud"] = 1
        base_time = ieee.loc[start, "TransactionDT"]
        ieee.loc[start:start + 4, "TransactionDT"] = base_time + np.array([0, 12, 31, 48, 75])
        ieee.loc[start:start + 4, "TransactionAmt"] = 88.88
        ieee.loc[start:start + 4, [
            "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain",
        ]] = [9999, 299, 150, "visa", 101, "credit", 19, 87, "risk.com"]

    identity = pd.DataFrame({
        "TransactionID": ieee["TransactionID"],
        "DeviceType": rng.choice(["desktop", "mobile"], size=n_ieee),
        "DeviceInfo": rng.choice(["Chrome", "Safari", "Android", "iOS"], size=n_ieee),
        "id_30": rng.choice(["Windows 10", "iOS 15", "Android 12"], size=n_ieee),
        "id_31": rng.choice(["chrome", "safari", "mobile safari"], size=n_ieee),
        "id_33": rng.choice(["1920x1080", "390x844", "1366x768"], size=n_ieee),
        "id_38": rng.choice(["T", "F"], size=n_ieee),
    })
    return ffsd, ieee, identity, labels


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic FFSD timestamps from IEEE-CIS rhythms.")
    parser.add_argument("--ffsd", help="Path to FFSD csv.")
    parser.add_argument("--ieee-transaction", help="Path to IEEE-CIS transaction csv.")
    parser.add_argument("--ieee-identity", help="Path to IEEE-CIS identity csv.", default=None)
    parser.add_argument("--ieee-label", help="Path to IEEE-CIS label csv.", default=None)
    parser.add_argument("--output", help="Output FFSD csv path.", default="FFSD_with_synthetic_timestamp.csv")
    parser.add_argument("--report", help="Output generation report json path.", default="synthetic_timestamp_generation_report.json")
    parser.add_argument("--demo", action="store_true", help="Run with built-in demo data.")
    parser.add_argument("--fraud-controller-mode", choices=["label_aware", "label_blind"], default="label_aware")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    config = GeneratorConfig(
        fraud_controller_mode=args.fraud_controller_mode,
        random_seed=args.random_seed,
    )

    if args.demo:
        ffsd, ieee_transaction, ieee_identity, ieee_label = make_demo_data(config)
    else:
        if not args.ffsd or not args.ieee_transaction:
            raise ValueError("Please provide --ffsd and --ieee-transaction, or use --demo.")
        ffsd = safe_read_csv(args.ffsd)
        ieee_transaction = safe_read_csv(args.ieee_transaction)
        ieee_identity = safe_read_csv(args.ieee_identity)
        ieee_label = safe_read_csv(args.ieee_label)

    out, report = generate_synthetic_timestamps(ffsd, ieee_transaction, ieee_identity, ieee_label, config)

    output_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    output_dir = os.path.dirname(output_path)
    report_dir = os.path.dirname(report_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if report_dir and not os.path.exists(report_dir):
        os.makedirs(report_dir)

    out.to_csv(output_path, index=False)
    write_json(report_path, report)
    print("Wrote:", output_path)
    print("Wrote:", report_path)
    if report["validation_problems"]:
        print("Validation problems:", report["validation_problems"])
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()
