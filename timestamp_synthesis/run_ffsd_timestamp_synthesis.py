#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project runner for FFSD timestamp synthesis.

Reads immutable inputs from data/ and writes module-1 outputs to work/:
- FFSD_with_synthetic_timestamp.csv
- FFSD_synthetic_time_core.csv
- FFSD_synthetic_time_detail.csv
- synthetic_timestamp_generation_report.json
"""

from __future__ import print_function

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
WORK_DIR = os.path.join(REPO_ROOT, "work")
IEEE_DIR = os.path.join(DATA_DIR, "ieee")

FFSD_RAW = os.path.join(DATA_DIR, "S-FFSD.csv")
IEEE_TRANSACTION = os.path.join(IEEE_DIR, "train_transaction.csv")
IEEE_IDENTITY = os.path.join(IEEE_DIR, "train_identity.csv")
FFSD_WITH_SYNTHETIC_TIMESTAMP = os.path.join(WORK_DIR, "FFSD_with_synthetic_timestamp.csv")
FFSD_SYNTHETIC_TIME_CORE = os.path.join(WORK_DIR, "FFSD_synthetic_time_core.csv")
FFSD_SYNTHETIC_TIME_DETAIL = os.path.join(WORK_DIR, "FFSD_synthetic_time_detail.csv")
SYNTHETIC_TIMESTAMP_REPORT = os.path.join(WORK_DIR, "synthetic_timestamp_generation_report.json")

FFSD_ORIGINAL_COLUMNS = [
    "Time", "Source", "Target", "Amount", "Location", "Type", "Labels",
]
CORE_TIME_COLUMNS = [
    "datetime", "date", "year", "month", "day", "hour", "minute", "second", "dayofweek",
]
DETAIL_TIME_COLUMNS = CORE_TIME_COLUMNS + [
    "is_weekend", "is_month_start", "is_month_middle", "is_month_end",
    "is_common_bill_day", "period_tag",
]

TIMESTAMP_DIR = os.path.join(REPO_ROOT, "timestamp_synthesis")
if TIMESTAMP_DIR not in sys.path:
    sys.path.insert(0, TIMESTAMP_DIR)

from synthetic_timestamp_generator import (
    GeneratorConfig,
    generate_synthetic_timestamps,
    write_json,
)


def write_timestamp_variants(full_df):
    os.makedirs(WORK_DIR, exist_ok=True)

    original_cols = [c for c in FFSD_ORIGINAL_COLUMNS if c in full_df.columns]
    core_cols = [c for c in CORE_TIME_COLUMNS if c in full_df.columns]
    detail_cols = [c for c in DETAIL_TIME_COLUMNS if c in full_df.columns]

    missing_original = [c for c in FFSD_ORIGINAL_COLUMNS if c not in full_df.columns]
    missing_core = [c for c in CORE_TIME_COLUMNS if c not in full_df.columns]
    missing_detail = [c for c in DETAIL_TIME_COLUMNS if c not in full_df.columns]
    if missing_original or missing_core or missing_detail:
        raise ValueError(
            f"Missing columns in synthesis output: "
            f"original={missing_original}, core={missing_core}, detail={missing_detail}"
        )

    full_df.to_csv(FFSD_WITH_SYNTHETIC_TIMESTAMP, index=False)
    full_df[original_cols + core_cols].to_csv(FFSD_SYNTHETIC_TIME_CORE, index=False)
    full_df[original_cols + detail_cols].to_csv(FFSD_SYNTHETIC_TIME_DETAIL, index=False)

    return {
        "full": FFSD_WITH_SYNTHETIC_TIMESTAMP,
        "core": FFSD_SYNTHETIC_TIME_CORE,
        "detail": FFSD_SYNTHETIC_TIME_DETAIL,
    }


def read_project_data():
    ffsd_cols = ["Time", "Source", "Target", "Amount", "Location", "Type", "Labels"]
    ieee_transaction_cols = [
        "TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    ]
    ieee_identity_cols = [
        "TransactionID", "DeviceType", "DeviceInfo",
        "id_30", "id_31", "id_33", "id_38",
    ]

    for path, label in [
        (FFSD_RAW, "S-FFSD.csv"),
        (IEEE_TRANSACTION, "ieee/train_transaction.csv"),
        (IEEE_IDENTITY, "ieee/train_identity.csv"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing dataset input: {path} ({label})")

    print(f"Reading {FFSD_RAW} ...")
    ffsd = pd.read_csv(FFSD_RAW, usecols=ffsd_cols)

    transaction_header = pd.read_csv(IEEE_TRANSACTION, nrows=0)
    if "isFraud" not in transaction_header.columns:
        raise ValueError(
            "data/ieee/train_transaction.csv must contain isFraud. "
            "Fraud labels are read directly from train_transaction.csv."
        )

    print(f"Reading {IEEE_TRANSACTION} selected columns (including isFraud) ...")
    ieee_transaction = pd.read_csv(IEEE_TRANSACTION, usecols=ieee_transaction_cols)

    print(f"Reading {IEEE_IDENTITY} selected columns ...")
    ieee_identity = pd.read_csv(IEEE_IDENTITY, usecols=ieee_identity_cols)

    return ffsd, ieee_transaction, ieee_identity, None


def main():
    os.makedirs(WORK_DIR, exist_ok=True)

    config = GeneratorConfig(
        ffsd_order_col="Time",
        ffsd_source_col="Source",
        ffsd_target_col="Target",
        ffsd_amount_col="Amount",
        ffsd_label_col="Labels",
        ffsd_fraud_positive_label=1,
        ieee_transaction_id_col="TransactionID",
        ieee_time_col="TransactionDT",
        ieee_amount_col="TransactionAmt",
        ieee_label_col="isFraud",
        synthetic_start="2021-01-01 00:00:00",
        synthetic_end="2021-10-31 23:59:59",
        fraud_controller_mode="label_aware",
        min_ffsd_source_tx_for_user_matching=20,
        min_ieee_user_tx_for_matching=5,
        max_ffsd_sources_for_matching=5000,
        max_ieee_users_for_matching=5000,
        max_block_size=20,
        random_seed=42,
    )

    ffsd, ieee_transaction, ieee_identity, ieee_label = read_project_data()
    out, report = generate_synthetic_timestamps(
        ffsd, ieee_transaction, ieee_identity, ieee_label, config
    )

    print("Writing module-1 outputs to work/ ...")
    written = write_timestamp_variants(out)
    write_json(SYNTHETIC_TIMESTAMP_REPORT, report)

    for key, path in written.items():
        print(f"Wrote [{key}]: {path}")
    print(f"Wrote: {SYNTHETIC_TIMESTAMP_REPORT}")

    if report["validation_problems"]:
        print("Validation problems:", report["validation_problems"])
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()
