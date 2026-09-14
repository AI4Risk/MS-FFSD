#!/usr/bin/env python3
"""Build and validate the five-table MS-FFSD publication archive."""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "input"
DEFAULT_WORK_DIR = ROOT / "work"
DEFAULT_OUTPUT_DIR = ROOT / "dataset"
ARCHIVE_NAME = "MS-FFSD.zip"
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

PUBLIC_TABLE_COLUMNS = {
    "transaction.csv": [
        "Datetime",
        "Source",
        "Target",
        "Amount",
        "Location",
        "Type",
        "Labels",
        "Transaction_province",
    ],
    "user.csv": [
        "Source",
        "Province",
        "Gender",
        "Age_group",
        "Education",
        "Occupation_industry",
    ],
    "merchant.csv": ["Target", "MCC_level1", "MCC_level2", "Merchant_name"],
    "user_description.csv": ["Source", "User_description"],
    "merchant_description.csv": ["Target", "Merchant_description"],
}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    frame = pd.read_csv(path, **kwargs)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    return frame


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def reject_missing(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    if frame[columns].isna().any().any():
        bad = frame[columns].isna().sum()
        bad = bad[bad > 0].to_dict()
        raise ValueError(f"{name} contains missing public values: {bad}")


def reject_cjk(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = frame[column].dropna().astype(str)
        matches = values[values.str.contains(CJK_PATTERN, regex=True)]
        if not matches.empty:
            raise ValueError(f"{name}.{column} contains CJK text; first value: {matches.iloc[0]!r}")


def build_public_tables(work_dir: Path) -> dict[str, pd.DataFrame]:
    transaction_source = read_csv(work_dir / "transaction_optimized.csv")
    user_source = read_csv(work_dir / "source_optimized.csv")
    merchant_source = read_csv(work_dir / "target_final.csv")
    user_text_source = read_csv(work_dir / "source_long_text.csv", dtype={"Source_id": str})
    merchant_text_source = read_csv(work_dir / "target_long_text.csv", dtype={"Target_id": str})

    require_columns(
        transaction_source,
        ["datetime", "Source", "Target", "Amount", "Location", "Type", "Labels", "transaction_province"],
        "transaction_optimized.csv",
    )
    require_columns(
        user_source,
        ["Source", "province", "gender", "age_group", "education", "occupation_industry"],
        "source_optimized.csv",
    )
    require_columns(merchant_source, ["Target", "behavior_level1", "behavior_level2"], "target_final.csv")
    require_columns(user_text_source, ["Source_id", "Source_description"], "source_long_text.csv")
    require_columns(
        merchant_text_source,
        ["Target_id", "Merchant", "Merchant_description"],
        "target_long_text.csv",
    )

    transaction = transaction_source[
        ["datetime", "Source", "Target", "Amount", "Location", "Type", "Labels", "transaction_province"]
    ].rename(columns={"datetime": "Datetime", "transaction_province": "Transaction_province"})

    user = user_source[
        ["Source", "province", "gender", "age_group", "education", "occupation_industry"]
    ].rename(
        columns={
            "province": "Province",
            "gender": "Gender",
            "age_group": "Age_group",
            "education": "Education",
            "occupation_industry": "Occupation_industry",
        }
    )

    merchant_names = merchant_text_source[["Target_id", "Merchant"]].rename(
        columns={"Target_id": "Target", "Merchant": "Merchant_name"}
    )
    merchant_names["Target"] = merchant_names["Target"].astype(str)
    merchant_source = merchant_source.copy()
    merchant_source["Target"] = merchant_source["Target"].astype(str)
    merchant = merchant_source.merge(merchant_names, on="Target", how="left", validate="one_to_one")
    merchant = merchant[["Target", "behavior_level1", "behavior_level2", "Merchant_name"]].rename(
        columns={"behavior_level1": "MCC_level1", "behavior_level2": "MCC_level2"}
    )

    user_description = user_text_source[["Source_id", "Source_description"]].rename(
        columns={"Source_id": "Source", "Source_description": "User_description"}
    )
    merchant_description = merchant_text_source[["Target_id", "Merchant_description"]].rename(
        columns={"Target_id": "Target"}
    )

    return {
        "transaction.csv": transaction,
        "user.csv": user,
        "merchant.csv": merchant,
        "user_description.csv": user_description,
        "merchant_description.csv": merchant_description,
    }


def validate_public_tables(tables: dict[str, pd.DataFrame], raw_path: Path) -> None:
    if set(tables) != set(PUBLIC_TABLE_COLUMNS):
        raise ValueError(f"Unexpected publication table set: {sorted(tables)}")

    for name, expected_columns in PUBLIC_TABLE_COLUMNS.items():
        actual_columns = tables[name].columns.tolist()
        if actual_columns != expected_columns:
            raise ValueError(f"{name} columns differ: expected {expected_columns}, got {actual_columns}")
        reject_missing(tables[name], expected_columns, name)

    transaction = tables["transaction.csv"]
    user = tables["user.csv"]
    merchant = tables["merchant.csv"]
    user_description = tables["user_description.csv"]
    merchant_description = tables["merchant_description.csv"]

    if user["Source"].duplicated().any() or user_description["Source"].duplicated().any():
        raise ValueError("User tables must contain one row per Source.")
    if merchant["Target"].duplicated().any() or merchant_description["Target"].duplicated().any():
        raise ValueError("Merchant tables must contain one row per Target.")
    if set(transaction["Source"].astype(str)) != set(user["Source"].astype(str)):
        raise ValueError("transaction.csv and user.csv have different Source sets.")
    if set(transaction["Target"].astype(str)) != set(merchant["Target"].astype(str)):
        raise ValueError("transaction.csv and merchant.csv have different Target sets.")
    if set(user["Source"].astype(str)) != set(user_description["Source"].astype(str)):
        raise ValueError("user.csv and user_description.csv have different Source sets.")
    if set(merchant["Target"].astype(str)) != set(merchant_description["Target"].astype(str)):
        raise ValueError("merchant.csv and merchant_description.csv have different Target sets.")

    reject_cjk(user, ["Province", "Gender", "Age_group", "Education", "Occupation_industry"], "user.csv")
    reject_cjk(merchant, ["MCC_level1", "MCC_level2", "Merchant_name"], "merchant.csv")
    reject_cjk(user_description, ["User_description"], "user_description.csv")
    reject_cjk(merchant_description, ["Merchant_description"], "merchant_description.csv")

    raw = read_csv(raw_path)
    core_columns = ["Source", "Target", "Amount", "Location", "Type", "Labels"]
    require_columns(raw, core_columns, raw_path.name)
    if len(raw) != len(transaction):
        raise ValueError(f"Transaction row count changed: raw={len(raw)}, publication={len(transaction)}")
    for column in ["Source", "Target", "Location", "Type", "Labels"]:
        left = raw[column].astype(str).reset_index(drop=True)
        right = transaction[column].astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(f"Published transaction column changed or was reordered: {column}")
    raw_amount = pd.to_numeric(raw["Amount"], errors="raise").to_numpy(dtype=float)
    published_amount = pd.to_numeric(transaction["Amount"], errors="raise").to_numpy(dtype=float)
    if not np.array_equal(raw_amount, published_amount):
        raise ValueError("Published transaction Amount values differ from S-FFSD.csv.")


def write_archive(tables: dict[str, pd.DataFrame], output_dir: Path, work_dir: Path) -> Path:
    resolved_work_dir = work_dir.resolve()
    stage_dir = (resolved_work_dir / "publication" / "MS-FFSD").resolve()
    if not stage_dir.is_relative_to(resolved_work_dir) or stage_dir.name != "MS-FFSD":
        raise ValueError(f"Unsafe publication staging path: {stage_dir}")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        frame.to_csv(stage_dir / name, index=False, encoding="utf-8-sig")

    archive_path = output_dir / ARCHIVE_NAME
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    if temporary_archive.exists():
        temporary_archive.unlink()
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in PUBLIC_TABLE_COLUMNS:
            archive.write(stage_dir / name, arcname=name)
    temporary_archive.replace(archive_path)

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if names != list(PUBLIC_TABLE_COLUMNS):
            raise ValueError(f"Archive members differ from the publication contract: {names}")
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f"Archive integrity check failed for {bad_file}")

    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = build_public_tables(args.work_dir)
    validate_public_tables(tables, args.data_dir / "S-FFSD.csv")
    archive_path = write_archive(tables, args.output_dir, args.work_dir)
    print("Publication validation passed.")
    for name, frame in tables.items():
        print(f"  {name}: {len(frame):,} rows, {len(frame.columns)} columns")
    print(f"Wrote archive: {archive_path}")


if __name__ == "__main__":
    main()
