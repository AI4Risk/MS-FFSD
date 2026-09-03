import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
INFORMATION_DIR = REPO_ROOT / "initialization" / "informations"
DEFAULT_SOURCE_PROVINCE_FILE = WORK_DIR / "source_province_information.csv"
DEFAULT_SOURCE_OUTPUT = WORK_DIR / "source.csv"

REGION_ORDER = [
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia",
    "Liaoning", "Jilin", "Heilongjiang", "Shanghai", "Jiangsu",
    "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong",
    "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi",
    "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan",
    "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang",
]

GENDERS = ["Male", "Female"]
AGE_CATEGORIES = ["Young Adults", "Middle-aged Adults", "Older Adults"]
EDUCATION_COLS = [
    "Junior College and Above",
    "Senior Secondary School",
    "Junior Secondary School",
    "Primary School",
]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def clean_string_cell(x) -> str:
    if pd.isna(x):
        return ""
    return (
        str(x)
        .replace("\ufeff", "")
        .replace(",", "")
        .replace("％", "%")
        .strip()
    )


def parse_numeric_cell(x) -> float:
    s = clean_string_cell(x)
    if s == "":
        return np.nan
    has_percent = s.endswith("%")
    s = s.replace("%", "").strip()
    value = float(s)
    if has_percent:
        value /= 100.0
    return value


def parse_probability_series(values: pd.Series) -> pd.Series:
    return values.apply(parse_numeric_cell)


def normalize_prob(values: pd.Series) -> np.ndarray:
    arr = parse_probability_series(values).fillna(0.0).to_numpy(dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probability values must be non-negative.")
    total = arr.sum()
    if total <= 0:
        raise ValueError("Probability values sum to zero. Please check the input table.")
    return arr / total


def parse_age2_relative_level_cell(x) -> float:
    """
    Parse age2.csv '65+ Relative Level'.

    Correct meanings:
      "98.52%" -> 0.9852
      "109.3%" -> 1.093
      "129.04%" -> 1.2904
      "0.9852" -> 0.9852
      "1.2904" -> 1.2904

    Defensive rule:
      If a value like 98.52 is saved without %, treat it as 98.52%.
    """
    s = clean_string_cell(x)
    if s == "":
        return np.nan
    has_percent = s.endswith("%")
    s_num = s.replace("%", "").strip()
    value = float(s_num)

    if has_percent:
        return value / 100.0
    if value > 3.0:
        return value / 100.0
    return value


def read_region_table(path: Path) -> pd.DataFrame:
    df = clean_column_names(pd.read_csv(path))
    if "Region" not in df.columns:
        raise ValueError(f"{path} must contain column: Region")
    df["Region"] = df["Region"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    missing = [r for r in REGION_ORDER if r not in set(df["Region"])]
    if missing:
        raise ValueError(f"{path} missing regions: {missing}")
    return df.set_index("Region").loc[REGION_ORDER].reset_index()


def sample_one(rng: np.random.Generator, candidates, probs):
    probs = np.asarray(probs, dtype=float)
    total = probs.sum()
    if total <= 0:
        raise ValueError("Sampling probabilities sum to zero.")
    probs = probs / total
    idx = int(rng.choice(len(candidates), p=probs))
    return candidates[idx]


def resolve_info_file(info_dir: Path, file_arg: str) -> Path:
    path = Path(file_arg)
    if path.is_absolute():
        return path
    return info_dir / path


def load_source_province(info_dir: Path, source_province_file: str) -> pd.DataFrame:
    path = resolve_info_file(info_dir, source_province_file)
    if not path.exists():
        raise FileNotFoundError(f"Source province file not found: {path}")

    df = clean_column_names(pd.read_csv(path))
    required = ["Source", "province"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"{path} must contain column: {col}")

    df = df[required].copy()
    df["Source"] = df["Source"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df["province"] = df["province"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    df = df[(df["Source"] != "") & (df["province"] != "")]

    if df.empty:
        raise ValueError(f"{path} contains no valid Source-province rows.")

    duplicated = df[df["Source"].duplicated(keep=False)]["Source"].unique().tolist()
    if duplicated:
        examples = duplicated[:10]
        raise ValueError(f"Source province file contains duplicated Source ids, examples: {examples}")

    valid_regions = set(REGION_ORDER)
    invalid_regions = sorted(set(df["province"]) - valid_regions)
    if invalid_regions:
        raise ValueError(
            "source_province_information.csv contains invalid province names. "
            f"Invalid values: {invalid_regions}. Expected names: {REGION_ORDER}"
        )

    return df.sort_values("Source").reset_index(drop=True)


def load_gender(info_dir: Path) -> pd.DataFrame:
    df = read_region_table(info_dir / "gender.csv")
    for col in GENDERS:
        if col not in df.columns:
            raise ValueError(f"gender.csv must contain column: {col}")
    rows = []
    for _, row in df.iterrows():
        probs = normalize_prob(row[GENDERS])
        rows.append({"Region": row["Region"], "Male": probs[0], "Female": probs[1]})
    return pd.DataFrame(rows)


def load_education(info_dir: Path) -> pd.DataFrame:
    df = read_region_table(info_dir / "education.csv")
    for col in EDUCATION_COLS:
        if col not in df.columns:
            raise ValueError(f"education.csv must contain column: {col}")
    rows = []
    for _, row in df.iterrows():
        probs = normalize_prob(row[EDUCATION_COLS])
        item = {"Region": row["Region"]}
        item.update({col: prob for col, prob in zip(EDUCATION_COLS, probs)})
        rows.append(item)
    return pd.DataFrame(rows)


def load_occupation(info_dir: Path, occupation_file: str = "occupation.csv"):
    path = info_dir / occupation_file
    if not path.exists():
        return None, []

    df = read_region_table(path)
    occupation_cols = [c for c in df.columns if c != "Region"]
    if not occupation_cols:
        raise ValueError(f"{path} must contain at least one occupation/industry column besides Region.")

    rows = []
    for _, row in df.iterrows():
        probs = normalize_prob(row[occupation_cols])
        item = {"Region": row["Region"]}
        item.update({col: prob for col, prob in zip(occupation_cols, probs)})
        rows.append(item)
    return pd.DataFrame(rows), occupation_cols


def load_age_base(info_dir: Path) -> pd.DataFrame:
    df = clean_column_names(pd.read_csv(info_dir / "age1.csv"))
    required = ["Age Category", "Male", "Female"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"age1.csv must contain column: {col}")
    df["Age Category"] = df["Age Category"].astype(str).str.replace("\ufeff", "", regex=False).str.strip()
    missing = [c for c in AGE_CATEGORIES if c not in set(df["Age Category"])]
    if missing:
        raise ValueError(f"age1.csv missing age categories: {missing}")
    df = df.set_index("Age Category").loc[AGE_CATEGORIES].reset_index()

    df["Male"] = normalize_prob(df["Male"])
    df["Female"] = normalize_prob(df["Female"])
    return df


def load_age_relative_level(info_dir: Path) -> pd.DataFrame:
    df = read_region_table(info_dir / "age2.csv")

    possible_cols = [c for c in df.columns if c != "Region" and "65" in c and "Relative" in c]
    if "65+ Relative Level" in df.columns:
        col = "65+ Relative Level"
    elif possible_cols:
        col = possible_cols[0]
    else:
        raise ValueError("age2.csv must contain column: 65+ Relative Level")

    df["65+ Relative Level"] = df[col].apply(parse_age2_relative_level_cell)

    if df["65+ Relative Level"].isna().any() or (df["65+ Relative Level"] <= 0).any():
        raise ValueError("age2.csv contains invalid 65+ Relative Level values.")

    too_large = df[df["65+ Relative Level"] > 3.0]
    if not too_large.empty:
        examples = too_large[["Region", "65+ Relative Level"]].head(5).to_dict("records")
        raise ValueError(
            "age2.csv '65+ Relative Level' is still too large after parsing. "
            "Expected values such as 98.52% -> 0.9852 or 1.2904. "
            f"Parsed examples: {examples}"
        )

    return df[["Region", "65+ Relative Level"]]


def build_age_distribution_by_region_and_gender(
    age_base_df: pd.DataFrame,
    age_relative_df: pd.DataFrame,
    old_factor_lower_bound: float = 0.5,
) -> dict:
    age_base = age_base_df.set_index("Age Category")
    result = {}

    for _, row in age_relative_df.iterrows():
        region = row["Region"]
        old_factor_raw = float(row["65+ Relative Level"])
        old_factor_used = max(old_factor_raw, old_factor_lower_bound)
        result[region] = {}

        for gender in GENDERS:
            base_young = float(age_base.loc["Young Adults", gender])
            base_middle = float(age_base.loc["Middle-aged Adults", gender])
            base_old = float(age_base.loc["Older Adults", gender])

            p_old = base_old * old_factor_used
            if p_old >= 0.80:
                raise ValueError(
                    f"Computed Older Adults probability is abnormal: region={region}, gender={gender}, "
                    f"base_old={base_old:.6f}, old_factor_raw={old_factor_raw:.6f}, "
                    f"old_factor_used={old_factor_used:.6f}, p_old={p_old:.6f}. "
                    "Please check age1.csv and age2.csv."
                )

            remaining = 1.0 - p_old
            young_middle_sum = base_young + base_middle
            if young_middle_sum <= 0:
                raise ValueError("base_young + base_middle must be positive.")

            p_young = remaining * base_young / young_middle_sum
            p_middle = remaining * base_middle / young_middle_sum

            probs = np.array([p_young, p_middle, p_old], dtype=float)
            probs = probs / probs.sum()

            result[region][gender] = {
                "Young Adults": probs[0],
                "Middle-aged Adults": probs[1],
                "Older Adults": probs[2],
                "65+ Relative Level Raw": old_factor_raw,
                "65+ Relative Level Used": old_factor_used,
            }

    return result


def initialize_user_information(
    source_province_df: pd.DataFrame,
    gender_df: pd.DataFrame,
    education_df: pd.DataFrame,
    age_dist: dict,
    occupation_df=None,
    occupation_cols=None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gender_lookup = gender_df.set_index("Region")
    education_lookup = education_df.set_index("Region")
    occupation_lookup = occupation_df.set_index("Region") if occupation_df is not None else None
    occupation_cols = occupation_cols or []

    records = []
    for _, src_row in source_province_df.iterrows():
        source_id = src_row["Source"]
        region = src_row["province"]

        gender_probs = gender_lookup.loc[region, GENDERS].to_numpy(dtype=float)
        gender = sample_one(rng, GENDERS, gender_probs)

        edu_probs = education_lookup.loc[region, EDUCATION_COLS].to_numpy(dtype=float)
        education = sample_one(rng, EDUCATION_COLS, edu_probs)

        age_probs = [age_dist[region][gender][cat] for cat in AGE_CATEGORIES]
        age_group = sample_one(rng, AGE_CATEGORIES, age_probs)

        record = {
            "Source": source_id,
            "province": region,
            "gender": gender,
            "age_group": age_group,
            "education": education,
        }

        if occupation_lookup is not None:
            occ_probs = occupation_lookup.loc[region, occupation_cols].to_numpy(dtype=float)
            record["occupation_industry"] = sample_one(rng, occupation_cols, occ_probs)

        records.append(record)

    return pd.DataFrame(records)


def build_source_table(user_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Source", "province", "gender", "age_group", "education"]
    if "occupation_industry" in user_df.columns:
        columns.append("occupation_industry")
    missing = [col for col in columns if col not in user_df.columns]
    if missing:
        raise ValueError(f"User information table missing columns: {missing}")
    return user_df[columns].copy()


def build_age_distribution_debug(age_dist: dict) -> pd.DataFrame:
    rows = []
    for region in REGION_ORDER:
        for gender in GENDERS:
            item = age_dist[region][gender]
            rows.append({
                "province": region,
                "gender": gender,
                "65+ Relative Level Raw": item["65+ Relative Level Raw"],
                "65+ Relative Level Used": item["65+ Relative Level Used"],
                "Young Adults": item["Young Adults"],
                "Middle-aged Adults": item["Middle-aged Adults"],
                "Older Adults": item["Older Adults"],
            })
    return pd.DataFrame(rows)


def ratio_table(df: pd.DataFrame, column: str, summary_type: str) -> pd.DataFrame:
    ratio = df[column].value_counts(normalize=True).rename("ratio").reset_index()
    ratio.columns = ["category", "ratio"]
    ratio.insert(0, "summary_type", summary_type)
    return ratio


def write_summary(df: pd.DataFrame, age_debug_df: pd.DataFrame, output_path: Path):
    output_path = Path(output_path)

    summary_tables = [
        ratio_table(df, "province", "province"),
        ratio_table(df, "gender", "gender"),
        ratio_table(df, "age_group", "age_group"),
        ratio_table(df, "education", "education"),
    ]
    if "occupation_industry" in df.columns:
        summary_tables.append(ratio_table(df, "occupation_industry", "occupation_industry"))

    province_age_sampled = pd.crosstab(df["province"], df["age_group"], normalize="index").reset_index()
    for col in AGE_CATEGORIES:
        if col not in province_age_sampled.columns:
            province_age_sampled[col] = 0.0
    province_age_sampled = province_age_sampled[["province"] + AGE_CATEGORIES]

    province_gender_sampled = pd.crosstab(df["province"], df["gender"], normalize="index").reset_index()
    for col in GENDERS:
        if col not in province_gender_sampled.columns:
            province_gender_sampled[col] = 0.0
    province_gender_sampled = province_gender_sampled[["province"] + GENDERS]

    province_education_sampled = pd.crosstab(df["province"], df["education"], normalize="index").reset_index()
    for col in EDUCATION_COLS:
        if col not in province_education_sampled.columns:
            province_education_sampled[col] = 0.0
    province_education_sampled = province_education_sampled[["province"] + EDUCATION_COLS]

    if output_path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output_path) as writer:
            for table in summary_tables:
                sheet_name = str(table["summary_type"].iloc[0])[:31]
                table.to_excel(writer, sheet_name=sheet_name, index=False)
            province_gender_sampled.to_excel(writer, sheet_name="province_gender_sampled", index=False)
            province_age_sampled.to_excel(writer, sheet_name="province_age_sampled", index=False)
            province_education_sampled.to_excel(writer, sheet_name="province_edu_sampled", index=False)
            age_debug_df.to_excel(writer, sheet_name="age_distribution_expected", index=False)
    else:
        compact = pd.concat(summary_tables, ignore_index=True)
        compact.to_csv(output_path, index=False, encoding="utf-8-sig")
        stem = output_path.with_suffix("")
        province_gender_sampled.to_csv(f"{stem}_province_gender_sampled.csv", index=False, encoding="utf-8-sig")
        province_age_sampled.to_csv(f"{stem}_province_age_sampled.csv", index=False, encoding="utf-8-sig")
        province_education_sampled.to_csv(f"{stem}_province_education_sampled.csv", index=False, encoding="utf-8-sig")
        age_debug_df.to_csv(f"{stem}_age_distribution_expected.csv", index=False, encoding="utf-8-sig")


def main():
    """
    Default one-click configuration.

    Put all required input files under ./information:
      - source_province_information.csv  columns: Source, province
      - gender.csv
      - education.csv
      - age1.csv
      - age2.csv
      - occupation.csv  optional

    Then run directly:
      python user_information_initialization_v8_folder_defaults.py

    Outputs are written to ./user_information_outputs by default.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Initialize Source user personal information using a pre-assigned "
            "Source-province table. Province is no longer randomly sampled. "
            "Default parameters are already configured in the script."
        )
    )

    # Default input configuration. In normal use, run the script without any arguments.
    parser.add_argument("--information_dir", type=str, default=str(INFORMATION_DIR))
    parser.add_argument(
        "--source_province_file",
        type=str,
        default=str(DEFAULT_SOURCE_PROVINCE_FILE),
        help="CSV file with columns Source,province. Relative paths are resolved under --information_dir.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_SOURCE_OUTPUT),
        help="Output path for source.csv in work/.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=None,
        help="Optional summary report path. Omitted by default.",
    )

    # Default sampling configuration.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--old_factor_lower_bound", type=float, default=0.5)
    parser.add_argument(
        "--disable_occupation",
        action="store_true",
        help="Disable occupation_industry sampling even if occupation.csv exists.",
    )
    args = parser.parse_args()

    info_dir = Path(args.information_dir)
    if not info_dir.exists():
        raise FileNotFoundError(f"Information directory not found: {info_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path = Path(args.summary_output) if args.summary_output else None

    source_province_df = load_source_province(info_dir, args.source_province_file)
    gender_df = load_gender(info_dir)
    education_df = load_education(info_dir)
    age_base_df = load_age_base(info_dir)
    age_relative_df = load_age_relative_level(info_dir)

    age_dist = build_age_distribution_by_region_and_gender(
        age_base_df=age_base_df,
        age_relative_df=age_relative_df,
        old_factor_lower_bound=args.old_factor_lower_bound,
    )
    age_debug_df = build_age_distribution_debug(age_dist)

    occupation_df, occupation_cols = (None, [])
    if not args.disable_occupation:
        occupation_df, occupation_cols = load_occupation(info_dir)

    output_df = build_source_table(
        initialize_user_information(
            source_province_df=source_province_df,
            gender_df=gender_df,
            education_df=education_df,
            age_dist=age_dist,
            occupation_df=occupation_df,
            occupation_cols=occupation_cols,
            seed=args.seed,
        )
    )

    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    if summary_output_path is not None:
        summary_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_summary(output_df, age_debug_df, summary_output_path)

    print("Done.")
    print(f"Number of Source users: {len(output_df)}")
    print(f"Province input: {resolve_info_file(info_dir, args.source_province_file)}")
    print(f"Output saved to: {output_path}")
    if summary_output_path is not None:
        print(f"Summary saved to: {summary_output_path}")
    if occupation_df is None:
        print("Occupation sampling: disabled or occupation.csv not found")
    else:
        print("Occupation sampling: enabled")


if __name__ == "__main__":
    main()
