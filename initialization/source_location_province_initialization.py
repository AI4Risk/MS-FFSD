import argparse
import math
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components

REPO_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = REPO_ROOT / "work"
INFORMATION_DIR = REPO_ROOT / "initialization" / "informations"


REGION_ORDER = [
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Liaoning",
    "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui",
    "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan",
    "Guangdong", "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou",
    "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang"
]


# =========================
# Default configuration
# You can run this script directly with:
#   python source_location_province_initialization.py
# Run step2 only after this step finishes.
# =========================
DEFAULT_TRANSACTION_FILE = str(WORK_DIR / "FFSD_synthetic_time_detail.csv")
DEFAULT_INFORMATION_DIR = str(INFORMATION_DIR)
DEFAULT_SOURCE_PROVINCE_OUTPUT = str(WORK_DIR / "source_province_information.csv")
DEFAULT_TRANSACTION_OUTPUT = str(WORK_DIR / "transaction.csv")
DEFAULT_SOURCE_COL = "Source"
DEFAULT_LOCATION_COL = "Location"
DEFAULT_AMOUNT_COL = "Amount"
DEFAULT_TRANSACTION_ID_COL = None
DEFAULT_LABEL_COL = "Labels"
DEFAULT_NORMAL_LABEL_VALUE = "0"
DEFAULT_FRAUD_LABEL_VALUE = "1"
DEFAULT_NORMAL_OUT_GLOBAL_PROB = 0.03
DEFAULT_FRAUD_OUT_GLOBAL_PROB = 0.09
DEFAULT_RANDOM_SEED = 42
DEFAULT_MATRIX_MODE = "binary"
DEFAULT_COSINE_THRESHOLD = 0.10
DEFAULT_MAX_NORMAL_BLOCK_SOURCE_RATIO = 0.06
DEFAULT_WARNING_BLOCK_SOURCE_RATIO = 0.05
DEFAULT_THRESHOLD_STEP = 0.05
DEFAULT_MAX_THRESHOLD = 0.95
DEFAULT_SOURCE_WEIGHT = 0.75
DEFAULT_AMOUNT_WEIGHT = 0.25


def clean_numeric_series(values: pd.Series) -> pd.Series:
    s = values.astype(str)
    s = (
        s.str.replace("\ufeff", "", regex=False)
         .str.replace(",", "", regex=False)
         .str.replace("％", "%", regex=False)
         .str.strip()
    )
    has_percent = s.str.contains("%", regex=False)
    s = s.str.replace("%", "", regex=False)
    numeric = pd.to_numeric(s, errors="coerce").fillna(0.0)
    numeric = numeric.where(~has_percent, numeric / 100.0)
    return numeric


def normalize_prob(values: pd.Series) -> np.ndarray:
    arr = clean_numeric_series(values).to_numpy(dtype=float)
    if np.any(arr < 0):
        raise ValueError("Probability values must be non-negative.")
    total = arr.sum()
    if total <= 0:
        raise ValueError("Probability values sum to zero.")
    return arr / total


def read_population_and_consumption(info_dir: Path) -> pd.DataFrame:
    pop_path = info_dir / "population.csv"
    con_path = info_dir / "consumption.csv"
    if not pop_path.exists():
        raise FileNotFoundError(f"Missing file: {pop_path}")
    if not con_path.exists():
        raise FileNotFoundError(f"Missing file: {con_path}")

    pop = pd.read_csv(pop_path)
    con = pd.read_csv(con_path)
    pop.columns = [c.strip().replace("\ufeff", "") for c in pop.columns]
    con.columns = [c.strip().replace("\ufeff", "") for c in con.columns]

    if "Region" not in pop.columns or "Population" not in pop.columns:
        raise ValueError("population.csv must contain columns: Region, Population")
    if "Region" not in con.columns or "Consumption" not in con.columns:
        raise ValueError("consumption.csv must contain columns: Region, Consumption")

    pop["Region"] = pop["Region"].astype(str).str.strip()
    con["Region"] = con["Region"].astype(str).str.strip()

    missing_pop = [r for r in REGION_ORDER if r not in set(pop["Region"])]
    missing_con = [r for r in REGION_ORDER if r not in set(con["Region"])]
    if missing_pop:
        raise ValueError(f"population.csv missing regions: {missing_pop}")
    if missing_con:
        raise ValueError(f"consumption.csv missing regions: {missing_con}")

    pop = pop.set_index("Region").loc[REGION_ORDER].reset_index()
    con = con.set_index("Region").loc[REGION_ORDER].reset_index()

    population_share = normalize_prob(pop["Population"])
    consumption_share = normalize_prob(con["Consumption"])
    activity = population_share * consumption_share
    activity_prob = activity / activity.sum()

    return pd.DataFrame({
        "Region": REGION_ORDER,
        "population_share": population_share,
        "per_capita_consumption_share": consumption_share,
        "consumption_activity_prob": activity_prob,
    })


def build_location_source_matrix(df: pd.DataFrame, source_col: str, location_col: str, matrix_mode: str):
    edge = df.groupby([location_col, source_col], observed=True).size().reset_index(name="tx_count")
    locations = sorted(edge[location_col].astype(str).unique().tolist())
    sources = sorted(edge[source_col].astype(str).unique().tolist())
    loc_to_idx = {loc: i for i, loc in enumerate(locations)}
    src_to_idx = {src: i for i, src in enumerate(sources)}

    rows = edge[location_col].astype(str).map(loc_to_idx).to_numpy()
    cols = edge[source_col].astype(str).map(src_to_idx).to_numpy()
    if matrix_mode == "binary":
        data = np.ones(len(edge), dtype=np.float32)
    elif matrix_mode == "log_count":
        data = np.log1p(edge["tx_count"].to_numpy(dtype=float)).astype(np.float32)
    else:
        raise ValueError("matrix_mode must be binary or log_count")

    mat = sparse.csr_matrix((data, (rows, cols)), shape=(len(locations), len(sources)), dtype=np.float32)
    return mat, locations, sources, loc_to_idx, src_to_idx


def cosine_components_for_indices(mat: sparse.csr_matrix, loc_indices, threshold: float):
    loc_indices = list(loc_indices)
    n = len(loc_indices)
    if n == 0:
        return []
    if n == 1:
        return [loc_indices]

    sub = mat[loc_indices].astype(np.float32)
    row_norm = np.sqrt(sub.multiply(sub).sum(axis=1)).A1
    row_norm[row_norm == 0] = 1.0
    sub_norm = sub.multiply(1.0 / row_norm[:, None])
    sim = (sub_norm @ sub_norm.T).tocoo()

    mask = (sim.row != sim.col) & (sim.data >= threshold)
    if not np.any(mask):
        return [[idx] for idx in loc_indices]

    adj = sparse.coo_matrix(
        (np.ones(mask.sum(), dtype=np.int8), (sim.row[mask], sim.col[mask])),
        shape=(n, n)
    ).tocsr()
    adj = adj.maximum(adj.T)
    comp_count, labels = connected_components(adj, directed=False)

    components = []
    for c in range(comp_count):
        local_members = np.where(labels == c)[0].tolist()
        components.append([loc_indices[i] for i in local_members])
    return components


def assign_sources_to_location_clusters(df, source_col, location_col, loc_to_cluster):
    tmp = df[[source_col, location_col]].copy()
    tmp[source_col] = tmp[source_col].astype(str)
    tmp[location_col] = tmp[location_col].astype(str)
    tmp["cluster_id"] = tmp[location_col].map(loc_to_cluster)
    counts = tmp.groupby([source_col, "cluster_id"], observed=True).size().reset_index(name="tx_count")
    total = counts.groupby(source_col, observed=True)["tx_count"].sum().rename("source_total_tx")
    counts = counts.merge(total, left_on=source_col, right_index=True)
    counts["cluster_ratio"] = counts["tx_count"] / counts["source_total_tx"]
    counts = counts.sort_values([source_col, "tx_count", "cluster_ratio", "cluster_id"], ascending=[True, False, False, True])
    dominant = counts.drop_duplicates(source_col, keep="first")
    return dominant.rename(columns={source_col: "Source", "cluster_id": "block_id", "cluster_ratio": "source_dominant_block_ratio"})


def summarize_blocks(df, source_col, location_col, amount_col, locations, clusters):
    loc_to_cluster = {}
    for cid, loc_indices in enumerate(clusters):
        for idx in loc_indices:
            loc_to_cluster[locations[idx]] = cid

    source_assign = assign_sources_to_location_clusters(df, source_col, location_col, loc_to_cluster)
    source_counts = source_assign.groupby("block_id", observed=True)["Source"].nunique().rename("assigned_source_count")

    tmp = df[[source_col, location_col, amount_col]].copy()
    tmp[source_col] = tmp[source_col].astype(str)
    tmp[location_col] = tmp[location_col].astype(str)
    tmp["block_id"] = tmp[location_col].map(loc_to_cluster)

    tx_counts = tmp.groupby("block_id", observed=True).size().rename("transaction_count")
    amounts = tmp.groupby("block_id", observed=True)[amount_col].sum().rename("total_amount")
    loc_counts = pd.Series({cid: len(locs) for cid, locs in enumerate(clusters)}, name="location_count")

    block_summary = pd.DataFrame({"block_id": range(len(clusters))})
    block_summary = block_summary.merge(loc_counts.reset_index().rename(columns={"index": "block_id"}), on="block_id", how="left")
    block_summary = block_summary.merge(source_counts.reset_index(), on="block_id", how="left")
    block_summary = block_summary.merge(tx_counts.reset_index(), on="block_id", how="left")
    block_summary = block_summary.merge(amounts.reset_index(), on="block_id", how="left")
    for c in ["assigned_source_count", "transaction_count", "total_amount"]:
        block_summary[c] = block_summary[c].fillna(0)
    block_summary["assigned_source_count"] = block_summary["assigned_source_count"].astype(int)
    block_summary["transaction_count"] = block_summary["transaction_count"].astype(int)
    block_summary["location_count"] = block_summary["location_count"].astype(int)
    return block_summary, source_assign, loc_to_cluster


def discover_and_refine_location_blocks(df, source_col, location_col, amount_col, mat, locations,
                                        cosine_threshold, max_normal_block_source_ratio,
                                        threshold_step, max_threshold):
    total_sources = df[source_col].astype(str).nunique()

    clusters = cosine_components_for_indices(mat, range(len(locations)), cosine_threshold)
    current_thresholds = {i: cosine_threshold for i in range(len(clusters))}

    changed = True
    while changed:
        changed = False
        block_summary, _, _ = summarize_blocks(df, source_col, location_col, amount_col, locations, clusters)
        new_clusters = []
        new_thresholds = {}

        for old_id, loc_indices in enumerate(clusters):
            row = block_summary.loc[block_summary["block_id"] == old_id].iloc[0]
            source_ratio = row["assigned_source_count"] / total_sources if total_sources else 0.0
            loc_count = int(row["location_count"])
            old_thr = current_thresholds.get(old_id, cosine_threshold)

            if source_ratio > max_normal_block_source_ratio and loc_count > 1:
                next_thr = min(old_thr + threshold_step, max_threshold)
                child_components = cosine_components_for_indices(mat, loc_indices, next_thr)

                if len(child_components) > 1:
                    changed = True
                    for comp in child_components:
                        nid = len(new_clusters)
                        new_clusters.append(comp)
                        new_thresholds[nid] = next_thr
                else:
                    if old_thr < max_threshold:
                        # Keep trying with a higher threshold in the next loop.
                        changed = True
                        nid = len(new_clusters)
                        new_clusters.append(loc_indices)
                        new_thresholds[nid] = next_thr
                    else:
                        # At the maximum threshold, force singleton split if possible.
                        changed = True
                        for loc_idx in loc_indices:
                            nid = len(new_clusters)
                            new_clusters.append([loc_idx])
                            new_thresholds[nid] = max_threshold
            else:
                nid = len(new_clusters)
                new_clusters.append(loc_indices)
                new_thresholds[nid] = old_thr

        clusters = new_clusters
        current_thresholds = new_thresholds

    block_summary, source_assign, loc_to_cluster = summarize_blocks(df, source_col, location_col, amount_col, locations, clusters)
    block_summary["assigned_source_ratio"] = block_summary["assigned_source_count"] / total_sources
    block_summary["is_global_shared_block"] = (
        (block_summary["location_count"] == 1) &
        (block_summary["assigned_source_ratio"] > max_normal_block_source_ratio)
    )
    return clusters, block_summary, source_assign, loc_to_cluster


def compute_normal_source_assignment(df, source_col, location_col, loc_to_cluster, global_blocks):
    tmp = df[[source_col, location_col]].copy()
    tmp[source_col] = tmp[source_col].astype(str)
    tmp[location_col] = tmp[location_col].astype(str)
    tmp["block_id"] = tmp[location_col].map(loc_to_cluster)
    tmp = tmp[~tmp["block_id"].isin(global_blocks)]

    if tmp.empty:
        return pd.DataFrame(columns=["Source", "block_id", "source_dominant_block_ratio"])

    counts = tmp.groupby([source_col, "block_id"], observed=True).size().reset_index(name="tx_count")
    total = counts.groupby(source_col, observed=True)["tx_count"].sum().rename("source_total_normal_tx")
    counts = counts.merge(total, left_on=source_col, right_index=True)
    counts["source_dominant_block_ratio"] = counts["tx_count"] / counts["source_total_normal_tx"]
    counts = counts.sort_values([source_col, "tx_count", "source_dominant_block_ratio", "block_id"], ascending=[True, False, False, True])
    dominant = counts.drop_duplicates(source_col, keep="first")
    return dominant.rename(columns={source_col: "Source"})[["Source", "block_id", "source_dominant_block_ratio"]]


def build_normal_block_summary(df, source_col, location_col, amount_col, loc_to_cluster, global_blocks, normal_source_assign):
    # Source count is based on dominant non-global block assignment.
    source_counts = normal_source_assign.groupby("block_id", observed=True)["Source"].nunique().rename("source_count")

    tmp = df[[location_col, amount_col]].copy()
    tmp[location_col] = tmp[location_col].astype(str)
    tmp["block_id"] = tmp[location_col].map(loc_to_cluster)
    tmp = tmp[~tmp["block_id"].isin(global_blocks)]

    tx_counts = tmp.groupby("block_id", observed=True).size().rename("transaction_count")
    amounts = tmp.groupby("block_id", observed=True)[amount_col].sum().rename("total_amount")
    loc_counts = tmp.groupby("block_id", observed=True)[location_col].nunique().rename("location_count")

    normal_ids = sorted(set(tmp["block_id"].dropna().astype(int).tolist()))
    out = pd.DataFrame({"block_id": normal_ids})
    out = out.merge(source_counts.reset_index(), on="block_id", how="left")
    out = out.merge(tx_counts.reset_index(), on="block_id", how="left")
    out = out.merge(amounts.reset_index(), on="block_id", how="left")
    out = out.merge(loc_counts.reset_index(), on="block_id", how="left")
    for c in ["source_count", "transaction_count", "total_amount", "location_count"]:
        out[c] = out[c].fillna(0)
    out["source_count"] = out["source_count"].astype(int)
    out["transaction_count"] = out["transaction_count"].astype(int)
    out["location_count"] = out["location_count"].astype(int)
    return out


def assign_seed_blocks_and_remaining(normal_block_summary, province_df, total_sources, source_weight=0.75, amount_weight=0.25):
    regions_by_pop = province_df.sort_values("population_share", ascending=False)["Region"].tolist()
    pop_share = dict(zip(province_df["Region"], province_df["population_share"]))
    act_share = dict(zip(province_df["Region"], province_df["consumption_activity_prob"]))

    total_amount = normal_block_summary["total_amount"].sum()
    target_source = {r: pop_share[r] * total_sources for r in REGION_ORDER}
    target_amount = {r: act_share[r] * total_amount for r in REGION_ORDER}

    assigned_source = {r: 0.0 for r in REGION_ORDER}
    assigned_amount = {r: 0.0 for r in REGION_ORDER}
    block_to_region = {}

    blocks = normal_block_summary[normal_block_summary["source_count"] > 0].copy()
    zero_source_blocks = normal_block_summary[normal_block_summary["source_count"] <= 0].copy()
    blocks = blocks.sort_values(["source_count", "total_amount", "location_count"], ascending=[False, False, False]).reset_index(drop=True)

    # Stage 1: seed matching. Largest 31 blocks -> 31 provinces sorted by population.
    seed_n = min(len(blocks), len(REGION_ORDER))
    seed_blocks = blocks.iloc[:seed_n].copy()
    remaining_blocks = blocks.iloc[seed_n:].copy()

    for i in range(seed_n):
        row = seed_blocks.iloc[i]
        region = regions_by_pop[i]
        bid = int(row["block_id"])
        block_to_region[bid] = region
        assigned_source[region] += float(row["source_count"])
        assigned_amount[region] += float(row["total_amount"])

    # If there are fewer than 31 non-empty blocks, some provinces will be filled later by global-only users.

    # Stage 2: remaining positive-source blocks, source gap first and amount gap second.
    for _, row in remaining_blocks.iterrows():
        bid = int(row["block_id"])
        bs = float(row["source_count"])
        ba = float(row["total_amount"])
        best_region = None
        best_score = -1e18

        for region in REGION_ORDER:
            sgap = target_source[region] - assigned_source[region]
            agap = target_amount[region] - assigned_amount[region]

            source_gap_fit = min(bs, max(sgap, 0.0)) / bs if bs > 0 else 0.0
            amount_gap_fit = min(ba, max(agap, 0.0)) / ba if ba > 0 else 0.0
            source_overfill = max(assigned_source[region] + bs - target_source[region], 0.0) / max(target_source[region], 1.0)

            score = source_weight * source_gap_fit + amount_weight * amount_gap_fit - 1.0 * source_overfill

            # Slight deterministic tie-breaker: prefer provinces with larger remaining source gap.
            score += 1e-9 * sgap

            if score > best_score:
                best_score = score
                best_region = region

        block_to_region[bid] = best_region
        assigned_source[best_region] += bs
        assigned_amount[best_region] += ba

    # Zero-source blocks do not affect user population. Assign by amount gap for location reporting only.
    if len(zero_source_blocks) > 0:
        zero_source_blocks = zero_source_blocks.sort_values(["total_amount", "location_count"], ascending=[False, False])
        for _, row in zero_source_blocks.iterrows():
            bid = int(row["block_id"])
            ba = float(row["total_amount"])
            best_region = max(REGION_ORDER, key=lambda r: target_amount[r] - assigned_amount[r])
            block_to_region[bid] = best_region
            assigned_amount[best_region] += ba

    block_assignment = pd.DataFrame([
        {"block_id": bid, "province": prov}
        for bid, prov in sorted(block_to_region.items(), key=lambda x: x[0])
    ])
    return block_assignment


def assign_global_only_sources(source_ids, source_province_map, province_df):
    pop_share = dict(zip(province_df["Region"], province_df["population_share"]))
    total_sources = len(source_ids)
    target_source = {r: pop_share[r] * total_sources for r in REGION_ORDER}
    assigned_counts = pd.Series(list(source_province_map.values())).value_counts().to_dict()
    assigned = {r: float(assigned_counts.get(r, 0)) for r in REGION_ORDER}

    missing_sources = [s for s in source_ids if s not in source_province_map]
    for s in sorted(missing_sources):
        # Fill largest relative source deficit.
        best_region = max(REGION_ORDER, key=lambda r: (target_source[r] - assigned[r]) / max(target_source[r], 1.0))
        source_province_map[s] = best_region
        assigned[best_region] += 1.0
    return source_province_map


def sample_weighted_region(candidates, weights, rng):
    candidates = list(candidates)
    weights = np.asarray(weights, dtype=float)
    if len(candidates) == 0:
        raise ValueError("No candidate regions are available for sampling.")
    if weights.shape[0] != len(candidates):
        raise ValueError("Candidate regions and weights must have the same length.")
    weights = np.where(weights < 0, 0.0, weights)
    total = weights.sum()
    if total <= 0:
        weights = np.ones(len(candidates), dtype=float) / len(candidates)
    else:
        weights = weights / total
    return str(rng.choice(candidates, p=weights))


def sample_out_province(source, source_province, source_out_province_pool, province_df, rng):
    pool = source_out_province_pool.get(source)
    if pool:
        candidates = list(pool.keys())
        weights = np.asarray([pool[p] for p in candidates], dtype=float)
        return sample_weighted_region(candidates, weights, rng)

    candidates = [r for r in REGION_ORDER if r != source_province]
    activity_lookup = dict(zip(province_df["Region"], province_df["consumption_activity_prob"]))
    weights = [activity_lookup.get(r, 0.0) for r in candidates]
    return sample_weighted_region(candidates, weights, rng)


def assign_transaction_provinces_with_fraud_control(
    df,
    source_col,
    location_col,
    label_col,
    normal_label_value,
    fraud_label_value,
    loc_to_cluster,
    block_to_region,
    global_blocks,
    source_province_df,
    province_df,
    normal_out_global_prob,
    fraud_out_global_prob,
    random_seed,
):
    if not (0.0 <= normal_out_global_prob <= 1.0):
        raise ValueError("normal_out_global_prob must be between 0 and 1.")
    if not (0.0 <= fraud_out_global_prob <= 1.0):
        raise ValueError("fraud_out_global_prob must be between 0 and 1.")
    if label_col not in df.columns:
        raise ValueError(f"transaction_file must contain label column: {label_col}")

    fraud_rng = np.random.default_rng(random_seed)
    normal_rng = np.random.default_rng(random_seed + 1)
    loc_block = df[location_col].map(loc_to_cluster)
    source_province_lookup = dict(zip(source_province_df["Source"].astype(str), source_province_df["province"]))
    source_out_province_pool = defaultdict(Counter)

    transaction_provinces = []
    fraud_control_candidate_count = 0
    fraud_control_triggered_count = 0
    normal_control_candidate_count = 0
    normal_control_triggered_count = 0

    source_values = df[source_col].astype(str).tolist()
    label_values = df[label_col].astype(str).str.strip().tolist()

    for src, bid, label in zip(source_values, loc_block.tolist(), label_values):
        bid = int(bid)
        source_province = source_province_lookup[str(src)]
        is_normal_transaction = label == str(normal_label_value)
        is_fraud_transaction = label == str(fraud_label_value)

        if bid in global_blocks:
            if is_normal_transaction:
                normal_control_candidate_count += 1
                if normal_rng.random() < normal_out_global_prob:
                    province = sample_out_province(
                        source=str(src),
                        source_province=source_province,
                        source_out_province_pool=source_out_province_pool,
                        province_df=province_df,
                        rng=normal_rng,
                    )
                    transaction_provinces.append(province)
                    source_out_province_pool[str(src)][province] += 1
                    normal_control_triggered_count += 1
                    continue

            if is_fraud_transaction:
                fraud_control_candidate_count += 1
                if fraud_rng.random() < fraud_out_global_prob:
                    province = sample_out_province(
                        source=str(src),
                        source_province=source_province,
                        source_out_province_pool=source_out_province_pool,
                        province_df=province_df,
                        rng=fraud_rng,
                    )
                    transaction_provinces.append(province)
                    source_out_province_pool[str(src)][province] += 1
                    fraud_control_triggered_count += 1
                    continue

            transaction_provinces.append(source_province)
        else:
            province = block_to_region.get(bid, source_province)
            transaction_provinces.append(province)
            if province != source_province:
                source_out_province_pool[str(src)][province] += 1

    return transaction_provinces, {
        "normal_global_block_candidate_count": normal_control_candidate_count,
        "normal_global_block_out_province_count": normal_control_triggered_count,
        "normal_out_global_prob": normal_out_global_prob,
        "fraud_global_block_candidate_count": fraud_control_candidate_count,
        "fraud_global_block_out_province_count": fraud_control_triggered_count,
        "fraud_out_global_prob": fraud_out_global_prob,
        "random_seed": random_seed,
    }


def build_transaction_table(transaction_df, source_col, transaction_provinces, source_province_df):
    df = transaction_df.copy()
    if "transaction_id" in df.columns:
        df = df.drop(columns=["transaction_id"])

    source_lookup = dict(
        zip(source_province_df["Source"].astype(str), source_province_df["province"])
    )
    source_provinces = df[source_col].astype(str).map(source_lookup)
    if source_provinces.isna().any():
        missing = int(source_provinces.isna().sum())
        raise ValueError(f"Failed to map source province for {missing} transactions")

    df["transaction_province"] = list(transaction_provinces)
    df["is_out_province_transaction"] = (
        source_provinces != df["transaction_province"]
    ).astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Source-Location block based province initialization with seed-block matching. "
            "All frequently used parameters have default values, so the script can be run directly."
        )
    )
    parser.add_argument("--transaction_file", default=DEFAULT_TRANSACTION_FILE)
    parser.add_argument("--information_dir", default=DEFAULT_INFORMATION_DIR)
    parser.add_argument("--source_province_output", default=DEFAULT_SOURCE_PROVINCE_OUTPUT)
    parser.add_argument("--transaction_output", default=DEFAULT_TRANSACTION_OUTPUT)
    parser.add_argument("--source_col", default=DEFAULT_SOURCE_COL)
    parser.add_argument("--location_col", default=DEFAULT_LOCATION_COL)
    parser.add_argument("--amount_col", default=DEFAULT_AMOUNT_COL)
    parser.add_argument("--transaction_id_col", default=DEFAULT_TRANSACTION_ID_COL)
    parser.add_argument("--label_col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--normal_label_value", default=DEFAULT_NORMAL_LABEL_VALUE)
    parser.add_argument("--fraud_label_value", default=DEFAULT_FRAUD_LABEL_VALUE)
    parser.add_argument("--normal_out_global_prob", type=float, default=DEFAULT_NORMAL_OUT_GLOBAL_PROB)
    parser.add_argument("--fraud_out_global_prob", type=float, default=DEFAULT_FRAUD_OUT_GLOBAL_PROB)
    parser.add_argument("--random_seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--matrix_mode", choices=["binary", "log_count"], default=DEFAULT_MATRIX_MODE)
    parser.add_argument("--cosine_threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    parser.add_argument("--max_normal_block_source_ratio", type=float, default=DEFAULT_MAX_NORMAL_BLOCK_SOURCE_RATIO)
    parser.add_argument("--warning_block_source_ratio", type=float, default=DEFAULT_WARNING_BLOCK_SOURCE_RATIO)  # reserved
    parser.add_argument("--threshold_step", type=float, default=DEFAULT_THRESHOLD_STEP)
    parser.add_argument("--max_threshold", type=float, default=DEFAULT_MAX_THRESHOLD)
    parser.add_argument("--source_weight", type=float, default=DEFAULT_SOURCE_WEIGHT)
    parser.add_argument("--amount_weight", type=float, default=DEFAULT_AMOUNT_WEIGHT)
    args = parser.parse_args()

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    source_province_output = Path(args.source_province_output)
    transaction_output = Path(args.transaction_output)
    source_province_output.parent.mkdir(parents=True, exist_ok=True)
    transaction_output.parent.mkdir(parents=True, exist_ok=True)

    print("Running with default/configured parameters:")
    print(f"  transaction_file: {args.transaction_file}")
    print(f"  information_dir: {args.information_dir}")
    print(f"  source_province_output: {source_province_output}")
    print(f"  transaction_output: {transaction_output}")
    print(f"  source_col: {args.source_col}")
    print(f"  location_col: {args.location_col}")
    print(f"  amount_col: {args.amount_col}")
    print(f"  label_col: {args.label_col}")
    print(f"  normal_label_value: {args.normal_label_value}")
    print(f"  fraud_label_value: {args.fraud_label_value}")
    print(f"  normal_out_global_prob: {args.normal_out_global_prob}")
    print(f"  fraud_out_global_prob: {args.fraud_out_global_prob}")
    print(f"  random_seed: {args.random_seed}")
    print(f"  matrix_mode: {args.matrix_mode}")
    print(f"  cosine_threshold: {args.cosine_threshold}")
    print(f"  max_normal_block_source_ratio: {args.max_normal_block_source_ratio}")
    print(f"  threshold_step: {args.threshold_step}")
    print(f"  max_threshold: {args.max_threshold}")

    original_df = pd.read_csv(args.transaction_file)
    df = original_df.copy()
    for col in [args.source_col, args.location_col, args.amount_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"transaction_file must contain column: {col}")

    df[args.source_col] = df[args.source_col].astype(str)
    df[args.location_col] = df[args.location_col].astype(str)
    df[args.amount_col] = clean_numeric_series(df[args.amount_col])

    if args.transaction_id_col and args.transaction_id_col in df.columns:
        df["transaction_id"] = df[args.transaction_id_col].astype(str)
    else:
        df["transaction_id"] = np.arange(len(df), dtype=np.int64)

    province_df = read_population_and_consumption(Path(args.information_dir))

    mat, locations, sources, loc_to_idx, src_to_idx = build_location_source_matrix(
        df, args.source_col, args.location_col, args.matrix_mode
    )

    clusters, initial_block_summary, _, loc_to_cluster = discover_and_refine_location_blocks(
        df=df,
        source_col=args.source_col,
        location_col=args.location_col,
        amount_col=args.amount_col,
        mat=mat,
        locations=locations,
        cosine_threshold=args.cosine_threshold,
        max_normal_block_source_ratio=args.max_normal_block_source_ratio,
        threshold_step=args.threshold_step,
        max_threshold=args.max_threshold,
    )

    global_blocks = set(initial_block_summary.loc[initial_block_summary["is_global_shared_block"], "block_id"].astype(int).tolist())

    normal_source_assign = compute_normal_source_assignment(df, args.source_col, args.location_col, loc_to_cluster, global_blocks)
    normal_block_summary = build_normal_block_summary(
        df, args.source_col, args.location_col, args.amount_col, loc_to_cluster, global_blocks, normal_source_assign
    )

    block_assignment = assign_seed_blocks_and_remaining(
        normal_block_summary=normal_block_summary,
        province_df=province_df,
        total_sources=df[args.source_col].nunique(),
        source_weight=args.source_weight,
        amount_weight=args.amount_weight,
    )
    block_to_region = dict(zip(block_assignment["block_id"], block_assignment["province"]))

    # Source province: dominant normal block -> province. Global-only sources are filled by source deficit.
    source_province_map = {}
    for _, row in normal_source_assign.iterrows():
        bid = int(row["block_id"])
        if bid in block_to_region:
            source_province_map[row["Source"]] = block_to_region[bid]
    all_sources = sorted(df[args.source_col].astype(str).unique().tolist())
    source_province_map = assign_global_only_sources(all_sources, source_province_map, province_df)

    source_province_df = pd.DataFrame({
        "Source": all_sources,
        "province": [source_province_map[s] for s in all_sources],
    })
    source_province_output.parent.mkdir(parents=True, exist_ok=True)
    source_province_df.to_csv(source_province_output, index=False, encoding="utf-8-sig")

    transaction_provinces, fraud_geo_control_report = assign_transaction_provinces_with_fraud_control(
        df=df,
        source_col=args.source_col,
        location_col=args.location_col,
        label_col=args.label_col,
        normal_label_value=args.normal_label_value,
        fraud_label_value=args.fraud_label_value,
        loc_to_cluster=loc_to_cluster,
        block_to_region=block_to_region,
        global_blocks=global_blocks,
        source_province_df=source_province_df,
        province_df=province_df,
        normal_out_global_prob=args.normal_out_global_prob,
        fraud_out_global_prob=args.fraud_out_global_prob,
        random_seed=args.random_seed,
    )

    transaction_df = build_transaction_table(
        transaction_df=original_df.copy(),
        source_col=args.source_col,
        transaction_provinces=transaction_provinces,
        source_province_df=source_province_df,
    )
    transaction_df.to_csv(transaction_output, index=False, encoding="utf-8-sig")

    print("Label-aware transaction province control:")
    print(f"  candidate normal transactions in global shared blocks: {fraud_geo_control_report['normal_global_block_candidate_count']}")
    print(f"  assigned out-province for normal transactions: {fraud_geo_control_report['normal_global_block_out_province_count']}")
    print(f"  candidate fraud transactions in global shared blocks: {fraud_geo_control_report['fraud_global_block_candidate_count']}")
    print(f"  assigned out-province by fraud control: {fraud_geo_control_report['fraud_global_block_out_province_count']}")
    print("Done.")
    print(f"Wrote source province table: {source_province_output}")
    print(f"Wrote transaction table: {transaction_output}")


if __name__ == "__main__":
    main()
