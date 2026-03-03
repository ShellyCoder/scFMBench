#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Task 6: Gene regulatory network (GRN) inference
  
  - Input matrices (per method × dataset):
    - Case GRN matrix: {dataset}_Case_geneEmbedding.csv
    - Control GRN matrix: {dataset}_Control_geneEmbedding.csv
    Each matrix:
      - index: gene symbols
      - columns: gene symbols
      - values: gene–gene association scores (float; from attention / gene embedding similarity / cosine baseline)
  
  - Datasets: TF perturbation experiments (formatted as {TF}_{MODE}, e.g., "BACH2_KD", "CDX1_OE").
    TF name is parsed as dataset.split("_")[0].

  - Ground truth (per dataset):
    - {TF}_{MODE}_GT.csv
    Each file contains a curated TF–target set (column: "gene"), derived from expression response with motif support.

  - Metrics:
    - Global (threshold-free): AUPRC
    - Top-K (fractional K): F1@5% and Jaccard@5%

  Outputs:
  - CSV (one row per method × dataset) with:
    - method, dataset, TF, and all metrics

"""

import argparse
import os
from typing import Dict, Set

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# -----------------------------
# Core utilities
# -----------------------------
def compute_tf_delta(case_mat: pd.DataFrame, ctrl_mat: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Compute delta vector for one TF: delta(g) = case(tf,g) - ctrl(tf,g)."""
    if tf not in case_mat.index:
        raise ValueError(f"TF '{tf}' not found in CASE matrix index.")
    if tf not in ctrl_mat.index:
        raise ValueError(f"TF '{tf}' not found in CTRL matrix index.")

    # Align columns defensively (same gene universe)
    common_genes = case_mat.columns.intersection(ctrl_mat.columns)
    if len(common_genes) == 0:
        raise ValueError("CASE/CTRL matrices have no overlapping gene columns.")
    case_vec = case_mat.loc[tf, common_genes]
    ctrl_vec = ctrl_mat.loc[tf, common_genes]

    delta = case_vec - ctrl_vec
    return pd.DataFrame({"gene": delta.index.astype(str), "delta": delta.values})


def prepare_ranked_df(df_delta: pd.DataFrame, tf_name: str) -> pd.DataFrame:
    """Remove self-loop, create absolute score, sort descending."""
    if "gene" not in df_delta.columns or "delta" not in df_delta.columns:
        raise ValueError("df_delta must contain columns: ['gene', 'delta'].")

    df = df_delta.copy()
    df["gene"] = df["gene"].astype(str)

    # remove TF self-loop
    df = df[df["gene"] != str(tf_name)].copy()

    # absolute score
    df["score"] = df["delta"].abs()

    # label must exist
    if "label" not in df.columns:
        raise ValueError("df_delta must contain a 'label' column.")
    df["label"] = df["label"].fillna(0).astype(int)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df


def frac_to_k(n: int, frac: float) -> int:
    return max(1, int(np.floor(frac * n)))


def global_metrics(df_ranked: pd.DataFrame) -> Dict[str, float]:
    y = df_ranked["label"].to_numpy()
    s = df_ranked["score"].to_numpy()
    # If only one class exists, AUROC will error. Handle gracefully.
    out = {}
    if len(np.unique(y)) < 2:
        out["AUROC"] = np.nan
    else:
        out["AUROC"] = float(roc_auc_score(y, s))
    out["AUPRC"] = float(average_precision_score(y, s))
    return out


def top_frac_f1(df_ranked: pd.DataFrame, top_frac: float) -> float:
    K = frac_to_k(len(df_ranked), top_frac)
    pred = df_ranked.head(K)

    tp = int((pred["label"] == 1).sum())
    fp = K - tp
    fn = int((df_ranked["label"] == 1).sum()) - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def top_frac_jaccard(df_ranked: pd.DataFrame, top_frac: float) -> float:
    K = frac_to_k(len(df_ranked), top_frac)
    pred_set = set(df_ranked.head(K)["gene"].astype(str))
    gt_set = set(df_ranked.loc[df_ranked["label"] == 1, "gene"].astype(str))
    denom = len(pred_set | gt_set)
    return (len(pred_set & gt_set) / denom) if denom > 0 else 0.0


def compute_aucell_auc(df_ranked: pd.DataFrame, gt_targets: Set[str], top_frac: float = 0.05) -> float:
    """AUCell-style recovery AUC (mean recovery within top K)."""
    genes = df_ranked["gene"].astype(str).values
    N = len(genes)
    K = frac_to_k(N, top_frac)

    gt_in = set(map(str, gt_targets)) & set(genes)
    if len(gt_in) < 5:
        return np.nan

    hits = np.isin(genes[:K], list(gt_in)).astype(int)
    if hits.sum() == 0:
        return 0.0

    recovery = np.cumsum(hits) / len(gt_in)  # recovery curve
    return float(recovery.mean())


# -----------------------------
# Main evaluation per dataset
# -----------------------------
def evaluate_grn_performance(
    method: str,
    dataset: str,
    input_dir: str,
    meta_info_dir: str,
    top_fracs=(0.05,),
    compute_extra=False
) -> pd.DataFrame:
    """
    Evaluate GRN inference performance for one TF perturbation dataset.

    Required files:
      {input_dir}/{dataset}_Case_geneEmbedding.csv
      {input_dir}/{dataset}_Control_geneEmbedding.csv
      {meta_info_dir}/{TF}_{MODE}_GT.csv
    where TF and MODE are inferred from `dataset` formatted as "TF_MODE".
    """
    # parse TF + perturbation mode
    parts = dataset.split("_")
    if len(parts) < 2:
        raise ValueError(f"Dataset name '{dataset}' must look like 'TF_MODE' (e.g., 'BACH2_KD').")
    tf_name = parts[0]
    tf_mode = parts[1]

    case_path = os.path.join(input_dir, f"{dataset}_Case_geneEmbedding.csv")
    ctrl_path = os.path.join(input_dir, f"{dataset}_Control_geneEmbedding.csv")
    gt_path = os.path.join(meta_info_dir, f"{tf_name}_{tf_mode}_GT.csv")

    for p in [case_path, ctrl_path, gt_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    case_mat = pd.read_csv(case_path, index_col=0)
    ctrl_mat = pd.read_csv(ctrl_path, index_col=0)

    # build delta table
    df_delta = compute_tf_delta(case_mat, ctrl_mat, tf=tf_name)

    # load GT
    gt = pd.read_csv(gt_path)
    if "gene" not in gt.columns:
        raise ValueError(f"GT file must contain a 'gene' column: {gt_path}")
    gt_genes = set(gt["gene"].astype(str))
    if len(gt_genes) == 0:
        raise ValueError(f"GT set is empty: {gt_path}")

    df_delta["label"] = df_delta["gene"].astype(str).isin(gt_genes).astype(int)
    if int(df_delta["label"].sum()) == 0:
        raise ValueError(f"No positive TF-targets found after aligning to matrix columns: {dataset}")

    df_ranked = prepare_ranked_df(df_delta, tf_name=tf_name)

    # metrics
    metrics = {}
    metrics.update(global_metrics(df_ranked))

    # Top-K metrics used in main text (default: 5%)
    for frac in top_fracs:
        pct = int(frac * 100)
        metrics[f"F1@{pct}%"] = top_frac_f1(df_ranked, frac)
        metrics[f"Jaccard@{pct}%"] = top_frac_jaccard(df_ranked, frac)

        if compute_extra:
            metrics[f"AUCellAUC@{pct}%"] = compute_aucell_auc(df_ranked, gt_genes, top_frac=frac)

    # round
    metrics = {k: (round(v, 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in metrics.items()}

    out = {
        "method": method,
        "dataset": dataset,
        "TF": tf_name,
        "mode": tf_mode,
        **metrics
    }
    return pd.DataFrame([out])


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Task6 GRN inference evaluation (zero-shot gene-gene relations).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--methods", nargs="+", required=True, help="Methods to evaluate, e.g. scGPT GeneCompass Cosine")
    p.add_argument("--datasets", nargs="+", required=True, help="Datasets to evaluate, e.g. BACH2_KD CDX1_OE ...")
    p.add_argument("--input-root", type=str, required=True,
                   help="Root directory that contains {method}_outputData/ folders.")
    p.add_argument("--meta-info-dir", type=str, required=True,
                   help="Directory containing ground truth files: {TF}_{MODE}_GT.csv")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path.")
    p.add_argument("--top-fracs", nargs="+", type=float, default=[0.05],
                   help="Top fractions for Top-K metrics, e.g. 0.05 0.1")
    p.add_argument("--compute-extra", action="store_true",
                   help="If set, also compute AUCellAUC@K% (optional).")
    return p.parse_args()


def main():
    args = parse_args()

    all_rows = []
    for method in args.methods:
        input_dir = os.path.join(args.input_root, f"{method}_outputData")
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found for method '{method}': {input_dir}")

        for ds in args.datasets:
            df_perf = evaluate_grn_performance(
                method=method,
                dataset=ds,
                input_dir=input_dir,
                meta_info_dir=args.meta_info_dir,
                top_fracs=tuple(args.top_fracs),
                compute_extra=args.compute_extra
            )
            all_rows.append(df_perf)

    final_df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    final_df.to_csv(args.out_csv, index=False)

    print(final_df)
    nan_datasets = final_df.loc[final_df.isna().any(axis=1), "dataset"].unique()
    print(f"\nDatasets with NaN metrics: {len(nan_datasets)}")
    if len(nan_datasets) > 0:
        print("  " + ", ".join(map(str, nan_datasets)))


if __name__ == "__main__":
    main()





