#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 1: Cell clustering (zero-shot cell embedding -> scIB clustering evaluation)

- Input:
  - cell embeddings: dataset1_cellEmbedding.csv
  - metadata:        dataset1_metaInfo.csv

- Output:
  - one CSV: overall results across methods x datasets
"""

import os
import sys
import argparse
import datetime as _dt
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scib
from joblib import Parallel, delayed
from tqdm import tqdm


def read_list_file(path: str) -> List[str]:
    items: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def parse_list_arg(csv_or_empty: str) -> List[str]:
    if not csv_or_empty.strip():
        return []
    return [x.strip() for x in csv_or_empty.split(",") if x.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def utcnow_str() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def load_embedding(embedding_path: str) -> pd.DataFrame:
    emb = pd.read_csv(embedding_path, index_col=0)
    # ensure numeric
    try:
        emb = emb.astype(float)
    except Exception as e:
        raise ValueError(f"Embedding contains non-numeric values: {embedding_path}") from e

    if emb.isnull().values.any():
        n_missing = int(emb.isnull().sum().sum())
        raise ValueError(f"Found {n_missing} NA values in embedding: {embedding_path}")
    if np.isinf(emb.values).any():
        raise ValueError(f"Found Inf values in embedding: {embedding_path}")
    return emb


def load_metadata(metadata_path: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path, index_col=0)
    return meta


def evaluate_one_dataset_method(
    dataset: str,
    method: str,
    embedding_dir: str,
    metadata_dir: str,
    n_neighbors: int,
    cluster_key: str,
    label_key: str,
    force_resolution: bool,
) -> Dict[str, Any]:
    """
    Evaluate one (dataset, method).
    Returns a dict ready for DataFrame row.
    """
    embedding_path = os.path.join(
        embedding_dir,
        f"{method}_outputData",
        "Augment_res",
        f"{dataset}_cellEmbedding.csv",
    )
    metadata_path = os.path.join(metadata_dir, f"{dataset}_metaInfo.csv")

    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    emb = load_embedding(embedding_path)
    meta = load_metadata(metadata_path)

    # Sanity checks: indices must match and in same order
    if not emb.index.equals(meta.index):
        raise ValueError(
            f"Cell IDs mismatch or order mismatch between embedding and metadata.\n"
            f"Embedding: {embedding_path}\nMetadata: {metadata_path}"
        )

    # Build AnnData with embedding in obsm
    adata = ad.AnnData(obs=meta.copy())
    adata.obs_names = emb.index
    adata.obsm["X_LLM_Embedding"] = emb.values

    # scIB standard: neighbors graph based on embedding
    sc.pp.neighbors(adata, use_rep="X_LLM_Embedding", n_neighbors=n_neighbors)

    # scIB: choose resolution that maximizes NMI (cluster stored in `cluster_key`)
    all_res = scib.me.cluster_optimal_resolution(
        adata,
        cluster_key=cluster_key,
        label_key=label_key,
        force=force_resolution,
        return_all=True,
    )

    # cluster_optimal_resolution returns
    best_res = all_res[0]

    result: Dict[str, Any] = {
        "dataset": dataset,
        "method": method,
        "best_resolution": float(best_res),
        "ARI": float(scib.me.ari(adata, cluster_key=cluster_key, label_key=label_key)),
        "NMI": float(scib.me.nmi(adata, cluster_key=cluster_key, label_key=label_key)),
        "ASW_celltype": float(scib.metrics.silhouette(adata, embed="X_LLM_Embedding", label_key=label_key)),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Task1 Cell clustering benchmark (utility on original datasets). "
                    "Requires explicit methods and datasets each run."
    )

    # Required: methods + datasets (or *_file)
    ap.add_argument("--methods", default="", help="Comma-separated method names. (required if --methods-file not set)")
    ap.add_argument("--datasets", default="", help="Comma-separated dataset names. (required if --datasets-file not set)")
    ap.add_argument("--methods-file", default="", help="Path to txt file: one method per line.")
    ap.add_argument("--datasets-file", default="", help="Path to txt file: one dataset per line.")

    # IO
    ap.add_argument("--embedding-dir", required=True, help="Root dir of embeddings (e.g. ./01_Cell_Identify/)")
    ap.add_argument("--metadata-dir", required=True, help="Dir containing <dataset>_metaInfo.csv")
    ap.add_argument("--output-csv", required=True, help="Path to save merged results CSV.")
    ap.add_argument("--log-file", default="", help="Optional log file path.")

    # scIB/scRNA parameters
    ap.add_argument("--n-neighbors", type=int, default=15, help="kNN neighbors for Scanpy graph (default: 15).")
    ap.add_argument("--cluster-key", default="cluster", help="Column name for clustering labels (default: cluster).")
    ap.add_argument("--label-key", default="celltype", help="Column name for ground-truth labels (default: celltype).")
    ap.add_argument("--force-resolution", action="store_true", help="Force re-clustering when selecting resolution.")

    # runtime
    ap.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs (default: 4).")
    ap.add_argument("--backend", default="loky", choices=["loky", "threading"], help="joblib backend (default: loky).")
    ap.add_argument("--fail-fast", action="store_true", help="Stop immediately if any task fails.")

    args = ap.parse_args()

    # ---- enforce explicit specification (NO DEFAULT METHODS/DATASETS) ----
    if not (args.methods.strip() or args.methods_file.strip()):
        ap.error("You must specify methods via --methods or --methods-file.")
    if not (args.datasets.strip() or args.datasets_file.strip()):
        ap.error("You must specify datasets via --datasets or --datasets-file.")

    methods = read_list_file(args.methods_file) if args.methods_file.strip() else parse_list_arg(args.methods)
    datasets = read_list_file(args.datasets_file) if args.datasets_file.strip() else parse_list_arg(args.datasets)

    if len(methods) == 0:
        ap.error("Parsed methods list is empty. Check --methods / --methods-file.")
    if len(datasets) == 0:
        ap.error("Parsed datasets list is empty. Check --datasets / --datasets-file.")

    # logging helper
    def _log(msg: str) -> None:
        line = f"[{utcnow_str()}] {msg}"
        print(line, flush=True)
        if args.log_file:
            ensure_dir(os.path.dirname(args.log_file) or ".")
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # Basic run summary
    ensure_dir(os.path.dirname(args.output_csv) or ".")
    _log("=== Task1 clustering benchmark started ===")
    _log(f"embedding_dir={args.embedding_dir}")
    _log(f"metadata_dir={args.metadata_dir}")
    _log(f"output_csv={args.output_csv}")
    _log(f"methods(n={len(methods)})={methods}")
    _log(f"datasets(n={len(datasets)})={datasets}")
    _log(f"n_neighbors={args.n_neighbors}, n_jobs={args.n_jobs}, backend={args.backend}")
    _log(f"cluster_key={args.cluster_key}, label_key={args.label_key}, force_resolution={args.force_resolution}")

    tasks: List[Tuple[str, str]] = [(d, m) for d in datasets for m in methods]

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    def _safe_eval(d: str, m: str) -> Dict[str, Any]:
        return evaluate_one_dataset_method(
            dataset=d,
            method=m,
            embedding_dir=args.embedding_dir,
            metadata_dir=args.metadata_dir,
            n_neighbors=args.n_neighbors,
            cluster_key=args.cluster_key,
            label_key=args.label_key,
            force_resolution=args.force_resolution,
        )

    if args.fail_fast:
        # Let exceptions bubble up immediately
        results = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
            delayed(_safe_eval)(d, m) for d, m in tqdm(tasks, desc="Evaluating", unit="task")
        )
    else:
        # Collect errors but keep running
        def _wrapped(d: str, m: str) -> Dict[str, Any]:
            try:
                return _safe_eval(d, m)
            except Exception as e:
                errors.append({"dataset": d, "method": m, "error": repr(e)})
                return {}

        raw = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
            delayed(_wrapped)(d, m) for d, m in tqdm(tasks, desc="Evaluating", unit="task")
        )
        results = [r for r in raw if r]  # drop empty dicts

    if errors:
        _log(f"WARNING: {len(errors)} task(s) failed. Writing error report next to output CSV.")
        err_path = os.path.splitext(args.output_csv)[0] + ".errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        _log(f"Saved errors to: {err_path}")

    if len(results) == 0:
        raise RuntimeError("No successful results to save. Check errors and input paths.")

    df = pd.DataFrame(results)
    # Round numeric metrics (keep dataset/method as is)
    for col in ["best_resolution", "ARI", "NMI", "ASW_celltype"]:
        if col in df.columns:
            df[col] = df[col].astype(float).round(4)

    df.to_csv(args.output_csv, index=False)
    _log(f"Saved results: {args.output_csv} (rows={df.shape[0]})")
    _log("=== Task1 clustering benchmark finished ===")


if __name__ == "__main__":
    main()
