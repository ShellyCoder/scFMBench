#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 4: Batch integration evaluation.

- Input: precomputed zero-shot cell embeddings 
- Evaluation: scIB metrics on integrated embeddings
  - Biology: ARI, NMI, Silhouette_celltype
  - Batch: Silhouette_batch, iLISI, kBET
- Clustering: scanpy neighbors (n_neighbors=15) + scib.me.cluster_optimal_resolution
  (scIB chooses NMI-optimal resolution; cluster stored in adata.obs["cluster"])

Outputs:
- One CSV file containing all (dataset, method) rows.
"""

import os
import argparse
from typing import List

import pandas as pd
import anndata as ad
import scanpy as sc
import scib
from joblib import Parallel, delayed
from tqdm import tqdm


# -----------------------
# CLI helpers
# -----------------------
def parse_csv_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def read_list_file(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x and not x.startswith("#"):
                items.append(x)
    return items


# -----------------------
# Core evaluation
# -----------------------
def evaluate_one(
    dataset_name: str,
    method: str,
    embedding_root: str,
    metadata_root: str,
    mode: str,
) -> dict:
    """
    mode:
      - baseline: <embedding_root>/<METHOD>_outputData/<DATASET>_cellEmbedding.csv
                 <metadata_root>/<DATASET>_metaInfo.csv
      - augment : <embedding_root>/<METHOD>_outputData/Augment_res/<DATASET>_cellEmbedding.csv
                 <metadata_root>/Augment_Data/csv/<DATASET>_metaInfo.csv
    """
    if mode == "baseline":
        emb_path = os.path.join(embedding_root, f"{method}_outputData", f"{dataset_name}_cellEmbedding.csv")
        meta_path = os.path.join(metadata_root, f"{dataset_name}_metaInfo.csv")
    elif mode == "augment":
        emb_path = os.path.join(embedding_root, f"{method}_outputData", "Augment_res", f"{dataset_name}_cellEmbedding.csv")
        meta_path = os.path.join(metadata_root, "Augment_Data", "csv", f"{dataset_name}_metaInfo.csv")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Missing embedding file: {emb_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing metadata file: {meta_path}")

    embedding_df = pd.read_csv(emb_path, index_col=0)
    metadata_df = pd.read_csv(meta_path, index_col=0)

    # strict alignment (content + order)
    if not embedding_df.index.equals(metadata_df.index):
        raise ValueError(f"Cell IDs mismatch or different order: dataset={dataset_name}, method={method}")

    # build AnnData with embeddings
    adata = ad.AnnData(obs=metadata_df.copy())
    adata.obs_names = embedding_df.index
    adata.obsm["X_pca_harmony"] = embedding_df.values  # keep your original key name

    # neighbors + optimal clustering resolution (scIB default strategy)
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=15)
    all_res = scib.me.cluster_optimal_resolution(
        adata,
        cluster_key="cluster",
        label_key="celltype",
        force=True,
        return_all=True
    )

    res = {
        "dataset": dataset_name,
        "method": method,
        "best_resolution": all_res[0],  # cluster stored in adata.obs["cluster"]
    }

    # Biology metrics
    res["ARI"] = scib.me.ari(adata, cluster_key="cluster", label_key="celltype")
    res["NMI"] = scib.me.nmi(adata, cluster_key="cluster", label_key="celltype")
    res["Silhouette_celltype"] = scib.metrics.silhouette(adata, embed="X_pca_harmony", label_key="celltype")

    # Batch metrics
    # NOTE: requires metadata columns: "tech" (batch), "celltype" (label)
    res["Silhouette_batch"] = scib.metrics.silhouette_batch(
        adata,
        embed="X_pca_harmony",
        batch_key="tech",
        label_key="celltype"
    )

    # iLISI + kBET require categorical batch/label columns (your original handling retained)
    adata.obs["celltype_category"] = adata.obs["celltype"].astype("category")
    adata.obs["tech_category"] = adata.obs["tech"].astype("category")

    res["iLISI"] = scib.me.ilisi_graph(
        adata,
        batch_key="tech_category",
        type_="embed",
        use_rep="X_pca_harmony"
    )

    adata.obs["tech_category"] = adata.obs["tech_category"].astype(str).astype("category")
    adata.obs["celltype_category"] = adata.obs["celltype_category"].astype(str).astype("category")

    res["kBET"] = scib.metrics.kBET(
        adata,
        embed="X_pca_harmony",
        type_="embed",
        batch_key="tech_category",
        label_key="celltype_category"
    )

    return res


def main():
    parser = argparse.ArgumentParser(description="Task 3: Batch integration evaluation (scIB)")

    group_m = parser.add_mutually_exclusive_group(required=True)
    group_m.add_argument("--methods", type=str, help="Comma-separated method names (required)")
    group_m.add_argument("--methods-file", type=str, help="Text file with one method per line")

    group_d = parser.add_mutually_exclusive_group(required=True)
    group_d.add_argument("--datasets", type=str, help="Comma-separated dataset names (required)")
    group_d.add_argument("--datasets-file", type=str, help="Text file with one dataset per line")

    parser.add_argument("--mode", type=str, choices=["baseline", "augment"], required=True,
                        help="baseline: original datasets; augment: augmented datasets")
    parser.add_argument("--embedding-root", type=str, required=True,
                        help="Root directory containing <METHOD>_outputData/")
    parser.add_argument("--metadata-root", type=str, required=True,
                        help="Root directory containing dataset metaInfo csv files")
    parser.add_argument("--output-csv", type=str, required=True,
                        help="Path to write a single merged CSV result")
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--backend", type=str, default="loky")

    args = parser.parse_args()

    methods = parse_csv_list_arg(args.methods) if args.methods else read_list_file(args.methods_file)
    datasets = parse_csv_list_arg(args.datasets) if args.datasets else read_list_file(args.datasets_file)
    if len(methods) == 0 or len(datasets) == 0:
        raise ValueError("Empty methods or datasets.")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    tasks = [(ds, m) for ds in datasets for m in methods]

    all_results = Parallel(n_jobs=args.n_jobs, backend=args.backend)(
        delayed(evaluate_one)(ds, m, args.embedding_root, args.metadata_root, args.mode)
        for ds, m in tqdm(tasks, desc="Evaluating", unit="task")
    )

    df = pd.DataFrame(all_results)
    metric_cols = [c for c in df.columns if c not in ("dataset", "method")]
    df.loc[:, metric_cols] = df.loc[:, metric_cols].astype(float).round(6)

    df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved results: {args.output_csv}")


if __name__ == "__main__":
    main()
