#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 5: Gene function prediction (HPA tissue-specific genes, single-label, 15 classes).

- Input embeddings: aligned gene embeddings (pickle), dict[str, pd.DataFrame]
  Each DataFrame:
    - index: gene symbols
    - values: embedding vectors (float)
- Labels: HPA tissue-specific gene sets (pickle), dict[tissue, list[gene]]
  Each gene maps to exactly one tissue (single-label multi-class).

Evaluation:
- Fixed downstream classifier: 2-layer MLP (Linear -> ReLU -> Dropout(0.3) -> Linear)
  hidden_dim=128, lr=1e-3, epochs<=100
- 5-fold CV (KFold shuffle=True, random_state=42)
- Metrics per fold: Macro-F1, Macro-AUROC, Macro-AUPRC

Outputs:
- Long-format CSV: one row per (model, fold) with metrics.
- Also writes the aligned label table for reproducibility.
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset / model / early stop
# -----------------------------
class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        # return True to continue, False to stop
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        return self.counter < self.patience


def safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int):

    auc_list = []
    aupr_list = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0:
            continue
        y_prob_c = y_prob[:, c]
        try:
            auc_list.append(roc_auc_score(y_true_c, y_prob_c))
            aupr_list.append(average_precision_score(y_true_c, y_prob_c))
        except Exception:
            continue

    if len(auc_list) == 0:
        return np.nan, np.nan
    return float(np.mean(auc_list)), float(np.mean(aupr_list))


# -----------------------------
# IO helpers
# -----------------------------
def parse_csv_list(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def read_list_file(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x and not x.startswith("#"):
                items.append(x)
    return items


# -----------------------------
# Core: build labels
# -----------------------------
def build_singlelabel_table(hpa_pickle_path: str, ref_gene_set: set) -> pd.DataFrame:
    """
    Returns df_label with columns: Gene, Tissue (string)
    Enforces single-label mapping; filters to genes present in ref_gene_set.
    """
    tissue_gene_dict = pd.read_pickle(hpa_pickle_path)

    gene_to_tissue = {}
    for tissue, gene_list in tissue_gene_dict.items():
        for g in gene_list:
            if g in gene_to_tissue:
                raise ValueError(f"Gene appears in multiple tissues: {g} ({gene_to_tissue[g]} vs {tissue})")
            gene_to_tissue[g] = tissue

    df_label = pd.DataFrame([(g, t) for g, t in gene_to_tissue.items()], columns=["Gene", "Tissue"])
    df_label = df_label[df_label["Gene"].isin(ref_gene_set)].copy()
    return df_label


def add_onehot_baseline(aligned_embeddings: dict, ref_model: str) -> dict:
    """
    Add a one-hot baseline embedding using the gene index of ref_model.
    """
    if ref_model not in aligned_embeddings:
        raise ValueError(f"ref_model '{ref_model}' not found in aligned_embeddings")

    genes = aligned_embeddings[ref_model].index.tolist()
    N = len(genes)
    one_hot_matrix = np.eye(N, dtype=np.float32)
    one_hot_df = pd.DataFrame(one_hot_matrix, index=genes)
    aligned_embeddings["onehot"] = one_hot_df
    return aligned_embeddings


# -----------------------------
# Evaluation (5-fold CV)
# -----------------------------
def evaluate_one_model_5fold(
    model_name: str,
    df_emb: pd.DataFrame,
    df_label_encoded: pd.DataFrame,
    num_classes: int,
    device: torch.device,
    batch_size_train: int = 64,
    batch_size_val: int = 128,
    lr: float = 1e-3,
    max_epochs: int = 50,
    patience: int = 10,
    min_delta: float = 1e-4,
    seed: int = 42,
):
    """
    Returns list of dicts, each dict corresponds to a fold result.
    """
    print(f"\n========== Model: {model_name} ==========")

    # keep only labeled genes that exist in embedding
    genes_emb = set(df_emb.index)
    df_use = df_label_encoded[df_label_encoded["Gene"].isin(genes_emb)].copy()

    print(f"Usable genes: {len(df_use)}")
    if len(df_use) < 100:
        print(f"WARNING: very few genes for model {model_name}")

    # X, y aligned by Gene
    df_use = df_use.sort_values("Gene")
    X = df_emb.loc[df_use["Gene"]].values
    y = df_use["label"].values.astype(int)

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        print(f"\n----- Fold {fold_id}/5 -----")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(EmbDataset(X_train, y_train), batch_size=batch_size_train, shuffle=True)
        val_loader = DataLoader(EmbDataset(X_val, y_val), batch_size=batch_size_val, shuffle=False)

        model = MLP(input_dim=X.shape[1], hidden_dim=128, num_classes=num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        early = EarlyStopping(patience=patience, min_delta=min_delta)

        # training loop
        for epoch in range(1, max_epochs + 1):
            model.train()
            train_losses = []
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        preds_prob, preds_cls, y_true = [], [], []

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                prob = torch.softmax(logits, dim=1).cpu().numpy()

                val_losses.append(criterion(logits, by).item())
                preds_prob.append(prob)
                preds_cls.append(prob.argmax(axis=1))
                y_true.append(by.cpu().numpy())

        preds_prob = np.concatenate(preds_prob)
        preds_cls = np.concatenate(preds_cls)
        y_true = np.concatenate(y_true)

        macro_f1 = f1_score(y_true, preds_cls, average="macro")
        auc_macro, aupr_macro = safe_macro_auc(y_true, preds_prob, num_classes)


        fold_results.append({
            "Model": model_name,
            "Fold": fold_id,
            "MacroF1": macro_f1,
            "MacroAUROC": auc_macro,
            "MacroAUPRC": aupr_macro,
            "n_genes_used": int(len(df_use)),
            "embed_dim": int(X.shape[1]),
        })

    return fold_results


def main():
    parser = argparse.ArgumentParser(description="Task 5: Gene function prediction (HPA tissue-specific genes)")

    group_m = parser.add_mutually_exclusive_group(required=True)
    group_m.add_argument("--methods", type=str, help="Comma-separated model names to evaluate (required)")
    group_m.add_argument("--methods-file", type=str, help="Text file with one model name per line")

    parser.add_argument("--aligned-embeddings-pkl", type=str, required=True,
                        help="Pickle file containing aligned embeddings dict[str, pd.DataFrame]")
    parser.add_argument("--hpa-tissue-genes-pkl", type=str, required=True,
                        help="Pickle file: dict[tissue, list[gene]]")
    parser.add_argument("--ref-onehot-model", type=str, required=True,
                        help="Reference model name whose gene index defines the one-hot baseline space")

    parser.add_argument("--output-csv", type=str, required=True,
                        help="Output CSV (long-format: one row per model per fold)")
    parser.add_argument("--output-label-csv", type=str, required=True,
                        help="Write the aligned Gene/Tissue/label table to this CSV")

    parser.add_argument("--device", type=str, default="cuda:0", help="e.g., cuda:0 or cpu")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    methods = parse_csv_list(args.methods) if args.methods else read_list_file(args.methods_file)
    if len(methods) == 0:
        raise ValueError("Empty --methods")

    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("Using device:", device)

    # load embeddings
    with open(args.aligned_embeddings_pkl, "rb") as f:
        aligned_embeddings = pickle.load(f)

    if not isinstance(aligned_embeddings, dict):
        raise ValueError("aligned_embeddings.pkl must be a dict[str, pd.DataFrame]")

    # add onehot baseline
    aligned_embeddings = add_onehot_baseline(aligned_embeddings, ref_model=args.ref_onehot_model)

    # label table built using ref gene set (same as your original: genes from ref model)
    ref_genes = set(aligned_embeddings[args.ref_onehot_model].index.tolist())
    df_label = build_singlelabel_table(args.hpa_tissue_genes_pkl, ref_gene_set=ref_genes)

    # encode labels
    le = LabelEncoder()
    df_label["label"] = le.fit_transform(df_label["Tissue"])
    num_classes = len(le.classes_)
    print("Tissue classes:", list(le.classes_))
    print("num_classes =", num_classes)

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_label_csv), exist_ok=True)
    df_label.to_csv(args.output_label_csv, index=False)

    # run selected methods
    all_fold_rows = []

    for model_name in methods:
        if model_name not in aligned_embeddings:
            raise ValueError(f"Model '{model_name}' not found in aligned_embeddings. Available: {list(aligned_embeddings.keys())[:10]}...")

        df_emb = aligned_embeddings[model_name]
        fold_rows = evaluate_one_model_5fold(
            model_name=model_name,
            df_emb=df_emb,
            df_label_encoded=df_label,
            num_classes=num_classes,
            device=device,
            seed=args.seed,
        )
        all_fold_rows.extend(fold_rows)

    results_df = pd.DataFrame(all_fold_rows)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n✅ Saved results: {args.output_csv}")
    print(f"✅ Saved labels : {args.output_label_csv}")


if __name__ == "__main__":
    main()













