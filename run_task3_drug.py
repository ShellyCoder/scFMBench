#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 4: Drug response prediction (binary classification) using zero-shot cell embeddings + drug fingerprints.

- Cell input: zero-shot cell embeddings (precomputed)
- Drug input: Morgan fingerprint from SMILES (RDKit)
- Model: Dual-tower (cell encoder + drug encoder) + fusion classifier
- Two modes:
  1) cv: 5-fold cross-validation on baseline datasets (fold-specific train/test)
  2) augment: single train/test split evaluation for augmented datasets

Outputs:
- One CSV per method under <output-dir>/results_<METHOD>.csv
"""

import os
import argparse
import random
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score


# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
# Labels / drug info
# -----------------------
def encode_condition(arr) -> np.ndarray:
    """
    Map labels into {0,1}: resistant/non-response -> 0; sensitive/response -> 1.
    Robust to case and minor spelling variants.
    """
    arr = np.array(arr, dtype=str)
    arr_clean = np.char.strip(np.char.lower(arr))

    mapping = {
        "resistant": 0,
        "non-response": 0,
        "nonresponse": 0,
        "nonsensitive": 0,
        "insensitive": 0,
        "sensitive": 1,
        "response": 1,
        "responsive": 1,
    }

    encoded = np.array([mapping.get(x, np.nan) for x in arr_clean], dtype=float)
    if np.isnan(encoded).any():
        bad_idx = np.where(np.isnan(encoded))[0]
        bad_labels = np.unique(arr_clean[bad_idx]).tolist()
        raise ValueError(f"Unrecognized label(s) in condition: {bad_labels}")
    return encoded.astype(int)


def smiles_to_fingerprint(smiles: str, fp_size: int = 1024) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
    arr = np.zeros((fp_size,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def build_drug2smiles(drug_info_path: str) -> Dict[str, str]:
    df = pd.read_csv(drug_info_path)
    # Keep first occurrence per drug_names
    m = (
        df[["drug_names", "PubChemSMILES"]]
        .dropna()
        .drop_duplicates("drug_names")
        .set_index("drug_names")["PubChemSMILES"]
        .to_dict()
    )
    return m


def get_single_dataset_smiles(dataset_name: str, drug_info_path: str) -> Tuple[str, str]:
    """
    For standard DRMref datasets: each dataset corresponds to a single drug.
    Match row by dataset == "<dataset_name>.rds".
    """
    df = pd.read_csv(drug_info_path)
    rec = df.loc[df["dataset"] == f"{dataset_name}.rds"]
    if rec.empty:
        raise ValueError(f"No drug record found for dataset: {dataset_name}")
    drug_name = rec.iloc[0]["drug_names"]
    smiles = rec.iloc[0]["PubChemSMILES"]
    if not isinstance(smiles, str) or len(smiles.strip()) == 0:
        raise ValueError(f"Missing SMILES for dataset={dataset_name}, drug={drug_name}")
    return str(drug_name), smiles.strip()


# -----------------------
# Data loading with strict ID checks
# -----------------------
def _check_id_match(train_ids: pd.DataFrame, test_ids: pd.DataFrame,
                    train_emb: pd.DataFrame, test_emb: pd.DataFrame,
                    dataset_name: str, fold: Optional[int] = None) -> None:
    # required columns
    for df, name in [(train_ids, "train_ids"), (test_ids, "test_ids")]:
        if "cell_id" not in df.columns:
            raise ValueError(f"{name} must contain column 'cell_id'")

    # set match
    if set(train_ids["cell_id"]) != set(train_emb.index):
        missing = set(train_ids["cell_id"]) - set(train_emb.index)
        extra = set(train_emb.index) - set(train_ids["cell_id"])
        raise ValueError(
            f"[Train ID mismatch] {dataset_name}"
            + (f" Fold{fold}" if fold is not None else "")
            + f" | Missing in embedding: {len(missing)} | Extra in embedding: {len(extra)}"
        )

    if set(test_ids["cell_id"]) != set(test_emb.index):
        missing = set(test_ids["cell_id"]) - set(test_emb.index)
        extra = set(test_emb.index) - set(test_ids["cell_id"])
        raise ValueError(
            f"[Test ID mismatch] {dataset_name}"
            + (f" Fold{fold}" if fold is not None else "")
            + f" | Missing in embedding: {len(missing)} | Extra in embedding: {len(extra)}"
        )

    # order match
    if list(train_ids["cell_id"]) != list(train_emb.index):
        raise ValueError(f"[Train order mismatch] {dataset_name}" + (f" Fold{fold}" if fold is not None else ""))

    if list(test_ids["cell_id"]) != list(test_emb.index):
        raise ValueError(f"[Test order mismatch] {dataset_name}" + (f" Fold{fold}" if fold is not None else ""))


def load_dataset_cv(
    dataset_name: str,
    fold: int,
    embed_dir: str,
    base_dir: str,
    drug_info_path: str,
) -> Dict[str, object]:
    """
    CV mode: read fold-specific train/test embeddings + fold ID files.
    """
    id_dir = os.path.join(base_dir, "fiveFold_ID")
    train_id_file = os.path.join(id_dir, f"{dataset_name}_fold{fold}_train_ids.csv")
    test_id_file = os.path.join(id_dir, f"{dataset_name}_fold{fold}_test_ids.csv")

    train_emb_file = os.path.join(embed_dir, f"{dataset_name}_fold{fold}_train_cellEmbedding.csv")
    test_emb_file = os.path.join(embed_dir, f"{dataset_name}_fold{fold}_test_cellEmbedding.csv")

    for p in [train_id_file, test_id_file, train_emb_file, test_emb_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_ids = pd.read_csv(train_id_file)
    test_ids = pd.read_csv(test_id_file)
    train_emb = pd.read_csv(train_emb_file, index_col=0)
    test_emb = pd.read_csv(test_emb_file, index_col=0)

    _check_id_match(train_ids, test_ids, train_emb, test_emb, dataset_name, fold=fold)

    drug_name, smiles = get_single_dataset_smiles(dataset_name, drug_info_path)

    return {
        "dataset": dataset_name,
        "fold": fold,
        "train_embedding": train_emb,
        "test_embedding": test_emb,
        "train_labels": train_ids,
        "test_labels": test_ids,
        "drug_name": drug_name,
        "smiles": smiles,
        "paths": {
            "train_ids": train_id_file,
            "test_ids": test_id_file,
            "train_emb": train_emb_file,
            "test_emb": test_emb_file,
        },
    }


def load_dataset_augment(
    dataset_name: str,
    embed_dir: str,
    base_dir: str,
    drug_info_path: str,
) -> Dict[str, object]:
    """
    Augment mode: one train/test split per dataset under base_dir/Augment_data,
    embeddings under embed_dir (typically .../Augment_res/).
    Supports two cases:
      1) single drug per dataset: use drug_info_path lookup by prefix
      2) DRMref_unseen-like datasets: per-cell drug column in meta, mapped to SMILES
    """
    id_dir = os.path.join(base_dir, "Augment_data")
    train_id_file = os.path.join(id_dir, f"{dataset_name}_train_meta.csv")
    test_id_file = os.path.join(id_dir, f"{dataset_name}_test_meta.csv")

    train_emb_file = os.path.join(embed_dir, f"{dataset_name}_train_cellEmbedding.csv")
    test_emb_file = os.path.join(embed_dir, f"{dataset_name}_test_cellEmbedding.csv")

    for p in [train_id_file, test_id_file, train_emb_file, test_emb_file]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing file: {p}")

    train_ids = pd.read_csv(train_id_file)
    test_ids = pd.read_csv(test_id_file)
    train_emb = pd.read_csv(train_emb_file, index_col=0)
    test_emb = pd.read_csv(test_emb_file, index_col=0)

    _check_id_match(train_ids, test_ids, train_emb, test_emb, dataset_name, fold=None)

    # Case A: multi drug per dataset
    if "drug" in train_ids.columns and "drug" in test_ids.columns:
        drug2smiles = build_drug2smiles(drug_info_path)
        train_smiles = train_ids["drug"].map(drug2smiles)
        test_smiles = test_ids["drug"].map(drug2smiles)
        if train_smiles.isnull().any() or test_smiles.isnull().any():
            missing_drugs = pd.concat([train_ids.loc[train_smiles.isnull(), "drug"],
                                       test_ids.loc[test_smiles.isnull(), "drug"]]).unique().tolist()
            raise ValueError(f"Missing SMILES for drugs: {missing_drugs}")

        return {
            "dataset": dataset_name,
            "train_embedding": train_emb,
            "test_embedding": test_emb,
            "train_labels": train_ids,
            "test_labels": test_ids,
            "train_smiles": train_smiles.astype(str).tolist(),
            "test_smiles": test_smiles.astype(str).tolist(),
            "paths": {
                "train_ids": train_id_file,
                "test_ids": test_id_file,
                "train_emb": train_emb_file,
                "test_emb": test_emb_file,
            },
        }

    # Case B: single drug per dataset
    prefix = dataset_name.split("_fold")[0] if "_fold" in dataset_name else dataset_name
    drug_name, smiles = get_single_dataset_smiles(prefix, drug_info_path)

    return {
        "dataset": dataset_name,
        "train_embedding": train_emb,
        "test_embedding": test_emb,
        "train_labels": train_ids,
        "test_labels": test_ids,
        "drug_name": drug_name,
        "smiles": smiles,
        "paths": {
            "train_ids": train_id_file,
            "test_ids": test_id_file,
            "train_emb": train_emb_file,
            "test_emb": test_emb_file,
        },
    }


def prepare_tensors(dataset_result: Dict[str, object], fp_size: int, device: str) -> Dict[str, torch.Tensor]:
    # labels
    if "condition" not in dataset_result["train_labels"].columns or "condition" not in dataset_result["test_labels"].columns:
        raise ValueError("train/test label files must contain column 'condition'")

    y_train = encode_condition(dataset_result["train_labels"]["condition"].values)
    y_test = encode_condition(dataset_result["test_labels"]["condition"].values)

    X_train = dataset_result["train_embedding"].values.astype(np.float32)
    X_test = dataset_result["test_embedding"].values.astype(np.float32)

    # drug fingerprints aligned to cells
    if "train_smiles" in dataset_result and "test_smiles" in dataset_result:
        drug_train = np.vstack([smiles_to_fingerprint(s, fp_size) for s in dataset_result["train_smiles"]]).astype(np.float32)
        drug_test = np.vstack([smiles_to_fingerprint(s, fp_size) for s in dataset_result["test_smiles"]]).astype(np.float32)
    else:
        drug_vec = smiles_to_fingerprint(dataset_result["smiles"], fp_size=fp_size).astype(np.float32)
        drug_train = np.repeat(drug_vec[None, :], X_train.shape[0], axis=0)
        drug_test = np.repeat(drug_vec[None, :], X_test.shape[0], axis=0)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32, device=device),
        "y_train": torch.tensor(y_train, dtype=torch.float32, device=device),
        "drug_train": torch.tensor(drug_train, dtype=torch.float32, device=device),
        "X_test": torch.tensor(X_test, dtype=torch.float32, device=device),
        "y_test": torch.tensor(y_test, dtype=torch.float32, device=device),
        "drug_test": torch.tensor(drug_test, dtype=torch.float32, device=device),
    }


# -----------------------
# Model
# -----------------------
class DualTowerDrugResponse(nn.Module):
    def __init__(self, cell_dim: int, drug_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_cell: torch.Tensor, x_drug: torch.Tensor) -> torch.Tensor:
        z_cell = self.cell_encoder(x_cell)
        z_drug = self.drug_encoder(x_drug)
        z = torch.cat([z_cell, z_drug], dim=1)
        logits = self.classifier(z).squeeze(-1)
        return torch.sigmoid(logits)


# -----------------------
# Train / Eval
# -----------------------
def train_with_early_stopping(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    drug_train: torch.Tensor,
    drug_val: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
) -> nn.Module:
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_auc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train, drug_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(X_val, drug_val).detach().cpu().numpy()
            y_val_true = y_val.detach().cpu().numpy()
            # AUROC requires both classes present
            try:
                auc = roc_auc_score(y_val_true, y_val_pred)
            except ValueError:
                auc = np.nan

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch:03d} | Loss={loss.item():.4f} | ValAUC={auc:.4f} | Best={best_auc:.4f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best AUC={best_auc:.4f})")
            break

    if best_state is None:
        # fallback: keep last state
        print("Warning: best_state is None (val AUROC may be undefined). Keeping last epoch weights.")
        return model

    model.load_state_dict(best_state)
    return model


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_bin = (y_prob > 0.5).astype(int)
    mf1 = f1_score(y_true, y_bin, average="macro")
    auc = roc_auc_score(y_true, y_prob)
    aupr = average_precision_score(y_true, y_prob)
    return {"Macro-F1": mf1, "AUROC": auc, "AUPRC": aupr}


# -----------------------
# Runners
# -----------------------
def run_one_fold_cv(
    dataset: str,
    fold: int,
    method: str,
    embed_root: str,
    base_dir: str,
    drug_info_path: str,
    fp_size: int,
    device: str,
    seed: int,
    epochs: int,
    lr: float,
    patience: int,
) -> Dict[str, object]:
    set_seed(seed)

    embed_dir = os.path.join(embed_root, f"{method}_outputData")
    ds = load_dataset_cv(dataset, fold, embed_dir=embed_dir, base_dir=base_dir, drug_info_path=drug_info_path)
    tensors = prepare_tensors(ds, fp_size=fp_size, device=device)

    model = DualTowerDrugResponse(cell_dim=tensors["X_train"].shape[1], drug_dim=fp_size).to(device)

    # NOTE: original code uses test set as "validation" for early stopping.
    # To stay consistent with your current implementation, we keep the same behavior.
    model = train_with_early_stopping(
        model,
        tensors["X_train"], tensors["y_train"],
        tensors["X_test"], tensors["y_test"],
        tensors["drug_train"], tensors["drug_test"],
        epochs=epochs, lr=lr, patience=patience
    )

    model.eval()
    with torch.no_grad():
        y_prob = model(tensors["X_test"], tensors["drug_test"]).detach().cpu().numpy()
    y_true = tensors["y_test"].detach().cpu().numpy().astype(int)

    metrics = evaluate_binary(y_true, y_prob)
    metrics.update({"Dataset": dataset, "Fold": fold, "Method": method})
    return metrics


def run_one_dataset_augment(
    dataset: str,
    method: str,
    embed_root: str,
    base_dir: str,
    drug_info_path: str,
    fp_size: int,
    device: str,
    seed: int,
    epochs: int,
    lr: float,
    patience: int,
) -> Dict[str, object]:
    set_seed(seed)

    embed_dir = os.path.join(embed_root, f"{method}_outputData", "Augment_res")
    ds = load_dataset_augment(dataset, embed_dir=embed_dir, base_dir=base_dir, drug_info_path=drug_info_path)
    tensors = prepare_tensors(ds, fp_size=fp_size, device=device)

    model = DualTowerDrugResponse(cell_dim=tensors["X_train"].shape[1], drug_dim=fp_size).to(device)

    model = train_with_early_stopping(
        model,
        tensors["X_train"], tensors["y_train"],
        tensors["X_test"], tensors["y_test"],
        tensors["drug_train"], tensors["drug_test"],
        epochs=epochs, lr=lr, patience=patience
    )

    model.eval()
    with torch.no_grad():
        y_prob = model(tensors["X_test"], tensors["drug_test"]).detach().cpu().numpy()
    y_true = tensors["y_test"].detach().cpu().numpy().astype(int)

    metrics = evaluate_binary(y_true, y_prob)
    metrics.update({"Dataset": dataset, "Fold": "NA", "Method": method})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Task 4: Drug response prediction (dual-tower, zero-shot embeddings)")

    group_m = parser.add_mutually_exclusive_group(required=True)
    group_m.add_argument("--methods", type=str, help="Comma-separated method names (required)")
    group_m.add_argument("--methods-file", type=str, help="Text file with one method per line")

    group_d = parser.add_mutually_exclusive_group(required=True)
    group_d.add_argument("--datasets", type=str, help="Comma-separated dataset names (required)")
    group_d.add_argument("--datasets-file", type=str, help="Text file with one dataset per line")

    parser.add_argument("--mode", type=str, choices=["cv", "augment"], required=True,
                        help="cv: 5-fold CV baseline datasets; augment: one train/test for augmented datasets")
    parser.add_argument("--embed-root", type=str, required=True,
                        help="Root directory containing <METHOD>_outputData/")
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base directory containing DRMref/{fiveFold_ID or Augment_data}")
    parser.add_argument("--drug-info", type=str, required=True,
                        help="CSV containing drug_names and PubChemSMILES for each dataset")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--fp-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    methods = parse_csv_list_arg(args.methods) if args.methods else read_list_file(args.methods_file)
    datasets = parse_csv_list_arg(args.datasets) if args.datasets else read_list_file(args.datasets_file)
    if len(methods) == 0 or len(datasets) == 0:
        raise ValueError("Empty methods or datasets.")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    for method in methods:
        rows = []
        print(f"\n=== Running method: {method} | mode={args.mode} ===")

        if args.mode == "cv":
            for dataset in datasets:
                for fold in range(1, 6):
                    print(f"[{method}] {dataset} fold={fold}")
                    row = run_one_fold_cv(
                        dataset=dataset, fold=fold, method=method,
                        embed_root=args.embed_root, base_dir=args.base_dir,
                        drug_info_path=args.drug_info, fp_size=args.fp_size,
                        device=device, seed=args.seed,
                        epochs=args.epochs, lr=args.lr, patience=args.patience
                    )
                    rows.append(row)
        else:
            for dataset in datasets:
                print(f"[{method}] {dataset} (augment)")
                row = run_one_dataset_augment(
                    dataset=dataset, method=method,
                    embed_root=args.embed_root, base_dir=args.base_dir,
                    drug_info_path=args.drug_info, fp_size=args.fp_size,
                    device=device, seed=args.seed,
                    epochs=args.epochs, lr=args.lr, patience=args.patience
                )
                rows.append(row)

        df = pd.DataFrame(rows)
        for c in ["Macro-F1", "AUROC", "AUPRC"]:
            if c in df.columns:
                df[c] = df[c].astype(float).round(6)

        out_path = os.path.join(args.output_dir, f"results_{method}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

