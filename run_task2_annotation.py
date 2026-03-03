#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Task 2: Cell annotation / classification (5-fold CV) using zero-shot cell embeddings.

- Input: precomputed embeddings for each dataset-fold split (train/test) + corresponding ID/label CSVs
- Downstream head: fixed MLP classifier (same hyperparameters across methods)
- Evaluation: Macro-F1, Macro-AUROC, Macro-AUPRC
- Output: one CSV per method, with one row per (dataset, fold)

"""

import os
import argparse
import random
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_csv_list_arg(s: str) -> List[str]:
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items


def read_list_file(path: str) -> List[str]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x and not x.startswith("#"):
                items.append(x)
    return items


# -----------------------
# I/O and checks
# -----------------------
def load_and_check_data(
    embedding_path_train: str,
    embedding_path_test: str,
    label_path_train: str,
    label_path_test: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    emb_train = pd.read_csv(embedding_path_train, index_col=0)
    emb_test = pd.read_csv(embedding_path_test, index_col=0)
    label_train = pd.read_csv(label_path_train)
    label_test = pd.read_csv(label_path_test)

    # required columns
    for df, name in [(label_train, "label_train"), (label_test, "label_test")]:
        if "cell_id" not in df.columns or "celltype" not in df.columns:
            raise ValueError(f"{name} must contain columns: cell_id, celltype")

    # ID order consistency
    if list(emb_train.index) != list(label_train["cell_id"]):
        raise ValueError(f"Train IDs mismatch between {embedding_path_train} and {label_path_train}")
    if list(emb_test.index) != list(label_test["cell_id"]):
        raise ValueError(f"Test IDs mismatch between {embedding_path_test} and {label_path_test}")

    # numeric integrity checks
    for df, name in [(emb_train, "train embedding"), (emb_test, "test embedding")]:
        arr = df.values
        if np.any(np.isnan(arr)):
            raise ValueError(f"Found NaN in {name}: {embedding_path_train if name.startswith('train') else embedding_path_test}")
        if np.any(np.isinf(arr)):
            raise ValueError(f"Found Inf in {name}: {embedding_path_train if name.startswith('train') else embedding_path_test}")

    X_train, y_train = emb_train.values.astype(np.float32), label_train["celltype"].values
    X_test, y_test = emb_test.values.astype(np.float32), label_test["celltype"].values
    return X_train, y_train, X_test, y_test


def split_train_val(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)


# -----------------------
# Model
# -----------------------
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, dropout: float = 0.3, h_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(512, h_dim)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(h_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act1(self.fc1(x)))
        x = self.drop2(self.act2(self.fc2(x)))
        x = self.fc3(x)
        return x


# -----------------------
# Train / Eval
# -----------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    patience: int = 5,
    device: str = "cuda",
) -> nn.Module:
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    best_val_acc = -1.0
    best_state = None
    counter = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=False):
            Xb = Xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # validation (accuracy)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                logits = model(Xb)
                probs = torch.softmax(logits, dim=1)
                preds = probs.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / max(total, 1)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch:03d} | TrainLoss={avg_train_loss:.4f} | "
            f"ValAcc={val_acc:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    if best_state is None:
        raise RuntimeError("Training failed: best_state is None.")
    model.load_state_dict(best_state)
    print(f"Training completed. Best Val Acc = {best_val_acc:.4f}")
    return model


def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray, le: LabelEncoder, device: str) -> Dict[str, float]:
    model.to(device)
    model.eval()

    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

    y_true = le.transform(y_test)

    mf1 = f1_score(y_true, preds, average="macro")

    y_bin = label_binarize(y_true, classes=range(len(le.classes_)))
    auroc = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    auprc = average_precision_score(y_bin, probs, average="macro")

    return {"Macro-F1": mf1, "Macro-AUROC": auroc, "Macro-AUPRC": auprc}


# -----------------------
# Main loop
# -----------------------
def run_one_dataset_fold(
    dataset: str,
    fold: int,
    method: str,
    embedding_root: str,
    label_root: str,
    device: str,
    seed: int,
    num_epochs: int,
    lr: float,
    patience: int,
    batch_size: int,
) -> Dict[str, object]:
    set_seed(seed)

    # Paths
    emb_dir = os.path.join(embedding_root, f"{method}_outputData")
    emb_train = os.path.join(emb_dir, f"{dataset}_fold{fold}_train_cellEmbedding.csv")
    emb_test = os.path.join(emb_dir, f"{dataset}_fold{fold}_test_cellEmbedding.csv")

    lab_train = os.path.join(label_root, f"{dataset}_fold{fold}_train_ids.csv")
    lab_test = os.path.join(label_root, f"{dataset}_fold{fold}_test_ids.csv")

    for p in [emb_train, emb_test, lab_train, lab_test]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    X_train, y_train, X_test, y_test = load_and_check_data(emb_train, emb_test, lab_train, lab_test)

    # label encode (fit on train+test labels to keep consistent class mapping)
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)

    # internal val split on training set only
    X_tr, X_val, y_tr, y_val = split_train_val(X_train, y_train_enc, seed=seed)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val, dtype=torch.long)),
        batch_size=max(batch_size * 2, 128),
        shuffle=False,
    )

    model = SimpleClassifier(input_dim=X_train.shape[1], out_dim=len(le.classes_))
    model = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, patience=patience, device=device)

    metrics = evaluate_model(model, X_test, y_test, le, device=device)
    metrics.update({"Dataset": dataset, "Fold": fold, "Method": method})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Task 2: Cell annotation / classification (5-fold CV)")
    # Require explicit methods/datasets every run
    group_m = parser.add_mutually_exclusive_group(required=True)
    group_m.add_argument("--methods", type=str, help="Comma-separated method names (e.g., scGNN,scGPT)")
    group_m.add_argument("--methods-file", type=str, help="Text file with one method per line")

    group_d = parser.add_mutually_exclusive_group(required=True)
    group_d.add_argument("--datasets", type=str, help="Comma-separated dataset names (e.g., Bladder,Blood)")
    group_d.add_argument("--datasets-file", type=str, help="Text file with one dataset per line")

    parser.add_argument("--embedding-root", type=str, required=True,
                        help="Root directory containing <METHOD>_outputData/ with fold embeddings")
    parser.add_argument("--label-root", type=str, required=True,
                        help="Directory containing <DATASET>_fold<FOLD>_{train,test}_ids.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for per-method summary CSVs")

    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    if args.methods:
        methods = parse_csv_list_arg(args.methods)
    else:
        methods = read_list_file(args.methods_file)

    if args.datasets:
        datasets = parse_csv_list_arg(args.datasets)
    else:
        datasets = read_list_file(args.datasets_file)

    if len(methods) == 0 or len(datasets) == 0:
        raise ValueError("Empty methods or datasets. Please provide non-empty --methods/--datasets.")

    os.makedirs(args.output_dir, exist_ok=True)

    # device availability check
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"
    else:
        device = args.device

    for method in methods:
        all_rows: List[Dict[str, object]] = []
        print(f"\n=== Running method: {method} ===")

        for dataset in datasets:
            for fold in range(1, 6):
                print(f"\n[Method={method}] Dataset={dataset} Fold={fold}")
                row = run_one_dataset_fold(
                    dataset=dataset,
                    fold=fold,
                    method=method,
                    embedding_root=args.embedding_root,
                    label_root=args.label_root,
                    device=device,
                    seed=args.seed,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    patience=args.patience,
                    batch_size=args.batch_size,
                )
                all_rows.append(row)

        df = pd.DataFrame(all_rows)
        metric_cols = ["Accuracy", "Macro-F1", "Macro-AUROC", "Macro-AUPRC"]
        for c in metric_cols:
            if c in df.columns:
                df[c] = df[c].astype(float).round(6)

        out_path = os.path.join(args.output_dir, f"results_{method}_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()




