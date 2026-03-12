import os
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================================================
# 1. Define Dataset / MLP
# =========================================================
class EmbDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def safe_macro_auc(y_true, y_prob, num_classes):
    auc_list = []
    aupr_list = []

    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        y_prob_c = y_prob[:, c]

        if y_true_c.sum() == 0:
            continue

        try:
            auc = roc_auc_score(y_true_c, y_prob_c)
            aupr = average_precision_score(y_true_c, y_prob_c)
            auc_list.append(auc)
            aupr_list.append(aupr)
        except:
            continue

    if len(auc_list) == 0:
        return np.nan, np.nan

    return np.mean(auc_list), np.mean(aupr_list)

# =========================================================
# 二. Gene Function Classification Evaluator Interface Class
# =========================================================
class GeneFunctionEvaluator:
    """
    Gene function classification evaluation interface class, responsible for 
    unifying data loading, label alignment, and evaluation processes.
    """
    def __init__(self, emb_file_path, tissue_genes_path, device=None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"💻 Using device: {self.device}")

        # 1. Load Embedding data
        print(f"📘 Loading aligned embeddings from {emb_file_path}...")
        with open(emb_file_path, "rb") as f:
            self.aligned_embeddings = pickle.load(f)
        print("   Available models:", list(self.aligned_embeddings.keys()))

        # 2. Load Tissue-specific genes label data
        print(f"📘 Loading HPA tissue-specific genes from {tissue_genes_path}...")
        self.tissue_gene_dict = pd.read_pickle(tissue_genes_path)

        # 3. Initialize and align labels
        self._prepare_labels()

    def _prepare_labels(self):
        """Internal method: Build single-label gene -> tissue label mapping and encode"""
        gene_to_tissue = {}
        for tissue, gene_list in self.tissue_gene_dict.items():
            for g in gene_list:
                if g in gene_to_tissue:
                    raise ValueError(f"Error: gene appears in multiple tissues: {g}")
                gene_to_tissue[g] = tissue

        df_label = pd.DataFrame(
            [(g, t) for g, t in gene_to_tissue.items()],
            columns=["Gene", "Tissue"]
        )

        some_model = list(self.aligned_embeddings.keys())[0]
        common_emb_genes = set(self.aligned_embeddings[some_model].index.tolist())
        
        self.df_label = df_label[df_label["Gene"].isin(common_emb_genes)].copy()

        # LabelEncoder
        self.le = LabelEncoder()
        self.df_label["label"] = self.le.fit_transform(self.df_label["Tissue"])
        self.num_classes = len(self.le.classes_)

        print(f"✅ Labels prepared: {len(self.df_label)} labeled genes across {self.num_classes} tissues.")

    def run_5_fold(self, model_name):
        """Execute 5-fold cross-validation for a specified model"""
        if model_name not in self.aligned_embeddings:
            raise ValueError(f"❌ Model '{model_name}' not found in the loaded embeddings!")

        df_emb = self.aligned_embeddings[model_name]
        print(f"\n========== 🚀 Running model: {model_name} (5-Fold CV) ==========")

        # Retain only genes present in the embeddings
        genes_emb = set(df_emb.index)
        df_use = self.df_label[self.df_label["Gene"].isin(genes_emb)].copy()

        print(f"Usable genes for this model: {len(df_use)}")
        if len(df_use) < 100:
            print(f"⚠️ WARNING: very few genes for model {model_name}")

        df_use = df_use.sort_values("Gene")
        X = df_emb.loc[df_use["Gene"]].values
        y = df_use["label"].values

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
            print(f"\n----- Fold {fold}/5 -----")
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            train_loader = DataLoader(EmbDataset(X_train, y_train), batch_size=64, shuffle=True)
            test_loader = DataLoader(EmbDataset(X_test, y_test), batch_size=128, shuffle=False)

            model = MLP(input_dim=X.shape[1], hidden_dim=128, num_classes=self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            # Train
            for epoch in range(1, 51):
                model.train()
                train_losses = []
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                if epoch == 1 or epoch % 10 == 0 or epoch == 50:
                    print(f"Epoch {epoch:03d} | TrainLoss={np.mean(train_losses):.4f}")

            # Evaluate
            model.eval()
            preds_cls, preds_prob, all_trues = [], [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(self.device)
                    logits = model(bx)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    preds_prob.append(prob)
                    preds_cls.append(prob.argmax(axis=1))
                    all_trues.append(by.numpy())

            preds_prob = np.concatenate(preds_prob)
            preds_cls = np.concatenate(preds_cls)
            all_trues = np.concatenate(all_trues)

            acc = accuracy_score(all_trues, preds_cls)
            macro_f1 = f1_score(all_trues, preds_cls, average="macro")
            auc_macro, aupr_macro = safe_macro_auc(all_trues, preds_prob, self.num_classes)

            print(f"✅ Fold {fold} Final | Acc={acc:.4f} | MacroF1={macro_f1:.4f} | AUC={auc_macro:.4f} | AUPR={aupr_macro:.4f}")
            
            results.append({
                "Model": model_name, "Fold": fold, "Accuracy": acc, 
                "MacroF1": macro_f1, "AUC": auc_macro, "AUPR": aupr_macro
            })

        return pd.DataFrame(results)