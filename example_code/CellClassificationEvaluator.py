import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score
)

# ==========================================
# 1. Define the classification network
# ==========================================
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, dropout=0.3, h_dim=256, out_dim=10):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, h_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(h_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ==========================================
# 2. Evaluator interface class
# ==========================================
class CellClassificationEvaluator:
    """
        Cell type classification evaluation interface
    """
    def __init__(self, base_data_path, label_data_path, device=None, seed=42, 
                 batch_size_train=64, batch_size_val=128, num_epochs=50, 
                 lr=1e-4, patience=5):
        """
            Initialize the evaluator and manage paths and hyperparameters
        """
        self.base_data_path = base_data_path
        self.label_data_path = label_data_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        
        # 训练超参数
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.num_epochs = num_epochs
        self.lr = lr
        self.patience = patience

    def _set_seed(self):
        """Fix the global random seed"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def _load_and_check_data(self, embedding_train, embedding_test, label_train, label_test):
        """Internal method for data loading and validation"""
        emb_train = pd.read_csv(embedding_train, index_col=0)
        emb_test  = pd.read_csv(embedding_test,  index_col=0)
        lbl_train = pd.read_csv(label_train)
        lbl_test  = pd.read_csv(label_test)

        assert list(emb_train.index) == list(lbl_train['cell_id']), "Train IDs mismatch!"
        assert list(emb_test.index) == list(lbl_test['cell_id']), "Test IDs mismatch!"

        X_train, y_train = emb_train.values, lbl_train['celltype'].values
        X_test, y_test   = emb_test.values,  lbl_test['celltype'].values
        return X_train, y_train, X_test, y_test

    def _train_model(self, model, train_loader, val_loader):
        """Internal training procedure (retains the original early stopping behavior)"""
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False)

        best_val_acc = 0.0
        counter = 0

        for epoch in range(self.num_epochs):
            model.train()
            # === Training phase ===
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}", leave=False):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)               
                loss = criterion(outputs, y_batch)     
                loss.backward()
                optimizer.step()

            # === Validation phase ===
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = model(X_batch)                  
                    probs = torch.softmax(outputs, dim=1)     
                    preds = probs.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()
                    val_total += y_batch.size(0)

            val_acc = val_correct / val_total
            scheduler.step(val_acc)  

            # === Early Stopping ===
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    break
        return model

    def _evaluate_model(self, model, X_test, y_test, label_encoder):
        """Internal evaluation procedure"""
        model.eval()
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

        y_true = label_encoder.transform(y_test)
        f1 = f1_score(y_true, preds, average='macro')

        y_bin = label_binarize(y_true, classes=range(len(label_encoder.classes_)))
        
        try:
            auroc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
        except ValueError:
            auroc = np.nan
            
        try:
            auprc = average_precision_score(y_bin, probs, average='macro')
        except ValueError:
            auprc = np.nan

        return {"Macro-F1": f1, "Macro-AUROC": auroc, "Macro-AUPRC": auprc}

    def run_one_fold(self, tissue_name, fold, method_name=""):
        """Run training and evaluation for a single fold"""
        self._set_seed()

        # Construct file paths
        emb_train = os.path.join(self.base_data_path, f"{tissue_name}_fold{fold}_train_cellEmbedding.csv")
        emb_test  = os.path.join(self.base_data_path, f"{tissue_name}_fold{fold}_test_cellEmbedding.csv")
        lbl_train = os.path.join(self.label_data_path, f"{tissue_name}_fold{fold}_train_ids.csv")
        lbl_test  = os.path.join(self.label_data_path, f"{tissue_name}_fold{fold}_test_ids.csv")

        X_train, y_train, X_test, y_test = self._load_and_check_data(emb_train, emb_test, lbl_train, lbl_test)

        # Label encoding and dataset split
        le = LabelEncoder()
        le.fit(np.concatenate([y_train, y_test]))
        y_train_enc = le.transform(y_train)
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train_enc, test_size=0.1, random_state=self.seed, stratify=y_train_enc
        )

        # DataLoader
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)),
            batch_size=self.batch_size_train, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=self.batch_size_val, shuffle=False
        )

        # Initialize, train, and evaluate the model
        model = SimpleClassifier(input_dim=X_train.shape[1], out_dim=len(le.classes_)).to(self.device)
        model = self._train_model(model, train_loader, val_loader)
        metrics = self._evaluate_model(model, X_test, y_test, le)

        # Add metadata
        metrics['Tissue'] = tissue_name
        metrics['Fold'] = fold
        metrics['Method_name'] = method_name
        
        return metrics

    def run_5_fold(self, tissue_name, method_name=""):
        """Public interface for five-fold cross-validation"""
        fold_metrics = []
        for fold in range(1, 6):  
            metrics = self.run_one_fold(tissue_name, fold, method_name)
            fold_metrics.append(metrics)
        return fold_metrics

