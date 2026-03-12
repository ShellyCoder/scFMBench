import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Basic preprocessing utilities
# ==========================================
def smiles_to_fingerprint(smiles, fp_size=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=fp_size)
    arr = np.zeros((fp_size,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def encode_condition(arr):
    arr = np.array(arr, dtype=str) 
    arr_clean = np.char.strip(np.char.lower(arr)) 
    mapping = {
        "resistant": 0, "non-response": 0, "nonresponse": 0,
        "nonsensitive": 0, "insensitive": 0, "sensitive": 1,
        "response": 1, "responsive": 1
    }
    encoded = np.array([mapping.get(x, np.nan) for x in arr_clean], dtype=float)
    if np.isnan(encoded).any():
        bad_indices = np.where(np.isnan(encoded))[0]
        bad_labels = np.unique(arr_clean[bad_indices])
        raise ValueError(f"⚠️ Unrecognized label(s): {bad_labels.tolist()}")
    return encoded.astype(int)

# ==========================================
# 2. Dual-tower model definition
# ==========================================
class DualTowerDrugResponse(nn.Module):
    def __init__(self, cell_dim=512, drug_dim=1024, hidden_dim=256):
        super().__init__()
        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_cell, x_drug):
        z_cell = self.cell_encoder(x_cell)
        z_drug = self.drug_encoder(x_drug)
        z_fuse = torch.cat([z_cell, z_drug], dim=1)
        logits = self.classifier(z_fuse).squeeze(-1)
        return logits

# ==========================================
# 3. Drug response evaluation interface class
# ==========================================
class DrugResponseEvaluator:
    """
    Drug response prediction evaluation interface.
    This class manages the workflow for both 5-fold cross-validation
    and single-dataset validation.
    """
    def __init__(self, base_dir, embed_dir, drug_info_path, 
                 device=None, fp_size=1024, epochs=100, lr=1e-3, 
                 patience=15, batch_size=64, val_ratio=0.1):
        self.base_dir = base_dir
        self.embed_dir = embed_dir
        self.drug_info_path = drug_info_path
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.fp_size = fp_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    # ---------------- Internal data loading logic ----------------
    def _load_fold_dataset(self, dataset_name, fold):
        print(f"📘 Loading {dataset_name} (Fold {fold})")
        id_dir = self.base_dir

        train_id_file = os.path.join(id_dir, f"{dataset_name}_fold{fold}_train_ids.csv")
        test_id_file  = os.path.join(id_dir, f"{dataset_name}_fold{fold}_test_ids.csv")
        train_emb_file = os.path.join(self.embed_dir, f"{dataset_name}_fold{fold}_train_cellEmbedding.csv")
        test_emb_file  = os.path.join(self.embed_dir, f"{dataset_name}_fold{fold}_test_cellEmbedding.csv")

        for f in [train_id_file, test_id_file, train_emb_file, test_emb_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"❌ Missing file: {f}")

        train_ids = pd.read_csv(train_id_file)
        test_ids  = pd.read_csv(test_id_file)
        if "cell_id" not in train_ids.columns or "cell_id" not in test_ids.columns:
            raise ValueError("❌ train/test csv must contain 'cell_id' column")

        train_emb = pd.read_csv(train_emb_file, index_col=0)
        test_emb  = pd.read_csv(test_emb_file, index_col=0)

        train_id_set, train_emb_set = set(train_ids["cell_id"]), set(train_emb.index)
        test_id_set, test_emb_set   = set(test_ids["cell_id"]), set(test_emb.index)

        if train_id_set != train_emb_set:
            missing, extra = train_id_set - train_emb_set, train_emb_set - train_id_set
            raise ValueError(f"❌ [Train ID mismatch] {dataset_name} Fold{fold}\n"
                             f"Missing in embedding: {len(missing)} | Extra in embedding: {len(extra)}")

        if test_id_set != test_emb_set:
            missing, extra = test_id_set - test_emb_set, test_emb_set - test_id_set
            raise ValueError(f"❌ [Test ID mismatch] {dataset_name} Fold{fold}\n"
                             f"Missing in embedding: {len(missing)} | Extra in embedding: {len(extra)}")

        train_id_order, test_id_order = list(train_ids["cell_id"]), list(test_ids["cell_id"])
        train_emb_order, test_emb_order = list(train_emb.index), list(test_emb.index)

        if train_id_order != train_emb_order:
            diff_indices = [i for i, (a,b) in enumerate(zip(train_id_order, train_emb_order)) if a != b][:10]
            raise ValueError(f"❌ [Train order mismatch] {dataset_name} Fold{fold}\nFound mismatched indices: {diff_indices}")

        if test_id_order != test_emb_order:
            diff_indices = [i for i, (a,b) in enumerate(zip(test_id_order, test_emb_order)) if a != b][:10]
            raise ValueError(f"❌ [Test order mismatch] {dataset_name} Fold{fold}\nFound mismatched indices: {diff_indices}")

        print(f"✅ {dataset_name} Fold{fold} | Cell IDs perfectly match (content + order) ✔️")

        summary_final = pd.read_csv(self.drug_info_path)
        record = summary_final.loc[summary_final["dataset"] == f"{dataset_name}.rds"]
        if record.empty:
            raise ValueError(f"❌ No record found for dataset: {dataset_name}")

        drug_name = record.iloc[0]["drug_names"]
        smiles = record.iloc[0]["PubChemSMILES"]
        smiles_complete = isinstance(smiles, str) and len(smiles.strip()) > 0

        print(f"✅ {dataset_name} Fold{fold} | included with drug: {smiles} ✔️")
        if not smiles_complete:
            raise ValueError(f"⚠️ {dataset_name}: SMILES missing for {drug_name}")

        return {
            "dataset": dataset_name, "fold": fold, "train_embedding": train_emb, "test_embedding": test_emb,
            "train_labels": train_ids, "test_labels": test_ids, "drug_name": drug_name,
            "smiles": smiles.strip(), "smiles_complete": smiles_complete
        }

    def _load_unseen_dataset(self, dataset_name):
        print(f"📘 Loading {dataset_name}")
        id_dir = self.base_dir

        train_id_file = os.path.join(id_dir, f"{dataset_name}_train_meta.csv")
        test_id_file  = os.path.join(id_dir, f"{dataset_name}_test_meta.csv")
        train_emb_file = os.path.join(self.embed_dir, f"{dataset_name}_train_cellEmbedding.csv")
        test_emb_file  = os.path.join(self.embed_dir, f"{dataset_name}_test_cellEmbedding.csv")

        for f in [train_id_file, test_id_file, train_emb_file, test_emb_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"❌ Missing file: {f}")

        train_ids = pd.read_csv(train_id_file)
        test_ids  = pd.read_csv(test_id_file)
        train_emb = pd.read_csv(train_emb_file, index_col=0)
        test_emb  = pd.read_csv(test_emb_file, index_col=0)

        print(f"✅ {dataset_name} | Cell IDs perfectly match (content + order) ✔️")

        summary_final = pd.read_csv(self.drug_info_path)

        if "DRMref_unseen" in dataset_name:
            drug2smiles = summary_final[["drug_names", "PubChemSMILES"]].drop_duplicates("drug_names").set_index("drug_names")["PubChemSMILES"].to_dict()
            train_drug_name = train_ids["drug"]
            train_smiles = train_ids["drug"].map(drug2smiles)
            test_drug_name = test_ids["drug"]
            test_smiles  = test_ids["drug"].map(drug2smiles)

            return {
                "dataset": dataset_name, "train_embedding": train_emb, "test_embedding": test_emb,
                "train_labels": train_ids, "test_labels": test_ids, "train_smiles": train_smiles, "test_smiles": test_smiles
            }
        elif "_fold" in dataset_name:
            prefix = dataset_name.split("_fold")[0]
            record = summary_final.loc[summary_final["dataset"] == f"{prefix}.rds"]
            drug_name = record.iloc[0]["drug_names"]
            smiles = record.iloc[0]["PubChemSMILES"]
            return {
                "dataset": dataset_name, "train_embedding": train_emb, "test_embedding": test_emb,
                "train_labels": train_ids, "test_labels": test_ids, "smiles": smiles.strip()
            }
        else:
            raise ValueError(f"❌ No record found for dataset: {dataset_name}")

    def _prepare_data_tensors(self, dataset_result):
        y_train = encode_condition(dataset_result["train_labels"]["condition"])
        y_test  = encode_condition(dataset_result["test_labels"]["condition"])

        X_train = torch.tensor(dataset_result["train_embedding"].values, dtype=torch.float32)
        X_test  = torch.tensor(dataset_result["test_embedding"].values, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test  = torch.tensor(y_test, dtype=torch.float32)

        if "train_smiles" in dataset_result and "test_smiles" in dataset_result:
            train_smiles_list = dataset_result["train_smiles"]
            test_smiles_list  = dataset_result["test_smiles"]
            drug_train = np.vstack([smiles_to_fingerprint(s, self.fp_size) for s in train_smiles_list])
            drug_test  = np.vstack([smiles_to_fingerprint(s, self.fp_size) for s in test_smiles_list])
            drug_train = torch.tensor(drug_train, dtype=torch.float32)
            drug_test  = torch.tensor(drug_test, dtype=torch.float32)
        else:
            smiles = dataset_result["smiles"]
            drug_vec = smiles_to_fingerprint(smiles, fp_size=self.fp_size)
            drug_vec = torch.tensor(drug_vec, dtype=torch.float32).view(1, -1)
            drug_train = drug_vec.repeat(X_train.shape[0], 1)
            drug_test  = drug_vec.repeat(X_test.shape[0], 1)

        return X_train, y_train, drug_train, X_test, y_test, drug_test

    def _split_train_val(self, X_train, drug_train, y_train, seed):
        X_np = X_train.detach().cpu().numpy()
        D_np = drug_train.detach().cpu().numpy()
        y_np = y_train.detach().view(-1).cpu().numpy().astype(int)

        X_tr, X_val, D_tr, D_val, y_tr, y_val = train_test_split(
            X_np, D_np, y_np, test_size=self.val_ratio, random_state=seed, stratify=y_np
        )

        X_tr  = torch.tensor(X_tr,  dtype=X_train.dtype,  device=self.device)
        X_val = torch.tensor(X_val, dtype=X_train.dtype,  device=self.device)
        D_tr  = torch.tensor(D_tr,  dtype=drug_train.dtype, device=self.device)
        D_val = torch.tensor(D_val, dtype=drug_train.dtype, device=self.device)
        y_tr  = torch.tensor(y_tr,  dtype=y_train.dtype, device=self.device).view(-1)
        y_val = torch.tensor(y_val, dtype=y_train.dtype, device=self.device).view(-1)

        return X_tr, y_tr, D_tr, X_val, y_val, D_val

    # ---------------- Core training and evaluation ----------------
    def _train_with_convergence_stopping(self, model, X_train, y_train, drug_train, X_val, y_val, drug_val):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.lr)
        best_val_acc = -1.0
        no_improve = 0
        weight_state = copy.deepcopy(model.state_dict())

        y_train = y_train.float().view(-1)
        y_val   = y_val.float().view(-1)

        train_dataset = TensorDataset(X_train, drug_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        for epoch in range(self.epochs):
            model.train()
            for batch_X, batch_D, batch_y in train_loader:
                optimizer.zero_grad()
                logits = model(batch_X, batch_D)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val, drug_val)
                val_prob = torch.sigmoid(val_logits).detach().cpu().numpy()
                val_true = y_val.detach().cpu().numpy()
                val_pred = (val_prob >= 0.5).astype(int)
                val_acc = accuracy_score(val_true, val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                weight_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"⏹️ Stopping triggered at epoch {epoch+1}")
                break

        model.load_state_dict(weight_state)
        return model

    def _evaluate_on_test(self, model, X_test, y_test, drug_test, threshold=0.5):
        model.eval()
        y_test = y_test.float().view(-1)

        with torch.no_grad():
            test_logits = model(X_test, drug_test)
            test_prob = torch.sigmoid(test_logits).detach().cpu().numpy()

        y_true = y_test.detach().cpu().numpy()
        y_bin = (test_prob > threshold).astype(int)

        f1 = f1_score(y_true, y_bin, average="macro")
        auc = roc_auc_score(y_true, test_prob)
        auprc = average_precision_score(y_true, test_prob)

        print("\n✅ Final Test Performance (test only):")
        print(f" Macro-F1 = {f1:.4f} | AUROC = {auc:.4f} | AUPRC = {auprc:.4f}")
        return {"MacroF1": f1, "AUROC": auc, "AUPRC": auprc}

    # ---------------- External API ----------------
    def run_5_fold(self, dataset_name, method_name):
        """Run 5-fold cross-validation"""
        results = []
        for fold in range(1, 6):
            print(f"\n===================== 🧩 Fold {fold}/5 =====================")
            dataset_info = self._load_fold_dataset(dataset_name, fold)
            X_train, y_train, drug_train, X_test, y_test, drug_test = self._prepare_data_tensors(dataset_info)
            
            X_train, y_train, drug_train = X_train.to(self.device), y_train.to(self.device), drug_train.to(self.device)
            X_test, y_test, drug_test = X_test.to(self.device), y_test.to(self.device), drug_test.to(self.device)

            X_tr, y_tr, drug_tr, X_val, y_val, drug_val = self._split_train_val(X_train, drug_train, y_train, seed=42 + fold)

            model = DualTowerDrugResponse(cell_dim=X_train.shape[1], drug_dim=self.fp_size).to(self.device)
            model = self._train_with_convergence_stopping(model, X_tr, y_tr, drug_tr, X_val, y_val, drug_val)
            metrics = self._evaluate_on_test(model, X_test, y_test, drug_test)

            metrics.update({"fold": fold, "dataset": dataset_name, "method": method_name})
            results.append(metrics)

        return pd.DataFrame(results)

    def run_validation(self, dataset_name, method_name):
        """Run single-validation"""
        print(f"\n===================== 🧩 Dataset {dataset_name} =====================")
        dataset_info = self._load_unseen_dataset(dataset_name)
        X_train, y_train, drug_train, X_test, y_test, drug_test = self._prepare_data_tensors(dataset_info)

        X_train, y_train, drug_train = X_train.to(self.device), y_train.to(self.device), drug_train.to(self.device)
        X_test, y_test, drug_test = X_test.to(self.device), y_test.to(self.device), drug_test.to(self.device)

        X_tr, y_tr, drug_tr, X_val, y_val, drug_val = self._split_train_val(X_train, drug_train, y_train, seed=42)

        model = DualTowerDrugResponse(cell_dim=X_train.shape[1], drug_dim=self.fp_size).to(self.device)
        model = self._train_with_convergence_stopping(model, X_tr, y_tr, drug_tr, X_val, y_val, drug_val)
        metrics = self._evaluate_on_test(model, X_test, y_test, drug_test)

        metrics.update({"dataset": dataset_name, "method": method_name})
        return pd.DataFrame([metrics])