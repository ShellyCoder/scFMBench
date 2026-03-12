import os
import pandas as pd
import numpy as np
import gseapy as gp
from sklearn.metrics import roc_auc_score, average_precision_score

# =========================================================
# 1. Calculations and Indicator Functions
# =========================================================
def compute_tf_delta(case_mat, ctrl_mat, tf):
    """
     Δ(TF → g) = case - control
    """
    if tf not in case_mat.index:
        raise ValueError(f"TF {tf} not found in case matrix")
    if tf not in ctrl_mat.index:
        raise ValueError(f"TF {tf} not found in control matrix")
        
    delta = case_mat.loc[tf] - ctrl_mat.loc[tf]
    df_delta = pd.DataFrame({
        "gene": delta.index,
        "delta": delta.values
    })
    return df_delta

def prepare_ranked_df(df_ranked: pd.DataFrame, tf_name: str = None) -> pd.DataFrame:
    df = df_ranked.copy()
    
    # 1) Remove TF itself
    if tf_name is not None and "gene" in df.columns:
        df = df[df["gene"] != tf_name].copy()
        
    # 2) Absolute value 
    df["score"] = df["delta"].abs()
    
    # 3) label column must exist and be of type int
    if "label" not in df.columns:
        raise ValueError("df_ranked must contain a 'label' column (0/1).")
    df["label"] = df["label"].fillna(0).astype(int)
    
    # 4) Sort by score
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df

def global_metrics(df: pd.DataFrame):
    y = df["label"].to_numpy()
    s = df["score"].to_numpy()
    return {
        "AUPRC": average_precision_score(y, s)
    }

def frac_to_k(n: int, frac: float) -> int:
    """Convert top fraction to absolute K."""
    return max(1, int(np.floor(frac * n)))

def top_frac_f1(df: pd.DataFrame, top_frac: float) -> float:
    K = frac_to_k(len(df), top_frac)
    pred = df.head(K)

    tp = int((pred["label"] == 1).sum())
    fp = K - tp
    fn = int((df["label"] == 1).sum()) - tp

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

def top_frac_jaccard(df: pd.DataFrame, top_frac: float) -> float:
    K = frac_to_k(len(df), top_frac)
    pred_set = set(df.head(K)["gene"])
    gt_set = set(df.loc[df["label"] == 1, "gene"])

    if len(pred_set | gt_set) == 0:
        return 0.0
    return len(pred_set & gt_set) / len(pred_set | gt_set)


# =========================================================
# 2. Core Assessment Process 
# =========================================================
def evaluate_grn_performance(
    method: str,
    dataset: str,
    input_dir: str,
    meta_info_dir: str
):
    """
    Evaluate GRN inference performance for a TF perturbation dataset.

    Parameters
    ----------
    method : str
        Algorithm name, e.g. "Geneformer", "scGPT"
    dataset : str
        Dataset name, e.g. "FERD3L_OE"
    input_dir : str
        Directory containing *_Case_geneEmbedding.csv and *_Control_geneEmbedding.csv
    meta_info_dir : str
        Directory containing *_GT.csv

    Returns
    -------
    pd.DataFrame
        One-row dataframe with evaluation metrics
    """

    # --------------------------------------------------
    # 0. basic info
    # --------------------------------------------------
    tf_name = dataset.split("_")[0]
    tf_model = dataset.split("_")[1]

    # --------------------------------------------------
    # 1. load GRN matrices
    # --------------------------------------------------
    case_mat = pd.read_csv(
        os.path.join(input_dir, f"{dataset}_Case_geneEmbedding.csv"),
        index_col=0
    )
    ctrl_mat = pd.read_csv(
        os.path.join(input_dir, f"{dataset}_Control_geneEmbedding.csv"),
        index_col=0
    )

    # --------------------------------------------------
    # 2. compute TF delta (case - control)
    # --------------------------------------------------
    df_delta = compute_tf_delta(case_mat, ctrl_mat, tf=tf_name)

    # --------------------------------------------------
    # 3. load ground truth TF-targets
    # --------------------------------------------------
    gt_path = os.path.join(meta_info_dir, f"{tf_name}_{tf_model}_GT.csv")
    gt = pd.read_csv(gt_path)
    gt_genes = set(gt["gene"].astype(str))

    df_delta["label"] = df_delta["gene"].isin(gt_genes).astype(int)
    assert df_delta["label"].sum() > 0, f"No positive TF-targets found for {tf_name}!"

    # --------------------------------------------------
    # 4. prepare ranked dataframe
    # --------------------------------------------------
    df_ranked = prepare_ranked_df(df_delta, tf_name)

    # --------------------------------------------------
    # 5. global metrics (threshold-free)
    # --------------------------------------------------
    metrics = global_metrics(df_ranked)

    # --------------------------------------------------
    # 6. Top-K metrics
    # --------------------------------------------------
    for frac in [0.05]:
        metrics[f"F1@{int(frac*100)}%"] = top_frac_f1(df_ranked, frac)
        metrics[f"Jaccard@{int(frac*100)}%"] = top_frac_jaccard(df_ranked, frac)

    metrics = {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}

    # --------------------------------------------------
    # 8. assemble result dataframe
    # --------------------------------------------------
    out = {
        "method": method,
        "dataset": dataset,
        "TF": tf_name,
        **metrics
    }

    return pd.DataFrame([out])


