import os
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scib

class BatchIntegrationEvaluator:
    """
    Batch Integration Evaluator for a single dataset and method.
    """
    def __init__(self, embedding_base_dir="./01_Cell_Identify/", metadata_dir="./01_Cell_Identify/TS_data/"):
        self.embedding_base_dir = embedding_base_dir
        self.metadata_dir = metadata_dir

    def evaluate_single_dataset(self, dataset_name, method_name, file_name):
        """
        Evaluate one specified dataset and one specified method.
        """
        print(f"\n==================================================")
        print(f"Starting Evaluation | Dataset:{dataset_name} | Method: {method_name}")
        print(f"==================================================")

        # 1. Dynamically build file paths
        embedding_path = os.path.join(
            self.embedding_base_dir, 
            f"{file_name}"
        )
        meta_path = os.path.join(self.metadata_dir, f"{dataset_name}_metaInfo.csv")

        # Check if files exist
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"❌ Embedding file not found: {embedding_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"❌ Metadata file not found: {meta_path}")

        # 2. Load embedding matrix and check for missing values
        embedding_df = pd.read_csv(embedding_path, index_col=0)
        if embedding_df.isnull().values.any():
            n_missing = embedding_df.isnull().sum().sum()
            raise ValueError(f"❌ Found {n_missing} missing values (NA) in {dataset_name}_cellEmbedding.csv!")
        else:
            print(f"✅ Embedding loaded successfully. Shape: {embedding_df.shape}")

        # 3.Load metadata and align Cell IDs
        metadata_df = pd.read_csv(meta_path, index_col=0)
        if not embedding_df.index.equals(metadata_df.index):
            raise ValueError(f"❌ Cell IDs in embedding and metadata do not match for {dataset_name}!")
        print(f"✅ Metadata loaded and aligned successfully.")

        # 4. Construct AnnData object
        adata = ad.AnnData(obs=metadata_df.copy())
        adata.obsm["X_LLM_Embedding"] = embedding_df.values
        adata.obs_names = embedding_df.index
        
        # 5. Perform clustering and scIB metric calculations
        print("Calculating neighbors graph (n_neighbors=15)...")
        sc.pp.neighbors(adata, use_rep="X_LLM_Embedding", n_neighbors=15) 
        
        print("Searching for optimal clustering resolution...")
        All_resolution = scib.metrics.cluster_optimal_resolution(
            adata, cluster_key="cluster", label_key="celltype", force=True, return_all=True
        )
        
        print("Calculating scores...")
        ari_score = scib.metrics.ari(adata, cluster_key="cluster", label_key="celltype")
        nmi_score = scib.metrics.nmi(adata, cluster_key="cluster", label_key="celltype")
        sil_score = scib.metrics.silhouette(adata, embed="X_LLM_Embedding", label_key="celltype")
        sil_batch_score = scib.metrics.silhouette_batch(adata, embed="X_LLM_Embedding", batch_key="tech", label_key="celltype")
        adata.obs["celltype_category"] = adata.obs["celltype"].astype("category")
        adata.obs["tech_category"] = adata.obs["tech"].astype("category")
        iLISI_score = scib.me.ilisi_graph(adata, batch_key="tech_category", type_="embed", use_rep="X_LLM_Embedding")
        kBET_score = scib.metrics.kBET(adata, embed="X_LLM_Embedding", type_="embed", batch_key="tech_category", label_key="celltype_category")
        
        # 6. Assemble and return results
        result = {
            "Dataset": dataset_name,
            "Method": method_name,
            "Best_Resolution": round(All_resolution[0], 4),
            "ARI": round(ari_score, 4),
            "NMI": round(nmi_score, 4),
            "Silhouette_celltype": round(sil_score, 4),
            "Silhouette_batch": round(sil_batch_score, 4),
            "iLISI": round(iLISI_score, 4),
            "kBET": round(kBET_score, 4),
        }

        print("Evaluation Complete")
        return result

