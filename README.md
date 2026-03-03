# scFMBench
Benchmarking single-cell foundation models under zero-shot

## `run_qc_seurat.R` - Seurat QC & preprocessing

This script applies a unified Seurat-based QC and preprocessing workflow for scRNA-seq datasets, serving as the shared upstream step for all downstream benchmarking tasks.

**QC rules**
- Gene filtering: keep genes expressed in ≥ **3** cells (`min.cells = 3`).
- Cell filtering: remove cells with < **1,000** detected genes (`min.features = 1000`).

**Input**: Seurat `.rds` or 10x directory  
**Output**: QCed Seurat object (`.rds`)

**Example**
```bash
Rscript run_qc_seurat.R \
  --input dataset1.rds \
  --output dataset1.qc.rds \
  --min_cells 3 --min_features 1000 --n_hvg 2000
```

# Task 1 — Cell clustering

This task benchmarks **cell clustering utility** using **zero-shot cell embeddings** produced by scFMs. For each `(dataset, method)`, we build a kNN graph in the embedding space and follow the **scIB** workflow to select the clustering resolution by maximizing NMI, then report **ARI**, **NMI**, and **ASW (cell-type silhouette)**.

## Expected inputs

### Embeddings
- CSV is `cells × embedding_dim`
- row names must be **cell IDs**
- values must be numeric with **no NA/Inf**

### Metadata
- row names must be **cell IDs**
- must contain at least the ground-truth label column (default: `celltype`)
- **cell IDs and order must exactly match** the embedding file

## Output

A merged CSV with one row per `(dataset, method)`:
- `dataset`, `method`, `best_resolution`, `ARI`, `NMI`, `ASW_celltype`

If some tasks fail (missing files / ID mismatch / NA/Inf), an additional `*.errors.csv` is written next to the output.

## Run

### Example
```bash
python run_task1_clustering.py \
  --embedding-dir ./01_Cell_Clustering/ \
  --metadata-dir ./01_Cell_Clustering/metaData/ \
  --output-csv ./01_Cell_Clustering/Task1_results.csv \
  --methods highVariable,scBERTCLS,scBERTMean,scGNN,Geneformer6L,Geneformer12L,scGPT,scFoundation,CellPLM,GeneCompass,tGPT,UCE4L,UCE33L,GenePTmin,GenePTlarge,CellFMmin,CellFMBase,SCimilarity \
  --datasets Bladder,Blood,Bone_Marrow,Ear,Eye,Fat,Heart,Large_Intestine,Liver,Lung,Lymph_Node,Mammary,Muscle,Ovary,Prostate,Salivary_Gland,Skin,Small_Intestine,Spleen,Stomach,Testis,Thymus,Tongue,Trachea,Uterus,Vasculature \
  --n-jobs 4
```

# Task 2 — Cell annotation

This module benchmarks **supervised cell-type annotation** using **zero-shot cell embeddings**. For each dataset, we run **stratified 5-fold cross-validation**. A **single standardized MLP classifier** is trained on the training embeddings and evaluated on the held-out test embeddings, ensuring fair comparisons across embedding methods.

## Inputs

### 1) Embeddings (per dataset × fold)
The script expects precomputed embeddings under:
<EMBEDDING_ROOT>/_outputData/

For each dataset and fold:
_fold_train_cellEmbedding.csv
_fold_test_cellEmbedding.csv

Format:
- CSV is `cells × embedding_dim`
- rownames must be `cell_id`
- numeric only (no NA/Inf)

### 2) Fold split files (IDs + labels)
Under:
<LABEL_ROOT>/

For each dataset and fold:
_fold_train_ids.csv
_fold_test_ids.csv

Required columns:
- `cell_id`
- `celltype`

Important:
- `cell_id` order must exactly match embedding CSV row order.

## Output
One summary CSV per method:
<OUTPUT_DIR>/results__summary.csv

Each row corresponds to one `(Dataset, Fold)`.

## Run

### Example
```bash
python src/tasks/task2_annotation/run_task2_annotation.py \
  --methods scFoundation \
  --datasets Bladder,Blood,Bone_Marrow \
  --embedding-root ./02_Cell_annotation \
  --label-root ./fiveFold_ID \
  --output-dir ./Five-fold \
  --device cuda:1 \
  --seed 42
```

# Task 3 — Drug response prediction (zero-shot cell embeddings)

This module benchmarks **binary drug response prediction** using:
- **zero-shot cell embeddings** (precomputed, no fine-tuning of foundation models)
- **drug Morgan fingerprints** derived from SMILES (RDKit)
- a fixed **dual-tower neural classifier** (cell tower + drug tower)

## Inputs

### 1) Embeddings
Expected directory structure:
<EMBED_ROOT>/_outputData/

**CV mode (`--mode cv`)** requires, for each dataset and fold:
_fold_train_cellEmbedding.csv
_fold_test_cellEmbedding.csv

Embedding CSV format:
- rows = `cell_id` (index)
- columns = embedding dimensions
- numeric only (no NA/Inf)

### 2) Labels (cell_id + condition)
**CV mode**:
<BASE_DIR>/fiveFold_ID/
_fold_train_ids.csv
_fold_test_ids.csv

Required columns:
- `cell_id`
- `condition` (mapped to {0,1}: resistant/non-response -> 0; sensitive/response -> 1)

For each split, `cell_id` order must **exactly match** the embedding row order.

### 3) Drug information (SMILES)
A CSV containing `dataset`, `drug_names`, `PubChemSMILES` (your `drug_response_14_datasets.csv`).

- For standard DRMref datasets: each dataset maps to a **single drug** by `dataset == "<name>.rds"`.
- For unseen/multi-drug augmented sets: if `train/test meta` contains a `drug` column, SMILES are mapped per cell.

## Model
Dual-tower network:
- Cell encoder: Linear → ReLU → BN → Dropout
- Drug encoder: Linear → ReLU → BN → Dropout
- Fusion: concat(cell, drug) → MLP → sigmoid

## Metrics
Computed on the test split:
- Accuracy
- Macro-F1
- AUROC
- AUPRC

## Output
One CSV per method:
<OUTPUT_DIR>/results_.csv

Each row corresponds to one `(Dataset, Fold)` (Fold is `NA` in augment mode).

## Run

Each run MUST explicitly provide `--methods` and `--datasets` (or list files). No defaults.

### Example
```bash
python /run_task3_drug.py \
  --mode cv \
  --methods scGNN \
  --datasets GDSC_A,GDSC_B \
  --embed-root ./drug_sensitivity \
  --base-dir ./drug_sensitivity/data \
  --drug-info ./drug_sensitivity/dataInfo/drug_response_14_datasets.csv \
  --output-dir ./drug_sensitivity/result/fiveFold \
  --device cuda:0 \
  --seed 42
```

# Task 4 — Batch integration

This module evaluates **batch integration quality** using **zero-shot cell embeddings** and **scIB** metrics.

## Inputs

### 1) Embeddings
Expected structure:

- Baseline mode:
<EMBEDDING_ROOT>/_outputData/_cellEmbedding.csv

Embedding CSV format:
- rows = `cell_id` (index)
- columns = embedding dimensions
- numeric only (no NA/Inf)

### 2) Metadata (metaInfo)
Required columns:
- `celltype` (biological label)
- `tech` (batch label)

Expected structure:

- Baseline mode:
<METADATA_ROOT>/_metaInfo.csv

**Important:** `cell_id` order must exactly match between embedding and metadata files.

## Run

Each run MUST explicitly provide `--methods` and `--datasets` (or list files). No defaults.

### Baseline datasets
```bash
python run_task4_batch.py \
  --mode baseline \
  --methods Harmony,scGPT,scFoundation \
  --datasets Pancreas,Lung,Immune \
  --embedding-root ./batch_effect \
  --metadata-root ./batch_effect/datasets \
  --output-csv ./batch_effect/result/Task4_results.csv \
  --n-jobs 24
```

# Task 5 — Gene function prediction

This module evaluates whether **zero-shot gene embeddings** encode stable **gene semantics**.

## Dataset
- Labels: **Human Protein Atlas (HPA)** tissue-specific genes  
- Task: **15-class** gene classification (each gene maps to exactly one tissue label)

Input label file:
- `HPA_tissue_specific_genes.pkl` (dict: tissue → list of genes)

## Inputs
### 1) Aligned gene embeddings
- `aligned_embeddings.pkl` (dict: model_name → `pandas.DataFrame`)
- Each DataFrame:
  - `index`: gene symbols
  - `values`: embedding vectors (float)

### 2) One-hot baseline
A one-hot baseline is constructed in the same gene space as a **reference model** (`--ref-onehot-model`) by creating an `N × N` identity matrix over the reference gene list.

## Output
- Long-format CSV (one row per model per fold):
  - `Model, Fold, MacroF1, MacroAUROC, MacroAUPRC, n_genes_used, embed_dim`
- A label table aligned to the reference gene space:
  - `Gene, Tissue, label`

## Run

```bash
python run_task5_gene_function.py \
  --methods GeneCompass,scGPT,onehot \
  --aligned-embeddings-pkl ./Gene_function/data/aligned_embeddings.pkl \
  --hpa-tissue-genes-pkl ./Gene_function/data/HPA_tissue_specific_genes.pkl \
  --ref-onehot-model GeneCompass \
  --output-csv ./Gene_function/result/GeneFunction_5_fold_results.csv \
  --output-label-csv ./Gene_function/data/tissue_specific_label.csv \
  --device cuda:0
```

# Task 6 — Gene regulatory network (GRN) inference

This module evaluates how zero-shot gene–gene relations extracted from foundation models recover ground-truth TF–target regulations in perturbation datasets.

## Overview
For each TF perturbation dataset (e.g., `BACH2_KD`, `CDX1_OE`), we load two condition-specific gene–gene matrices:
- `*_Case_geneEmbedding.csv`
- `*_Control_geneEmbedding.csv`

## Expected inputs
For each method:
{input_root}/{dataset}_Case_geneEmbedding.csv
{input_root}/{dataset}_Control_geneEmbedding.csv

For ground truth:
{meta_info_dir}/{TF}_{MODE}_GT.csv

`dataset` must follow the format `{TF}_{MODE}` (e.g., `BACH2_KD`, `CDX1_OE`).

## Usage
You must explicitly specify both `--methods` and `--datasets`.

Example:
```bash
python run_task6_grn_inference.py \
  --methods Cosine scGPT GeneCompass \
  --datasets BACH2_KD CDX1_OE CDX2_OE \
  --input-root ./GRN_infer \
  --meta-info-dir ./GRN_infer/data/perturb_data \
  --out-csv ./GRN_infer/result/Task6_result.csv \
  --top-fracs 0.05
```



