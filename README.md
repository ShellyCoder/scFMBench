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
Rscript src/qc/run_qc_seurat.R \
  --input dataset1.rds \
  --output dataset1.qc.rds \
  --min_cells 3 --min_features 1000 --n_hvg 2000
```

# Task 1 — Cell clustering (zero-shot cell embedding)

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

### Option A: comma-separated lists
```bash
python src/tasks/task1_clustering/run_task1_clustering.py \
  --embedding-dir ./01_Cell_Identify/ \
  --metadata-dir ./01_Cell_Identify/TS_data/Augment_Data/ \
  --output-csv ./01_Cell_Identify/A_Task1_evaluation_result/All_Methods_Task1_results.csv \
  --methods highVariable,scBERTCLS,scBERTMean,scGNN,Geneformer6L,Geneformer12L,scGPT,scFoundation,CellPLM,GeneCompass,tGPT,UCE4L,UCE33L,GenePTmin,GenePTlarge,CellFMmin,CellFMBase,SCimilarity \
  --datasets Bladder,Blood,Bone_Marrow,Ear,Eye,Fat,Heart,Large_Intestine,Liver,Lung,Lymph_Node,Mammary,Muscle,Ovary,Prostate,Salivary_Gland,Skin,Small_Intestine,Spleen,Stomach,Testis,Thymus,Tongue,Trachea,Uterus,Vasculature \
  --n-jobs 4

python src/tasks/task1_clustering/run_task1_clustering.py \
  --embedding-dir ./01_Cell_Identify/ \
  --metadata-dir ./01_Cell_Identify/TS_data/Augment_Data/ \
  --output-csv ./01_Cell_Identify/A_Task1_evaluation_result/All_Methods_Task1_results.csv \
  --methods-file methods.txt \
  --datasets-file datasets.txt \
  --n-jobs 4
```
