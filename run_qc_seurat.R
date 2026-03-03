suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
})

# ---------------------------
# 1) QC rules
# ---------------------------

apply_qc_rules <- function(seu, min_cells = 3, min_features = 1000) {
  # Ensure RNA assay exists
  DefaultAssay(seu) <- "RNA"
  
  # Gene filter: keep genes expressed in >= min_cells cells
  counts <- GetAssayData(seu, slot = "counts")
  keep_genes <- Matrix::rowSums(counts > 0) >= min_cells
  seu <- subset(seu, features = rownames(counts)[keep_genes])
  
  # Cell filter: keep cells with >= min_features detected genes
  # Seurat stores nFeature_RNA in meta.data after object creation;
  # but we re-compute to be safe.
  seu[["nFeature_RNA"]] <- Matrix::colSums(GetAssayData(seu, slot = "counts") > 0)
  seu <- subset(seu, subset = nFeature_RNA >= min_features)
  
  return(seu)
}

# ---------------------------
# 2) Preprocess
# ---------------------------

run_preprocess <- function(seu, n_hvg = 2000) {
  DefaultAssay(seu) <- "RNA"
  
  # Seurat defaults
  seu <- NormalizeData(seu, verbose = FALSE)
  seu <- FindVariableFeatures(seu, selection.method = "vst", nfeatures = n_hvg, verbose = FALSE)
  seu <- ScaleData(seu, verbose = FALSE)
  seu <- RunPCA(seu, verbose = FALSE)
  
  return(seu)
}

# ---------------------------
# 3) IO helpers
# ---------------------------

read_input <- function(input_path, input_format = c("h5seurat", "rds", "10x")) {
  input_format <- match.arg(input_format)
  
  if (input_format == "rds") {
    obj <- readRDS(input_path)
    if (!inherits(obj, "Seurat")) stop("Input RDS is not a Seurat object.")
    return(obj)
  }
  
  if (input_format == "10x") {
    # input_path should be a 10X directory containing matrix.mtx(.gz), features.tsv(.gz), barcodes.tsv(.gz)
    counts <- Read10X(data.dir = input_path)
    seu <- CreateSeuratObject(counts = counts, assay = "RNA", project = "scQC")
    return(seu)
  }
  
  if (input_format == "h5seurat") {
    # Requires SeuratDisk package;
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Reading .h5seurat requires SeuratDisk. Please install SeuratDisk or use --input_format rds/10x.")
    }
    obj <- SeuratDisk::LoadH5Seurat(input_path, assays = "RNA")
    if (!inherits(obj, "Seurat")) stop("Loaded object is not a Seurat object.")
    return(obj)
  }
  
  stop("Unsupported input_format.")
}

write_output <- function(seu, output_path, output_format = c("rds", "h5seurat")) {
  output_format <- match.arg(output_format)
  
  dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
  
  if (output_format == "rds") {
    saveRDS(seu, file = output_path)
    return(invisible(TRUE))
  }
  
  if (output_format == "h5seurat") {
    if (!requireNamespace("SeuratDisk", quietly = TRUE)) {
      stop("Writing .h5seurat requires SeuratDisk. Please install SeuratDisk or use --output_format rds.")
    }
    # SaveSeuratDisk writes an .h5seurat file (output_path should end with .h5seurat)
    SeuratDisk::SaveH5Seurat(seu, filename = output_path, overwrite = TRUE)
    return(invisible(TRUE))
  }
  
  stop("Unsupported output_format.")
}

# ---------------------------
# 4) Main (run_qc.py equivalent)
# ---------------------------

main <- function() {
  option_list <- list(
    make_option(c("-i", "--input"), type = "character", help = "Input path: Seurat RDS / 10X dir / h5seurat"),
    make_option(c("-o", "--output"), type = "character", help = "Output path"),
    make_option(c("--input_format"), type = "character", default = "rds",
                help = "Input format: rds | 10x | h5seurat [default %default]"),
    make_option(c("--output_format"), type = "character", default = "rds",
                help = "Output format: rds | h5seurat [default %default]"),
    make_option(c("--min_cells"), type = "integer", default = 3,
                help = "min.cells: gene expressed in >= min_cells cells [default %default]"),
    make_option(c("--min_features"), type = "integer", default = 1000,
                help = "min.features: cells with >= detected genes [default %default]"),
    make_option(c("--n_hvg"), type = "integer", default = 2000,
                help = "Number of HVGs [default %default]")
  )
  
  opt <- parse_args(OptionParser(option_list = option_list))
  
  if (is.null(opt$input) || is.null(opt$output)) {
    stop("Please provide --input and --output.")
  }
  
  message("[QC] Reading input: ", opt$input, " (", opt$input_format, ")")
  seu <- read_input(opt$input, input_format = opt$input_format)
  
  n_cells0 <- ncol(seu); n_genes0 <- nrow(seu)
  message(sprintf("[QC] Before QC: %d cells, %d genes", n_cells0, n_genes0))
  
  seu <- apply_qc_rules(seu, min_cells = opt$min_cells, min_features = opt$min_features)
  
  n_cells1 <- ncol(seu); n_genes1 <- nrow(seu)
  message(sprintf("[QC] After QC:  %d cells, %d genes", n_cells1, n_genes1))
  
  message("[QC] Running preprocess (NormalizeData + HVG(2000) + ScaleData + PCA)")
  seu <- run_preprocess(seu, n_hvg = opt$n_hvg)
  
  message("[QC] Writing output: ", opt$output, " (", opt$output_format, ")")
  write_output(seu, opt$output, output_format = opt$output_format)
  
  message("[QC] Done.")
}

main()











