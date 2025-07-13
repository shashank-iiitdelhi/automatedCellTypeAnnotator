import scanpy as sc
import pandas as pd
from pathlib import Path

# === Step 1: Load count matrix ===
print("Reading count matrix...")
counts = pd.read_csv("datasets/GSE255460/sc_counts.txt", sep="\t", index_col=0)

# Transpose if needed (we want cells as rows and genes as columns)
if counts.shape[0] > counts.shape[1]:
    counts = counts.T

print("Original matrix shape:", counts.shape)

# === Step 2: Load metadata ===
print("Reading metadata...")
metadata = pd.read_csv("datasets/GSE255460/GSE255460_metadata.csv", index_col=0)

# === Step 3: Fix index mismatch ===
print("Fixing cell name mismatch...")
# Convert count matrix index: .1 -> -1
counts.index = counts.index.str.replace(r"\.1$", "-1", regex=True)

# (Optional) Strip trailing whitespace in metadata index
metadata.index = metadata.index.str.strip()

# === Step 4: Intersect and filter ===
shared_cells = counts.index.intersection(metadata.index)
print(f"Number of shared cells: {len(shared_cells)}")

# Subset both
counts = counts.loc[shared_cells]
metadata = metadata.loc[shared_cells]

print("Filtered matrix shape:", counts.shape)
print("Filtered metadata shape:", metadata.shape)

# === Step 5: Create AnnData object ===
print("Creating AnnData object...")
adata = sc.AnnData(X=counts)
adata.obs = metadata
adata.var_names = counts.columns
adata.obs_names = counts.index

# === Step 6: Save to .h5ad ===
output_file = "datasets/osteoarthritis.h5ad"
print(f"Saving to {output_file}...")
adata.write(output_file)
print("Done.")
