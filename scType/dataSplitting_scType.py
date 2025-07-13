import scanpy as sc
import pandas as pd
import numpy as np
import os

# Load the annotated AnnData object
# DATASET = "PSC_Liver"
# DATASET = "DCMACM_heart_cell"
DATASET = "kidney"
adata = sc.read_h5ad(f"datasets/annotated_data_{DATASET}.h5ad")
label_mapping = {}
# Mapping dictionary (same as your previous one)
if DATASET == "PSC_Liver":
    label_mapping = {
        'hepatocyte': 'Hepatocytes',
        'centrilobular region hepatocyte': 'Hepatocytes',
        'periporta l region hepatocyte': 'Hepatocytes',
        'midzonal region hepatocyte': 'Hepatocytes',

        'endothelial cell of artery': 'Endothelial cell',
        'endothelial cell of pericentral hepatic sinusoid': 'Endothelial cell',
        'endothelial cell of periportal hepatic sinusoid': 'Endothelial cell',
        'vein endothelial cell': 'Endothelial cell',

        'Kupffer cell': 'Kupffer cells',

        'macrophage': 'Immune system cells',
        'monocyte': 'Immune system cells',
        'neutrophil': 'Immune system cells',
        'mast cell': 'Immune system cells',
        'T cell': 'Immune system cells',
        'CD4-positive, alpha-beta T cell': 'Immune system cells',
        'CD8-positive, alpha-beta T cell': 'Immune system cells',
        'natural killer cell': 'Immune system cells',
        'hepatic pit cell': 'Immune system cells',
        'conventional dendritic cell': 'Immune system cells',
        'plasmacytoid dendritic cell': 'Immune system cells',
        'mature B cell': 'Immune system cells',
        'plasma cell': 'Immune system cells',

        'hepatic stellate cell': 'Unknown',
        'intrahepatic cholangiocyte': 'Unknown',
        'fibroblast': 'Unknown',
        'erythrocyte': 'Unknown',
        'unknown': 'Unknown',
    }
elif DATASET== "DCMACM_heart_cell":
    label_mapping = {
        'mural cell': 'Smooth muscle cells',  # mural cells include pericytes/smooth muscle lineage
        'cardiac muscle cell': 'Smooth muscle cells',  # broad match to muscle lineage, closest is smooth
        'endothelial cell': 'Vascular endothelial cells',
        'myeloid cell': 'Myeloid cells',
        'lymphocyte': 'Lymphoid cells',
        'fibroblast of cardiac tissue': 'Stromal cells',  # fibroblasts contribute to the stromal support
        'cardiac neuron': 'Schwann cells',  # closest match since Schwann cells support peripheral neurons
        'mast cell': 'Myeloid cells',  # mast cells are derived from myeloid lineage
        'adipocyte': 'Stromal cells',  # often considered part of the stromal compartment

        'unknown': 'Unknown'  # if you have 'unknown' in author labels
    }

elif DATASET == "kidney":
    label_mapping = {
        # Proximal tubule
        'epithelial cell of proximal tubule': 'Proximal tubule cells',

        # Loop of Henle
        'kidney loop of Henle thick ascending limb epithelial cell': 'Loop of Henle cells',
        'kidney loop of Henle thin ascending limb epithelial cell': 'Loop of Henle cells',
        'kidney loop of Henle thin descending limb epithelial cell': 'Loop of Henle cells',

        # Principal cells (collecting duct)
        'kidney collecting duct principal cell': 'Principal cells (Collecting duct system)',
        'kidney connecting tubule epithelial cell': 'Principal cells (Collecting duct system)',

        # Intercalated cells
        'kidney collecting duct intercalated cell': 'α-intercalated cells (Collecting duct system)',

        # Distal tubule
        'kidney distal convoluted tubule epithelial cell': 'Distal tubule cells',

        # Endothelial
        'endothelial cell': 'Endothelial cells',

        # Immune (grouped broadly)
        'T cell': 'Immune cells',
        'cytotoxic T cell': 'Immune cells',
        'natural killer cell': 'Immune cells',
        'B cell': 'Immune cells',
        'plasma cell': 'Immune cells',
        'monocyte': 'Immune cells',
        'non-classical monocyte': 'Immune cells',
        'mononuclear phagocyte': 'Immune cells',
        'mast cell': 'Immune cells',
        'conventional dendritic cell': 'Immune cells',
        'plasmacytoid dendritic cell, human': 'Immune cells',
        'mature NK T cell': 'Immune cells',
        'kidney interstitial alternatively activated macrophage': 'Immune cells',

        # Hematopoietic cells (likely overlap with immune)
        'hematopoietic cell': 'Hematopoietic cells',

        # Mesangial or related interstitial
        'kidney interstitial cell': 'Mesangial cells',
        'podocyte': 'Mesangial cells',  # approximate, based on proximity and clustering

        # Rare/unmapped types
        'parietal epithelial cell': 'Unknown',
        'unknown': 'Unknown'
    }
else:
    print(f"⚠️  No mapping defined for dataset '{DATASET}'. Setting 'mapped_cell_type' to empty string.")
    label_mapping = {}
    adata.obs['mapped_cell_type'] = ""
    
# Apply mapping
if label_mapping:
    # Ensure 'cell_type' exists in obs
    if 'cell_type' not in adata.obs.columns:
        raise ValueError("Expected 'cell_type' column not found in adata.obs. Available columns: "
                         f"{list(adata.obs.columns)}")
    
    # Map cell types to new labels
    adata.obs['mapped_cell_type'] = adata.obs['cell_type'].map(label_mapping)

    # Drop cells with missing or Unknown mapping
    valid_mask = adata.obs['sctype_classification'].notna() & (adata.obs['sctype_classification'] != 'Unknown')
    adata_valid = adata[valid_mask].copy()

    # Split into train and test based on top 50% sctype_score per mapped class
    train_indices = []
    test_indices = []

    for label in adata_valid.obs['sctype_classification'].unique():
        subset = adata_valid[adata_valid.obs['sctype_classification'] == label]

        # Sort by sctype_score descending
        sorted_scores = subset.obs['sctype_score'].sort_values(ascending=False)
        cutoff = int(np.ceil(0.5 * len(sorted_scores)))

        # Top 50% for training — only if markerFound == True
        top_half = sorted_scores[:cutoff]
        top_half = top_half[subset.obs.loc[top_half.index, 'markerFound'] == True]

        # Bottom 50% (regardless of markerFound) for testing
        bottom_half = sorted_scores[cutoff:]

        train_indices.extend(top_half.index.tolist())
        test_indices.extend(bottom_half.index.tolist())

    # Create train/test AnnData objects
    adata_train = adata_valid[train_indices].copy()
    adata_test = adata_valid[test_indices].copy()

    # Save splits
    os.makedirs("output", exist_ok=True)
    adata_train.write(f"datasets/{DATASET}_train_top50pct.h5ad")
    adata_test.write(f"datasets/{DATASET}_test_bottom50pct.h5ad")

    print("✅ Data split complete.")
    print(f"Train set: {adata_train.n_obs} cells")
    print(f"Test set:  {adata_test.n_obs} cells")
    print(adata_test.obs.columns)
