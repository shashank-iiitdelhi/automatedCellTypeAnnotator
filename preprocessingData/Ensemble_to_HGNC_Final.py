import scanpy as sc
import pandas as pd
import numpy as np
import mygene

# Step 1: Convert Ensembl IDs in AnnData var_names to HGNC symbols and save new file
#DATASET = "DCMACM_heart_cell"
DATASET =  "kidney"
adata = sc.read_h5ad(f"datasets/{DATASET}_processed.h5ad")
#adata = adata[adata.obs['disease'] != '10x 5\' v1'].copy()

ensembl_ids = adata.var_names.tolist()  # Extract Ensembl gene IDs

mg = mygene.MyGeneInfo()
res = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

# Create mapping dict from Ensembl ID to HGNC symbol (only valid ones)
id_map = {r['query']: r.get('symbol') for r in res if 'symbol' in r}

valid_genes = [g for g in adata.var_names if g in id_map and id_map[g]]
new_names = [id_map[g] for g in valid_genes]

# Subset AnnData to valid genes and rename var_names
adata = adata[:, valid_genes].copy()
adata.var_names = new_names

# Remove duplicate gene symbols
adata = adata[:, ~adata.var_names.duplicated()].copy()

adata.write(f"datasets/{DATASET}_processed.h5ad")

