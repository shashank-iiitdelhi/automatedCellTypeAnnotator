import anndata
import scanpy as sc
import pandas as pd
DATASET = "osteoarthritis"
#DATASET = "kidney"
#DATASET = "DCMACM_heart_cell"
adata = sc.read_h5ad(f'datasets/{DATASET}.h5ad')
print(adata)
print("_______________"*5)
print(adata.obs)
print("_______________"*5)
print(adata.var)
