import scanpy as sc
import pandas as pd
import urllib.request
import os
from sctype_py import gene_sets_prepare, sctype_score, process_cluster
import numpy as np
# Load ScType functions (if not already in local sctype_py.py)
# Uncomment the line below if you need to download ScType dynamically.
# exec(urllib.request.urlopen("https://raw.githubusercontent.com/kris-nader/sc-type-py/main/sctype_py.py").read().decode())

# Load AnnData object
# DATASET = "DCMACM_heart_cell"
DATASET = "kidney"
adata = sc.read_h5ad(f"datasets/{DATASET}_processed.h5ad")

# Use the correct clustering column
cluster_column = 'leiden_res0.25'
if cluster_column not in adata.obs.columns:
    raise ValueError(f"Expected clustering column '{cluster_column}' not found in adata.obs. "
                     f"Available columns: {list(adata.obs.columns)}")

# Optional preprocessing (skip if already done)
if 'X_pca' not in adata.obsm:
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
if 'neighbors' not in adata.uns:
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10)
if 'X_umap' not in adata.obsm:
    sc.tl.umap(adata)

# Convert AnnData to DataFrame
# Use this when adata.X is (n_cells, n_genes) shape
scRNAseqData = pd.DataFrame(adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X,
                            index=adata.obs_names,
                            columns=adata.var_names).T

# Load marker genes
CELL_TYPE = ""
if DATASET == "DCMACM_heart_cell":
    CELL_TYPE = "Heart"
elif DATASET == "PSC_Liver":
    CELL_TYPE = "Liver"
elif DATASET == "kidney":
    CELL_TYPE = "Kidney"
gs_list = gene_sets_prepare(path_to_db_file="ScTypeDB_full.xlsx", cell_type=CELL_TYPE)

# Compute ScType scores
es_max, marker_found_series = sctype_score(scRNAseqData, scaled=True, gs=gs_list['gs_positive'], gs2=gs_list['gs_negative'])

adata.obs['markerFound'] = marker_found_series.loc[adata.obs_names]
# Make sure cluster labels are strings
adata.obs[cluster_column] = adata.obs[cluster_column].astype(str)

# Annotate clusters
unique_clusters = adata.obs[cluster_column].unique()
cL_results = pd.concat([process_cluster(cluster, adata, es_max, cluster_column) for cluster in unique_clusters])
sctype_scores = cL_results.groupby('cluster').apply(lambda x: x.nlargest(1, 'scores')).reset_index(drop=True)
sctype_scores['cluster'] = sctype_scores['cluster'].astype(str)

# Mark unknowns below threshold
sctype_scores.loc[sctype_scores['scores'] < sctype_scores['ncells'] / 4, 'type'] = 'Unknown'

# Add ScType classifications to AnnData
adata.obs['sctype_classification'] = ""
adata.obs['sctype_score'] = np.nan  # Initialize with NaNs

for cluster in sctype_scores['cluster'].unique():
    cl_info = sctype_scores[sctype_scores['cluster'] == cluster].iloc[0]
    adata.obs.loc[adata.obs[cluster_column] == cluster, 'sctype_classification'] = cl_info['type']
    adata.obs.loc[adata.obs[cluster_column] == cluster, 'sctype_score'] = cl_info['scores']

# Plot UMAP
sc.pl.umap(
    adata,
    color='sctype_classification',
    legend_loc='on data',
    title='ScType Cell Type Annotation',
    save=f"_sctype_umap_{DATASET}.png"  # This saves to `figures/` by default
)

# Save results
os.makedirs("output", exist_ok=True)
adata.obs[['sctype_classification', 'sctype_score', 'cell_type']].to_csv(f"output/csv/sctype_annotations_{DATASET}.csv")
adata.write(f"datasets/annotated_data_{DATASET}.h5ad")
