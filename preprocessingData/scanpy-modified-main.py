import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import pandas as pd
import numpy as np
import logging
import scvi
import torch
import os
import anndata as ad
import scvi
import torch
import gc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import median_abs_deviation
import plotly.express as px

warnings.filterwarnings("ignore")
sc.settings.verbosity = 3  # Increase verbosity for more detailed logs
sc.settings.n_jobs = 20  # Limit to 4 threads

# DATASET PATH
# DATASET = "DCMACM_heart_cell"
DATASET = "kidney"
adata = sc.read_h5ad(f"datasets/{DATASET}.h5ad")

# Data Subsampling
if(DATASET == "DCMACM_heart_cell"):
    filt = (
        (adata.obs['assay'] == "10x 3' v3") &
        (adata.obs['sex'] == "female") &
        (adata.obs['cell_type'] != "unknown") &  # Fixed typo
        (adata.obs['disease'].isin(['normal', 'dilated cardiomyopathy'])) &
        (adata.obs['self_reported_ethnicity'] == 'European') &  # Fixed case
        (adata.obs['tissue'] == 'heart left ventricle')
    )

    adata = adata[filt].copy()

elif(DATASET == "PSC_Liver"):
    adata = adata[adata.obs['disease'] != 'primary biliary cholangitis'].copy()
    adata = adata[adata.obs['assay'] != '10x 5\' v1'].copy()
elif DATASET == "kidney":
    filt = (
        (adata.obs['disease']!= 'acute kidney failure') 
        &
        (adata.obs['self_reported_ethnicity'] == 'European') &
        (adata.obs['sex'] == 'male') &
        (adata.obs['Race'] == 'White') &
        (adata.obs['development_stage'].isin(['39-year-old stage', '47-year-old stage', '48-year-old stage', '52-year-old stage', '56-year-old stage', '66-year-old stage'])) &
        (adata.obs['diabetes_history']!= 'Don\'t know')
    )
    adata = adata[filt].copy()

print(adata)

#Data doublet detection
torch.set_num_threads(4) 
def pp(adata_org):
    # Work on view, avoid deep copy unless needed
    adata = adata_org.copy()  # Don't use .copy() to avoid memory issues

    # Filter genes in-place to reduce memory usage
    sc.pp.filter_genes(adata, min_cells=10, inplace=True)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor='seurat_v3')

    # Setup and train SCVI model
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata)
    vae.train()

    # Setup and train SOLO doublet detection
    solo = scvi.external.SOLO.from_scvi_model(vae)
    solo.train()

    # Predict doublets
    df = solo.predict()
    df['prediction'] = solo.predict(soft=False)

    # Format index to match AnnData
    df.index = df.index.map(lambda x: x[:-2])
    df['dif'] = df.doublet - df.singlet
    doublets = df[(df.prediction == 'doublet') & (df.dif > 1)]
    del adata
    # Tag and filter doublets in original AnnData
    adata_org.obs['doublet'] = adata_org.obs.index.isin(doublets.index)
    adata_org = adata_org[~adata_org.obs.doublet].copy()
    
    # Cleanup
    # del vae, solo, df, doublets
    # gc.collect()

    return adata_org

adata = pp(adata)
adata.var_names_make_unique()
# Voilin Plot
adata.var["mt"] = adata.var_names.str.startswith("MT-")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt"], inplace=True, log1p=True
)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
             jitter=0.4, multi_panel=True)

def is_outlier(adata, metric: str, nmads: int):
    M = adata.obs[metric]
    outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
        np.median(M) + nmads * median_abs_deviation(M) < M
    )
    return outlier

adata.obs["outlier"] = (
    is_outlier(adata, "log1p_total_counts", 5)
    | is_outlier(adata, "log1p_n_genes_by_counts", 5)
    | is_outlier(adata, "pct_counts_in_top_50_genes", 5)
)
adata.obs.outlier.value_counts()

adata.obs["mt_outlier"] =  is_outlier(adata, "pct_counts_mt", 3) | (
    adata.obs["pct_counts_mt"] > 5
)
adata.obs.mt_outlier.value_counts()

print(f"Total number of cells: {adata.n_obs}")
adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()

print(f"Number of cells after filtering of low quality cells: {adata.n_obs}")

sc.pp.filter_genes(adata, min_cells=3)

# Normalization
fig = px.histogram(adata.obs["total_counts"], x="total_counts", title="Original Distribution")
# fig.show()
#### Shifted Logarith (Delta Method) #####
scales_counts = sc.pp.normalize_total(adata, target_sum=None, inplace=False)
# log1p transform
adata.layers["log1p_norm"] = sc.pp.log1p(scales_counts["X"], copy=True)

fig2 = px.histogram(adata.layers["log1p_norm"].sum(1) ,title="Shifted Logarithm")
# fig2.show()

# Highly Variable Genes
sc.pp.highly_variable_genes(adata, layer="log1p_norm",n_top_genes=3000)
# Extract highly variable genes data into a DataFrame
hvg_df = pd.DataFrame({
    'gene': adata.var_names,
    'highly_variable': adata.var['highly_variable'],
    'means': adata.var['means'],  # Already present
    'dispersions_norm': adata.var['dispersions_norm']  # Use dispersions_norm instead of variances_norm
})

# Sort the DataFrame by normalized dispersion (most variable genes)
hvg_df_sorted = hvg_df[hvg_df['highly_variable']].sort_values(by='dispersions_norm', ascending=False)

# Get the top 20 most variable genes
top_20_genes = hvg_df_sorted.head(20)

# Create a scatter plot of the normalized dispersion vs. mean expression
fig = px.scatter(
    hvg_df, 
    x='means', 
    y='dispersions_norm', 
    color='highly_variable', 
    title="Highly Variable Genes",
    labels={"means": "Mean Expression", "dispersions_norm": "Normalized Dispersion"},
    hover_name='gene'
)

# Add annotations for the top 20 genes
for i, row in top_20_genes.iterrows():
    fig.add_annotation(
        x=row['means'], 
        y=row['dispersions_norm'], 
        text=row['gene'], 
        showarrow=True, 
        arrowhead=2, 
        ax=0, 
        ay=-40
    )
# Show the plot
# fig.show()
sc.pp.neighbors(adata)
# run UMAP
sc.tl.umap(adata)
adata.obsm['X_umap_org'] = adata.obsm['X_umap']
# Dimensionality Reduction
sc.pp.pca(adata, svd_solver="arpack", layer='log1p_norm', mask_var='highly_variable') # mask_var='highly_variable'
#### log = True. Use only this if the first few pcs have a larger gap in between
import numpy as np
sc.pl.pca_variance_ratio(adata, n_pcs=20, log=True)
pca_variance_ratio = adata.uns['pca']['variance_ratio']

### for plotting using plotty we have to use np.log() to apply lograthmic scale to it.
variance_ratio_log = np.log(pca_variance_ratio)
fig = px.line(
    x=range(1, len(variance_ratio_log[:20]) + 1),
    y=variance_ratio_log[:20],
    labels={'x': 'Principal Components', 'y': 'Variance Explained'},
    title="Elbow Plot"
)

# Optionally, add markers at the points for better visualization
fig.update_traces(mode="lines+markers")
# fig.show()
### Heatmap ##########

top_genes_per_pc = 200
pcs_to_plot = range(1, 11)  # PCs 1 to 20

# Get the loadings of each gene on the PCs
loadings = adata.varm['PCs']

# Get the top contributing genes for each PC
top_genes = {}
for pc in pcs_to_plot:
    # Get absolute values of loadings for the current PC
    pc_loadings = np.abs(loadings[:, pc-1])
    
    # Sort genes by their loading values, and select top genes
    top_gene_indices = np.argsort(pc_loadings)[-top_genes_per_pc:]
    top_genes[pc] = adata.var_names[top_gene_indices].tolist()

cells_to_plot = list(adata.obs_names)[:50]
adata_subset = adata[cells_to_plot, :]

gene_expression_matrix = []
gene_labels = []

for pc, genes in top_genes.items():
    gene_expression = adata_subset[:, genes].X.toarray()  # Extract gene expression data
    gene_expression_matrix.append(gene_expression)
    gene_labels.extend(genes)

# Combine data for all PCs into a single matrix
gene_expression_matrix = np.concatenate(gene_expression_matrix, axis=1)

fig = make_subplots(
    rows=5, cols=4, 
    subplot_titles=[f'PC {pc}' for pc in pcs_to_plot],
    vertical_spacing=0.1, horizontal_spacing=0.1
)

# Loop through each PC and plot the top genes in a subplot
row = 1
col = 1
for pc in pcs_to_plot:
    genes = top_genes[pc]
    gene_expression = adata_subset[:, genes].X.toarray()  # Extract gene expression data

    # Add the heatmap for this PC to the subplot
    fig.add_trace(
        go.Heatmap(
            z=gene_expression.T,  # Transpose for correct orientation
            x=cells_to_plot,
            y=genes,
            colorscale='Viridis',
            showscale=False,  # Disable individual colorbars
        ),
        row=row, col=col
    )

    # Update subplot position
    col += 1
    if col > 4:  # Move to the next row after 4 columns
        col = 1
        row += 1

# Update layout
fig.update_layout(
    title='PC Heatmaps (Top 20 Genes per PC)',
    height=1000, width=1200,
    showlegend=False
)

# fig.show()
adata.obs['batch'] = adata.obs['disease']
sc.external.pp.harmony_integrate(adata,key = 'batch')
### tsne ####
# adata.obsm['X_pca_6'] = adata.obsm['X_pca'][:, :6]
sc.tl.tsne(adata, use_rep="X_pca")
sc.pl.tsne(adata, color="total_counts")
### UMAP ###
sc.pp.neighbors(adata,n_pcs=6)
sc.tl.umap(adata)
sc.pl.umap(adata, color="total_counts")
sc.pp.neighbors(adata, n_pcs=9, n_neighbors=15, use_rep='X_pca')
sc.tl.umap(adata, min_dist=0.1, spread=1.0, random_state=42)
sc.tl.leiden(adata, key_added="leiden_res0.25", resolution=0.25)
adata.obs['cell_type'].value_counts()
# adata.write_h5ad('datasets/alzheimer_processed.h5ad')
adata.write_h5ad(f'datasets/{DATASET}_processed.h5ad')
print(adata.obs.columns)
print("_"*50)
print(f"Preprocessing of {DATASET} dataset has been completed")
