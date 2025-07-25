import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, MinMaxScaler
import requests
import xml.etree.ElementTree as ET
import scanpy as sc
from collections import defaultdict
import concurrent.futures
import multiprocessing
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry





def gene_sets_prepare(path_to_db_file, cell_type):
    # Read data from Excel file
    cell_markers = pd.read_excel(path_to_db_file)
    # Filter by cell_type
    cell_markers = cell_markers[cell_markers['tissueType'] == cell_type]
    # Preprocess geneSymbolmore1 and geneSymbolmore2 columns
    #for col in ['geneSymbolmore1', 'geneSymbolmore2']:
    #    cell_markers[col] = cell_markers[col].str.replace(" ", "").str.upper()
    for col in ['geneSymbolmore1', 'geneSymbolmore2']:
        if col in cell_markers.columns:
            cell_markers[col] = cell_markers[col].fillna('').str.replace(" ", "").str.upper()
    # Stack and drop duplicates to get unique gene names
    gene_names = pd.concat([cell_markers['geneSymbolmore1'], cell_markers['geneSymbolmore2']]).str.split(',', expand=True).stack().drop_duplicates().reset_index(drop=True)
    gene_names = gene_names[gene_names.str.strip() != '']
    gene_names = gene_names[gene_names != 'None'].unique()
    # Get approved symbols for gene names
    res = get_gene_symbols(set(gene_names))
    res = dict(zip(res['Gene'], res['Symbol']))
    # Process gene symbols
    for col in ['geneSymbolmore1', 'geneSymbolmore2']:
        cell_markers[col] = cell_markers[col].apply(lambda row: process_gene_symbols(row, res)).str.replace("///", ",").str.replace(" ", "")
    # Group by cellName and create dictionaries of gene sets
    gs_positive = cell_markers.groupby('cellName')['geneSymbolmore1'].apply(lambda x: list(set(','.join(x).split(',')))).to_dict()
    gs_negative = cell_markers.groupby('cellName')['geneSymbolmore2'].apply(lambda x: list(set(','.join(x).split(',')))).to_dict()
    return {'gs_positive': gs_positive, 'gs_negative': gs_negative}


def process_gene_symbols(gene_symbols, res):
    if pd.isnull(gene_symbols):
        return ""
    markers_all = gene_symbols.upper().split(',')
    markers_all = [marker.strip().upper() for marker in markers_all if marker.strip().upper() not in ['NA', '']]
    markers_all = sorted(markers_all)
    if len(markers_all) > 0:
        markers_all = [res.get(marker) for marker in markers_all]
        markers_all = [symbol for symbol in markers_all if symbol is not None]
        markers_all = list(set(markers_all))
        return ','.join(markers_all)
    else:
        return ""



# =============================================================================
# def get_gene_symbols(genes):
#     data = {"Gene": [], "Symbol": []}
#     for gene in genes:
#         session = requests.Session()
#         retry = Retry(connect=3, backoff_factor=0.5)
#         adapter = HTTPAdapter(max_retries=retry)
#         session.mount('http://', adapter)
#         session.mount('https://', adapter)
#         url = f"https://rest.genenames.org/fetch/symbol/{gene}"
#         response = session.get(url)
#         if response.status_code == 200:
#             root = ET.fromstring(response.content)
#             result_elem = root.find("result")
#             if result_elem.get("numFound") == "0":
#                 url = f"https://rest.genenames.org/search/alias_symbol/{gene}"
#                 response = session.get(url)
#                 if response.status_code == 200:
#                     root = ET.fromstring(response.content)
#                     result_elem = root.find("result")
#                     if result_elem is not None and result_elem.get("numFound") != "0":
#                         symbols = [doc.find('str[@name="symbol"]').text for doc in root.findall('.//doc')]
#                         data["Gene"].append(gene)
#                         data["Symbol"].append(','.join(symbols))
#                     if result_elem is not None and result_elem.get("numFound") == "0":
#                         url = f"https://rest.genenames.org/search/prev_symbol/{gene}"
#                         response = session.get(url)
#                         if response.status_code == 200:
#                             root = ET.fromstring(response.content)
#                             result_elem = root.find("result")
#                             if result_elem is not None and result_elem.get("numFound") == "0":
#                                 symbol_element = root.find('.//str[@name="symbol"]')
#                                 data["Gene"].append(gene)
#                                 data["Symbol"].append(symbol_element)
#             else:
#                 data["Gene"].append(gene)
#                 data["Symbol"].append(gene)
#         else:
#             print(f"Failed to retrieve data for gene {gene}. Status code:", response.status_code)
#     df = pd.DataFrame(data)
#     return df
# 
# 
# =============================================================================


def get_gene_symbols(genes):
    data = {"Gene": [], "Symbol": []}
    for gene in genes:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        url = f"https://rest.genenames.org/fetch/symbol/{gene}"
        response = session.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            result_elem = root.find("result")
            if result_elem is not None and result_elem.get("numFound") == "0":
                url = f"https://rest.genenames.org/search/alias_symbol/{gene}"
                response = session.get(url)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    result_elem = root.find("result")
                    if result_elem is not None and result_elem.get("numFound") != "0":
                        symbols = [doc.find('str[@name="symbol"]').text for doc in root.findall('.//doc')]
                        data["Gene"].append(gene)
                        data["Symbol"].append(','.join(symbols))
                    elif result_elem is not None and result_elem.get("numFound") == "0":
                        url = f"https://rest.genenames.org/search/prev_symbol/{gene}"
                        response = session.get(url)
                        if response.status_code == 200:
                            root = ET.fromstring(response.content)
                            result_elem = root.find("result")
                            if result_elem is not None and result_elem.get("numFound") != "0":
                                symbol_element = root.find('.//str[@name="symbol"]').text
                                data["Gene"].append(gene)
                                data["Symbol"].append(symbol_element)
                else:
                    print(f"Failed to retrieve data for gene {gene}. Status code:", response.status_code)
            else:
                symbol_element = root.find('.//str[@name="symbol"]').text
                data["Gene"].append(gene)
                data["Symbol"].append(gene)
        else:
            print(f"Failed to retrieve data for gene {gene}. Status code:", response.status_code)
    df = pd.DataFrame(data)
    return df



# def sctype_score(scRNAseqData, scaled=True, gs=None, gs2=None, gene_names_to_uppercase=True, *args, **kwargs):
#     marker_stat = defaultdict(int, {gene: sum(gene in genes for genes in gs.values())
#                                     for gene in set(gene for genes in gs.values() for gene in genes)})
#     marker_sensitivity = pd.DataFrame({'gene_': list(marker_stat.keys()), 'score_marker_sensitivity': list(marker_stat.values())})

#     # Fix for potential divide by zero
#     min_value = 1
#     max_value = len(gs)
#     denominator = max_value - min_value

#     if denominator == 0:
#         marker_sensitivity['score_marker_sensitivity'] = 1.0
#     else:
#         marker_sensitivity['score_marker_sensitivity'] = 1 - (marker_sensitivity['score_marker_sensitivity'] - min_value) / denominator

#     if gene_names_to_uppercase:
#         scRNAseqData.index = scRNAseqData.index.str.upper()

#     names_gs_cp = list(gs.keys())
#     names_gs_2_cp = list(gs2.keys())

#     gs_ = {key: [gene for gene in scRNAseqData.index if gene in gs[key]] for key in gs}
#     gs2_ = {key: [gene for gene in scRNAseqData.index if gene in gs2[key]] for key in gs2}

#     gs__ = dict(zip(names_gs_cp, gs_.values()))
#     gs2__ = dict(zip(names_gs_2_cp, gs2_.values()))

#     cell_markers_genes_score = marker_sensitivity[marker_sensitivity['gene_'].isin(set.union(*map(set, gs__.values())))]

#     if not scaled:
#         Z = scale(scRNAseqData.T).T
#     else:
#         Z = scRNAseqData

#     for _, row in cell_markers_genes_score.iterrows():
#         if row['gene_'] in Z.index:
#             Z.loc[row['gene_']] *= row['score_marker_sensitivity']

#     marker_genes = list(set().union(*gs__.values(), *gs2__.values()))
#     Z = Z.loc[Z.index.intersection(marker_genes)]

#     es = pd.DataFrame(index=gs__.keys(), columns=Z.columns, data=np.zeros((len(gs__), Z.shape[1])))

#     for gss_, genes in gs__.items():
#         for j in range(Z.shape[1]):
#             gs_z = Z.loc[genes, Z.columns[j]] if genes else pd.Series(dtype=np.float64)
#             gz_2 = Z.loc[gs2__.get(gss_, []), Z.columns[j]] * -1 if gs2__ and gss_ in gs2__ else pd.Series(dtype=np.float64)
#             sum_t1 = np.sum(gs_z) / np.sqrt(len(gs_z)) if len(gs_z) > 0 else 0
#             sum_t2 = np.sum(gz_2) / np.sqrt(len(gz_2)) if len(gz_2) > 0 else 0
#             es.loc[gss_, Z.columns[j]] = sum_t1 + sum_t2

#     es = es.dropna(how='all')
#     return es
def sctype_score(scRNAseqData, scaled=True, gs=None, gs2=None, gene_names_to_uppercase=True, *args, **kwargs):
    # Create a dictionary counting how many cell types each gene appears in (for sensitivity penalty)
    marker_stat = defaultdict(int, {
        gene: sum(gene in genes for genes in gs.values())
        for gene in set(gene for genes in gs.values() for gene in genes)
    })

    # Convert to DataFrame: gene_ (marker gene), score_marker_sensitivity (used to weigh rare vs common markers)
    marker_sensitivity = pd.DataFrame({
        'gene_': list(marker_stat.keys()),
        'score_marker_sensitivity': list(marker_stat.values())
    })

    # Normalize marker sensitivity score between 0 and 1 (rare marker = higher weight)
    min_value = 1
    max_value = len(gs)
    denominator = max_value - min_value
    if denominator == 0:
        marker_sensitivity['score_marker_sensitivity'] = 1.0
    else:
        marker_sensitivity['score_marker_sensitivity'] = 1 - (
            (marker_sensitivity['score_marker_sensitivity'] - min_value) / denominator
        )

    # Make gene names uppercase for consistency (since marker sets use uppercase)
    if gene_names_to_uppercase:
        scRNAseqData.index = scRNAseqData.index.str.upper()

    # Get marker gene sets for each cell type
    names_gs_cp = list(gs.keys())       # positive markers
    names_gs_2_cp = list(gs2.keys())    # negative markers

    # Filter marker genes to only those found in the dataset
    gs_ = {key: [gene for gene in scRNAseqData.index if gene in gs[key]] for key in gs}
    gs2_ = {key: [gene for gene in scRNAseqData.index if gene in gs2[key]] for key in gs2}

    # Remap back into dicts (key: cell type, value: list of matched genes)
    gs__ = dict(zip(names_gs_cp, gs_.values()))
    gs2__ = dict(zip(names_gs_2_cp, gs2_.values()))

    # Filter sensitivity scores to only those genes present in filtered sets
    marker_genes_union = set.union(*map(set, gs__.values()))
    cell_markers_genes_score = marker_sensitivity[marker_sensitivity['gene_'].isin(marker_genes_union)]

    # Optionally scale the expression data (row-wise)
    if not scaled:
        from sklearn.preprocessing import scale
        Z = scale(scRNAseqData.T).T  # scale along genes
        Z = pd.DataFrame(Z, index=scRNAseqData.index, columns=scRNAseqData.columns)
    else:
        Z = scRNAseqData.copy()

    # Apply sensitivity scaling to each gene (weighting less common genes higher)
    for _, row in cell_markers_genes_score.iterrows():
        if row['gene_'] in Z.index:
            Z.loc[row['gene_']] *= row['score_marker_sensitivity']

    # Keep only marker genes (both positive and negative)
    marker_genes = list(set().union(*gs__.values(), *gs2__.values()))
    Z = Z.loc[Z.index.intersection(marker_genes)]

    # ✅ New Part: Mark each cell as True/False if any marker gene has non-zero expression
    markerFound = (Z.sum(axis=0) != 0)  # axis=0 means across genes per cell
    markerFound.name = "markerFound"   # nice label for adding to AnnData later

    # Initialize scores matrix (cell type × cells)
    es = pd.DataFrame(index=gs__.keys(), columns=Z.columns, data=np.zeros((len(gs__), Z.shape[1])))

    # Compute enrichment score per cell type and cell
    for gss_, genes in gs__.items():
        for j in range(Z.shape[1]):
            col_name = Z.columns[j]
            gs_z = Z.loc[genes, col_name] if genes else pd.Series(dtype=np.float64)
            gz_2 = Z.loc[gs2__.get(gss_, []), col_name] * -1 if gs2__ and gss_ in gs2__ else pd.Series(dtype=np.float64)
            sum_t1 = np.sum(gs_z) / np.sqrt(len(gs_z)) if len(gs_z) > 0 else 0
            sum_t2 = np.sum(gz_2) / np.sqrt(len(gz_2)) if len(gz_2) > 0 else 0
            es.loc[gss_, col_name] = sum_t1 + sum_t2

    # Remove any all-NaN rows (edge case)
    es = es.dropna(how='all')

    # Return enrichment scores + marker presence boolean per cell
    return es, markerFound

def process_cluster(cluster,adata,es_max,clustering):
    cluster_data = es_max.loc[:, adata.obs.index[adata.obs[clustering] == cluster]]
    es_max_cl = cluster_data.sum(axis=1).sort_values(ascending=False)
    top_scores = es_max_cl.head(10)
    ncells = sum(adata.obs[clustering] == cluster)
    return pd.DataFrame({
        'cluster': cluster,
        'type': top_scores.index,
        'scores': top_scores.values,
        'ncells': ncells
    })