import os
import random
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd
import math
import numba
import time
import pickle
import mygene
import argparse
import joblib
from joblib import numpy_pickle, dump
from pybiomart import Dataset
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse, csr_matrix
from sklearn.utils import sparsefuncs
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from config.config_cdniche import specie_dict
from config.config_cdniche import technology_dict
from config.config_cdniche import MAX_LENGTH
from config.config_cdniche import AUX_TOKEN
from config.config_cdniche import cell_density_bin_dict
from config.config_train_cdniche import config_train



config_train["single_context_length"] = config_train["context_length"]

platform_resolution_um = {
    "XENIUM": 0.2,
    "STARMAP": 0.1,
    "SLIDESEQV2": 10.0,
    "STEREO": 0.5,
    # MERFISH use auto estimation
}

# Specify the radius for each platform (unit can be customized)
platform_radius_map = {
    "STARMAP": 120,
    "MERFISH": 150,
    "SLIDESEQV2": 80,
    "XENIUM": 10,
    "STEREO": 200,
}

def set_seed(seed: int, deterministic: bool = True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # Convert X to float64 if it's not already
    if not np.issubdtype(X.dtype, np.floating):
        X = X.astype(np.float64)

    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


def estimate_resolution(spatial, expected_physical_dist=20.0):
    """
    automatically estimate the spatial resolution (μm / bin) of MERFISH and other platforms
    """
    nbrs = NearestNeighbors(n_neighbors=2).fit(spatial)
    dists, _ = nbrs.kneighbors(spatial)
    return expected_physical_dist / np.mean(dists[:, 1])

def convert_spatial_to_um(adata, platform_name):
    """
    convert adata.obsm["spatial"] to the same unit (μm), and store the result in adata.obsm["spatial_um"]
    """
    if platform_name == "MERFISH":
        factor = estimate_resolution(adata.obsm["spatial"])
    else:
        factor = platform_resolution_um.get(platform_name, 1.0)
    adata.obsm["spatial_um"] = adata.obsm["spatial"] * factor
    return adata

def compute_density_token(adata, radius_um=100, n_bins=5):
    """
    compute the density token for each cell
    """  
    # get spatial coordinates
    if isinstance(adata.obsm["spatial_um"], pd.DataFrame):
        coords = adata.obsm["spatial_um"].to_numpy()
    elif isinstance(adata.obsm["spatial_um"], np.ndarray):
        coords = adata.obsm["spatial_um"]
    else:
        raise TypeError(f"Unsupported type for spatial_um: {type(adata.obsm['spatial_um'])}")
    
    density_tokens = np.zeros(adata.n_obs, dtype=np.int8)
    raw_density = np.zeros(adata.n_obs, dtype=np.float32)

    for sid in adata.obs["slice"].unique():
        idx = adata.obs["slice"] == sid
        coords = adata.obsm["spatial_um"][idx]

        if isinstance(coords, pd.DataFrame):
            coords = coords.to_numpy()

        nbrs = NearestNeighbors(radius=radius_um).fit(coords)
        density = [len(nbrs.radius_neighbors(pt.reshape(1, -1), return_distance=False)[0]) for pt in coords]
        density_log = np.log1p(density)

        # binning
        min_val = max(0, density_log.min())
        max_val = density_log.max()
        if min_val == max_val:
            max_val = min_val + 1e-3  # avoid duplicate bin edges
        bins = np.linspace(min_val, max_val, n_bins + 1)
        token = pd.cut(density_log, bins=bins, labels=False, include_lowest=True)

        if isinstance(token, pd.Series):
            token = token.fillna(0).to_numpy()
        else:
            token = np.nan_to_num(token, nan=0)

        density_tokens[idx] = token
        raw_density[idx] = density

    adata.obs["density_token"] = density_tokens
    # print(f"density_tokens: {density_tokens}")
    return adata, density_tokens


def spatial_expression_imputation_yyw(adata, spatial_key='spatial', expr_key='X',
                                  n_neighbors=20, spatial_weight=0.5,
                                  min_genes=50, min_cells=50, n_pcs=50,
                                  use_raw_counts=True,
                                  chunk_size=1000,
                                  progress_bar=True):
    
    """
    based on spatial and expression similarity to impute gene expression values.

    Args:
        adata: AnnData object
        spatial_key: The key for spatial coordinates in adata.obsm
        expr_key: The key for expression matrix
        n_neighbors: The number of neighbors to use for imputation
        spatial_weight: The weight for spatial similarity (0-1)
        min_genes: The minimum number of genes required when filtering cells
        min_cells: The minimum number of cells required when filtering genes
        n_pcs: The number of principal components to use for PCA
        use_raw_counts: Whether to use raw counts
        chunk_size: The number of cells to process in each chunk
        progress_bar: Whether to show a progress bar
    """

    start_time = time.time()

    print("data preprocessing...")
    adata = adata.copy()

    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_cells)
    sc.pp.filter_genes(adata, min_cells=min_genes)
    # adata = adata[:2000, :].copy()
    if use_raw_counts and adata.raw is None:
        adata.raw = adata.copy()

    # Normalize expression matrix for similarity computation
    norm_adata = adata.copy()
    sc.pp.normalize_total(norm_adata, target_sum=1e4)
    # sc.pp.log1p(norm_adata)

    print("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(norm_adata, flavor='seurat_v3', n_top_genes=2000)
    norm_adata = norm_adata[:, norm_adata.var.highly_variable]

    print("PCA...")
    sc.pp.scale(norm_adata, max_value=10)
    sc.tl.pca(norm_adata, n_comps=n_pcs)

    print("Computing spatial neighbors...")
    spatial_coords = adata.obsm[spatial_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(spatial_coords)
    spatial_distances, spatial_indices = nbrs.kneighbors(spatial_coords)

    # Get expression matrix for imputation
    if use_raw_counts and adata.raw is not None:
        expr_matrix = adata.raw.X
    else:
        expr_matrix = adata.X

    # Ensure expression matrix is in sparse format to save memory
    if not issparse(expr_matrix):
        expr_matrix = csr_matrix(expr_matrix)

    # Initialize imputation result matrix
    imputed_expr = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)

    print("Executing expression imputation...")
    n_chunks = int(np.ceil(adata.n_obs / chunk_size))

    chunk_iter = range(n_chunks)
    if progress_bar:
        chunk_iter = tqdm(chunk_iter, desc="Imputation Progress")

    for chunk_idx in chunk_iter:
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, adata.n_obs)

        for i in range(start_idx, end_idx):
            # Get neighbor information
            neighbors = spatial_indices[i]

            # Compute expression similarity
            expr_similarity = cosine_similarity(
                norm_adata.obsm['X_pca'][i].reshape(1, -1),
                norm_adata.obsm['X_pca'][neighbors]
            ).flatten()

            # Compute spatial similarity (using Gaussian kernel)
            spatial_similarity = np.exp(-spatial_distances[i] ** 2 / (2 * np.mean(spatial_distances[i]) ** 2))

            # Normalize weights
            expr_similarity = expr_similarity / np.max(expr_similarity)
            spatial_similarity = spatial_similarity / np.max(spatial_similarity)

            # Combine weights
            weights = (1 - spatial_weight) * expr_similarity + spatial_weight * spatial_similarity
            weights = weights / np.sum(weights)

            # Extract neighbor expression values and compute weighted average
            neighbor_expr = expr_matrix[neighbors].toarray()
            imputed_expr[i] = np.average(neighbor_expr, axis=0, weights=weights)

            # Ensure non-negativity
            imputed_expr[i] = np.maximum(imputed_expr[i], 0)
    # Save imputed expression back to AnnData
    adata.X = csr_matrix(imputed_expr)

    print(f"Done! Processed {adata.n_obs} cells")
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return adata


def spatial_expression_imputation(adata, spatial_key='spatial', expr_key='X',
                                  n_neighbors=20, spatial_weight=0.5,
                                  min_genes=50, min_cells=50, n_pcs=50,
                                  use_raw_counts=True,
                                  chunk_size=1000,
                                  progress_bar=True):
    """
    Spatial gene expression imputation using spatial and expression similarity.
    """
    start_time = time.time()
    print("Preprocessing...")
    adata = adata.copy()

    # Basic filtering
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # Save raw counts
    if use_raw_counts and adata.raw is None:
        adata.raw = adata.copy()

    # Normalization for similarity calculation
    norm_adata = adata.copy()
    sc.pp.normalize_total(norm_adata, target_sum=1e4)

    # HVG selection
    sc.pp.highly_variable_genes(norm_adata, flavor='seurat_v3', n_top_genes=2000)
    norm_adata = norm_adata[:, norm_adata.var.highly_variable]

    # PCA
    sc.pp.scale(norm_adata, max_value=10)
    sc.tl.pca(norm_adata, n_comps=n_pcs)

    # Compute spatial neighbors
    spatial_coords = adata.obsm[spatial_key]
    mask = np.isfinite(spatial_coords).all(axis=1)
    adata = adata[mask].copy()
    spatial_coords = adata.obsm[spatial_key]

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(spatial_coords)
    spatial_distances, spatial_indices = nbrs.kneighbors(spatial_coords)

    # Expression matrix
    expr_matrix = adata.raw.X if (use_raw_counts and adata.raw is not None) else adata.X
    if not issparse(expr_matrix):
        expr_matrix = csr_matrix(expr_matrix)

    # Init output
    imputed_expr = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)

    print("Imputing...")
    n_chunks = int(np.ceil(adata.n_obs / chunk_size))
    chunk_iter = range(n_chunks)
    if progress_bar:
        chunk_iter = tqdm(chunk_iter, desc="imputation")

    for chunk_idx in chunk_iter:
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, adata.n_obs)

        for i in range(start_idx, end_idx):
            neighbors = spatial_indices[i]

            # Expression similarity
            expr_similarity = cosine_similarity(
                norm_adata.obsm['X_pca'][i].reshape(1, -1),
                norm_adata.obsm['X_pca'][neighbors]
            ).flatten()
            if np.max(expr_similarity) > 0:
                expr_similarity /= np.max(expr_similarity)

            # Spatial similarity
            spatial_similarity = np.exp(-spatial_distances[i] ** 2 / (2 * np.mean(spatial_distances[i]) ** 2))
            if np.max(spatial_similarity) > 0:
                spatial_similarity /= np.max(spatial_similarity)

            # Combine weights
            weights = (1 - spatial_weight) * expr_similarity + spatial_weight * spatial_similarity
            weights /= np.sum(weights)

            # Weighted average
            neighbor_expr = expr_matrix[neighbors].toarray()
            imputed_expr[i] = np.average(neighbor_expr, axis=0, weights=weights)
            imputed_expr[i] = np.maximum(imputed_expr[i], 0)

    # Save back
    adata.X = csr_matrix(imputed_expr)

    print(f"Finished {adata.n_obs} cells")
    print(f"Time: {time.time() - start_time:.2f} sec")

    return adata

def ensure_ensembl_ids_raw(adata, species="hsapiens"):
    """
    Ensure gene IDs are Ensembl.
    If current var_names are symbols, convert them to Ensembl IDs.

    """
    print(f"[INFO] Converting gene symbols to Ensembl IDs for {species} ...")

    # Get biomart dataset
    dataset = Dataset(name=f"{species}_gene_ensembl",
                      host="http://www.ensembl.org")

    mapping = dataset.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    mapping = mapping.dropna().drop_duplicates()
    symbol_to_ensembl = dict(zip(mapping['Gene name'], mapping['Gene stable ID']))

    # Map
    adata.var["ensembl_id"] = adata.var_names.map(symbol_to_ensembl)

    # Drop genes without mapping
    mask = adata.var["ensembl_id"].notna()
    adata = adata[:, mask].copy()
    adata.var_names = adata.var["ensembl_id"]

    print(f"[INFO] Converted {mask.sum()} / {len(mask)} genes to Ensembl IDs.")

    return adata


def ensure_ensembl_ids(adata, species="human"):
    """
    Ensure gene IDs are Ensembl IDs.
    Input species: "human", "mouse", "macaque", "marmoset"
    Priority: local CSV -> BioMart -> MyGene.info
    """
    print(f"[INFO] Converting gene symbols to Ensembl IDs for {species} ...")
    symbol_to_ensembl = {}

    # === Step 1: Try local mapping ===
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    local_path = os.path.join(base_dir, "prior_knowledge", "id_mapping", f"{species}_symbol2ensembl.csv")

    if os.path.exists(local_path):
        print(f"[INFO] Using local mapping file: {local_path}")
        df = pd.read_csv(local_path)
        symbol_to_ensembl = dict(zip(df["gene_symbol"], df["ensembl_id"]))
    else:
        print(f"[WARN] Local mapping not found: {local_path}")

        # === Step 2: Try BioMart ===
        biomart_map = {
            "human": "hsapiens",
            "mouse": "mmusculus",
            "macaque": "mfascicularis",
            "marmoset": "cjacchus",
        }
        try:
            bm_species = biomart_map[species]
            dataset = Dataset(name=f"{bm_species}_gene_ensembl", host="http://www.ensembl.org")
            mapping = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
            mapping = mapping.dropna().drop_duplicates()
            symbol_to_ensembl = dict(zip(mapping["Gene name"], mapping["Gene stable ID"]))
            print(f"[INFO] BioMart mapping retrieved for {species}.")
        except Exception as e:
            print(f"[WARN] BioMart failed: {e}. Falling back to MyGene.info ...")

            # === Step 3: MyGene fallback ===
            mg = mygene.MyGeneInfo()
            symbols = list(adata.var_names)
            mygene_map = {
                "human": "human",
                "mouse": "mouse",
                "macaque": 9541,
                "marmoset": 9483,
            }
            species_query = mygene_map.get(species, "human")
            print(f"[INFO] MyGene species query = {species_query}")
            out = mg.querymany(symbols, scopes="symbol", fields="ensembl.gene", species=species_query)
            for rec in out:
                if rec.get("notfound"):
                    continue
                ensg = rec.get("ensembl")
                if isinstance(ensg, list):
                    ensg = ensg[0].get("gene")
                elif isinstance(ensg, dict):
                    ensg = ensg.get("gene")
                if ensg:
                    symbol_to_ensembl[rec["query"]] = ensg
            print(f"[INFO] MyGene mapping retrieved for {species}.")

    # === Map into AnnData ===
    adata.var["ensembl_id"] = adata.var_names.map(symbol_to_ensembl)
    mask = adata.var["ensembl_id"].notna()
    adata = adata[:, mask].copy()
    adata.var_names = adata.var["ensembl_id"]
    adata.var.drop(columns=["ensembl_id"], inplace=True)

    print(f"[INFO] Converted {mask.sum()} / {len(mask)} genes to Ensembl IDs.")
    return adata


def align_adata_and_mean_matrix(
    adata: ad.AnnData,
    gene_dict: ad.AnnData,
    mean_matrix: np.ndarray,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Align adata and mean_matrix to gene_dict:
    - Keep only shared genes
    - Reorder to match gene_dict.var.index
    - Replace adata.var with gene_dict.var
    - Remove any genes whose mean vector contains zero
    """
    # Get shared genes
    shared_genes = gene_dict.var.index.intersection(adata.var.index)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes found between adata and gene_dict!")

    # Sort genes to match gene_dict order
    ordered_shared_genes = [g for g in gene_dict.var.index if g in shared_genes]

    # Align adata
    adata = adata[:, ordered_shared_genes].copy()
    adata.var = gene_dict.var.loc[ordered_shared_genes].copy()

    # Align mean matrix
    gene_indices = [gene_dict.var.index.get_loc(g) for g in ordered_shared_genes]
    mean_matrix_aligned = mean_matrix[gene_indices]
    
    # Ensure mean_matrix is 2D
    if mean_matrix_aligned.ndim == 1:
        mean_matrix_aligned = mean_matrix_aligned.reshape(-1, 1)

    # Remove genes where mean vector contains 0
    nonzero_mask = ~(mean_matrix_aligned == 0).any(axis=1)
    if nonzero_mask.sum() == 0:
        raise ValueError("All aligned mean vectors contain zero! Cannot proceed.")
    
    adata = adata[:, nonzero_mask].copy()
    mean_matrix_aligned = mean_matrix_aligned[nonzero_mask]

    return adata, mean_matrix_aligned


def compute_deviation_bin(adata_output, n_neighbors=50, n_bins=5):
    assert "x" in adata_output.obs.columns and "y" in adata_output.obs.columns, "Spatial coordinates 'x', 'y' not found in .obs"
    assert "brain_region" in adata_output.obs.columns and "brain_region_main" in adata_output.obs.columns, "Missing region annotations"
    adata_output.obs['slice_brain_area'] = adata_output.obs.apply(
        lambda x: f"{x['brain_region']}_{x['brain_region_main']}", axis=1
    )
    adata_output.obsm['deviation_bin'] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.int8)
    adata_output.obsm['neighbor_gene_distribution'] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.float32)
    X_raw = adata_output.X.copy()
    adata_output.X = sf_normalize(adata_output.X)
    group_results = {}

    for idx in adata_output.obs["slice_brain_area"].unique():
        knn_sample_obs = adata_output.obs.loc[adata_output.obs["slice_brain_area"] == idx, :]
        num_sample = min(knn_sample_obs.shape[0], n_neighbors)

        nbrs = NearestNeighbors(n_neighbors=num_sample, algorithm='ball_tree', n_jobs=16).fit(
            knn_sample_obs[["x", "y"]]
        )
        _, indices = nbrs.kneighbors(knn_sample_obs[["x", "y"]], n_neighbors=num_sample)
        index_array = knn_sample_obs.index.to_numpy()

        neighbor_gene_distribution = []
        for neighbor_idx_list in indices:
            neighbor_expr = adata_output.X[neighbor_idx_list]
            neighbor_gene_distribution.append(np.asarray(neighbor_expr.mean(axis=0)).flatten())
        neighbor_gene_distribution_matrix = np.stack(neighbor_gene_distribution, axis=0)
        
        cell_expr_matrix = adata_output.X[index_array]
        if hasattr(cell_expr_matrix, "todense"):
            cell_expr_matrix = cell_expr_matrix.todense()
        cell_expr_matrix = np.asarray(cell_expr_matrix)
        
        deviation_matrix = cell_expr_matrix - neighbor_gene_distribution_matrix
        mask_all_zero = (cell_expr_matrix == 0) & (neighbor_gene_distribution_matrix == 0)
        deviation_matrix[mask_all_zero] = np.nan

        group_results[idx] = {
            "index_array": index_array,
            "neighbor_gene_distribution_matrix": neighbor_gene_distribution_matrix,
            "deviation_matrix": deviation_matrix
        }

    # all deviation matrix
    full_deviation = np.full((adata_output.n_obs, adata_output.n_vars), np.nan, dtype=np.float32)
    for idx, result in group_results.items():
        index_array = result["index_array"]
        full_deviation[index_array] = result["deviation_matrix"]
        adata_output.obsm['neighbor_gene_distribution'][index_array] = result["neighbor_gene_distribution_matrix"]

    # calculate quantiles for each gene
    deviation_bin = np.zeros_like(full_deviation, dtype=np.int8)
    for j in range(full_deviation.shape[1]):
        col = full_deviation[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue  # skip if no valid values
        quantiles = np.quantile(col[valid], np.linspace(0, 1, n_bins + 1)[1:-1])

        for i in range(full_deviation.shape[0]):
            val = full_deviation[i, j]
            if np.isnan(val):
                deviation_bin[i, j] = 0  # 0 no deviation signal
            else:
                for b, q in enumerate(quantiles):
                    if val <= q:
                        deviation_bin[i, j] = b + 1
                        break
                else:
                    deviation_bin[i, j] = n_bins

    adata_output.obsm['deviation_bin'] = deviation_bin
    adata_output.X = X_raw
    return adata_output


def compute_deviation_bin_rapid(adata_output, n_neighbors=50, n_bins=5, batch_size=2000):
    assert "x" in adata_output.obs.columns and "y" in adata_output.obs.columns, "Spatial coordinates 'x', 'y' not found in .obs"
    assert "brain_region" in adata_output.obs.columns and "brain_region_main" in adata_output.obs.columns, "Missing region annotations"

    adata_output.obs['slice_brain_area'] = adata_output.obs.apply(
        lambda x: f"{x['brain_region']}_{x['brain_region_main']}", axis=1
    )

    adata_output.obsm['deviation_bin'] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.int8)
    adata_output.obsm['neighbor_gene_distribution'] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.float32)

    X_raw = adata_output.X.copy()
    adata_output.X = sf_normalize(adata_output.X)
    X = adata_output.X.tocsr() if scipy.sparse.issparse(adata_output.X) else np.asarray(adata_output.X)

    full_deviation = np.full((adata_output.n_obs, adata_output.n_vars), np.nan, dtype=np.float32)

    for idx in adata_output.obs["slice_brain_area"].unique():
        knn_sample_obs = adata_output.obs.loc[adata_output.obs["slice_brain_area"] == idx]
        coords = knn_sample_obs[["x", "y"]].values
        index_array = adata_output.obs.index.get_indexer(knn_sample_obs.index)
        num_sample = min(len(index_array), n_neighbors)

        nbrs = NearestNeighbors(n_neighbors=num_sample, algorithm='ball_tree', n_jobs=4).fit(coords)
        _, all_indices = nbrs.kneighbors(coords)

        # batch-wise processing to avoid memory issues
        for start in range(0, len(index_array), batch_size):
            end = min(start + batch_size, len(index_array))
            batch_idx = index_array[start:end]
            batch_neighbors = all_indices[start:end]

            # get the mean expression of neighbors
            if scipy.sparse.issparse(X):
                batch_neighbor_expr = np.stack([
                    X[ni].mean(axis=0).A1 for ni in batch_neighbors
                ])
                batch_cell_expr = X[batch_idx].toarray()
            else:
                batch_neighbor_expr = X[batch_neighbors].mean(axis=1)
                batch_cell_expr = X[batch_idx]

            deviation = batch_cell_expr - batch_neighbor_expr
            zero_mask = (batch_cell_expr == 0) & (batch_neighbor_expr == 0)
            deviation[zero_mask] = np.nan

            full_deviation[batch_idx] = deviation
            adata_output.obsm['neighbor_gene_distribution'][batch_idx] = batch_neighbor_expr

    # calculate quantiles for each gene (0 = no deviation signal)
    deviation_bin = np.zeros_like(full_deviation, dtype=np.int8)
    for j in range(full_deviation.shape[1]):
        col = full_deviation[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        quantiles = np.quantile(col[valid], np.linspace(0, 1, n_bins + 1)[1:-1])
        deviation_bin[valid, j] = np.digitize(col[valid], quantiles, right=True) + 1

    adata_output.obsm['deviation_bin'] = deviation_bin
    adata_output.X = X_raw
    return adata_output


def compute_deviation_bin_rapid_v2(adata_output, n_neighbors=50, n_bins=5, batch_size=2000,
                                   use_abs=True):  # for virtual perturbation task
    assert "x" in adata_output.obs.columns and "y" in adata_output.obs.columns, "Spatial coordinates 'x', 'y' not found in .obs"
    assert "brain_region" in adata_output.obs.columns and "brain_region_main" in adata_output.obs.columns, "Missing region annotations"

    adata_output.obs['slice_brain_area'] = adata_output.obs.apply(
        lambda x: f"{x['brain_region']}_{x['brain_region_main']}", axis=1
    )

    adata_output.obsm['deviation_bin'] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.int8)
    adata_output.obsm['neighbor_gene_distribution'] = np.zeros((adata_output.n_obs, adata_output.n_vars),
                                                               dtype=np.float32)

    X_raw = adata_output.X.copy()
    adata_output.X = sf_normalize(adata_output.X)
    X = adata_output.X.tocsr() if scipy.sparse.issparse(adata_output.X) else np.asarray(adata_output.X)

    full_deviation = np.full((adata_output.n_obs, adata_output.n_vars), np.nan, dtype=np.float32)

    for idx in adata_output.obs["slice_brain_area"].unique():
        knn_sample_obs = adata_output.obs.loc[adata_output.obs["slice_brain_area"] == idx]
        coords = knn_sample_obs[["x", "y"]].values
        index_array = adata_output.obs.index.get_indexer(knn_sample_obs.index)
        num_sample = min(len(index_array), n_neighbors)

        nbrs = NearestNeighbors(n_neighbors=num_sample, algorithm='ball_tree', n_jobs=4).fit(coords)
        _, all_indices = nbrs.kneighbors(coords)

        # batch-wise processing to avoid memory issues
        for start in range(0, len(index_array), batch_size):
            end = min(start + batch_size, len(index_array))
            batch_idx = index_array[start:end]
            batch_neighbors = all_indices[start:end]

            # get the mean expression of neighbors
            if scipy.sparse.issparse(X):
                batch_neighbor_expr = np.stack([
                    X[ni].mean(axis=0).A1 for ni in batch_neighbors
                ])
                batch_cell_expr = X[batch_idx].toarray()
            else:
                batch_neighbor_expr = X[batch_neighbors].mean(axis=1)
                batch_cell_expr = X[batch_idx]
            if use_abs:
                deviation = np.abs(batch_cell_expr - batch_neighbor_expr)
            else:
                deviation = batch_cell_expr - batch_neighbor_expr

            zero_threshold = 1e-4
            both_low_mask = (batch_cell_expr < zero_threshold) & (batch_neighbor_expr < zero_threshold)

            deviation[both_low_mask] = np.nan

            full_deviation[batch_idx] = deviation
            adata_output.obsm['neighbor_gene_distribution'][batch_idx] = batch_neighbor_expr
            batch_neighbor_expr[np.abs(batch_neighbor_expr) < 1e-6] = 0
            adata_output.obsm['neighbor_gene_distribution'][batch_idx] = batch_neighbor_expr
    # calculate quantiles for each gene (0 = no deviation signal)
    deviation_bin = np.zeros_like(full_deviation, dtype=np.int8)
    for j in range(full_deviation.shape[1]):
        col = full_deviation[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        quantiles = np.quantile(col[valid], np.linspace(0, 1, n_bins + 1)[1:-1])
        deviation_bin[valid, j] = np.digitize(col[valid], quantiles, right=True) + 1

    adata_output.obsm['deviation_bin'] = deviation_bin
    adata_output.X = X_raw
    return adata_output

@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(
    x: np.array,
    gene_id:np.array,
    gene_connect_comp: np.array,
    rna_type_id: np.array,
    deviation_bin: np.array,
    max_seq_len: int = -1,
    aux_tokens: int = 30
):
    n_cells, n_genes = x.shape
    seq_len = max_seq_len if max_seq_len > 0 else n_genes

    # Initialize output arrays (fixed-length, zero-padded)
    scores_final = np.zeros((n_cells, seq_len), dtype=np.int32)
    scores_connect_comp_final = np.zeros((n_cells, seq_len), dtype=np.int32)
    scores_rna_type_id_final = np.zeros((n_cells, seq_len), dtype=np.int32)
    scores_deviation_bin_final = np.zeros((n_cells, seq_len), dtype=np.int32)
    exp_final = np.zeros((n_cells, seq_len), dtype=np.float32)
    for i, cell in enumerate(x):
        # Ensure cell is a one-dimensional array and handle sparse matrices
        if hasattr(cell, 'todense'):
            cell = cell.todense()
        cell = np.asarray(cell).flatten()

        # Select non-zero genes and sort by expression
        nonzero_mask = np.nonzero(cell)[0]
        if len(nonzero_mask) == 0:
            continue  # skip empty cells
        
        # Get expression values and sort
        expr_values = cell[nonzero_mask]
        sorted_idx = np.argsort(-expr_values)
        real_seq_len = min(seq_len, len(nonzero_mask))
        sorted_indices = nonzero_mask[sorted_idx[:real_seq_len]]

        # Lookup and offset auxiliary info
        # gene_ids = sorted_indices + aux_tokens
        gene_connect = gene_connect_comp[sorted_indices] + 1
        rna_type = rna_type_id[sorted_indices] + 1
        dev_bins = deviation_bin[i][sorted_indices]  # already tokenized
        gene_ids = gene_id[sorted_indices] + aux_tokens

        # Assign to padded arrays
        scores_final[i, :real_seq_len] = gene_ids
        scores_connect_comp_final[i, :real_seq_len] = gene_connect
        scores_rna_type_id_final[i, :real_seq_len] = rna_type
        scores_deviation_bin_final[i, :real_seq_len] = dev_bins
        exp_final[i, :real_seq_len] = expr_values[sorted_idx[:real_seq_len]]

    return (
        scores_final,                    # gene token ids (offset by aux_tokens)
        scores_connect_comp_final,       # homology group token
        scores_rna_type_id_final,        # gene type token
        scores_deviation_bin_final,      # deviation bin token (0 = NaN, 1~5 = binned)
        exp_final                        # expression values
    )

def tokenize_data(
        x: np.array,
        gene_id:np.array,
        gene_connect_comp: np.array,
        gene_type_id: np.array,
        deviation_bin: np.array,
        mean_matrix: np.array,
        max_seq_len: int,
        aux_token_len: int
):
    """Tokenize the input gene vector to a vector of 32-bit integers."""

    x = np.nan_to_num(x)  # is NaN values, fill with 0s
    x = sf_normalize(x)
    out = x / mean_matrix.reshape((1, -1))
    out = np.asarray(out)
    scores_final, scores_connect_comp_final, scores_rna_type_id, scores_deviation_bin, exp_final = _sub_tokenize_data(
        out, gene_id,gene_connect_comp, gene_type_id, deviation_bin, max_seq_len, aux_token_len
    )
    return (
        scores_final.astype(np.int32), 
        scores_connect_comp_final.astype(np.int32), 
        scores_rna_type_id.astype(np.int32), 
        scores_deviation_bin.astype(np.int32),
        exp_final.astype(np.float32)
    )

def convert_dtypes_for_parquet(df):
    """Convert the data types of the DataFrame to ensure they can be serialized to parquet format"""
    for col in df.columns:
        # Handle category type
        if df[col].dtype.name == 'category':
            df[col] = df[col].astype(str)
        # Handle numpy bool type
        elif df[col].dtype.name == 'bool':
            df[col] = df[col].astype('boolean')
        # Handle numpy numeric type
        elif 'int' in df[col].dtype.name:
            df[col] = df[col].astype('int64')
        elif 'float' in df[col].dtype.name:
            df[col] = df[col].astype('float64')
    return df

def standardize_adata_obs(adata: ad.AnnData, gene_dict: ad.AnnData, mean_matrix: np.array, specie: str, assay: str, cell_density: bool = True) -> ad.AnnData:
    """
    Standardize the observation (obs) attributes of an AnnData object and align it with a gene dictionary.
    
    Args:
        adata: Input AnnData object to standardize
        gene_dict: Reference gene dictionary AnnData object
        specie: Species identifier
        assay: Assay type identifier
        density: Whether density token is included
    
    Returns:
        Standardized AnnData object with aligned genes and normalized observations
    """
    # Add missing columns with default values
    if 'slice' not in adata.obs.columns:
        adata.obs['slice'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='slice')
    if 'region' not in adata.obs.columns:
        adata.obs['region'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='region')
    if 'brain_region' not in adata.obs.columns:
        adata.obs['brain_region'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='brain_region')    
    
    # Add spatial coordinates if not present
        # Add spatial coordinates if not present
    if assay == "snrna":
        adata.obs["x"] = np.zeros(adata.shape[0])
        adata.obs["y"] = np.zeros(adata.shape[0])
    else:
        if 'x' not in adata.obs.columns:
            if isinstance(adata.obsm["spatial"], pd.DataFrame):
                adata.obs["x"] = adata.obsm["spatial"].to_numpy()[:, 0]
                adata.obs["y"] = adata.obsm["spatial"].to_numpy()[:, 1]
            elif isinstance(adata.obsm["spatial"], np.ndarray):
                adata.obs["x"] = adata.obsm["spatial"][:, 0]
                adata.obs["y"] = adata.obsm["spatial"][:, 1]
            else:
                raise TypeError(f"Unsupported type for adata.obsm['spatial']: {type(adata.obsm['spatial'])}")

    # Define and filter columns to keep
    keys_to_keep = ['brain_region', 'x', 'y', 'original_index', 'slice', "cell_label", "region"]
    if isinstance(cell_density, bool) and cell_density:
        print("Computing cell density...")
        import time
        time0 = time.time()
        # modify the density analysis result save path
        platform_name = assay.upper()
        adata = convert_spatial_to_um(adata, platform_name)
        radius = platform_radius_map.get(platform_name, 200)
        adata, _ = compute_density_token(adata, radius_um=radius, n_bins=5) # radius_um=100
        adata.obs.replace({'density_token': {i: cell_density_bin_dict[f"cell_density_bin_{i}"] for i in range(5)}}, inplace=True)
        keys_to_keep.append("density_token")
        time1 = time.time()
        print("compute_density_token time: ", (time1 - time0) / 60, "min")
    columns_to_delete = [col for col in adata.obs.columns if col not in keys_to_keep]
    adata.obs.drop(columns=columns_to_delete, inplace=True)
    # Clear unnecessary data
    adata.uns = {}
    adata.obsm = {}
    
    # Align with gene dictionary and filter invalid coordinates
    adata_output, mean_matrix_aligned = align_adata_and_mean_matrix(adata, gene_dict, mean_matrix)
    adata_output.obs = adata_output.obs.reset_index(drop=True)
    
    adata_output.obs['specie'] = specie
    adata_output.obs['assay'] = assay
    
    # Replace values with dictionary mappings
    adata_output.obs.replace({'specie': specie_dict}, inplace=True)
    adata_output.obs.replace({'assay': technology_dict}, inplace=True)
    
    return adata_output, mean_matrix_aligned

def tokenization_h5ad(adata_path, gene_dict_path, mean_path, specie=None, assay=None, output_path=None, anno=False,
                      split="train", label=False, cell_density=True, gene_niche=True,
                      use_hvg=False, n_hvg=1000, min_genes=3, min_cells=3, spatial_imputation=False,
                      use_dev_abs=False):
    """
    Brainbeacon input tokenization
    Conver H5ad to Joblib
    """
    assert gene_dict_path, "Input `gene_dict_path` cannot be empty."
    gene_dict = sc.read_h5ad(gene_dict_path)
    mean_matrix = np.load(mean_path)
    print(f"path to process: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    if assay == "snrna":
        cell_density = False  # snRNA-seq does not have spatial coordinates
        gene_niche = False
        spatial_imputation = False
    if assay == "stereo" and spatial_imputation:
        adata = spatial_expression_imputation(
            adata,
            spatial_key='spatial',
            n_neighbors=50,
        )
    print(f"before quality control adata shape: {adata.shape}")
    # Filter genes and cells
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if use_hvg:
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        # sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat")
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat_v3")
        # sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat_v3", batch_key="slice")
        adata = adata[:, tmp.var.highly_variable].copy()
        print(f"After HVG ({n_hvg}) selection: {adata.shape}")

    if anno:
        adata = adata[adata.obs["split"] == split]
        if label:
            adata.obs['cell_label'] = adata.obs[label]
            # 保存 LabelEncoder 到 `output_path`
            assert output_path, "Output path must be provided."
            parent_output_path = os.path.dirname(output_path)  # 获取上一级目录
            le_path = os.path.join(parent_output_path, "label_encoder.pkl")  # 在父目录存储
            # 如果 label_encoder.pkl 存在，则加载，否则重新训练
            if os.path.exists(le_path):
                with open(le_path, "rb") as f:
                    le = pickle.load(f)
                cell_labels_int = le.transform(adata.obs['cell_label'])  # 只 transform
                print(f"Loaded existing LabelEncoder from {le_path}")
            else:
                le = LabelEncoder()
                cell_labels_int = le.fit_transform(adata.obs['cell_label'])
                with open(le_path, "wb") as f:
                    pickle.dump(le, f)
                print(f"Trained and saved new LabelEncoder to {le_path}")    

            cell_labels_tensor = torch.tensor(cell_labels_int)
            adata.obs['cell_label'] = cell_labels_tensor
    else:
        adata.obs['cell_label'] = pd.Series(np.zeros(adata.shape[0], dtype=np.int64), index=adata.obs.index,
                                    name='cell_label')
    adata.obs["original_index"] = adata.obs.index
    # ensure adata.var.index is Ensembl IDs
    is_ensembl = adata.var.index.str.startswith(('ENS'))
    if not is_ensembl.all():
        raise ValueError(
            "adata must contain Ensembl IDs in `var.index`. "
            "Please convert gene names to Ensembl IDs before proceeding."
        )
    adata_output, mean_matrix = standardize_adata_obs(adata, gene_dict, mean_matrix, specie, assay, cell_density)

    # No brain_region
    if use_dev_abs:
        adata_output.obs['brain_region'] = adata_output.obs["slice"] 
        adata_output.obs['brain_region_main'] = adata_output.obs["slice"] # user can change the cell label to other annotation
    else:
        adata_output.obs['brain_region'] = adata_output.obs["slice"] 
        adata_output.obs['brain_region_main'] = adata_output.obs["cell_label"] # user can change the cell label to other annotation

    if gene_niche:
        adata_output = (
            compute_deviation_bin_rapid_v2(adata_output, n_neighbors=50, n_bins=5)
            if use_dev_abs
            else compute_deviation_bin_rapid(adata_output, n_neighbors=50, n_bins=5)
        )

    else:
        adata_output.obsm["deviation_bin"] = np.zeros((adata_output.shape[0], adata_output.shape[1]), dtype=np.int8)
        adata_output.obs["density_token"] = np.zeros((adata_output.shape[0]), dtype=np.int8)

    obs_adata_output = adata_output.obs
    N_BATCHES = math.ceil(obs_adata_output.shape[0] / 10_000)
    batch_indices = np.array_split(obs_adata_output.index, N_BATCHES)
    chunk_len = len(batch_indices[0])

    obs_adata_output = obs_adata_output.reset_index().rename(columns={'index': 'idx'})
    obs_adata_output['idx'] = obs_adata_output['idx'].astype('i8')

    for batch in tqdm(range(N_BATCHES), desc="Processing data batches"): 
        X_chunk = adata_output.X[batch * chunk_len:chunk_len * (batch + 1)]
        if issparse(X_chunk):
            X_chunk = X_chunk.todense()
        obs_tokens = obs_adata_output.iloc[batch * chunk_len:chunk_len * (batch + 1)].copy()
        tokenized, tokenized_connect_comp, tokenized_rna_type, tokenized_deviation_bin, tokenized_exp = tokenize_data(
            x=X_chunk,
            gene_connect_comp=adata_output.var["homo_connect_id"].values,
            gene_id=adata_output.var["gene_id"].values,
            gene_type_id=adata_output.var["Gene_type_id"].values,
            deviation_bin=adata_output.obsm["deviation_bin"],
            mean_matrix=mean_matrix,
            max_seq_len=MAX_LENGTH,
            aux_token_len=AUX_TOKEN,
        )
    
        available_columns = []
        for col in ['brain_region', 'brain_region_main', 'x', 'y', 'assay', 'specie', 'idx', "original_index", "cell_label", "density_token"]:
            if col in obs_tokens.columns:
                available_columns.append(col)
        
        obs_tokens = obs_tokens[available_columns]        
        # concatenate dataframes
        obs_tokens['X'] = [tokenized[i, :] for i in range(tokenized.shape[0])]
        obs_tokens['X_connect_comp'] = [tokenized_connect_comp[i, :] for i in
                                        range(tokenized_connect_comp.shape[0])]
        obs_tokens['X_rna_type'] = [tokenized_rna_type[i, :] for i in range(tokenized_rna_type.shape[0])]
        obs_tokens['X_deviation_bin'] = [tokenized_deviation_bin[i, :] for i in range(tokenized_deviation_bin.shape[0])]
        obs_tokens['X_exp'] = [tokenized_exp[i, :] for i in range(tokenized_exp.shape[0])]
        if anno:
            obs_tokens = obs_tokens.sample(frac=1)
        obs_tokens = convert_dtypes_for_parquet(obs_tokens)
        # Convert pandas DataFrame to pyarrow Table
        table = pa.Table.from_pandas(obs_tokens)
        
        pq.write_table(
            table, os.path.join(output_path, f"tokens-{batch:04d}.parquet"),
            row_group_size=1024
        )
    return output_path

def split_iter(a: list, n: int):
    """Pack a dataset (array of samples) into an array of batches"""
    q = math.ceil(len(a) / n) 

    for i in range(q - 1):  
        yield a[i * n:(i + 1) * n]

    # Process the last batch to avoid empty batch
    last_batch = a[(q - 1) * n:]
    if isinstance(last_batch, np.ndarray):  # 
        if last_batch.size > 0:
            yield last_batch
    else:  # 
        if len(last_batch) > 0:
            yield last_batch

def batches(data, batch_size=36):
    return list(split_iter(data, batch_size))

def do_masking(adata, p, n_tokens):
    padding_token = 1
    indices = torch.as_tensor(adata.obsm["X"], dtype=torch.long)
    indices = torch.where(indices == 0, torch.as_tensor(padding_token, dtype=torch.long), indices)
    adata.obsm["X"] = indices.numpy()

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p)  # mask indices with probability p
    mask = torch.where(indices > config_train['n_aux'], mask, torch.ones_like(mask))

    masked_indices = indices * mask  # masked_indices

    masked_indices = torch.where(indices != padding_token, masked_indices, indices)
    mask = torch.where(indices == padding_token, torch.as_tensor(padding_token, dtype=torch.long), mask)
    random_tokens = torch.randint(config_train['n_aux'] + 1, n_tokens + config_train['n_aux'], size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, random_tokens,
                                 masked_indices) 
    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens,
                                 masked_indices) 
    adata.obsm['masked_indices'] = masked_indices.numpy()
    adata.obsm['mask'] = mask.numpy()

    attention_mask = (masked_indices == padding_token)
    adata.obsm['attention_mask'] = attention_mask.type(torch.bool).numpy()

    return adata

def process_parquet(input_file, output_path):
    """
    Reads a .parquet file and converts it to AnnData with required keys.
    """
    if os.path.basename(input_file) == "tokens-0000.parquet":
        print(f"Begin processing: {input_file}")
    

    table = pq.read_table(input_file)
    if os.path.basename(input_file) == "tokens-0000.parquet":
        print(f"Table shape from parquet = {table.num_rows}")

    required_obs_cols = {
        "brain_region": False,
        "brain_region_main": False,
        "x": False,
        "y": False,
        "assay": False,
        "specie": False,
        "idx": False,
        "original_index": False,
        "cell_label": False,
        "density_token": False
    }
    
    # 3) Check the existence of all columns at once
    for col in table.column_names:
        if col in required_obs_cols:
            required_obs_cols[col] = True
    

    obs_cols = [col for col, exists in required_obs_cols.items() if exists]
    obs_df = table.select(obs_cols).to_pandas() if obs_cols else None
    

    if "X" not in table.column_names:
        raise ValueError("No 'X' column in parquet; cannot proceed.")

    X_stack = table["X"].to_numpy()
    
    if len(X_stack.shape) == 1:
        X_stack = np.vstack(X_stack)  
    
    adata = ad.AnnData(
        X=X_stack,
        obs=obs_df
    )

    data_connect_comp_key = 'X_connect_comp'
    data_rna_type_key = 'X_rna_type'
    data_neighbor_gene_distribution_key = 'X_deviation_bin'
    data_exp_key = 'X_exp'
    if data_connect_comp_key not in table.column_names:
        raise ValueError(f"No '{data_connect_comp_key}' in parquet.")
    if data_rna_type_key not in table.column_names:
        raise ValueError(f"No '{data_rna_type_key}' in parquet.")
    if data_neighbor_gene_distribution_key not in table.column_names:
        raise ValueError(f"No '{data_neighbor_gene_distribution_key}' in parquet.")
    if data_exp_key not in table.column_names:
        raise ValueError(f"No '{data_exp_key}' in parquet.")
    adata.obsm[data_connect_comp_key] = table[data_connect_comp_key].to_numpy()
    adata.obsm[data_rna_type_key] = table[data_rna_type_key].to_numpy()
    adata.obsm[data_neighbor_gene_distribution_key] = table[data_neighbor_gene_distribution_key].to_numpy()
    adata.obsm[data_exp_key] = table[data_exp_key].to_numpy()
    data_key = 'X'
    X = adata.X.copy()
    X = torch.as_tensor(X, dtype=torch.float32) 
    
    if adata.obsm[data_connect_comp_key].dtype == object:
        adata.obsm[data_connect_comp_key] = np.vstack(adata.obsm[data_connect_comp_key])
    
    if adata.obsm[data_rna_type_key].dtype == object:
        adata.obsm[data_rna_type_key] = np.vstack(adata.obsm[data_rna_type_key])
    
    if adata.obsm[data_neighbor_gene_distribution_key].dtype == object:
        adata.obsm[data_neighbor_gene_distribution_key] = np.vstack(adata.obsm[data_neighbor_gene_distribution_key])
    if adata.obsm[data_exp_key].dtype == object:
        adata.obsm[data_exp_key] = np.vstack(adata.obsm[data_exp_key])
    X_connect_comp = torch.as_tensor(adata.obsm[data_connect_comp_key], dtype=torch.float32)
    X_rna_type = torch.as_tensor(adata.obsm[data_rna_type_key], dtype=torch.float32)
    X_neighbor_gene_distribution = torch.as_tensor(adata.obsm[data_neighbor_gene_distribution_key], dtype=torch.float32)
    X_exp = torch.as_tensor(adata.obsm[data_exp_key], dtype=torch.float32)
    # truncate single context length
    X = X[:, :config_train["single_context_length"]]
    X_connect_comp = X_connect_comp[:, :config_train["single_context_length"]]
    X_rna_type = X_rna_type[:, :config_train["single_context_length"]]
    X_neighbor_gene_distribution = X_neighbor_gene_distribution[:, :config_train["single_context_length"]]
    X_exp = X_exp[:, :config_train["single_context_length"]]
    # cell density token
    if 'density_token' in adata.obs.columns:
        density_token = torch.as_tensor(adata.obs['density_token'], dtype=torch.float32).view(-1, 1)
        X = torch.cat((density_token, X), dim=1)
        zero_tensor = torch.zeros_like(density_token)
        X_connect_comp = torch.cat((zero_tensor, X_connect_comp), dim=1)
        X_rna_type = torch.cat((zero_tensor, X_rna_type), dim=1)
        X_neighbor_gene_distribution = torch.cat((zero_tensor, X_neighbor_gene_distribution), dim=1)
        X_exp = torch.cat((zero_tensor, X_exp), dim=1)

    if config_train["assay"] and 'assay' in adata.obs.columns:
        assay = torch.as_tensor(adata.obs['assay'], dtype=torch.float32).view(-1, 1)
        X = torch.cat((assay, X), dim=1)
        zero_tensor = torch.zeros_like(assay)
        X_connect_comp = torch.cat((zero_tensor, X_connect_comp), dim=1)
        X_rna_type = torch.cat((zero_tensor, X_rna_type), dim=1)
        X_neighbor_gene_distribution = torch.cat((zero_tensor, X_neighbor_gene_distribution), dim=1)
        X_exp = torch.cat((zero_tensor, X_exp), dim=1)

    if config_train["specie"] and 'specie' in adata.obs.columns:
        specie = torch.as_tensor(adata.obs['specie'], dtype=torch.float32).view(-1, 1)
        X = torch.cat((specie, X), dim=1)
        zero_tensor = torch.zeros_like(specie)
        X_connect_comp = torch.cat((zero_tensor, X_connect_comp), dim=1)
        X_rna_type = torch.cat((zero_tensor, X_rna_type), dim=1)   
        X_neighbor_gene_distribution = torch.cat((zero_tensor, X_neighbor_gene_distribution), dim=1)
        X_exp = torch.cat((zero_tensor, X_exp), dim=1)
    # Store back into adata.obsm
    adata.obsm[data_key] = X.numpy()
    adata.obsm[data_connect_comp_key] = X_connect_comp.numpy()
    adata.obsm[data_rna_type_key] = X_rna_type.numpy()
    adata.obsm[data_neighbor_gene_distribution_key] = X_neighbor_gene_distribution.numpy()
    adata.obsm[data_exp_key] = X_exp.numpy()
    # Masking
    adata = do_masking(adata, config_train["masking_p"], config_train["n_tokens"])

 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prefix = os.path.basename(input_file).replace(".parquet", "")
    if not os.path.exists(os.path.join(output_path, prefix)):
        os.makedirs(os.path.join(output_path, prefix))

    masked_indices_batches = batches(adata.obsm["masked_indices"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(masked_indices_batches), os.path.join(
        output_path, prefix, f'masked_indices_{config_train["batch_size"]}.job'))
    mask_batches = batches(adata.obsm["mask"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(mask_batches), os.path.join(
        output_path, prefix, f'mask_{config_train["batch_size"]}.job'))
    real_indices_batches = batches(adata.obsm["X"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(real_indices_batches), os.path.join(
        output_path, prefix, f'real_indices_{config_train["batch_size"]}.job'))
    attention_mask_batches = batches(adata.obsm["attention_mask"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(attention_mask_batches), os.path.join(
        output_path, prefix, f'attention_mask_{config_train["batch_size"]}.job'))
    connect_comp_batches = batches(adata.obsm[data_connect_comp_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(connect_comp_batches), os.path.join(
        output_path, prefix, f'connect_comp_{config_train["batch_size"]}.job'))
    rna_type_batches = batches(adata.obsm[data_rna_type_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(rna_type_batches), os.path.join(
        output_path, prefix, f'rna_type_{config_train["batch_size"]}.job'))
    cell_raw_index = batches(adata.obs["original_index"].values, config_train["batch_size"])
    joblib.numpy_pickle.dump(list(cell_raw_index), os.path.join(
        output_path, prefix, f'cell_raw_index_{config_train["batch_size"]}.job'))    
    neighbor_gene_distribution_batches = batches(adata.obsm[data_neighbor_gene_distribution_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(neighbor_gene_distribution_batches), os.path.join(
        output_path, prefix, f'neighbor_gene_distribution_{config_train["batch_size"]}.job'))
    cell_labels_batches = batches(adata.obs['cell_label'].values, config_train["batch_size"])
    joblib.numpy_pickle.dump(list(cell_labels_batches), os.path.join(
        output_path, prefix, f'cell_labels_{config_train["batch_size"]}.job'))
    exp_batches = batches(adata.obsm[data_exp_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(exp_batches), os.path.join(
        output_path, prefix, f'exp_{config_train["batch_size"]}.job'))


def get_gene_mean_path(base_dir: str, assay: str, use_metacell: bool = False) -> str:

    assay = assay.lower()
    fname = None

    if assay == "stereo" or assay == "snrna":
        if use_metacell:
            fname = "stereo-seq_gene_nonzero_means_metacell.npy"
        else:
            fname = "stereo-seq_gene_nonzero_means.npy"
    else:
        fname_map = {
            "merfish": "merfish_gene_nonzero_means.npy",
            "xenium": "Xenium_gene_nonzero_means.npy",
            "starmap": "STARmap_gene_nonzero_means.npy",
            "slideseqv2": "SlideseqV2_gene_nonzero_means.npy",
        }
        fname = fname_map.get(assay)

    if fname is None:
        raise ValueError(f"Unknown assay: {assay}. Please update assay_map.")

    path = os.path.join(base_dir, "prior_knowledge", fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"gene_mean_path not found at: {path}")

    return path


