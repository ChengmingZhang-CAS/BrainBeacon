import os
import random
import warnings
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd
import math
import numba
import time
import pickle
import joblib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import scipy.sparse
from scipy import sparse as scipy_sparse
from scipy.sparse import issparse, csr_matrix
from sklearn.utils import sparsefuncs
from sklearn.neighbors import BallTree, NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from brainbeacon.configs.config import species_dict
from brainbeacon.configs.config import technology_dict
from brainbeacon.configs.config import MAX_LENGTH
from brainbeacon.configs.config import AUX_TOKEN
from brainbeacon.configs.config import cell_density_bin_dict
from brainbeacon.configs.stage1_config import stage1_config

config_train = stage1_config
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

assay_alias_map = {
    "merfish": "merfish",
    "xenium": "xenium",
    "starmap": "starmap",
    "slideseqv2": "slideseqv2",
    "slideseqv2": "slideseqv2",
    "slideseq": "slideseqv2",
    "stereo": "stereo",
    "stereoseq": "stereo",
    "stereoseqv1": "stereo",
    "snrna": "snrna",
    "scrna": "snrna",
}

mean_var_column_by_assay = {
    "merfish": ("mean_merfish",),
    "xenium": ("mean_xenium",),
    "starmap": ("mean_starmap",),
    "slideseqv2": ("mean_slideseqv2",),
    "stereo": ("mean_stereo",),
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
    resolved_platform = assay_to_platform_name(platform_name)
    if resolved_platform == "MERFISH":
        factor = estimate_resolution(adata.obsm["spatial"])
    else:
        factor = platform_resolution_um.get(resolved_platform, 1.0)
    adata.obsm["spatial_um"] = adata.obsm["spatial"] * factor
    return adata


def _obsm_to_numpy(value):
    if isinstance(value, pd.DataFrame):
        array = value.to_numpy()
    else:
        array = np.asarray(value)
    if array.ndim != 2:
        raise TypeError(f"Expected a 2D array, got shape {array.shape}")
    return array


def _gene_type_column(var):
    if "Gene_type_id" in var.columns:
        return "Gene_type_id"
    if "gene_type_id" in var.columns:
        return "gene_type_id"
    raise KeyError("Neither 'Gene_type_id' nor 'gene_type_id' found in adata.var.")


def normalize_assay_name(assay):
    if assay is None:
        return None
    normalized = str(assay).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    return assay_alias_map.get(normalized, str(assay).strip().lower())


def assay_to_platform_name(assay):
    normalized = normalize_assay_name(assay)
    if normalized is None:
        return None
    platform_name_map = {
        "merfish": "MERFISH",
        "xenium": "XENIUM",
        "starmap": "STARMAP",
        "slideseqv2": "SLIDESEQV2",
        "stereo": "STEREO",
        "snrna": "STEREO",
    }
    return platform_name_map.get(normalized, str(assay).strip().upper())


def mean_var_column(assay, available_columns=None):
    normalized = normalize_assay_name(assay)
    if normalized not in mean_var_column_by_assay:
        return None
    candidates = mean_var_column_by_assay[normalized]
    if isinstance(candidates, str):
        candidates = (candidates,)
    if available_columns is None:
        return candidates[0]
    available = set(available_columns)
    for column in candidates:
        if column in available:
            return column
    candidate_text = ", ".join(candidates)
    raise ValueError(
        f"gene_dict is missing assay mean columns for assay={assay}. "
        f"Tried: {candidate_text}"
    )


def normalize_gene_dict_var(gene_dict):
    rename_candidates = {
        "Gene_type_id": "gene_type_id",
        "Gene type": "gene_type",
    }
    for source, target in rename_candidates.items():
        if target not in gene_dict.var.columns and source in gene_dict.var.columns:
            gene_dict.var[target] = gene_dict.var[source]
    # required_columns = {"gene_id", "homo_connect_id", "gene_type_id"}
    required_columns = {"gene_id", "homo_connect_id_old", "gene_type_id"}
    missing = required_columns.difference(gene_dict.var.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"gene_dict is missing required columns: {missing_list}")
    return gene_dict


def resolve_mean_var_column(assay, available_columns):
    normalized = normalize_assay_name(assay)
    if normalized not in mean_var_column_by_assay:
        return None, False, ()
    candidates = mean_var_column_by_assay[normalized]
    if isinstance(candidates, str):
        candidates = (candidates,)
    column = mean_var_column(assay, available_columns)
    used_fallback = column != candidates[0]
    return column, used_fallback, candidates


def load_gene_dict_and_mean(gene_dict_path, assay):
    gene_dict = normalize_gene_dict_var(sc.read_h5ad(gene_dict_path))
    mean_column, used_fallback, candidates = resolve_mean_var_column(assay, gene_dict.var.columns)
    if mean_column is None:
        return gene_dict, np.ones(gene_dict.n_vars, dtype=np.float32)
    mean_values = np.nan_to_num(
        gene_dict.var[mean_column].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return gene_dict, mean_values


def compute_density_token(adata, radius_um=100, n_bins=5):
    """
    compute the density token for each cell
    """
    coords_all = _obsm_to_numpy(adata.obsm["spatial_um"])
    density_tokens = np.zeros(adata.n_obs, dtype=np.int8)

    for sid in adata.obs["slice"].unique():
        idx = np.flatnonzero((adata.obs["slice"] == sid).to_numpy())
        if idx.size == 0:
            continue
        coords = np.asarray(coords_all[idx], dtype=np.float32)
        counts = BallTree(coords).query_radius(coords, r=radius_um, count_only=True)
        density_log = np.log1p(counts.astype(np.float32))

        min_val = max(0.0, float(density_log.min()))
        max_val = float(density_log.max())
        if min_val == max_val:
            max_val = min_val + 1e-3
        bins = np.linspace(min_val, max_val, n_bins + 1)
        token = np.digitize(density_log, bins[1:-1], right=True).astype(np.int8)
        density_tokens[idx] = token

    adata.obs["density_token"] = density_tokens
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
    from pybiomart import Dataset
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
            from pybiomart import Dataset
            bm_species = biomart_map[species]
            dataset = Dataset(name=f"{bm_species}_gene_ensembl", host="http://www.ensembl.org")
            mapping = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
            mapping = mapping.dropna().drop_duplicates()
            symbol_to_ensembl = dict(zip(mapping["Gene name"], mapping["Gene stable ID"]))
            print(f"[INFO] BioMart mapping retrieved for {species}.")
        except Exception as e:
            print(f"[WARN] BioMart failed: {e}. Falling back to MyGene.info ...")

            # === Step 3: MyGene fallback ===
            import mygene
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
    adata.var = adata.var.drop(columns=["ensembl_id"])

    print(f"[INFO] Converted {mask.sum()} / {len(mask)} genes to Ensembl IDs.")
    return adata


def align_adata_and_mean_matrix(
        adata: ad.AnnData,
        gene_dict: ad.AnnData,
        mean_matrix: np.ndarray,
        fill_miss_mean: bool = False,
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
        if fill_miss_mean:
            mean_matrix_aligned = np.ones_like(mean_matrix_aligned)
            return adata, mean_matrix_aligned
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
    adata_output.obsm['neighbor_gene_distribution'] = np.zeros((adata_output.n_obs, adata_output.n_vars),
                                                               dtype=np.float32)
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


def compute_deviation_bin_rapid(
        adata_output,
        n_neighbors=50,
        n_bins=5,
        batch_size=2000,
        store_neighbor_gene_distribution=True,
        neighbor_jobs=4,
):
    return compute_deviation_bin_rapid_v2(
        adata_output,
        n_neighbors=n_neighbors,
        n_bins=n_bins,
        batch_size=batch_size,
        use_abs=False,
        store_neighbor_gene_distribution=store_neighbor_gene_distribution,
        zero_threshold=0.0,
        neighbor_jobs=neighbor_jobs,
    )


def compute_deviation_bin_rapid_v2(
        adata_output,
        n_neighbors=50,
        n_bins=5,
        batch_size=2000,
        use_abs=True,
        store_neighbor_gene_distribution=True,
        zero_threshold=1e-4,
        neighbor_jobs=4,
):
    assert "x" in adata_output.obs.columns and "y" in adata_output.obs.columns, "Spatial coordinates 'x', 'y' not found in .obs"
    assert "brain_region" in adata_output.obs.columns and "brain_region_main" in adata_output.obs.columns, "Missing region annotations"

    slice_brain_area = (
            adata_output.obs["brain_region"].astype(str) + "_" + adata_output.obs["brain_region_main"].astype(str)
    )
    adata_output.obs["slice_brain_area"] = slice_brain_area
    adata_output.obsm["deviation_bin"] = np.zeros((adata_output.n_obs, adata_output.n_vars), dtype=np.int8)

    if store_neighbor_gene_distribution:
        adata_output.obsm["neighbor_gene_distribution"] = np.zeros(
            (adata_output.n_obs, adata_output.n_vars),
            dtype=np.float32,
        )

    x_raw = adata_output.X.copy()
    x_norm = sf_normalize(adata_output.X)
    x_norm = x_norm.tocsr() if scipy.sparse.issparse(x_norm) else np.asarray(x_norm, dtype=np.float32)
    full_deviation = np.full((adata_output.n_obs, adata_output.n_vars), np.nan, dtype=np.float32)
    coords_all = adata_output.obs[["x", "y"]].to_numpy(dtype=np.float32, copy=False)

    for group in pd.unique(slice_brain_area):
        index_array = np.flatnonzero((slice_brain_area == group).to_numpy())
        if index_array.size == 0:
            continue

        coords = coords_all[index_array]
        num_sample = min(index_array.size, n_neighbors)
        if num_sample == 0:
            continue

        nbrs = NearestNeighbors(
            n_neighbors=num_sample,
            algorithm="ball_tree",
            n_jobs=neighbor_jobs,
        ).fit(coords)
        _, all_indices = nbrs.kneighbors(coords)

        if scipy_sparse.issparse(x_norm):
            x_group = x_norm[index_array]
            for start in range(0, index_array.size, batch_size):
                end = min(start + batch_size, index_array.size)
                current_size = end - start
                batch_neighbors = all_indices[start:end]

                row_idx = np.repeat(np.arange(current_size), num_sample)
                col_idx = batch_neighbors.reshape(-1)
                weights = np.full(row_idx.shape[0], 1.0 / num_sample, dtype=np.float32)
                weight_matrix = scipy_sparse.csr_matrix(
                    (weights, (row_idx, col_idx)),
                    shape=(current_size, index_array.size),
                )

                batch_neighbor_expr = (weight_matrix @ x_group).toarray().astype(np.float32, copy=False)
                batch_cell_expr = x_group[start:end].toarray().astype(np.float32, copy=False)
                deviation = np.abs(
                    batch_cell_expr - batch_neighbor_expr) if use_abs else batch_cell_expr - batch_neighbor_expr

                if zero_threshold > 0:
                    both_low_mask = (batch_cell_expr < zero_threshold) & (batch_neighbor_expr < zero_threshold)
                else:
                    both_low_mask = (batch_cell_expr == 0) & (batch_neighbor_expr == 0)
                deviation[both_low_mask] = np.nan
                full_deviation[index_array[start:end]] = deviation

                if store_neighbor_gene_distribution:
                    batch_neighbor_expr[np.abs(batch_neighbor_expr) < 1e-6] = 0
                    adata_output.obsm["neighbor_gene_distribution"][index_array[start:end]] = batch_neighbor_expr
        else:
            x_group = np.asarray(x_norm[index_array], dtype=np.float32)
            for start in range(0, index_array.size, batch_size):
                end = min(start + batch_size, index_array.size)
                batch_neighbors = all_indices[start:end]
                batch_cell_expr = x_group[start:end]
                batch_neighbor_expr = x_group[batch_neighbors].mean(axis=1, dtype=np.float32)
                deviation = np.abs(
                    batch_cell_expr - batch_neighbor_expr) if use_abs else batch_cell_expr - batch_neighbor_expr

                if zero_threshold > 0:
                    both_low_mask = (batch_cell_expr < zero_threshold) & (batch_neighbor_expr < zero_threshold)
                else:
                    both_low_mask = (batch_cell_expr == 0) & (batch_neighbor_expr == 0)
                deviation[both_low_mask] = np.nan
                full_deviation[index_array[start:end]] = deviation

                if store_neighbor_gene_distribution:
                    batch_neighbor_expr[np.abs(batch_neighbor_expr) < 1e-6] = 0
                    adata_output.obsm["neighbor_gene_distribution"][index_array[start:end]] = batch_neighbor_expr

    quantile_points = np.linspace(0, 1, n_bins + 1, dtype=np.float32)[1:-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quantiles = np.nanquantile(full_deviation, quantile_points, axis=0)

    deviation_bin = np.zeros_like(full_deviation, dtype=np.int8)
    valid_mask = ~np.isnan(full_deviation)
    deviation_bin[valid_mask] = 1
    for quantile in quantiles:
        deviation_bin += ((full_deviation > quantile.reshape(1, -1)) & valid_mask).astype(np.int8)

    adata_output.obsm["deviation_bin"] = deviation_bin
    adata_output.X = x_raw
    return adata_output


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(
        x: np.array,
        gene_id: np.array,
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
        scores_final,  # gene token ids (offset by aux_tokens)
        scores_connect_comp_final,  # homology group token
        scores_rna_type_id_final,  # gene type token
        scores_deviation_bin_final,  # deviation bin token (0 = NaN, 1~5 = binned)
        exp_final  # expression values
    )


def tokenize_data(
        x: np.array,
        gene_id: np.array,
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
        out, gene_id, gene_connect_comp, gene_type_id, deviation_bin, max_seq_len, aux_token_len
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


def standardize_adata_obs(
        adata: ad.AnnData,
        gene_dict: ad.AnnData,
        mean_matrix: np.array,
        species: str,
        assay: str,
        cell_density: bool = True,
) -> tuple[ad.AnnData, np.ndarray]:
    """
    Standardize the observation (obs) attributes of an AnnData object and align it with a gene dictionary.

    Args:
        adata: Input AnnData object to standardize
        gene_dict: Reference gene dictionary AnnData object
        species: Species identifier
        assay: Assay type identifier
        density: Whether density token is included

    Returns:
        Standardized AnnData object with aligned genes and normalized observations
    """
    if 'slice' not in adata.obs.columns:
        adata.obs['slice'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='slice')
    if 'region' not in adata.obs.columns:
        adata.obs['region'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='region')
    if 'brain_region' not in adata.obs.columns:
        adata.obs['brain_region'] = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='brain_region')

    normalized_assay = normalize_assay_name(assay)
    if normalized_assay == "snrna":
        adata.obs["x"] = np.zeros(adata.shape[0])
        adata.obs["y"] = np.zeros(adata.shape[0])
    else:
        if 'x' not in adata.obs.columns:
            spatial_array = _obsm_to_numpy(adata.obsm["spatial"])
            adata.obs["x"] = spatial_array[:, 0]
            adata.obs["y"] = spatial_array[:, 1]

    keys_to_keep = ['brain_region', 'x', 'y', 'original_index', 'slice', "cell_label", "region"]
    if isinstance(cell_density, bool) and cell_density:
        print("Computing cell density...")
        time0 = time.time()
        platform_name = assay_to_platform_name(assay)
        adata = convert_spatial_to_um(adata, platform_name)
        if platform_name not in platform_radius_map:
            raise KeyError(f"Unsupported assay/platform for density token: {assay}")
        radius = platform_radius_map[platform_name]
        adata, _ = compute_density_token(adata, radius_um=radius, n_bins=5)
        density_map = {i: cell_density_bin_dict[f"cell_density_bin_{i}"] for i in range(5)}
        adata.obs["density_token"] = adata.obs["density_token"].map(density_map).astype(int)
        keys_to_keep.append("density_token")
        time1 = time.time()
        print(f"compute_density_token time: {(time1 - time0):.4f} seconds")

    columns_to_delete = [col for col in adata.obs.columns if col not in keys_to_keep]
    adata.obs = adata.obs.drop(columns=columns_to_delete)
    adata.uns = {}
    adata.obsm = {}

    adata_output, mean_matrix_aligned = align_adata_and_mean_matrix(adata, gene_dict, mean_matrix)
    adata_output.obs = adata_output.obs.reset_index(drop=True)
    adata_output.obs['species'] = species_dict.get(species, species)
    adata_output.obs['assay'] = technology_dict.get(normalized_assay, normalized_assay)

    return adata_output, mean_matrix_aligned


def tokenization_h5ad(adata, gene_dict_path, species=None, assay=None, output_path=None, anno=False,
                      split="train", label=False, cell_density=True, gene_niche=True,
                      use_hvg=True, n_hvg=2000, min_genes=3, min_cells=3, spatial_imputation=False,
                      use_dev_abs=True):
    """
    Brainbeacon input tokenization.
    Convert H5ad directly to batched .job outputs.
    """
    assert gene_dict_path, "Input `gene_dict_path` cannot be empty."
    normalized_assay = normalize_assay_name(assay)
    gene_dict, mean_matrix = load_gene_dict_and_mean(gene_dict_path, normalized_assay)
    print(f"adata to process: {adata.shape}")
    # adata = sc.read_h5ad(adata_path)

    if normalized_assay == "snrna":
        cell_density = False  # snRNA-seq does not have spatial coordinates
        gene_niche = False
        spatial_imputation = False
    if normalized_assay == "stereo" and spatial_imputation:
        print("performing spatial imputation...")
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
        # sc.pp.log1p(tmp)
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
    adata_output, mean_matrix = standardize_adata_obs(adata, gene_dict, mean_matrix, species, normalized_assay,
                                                      cell_density)

    # No brain_region
    if use_dev_abs:
        adata_output.obs['brain_region'] = adata_output.obs["slice"]
        adata_output.obs['brain_region_main'] = adata_output.obs[
            "slice"]  # user can change the cell label to other annotation
    else:
        adata_output.obs['brain_region'] = adata_output.obs["slice"]
        adata_output.obs['brain_region_main'] = adata_output.obs[
            "cell_label"]  # user can change the cell label to other annotation

    if gene_niche:
        if use_dev_abs:
            adata_output = compute_deviation_bin_rapid_v2(
                adata_output,
                n_neighbors=50,
                n_bins=5,
                use_abs=True,
                store_neighbor_gene_distribution=False,
                zero_threshold=1e-4,
            )
        else:
            adata_output = compute_deviation_bin_rapid(
                adata_output,
                n_neighbors=50,
                n_bins=5,
                store_neighbor_gene_distribution=False,
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
    gene_type_col = _gene_type_column(adata_output.var)
    connect_comp_lookup, rna_type_lookup = _build_global_feature_lookups(
        adata_output.var,
        n_aux=config_train["n_aux"],
        n_tokens=config_train["n_tokens"],
    )
    rng = _new_rng()

    if output_path is None:
        raise ValueError("Output path must be provided.")
    os.makedirs(output_path, exist_ok=True)
    for item in os.listdir(output_path):
        item_path = os.path.join(output_path, item)
        if item.endswith(".parquet") and os.path.isfile(item_path):
            os.remove(item_path)
        elif item.startswith("tokens-") and os.path.isdir(item_path):
            import shutil
            shutil.rmtree(item_path)

    for batch in tqdm(range(N_BATCHES), desc="Processing data batches"):
        X_chunk = adata_output.X[batch * chunk_len:chunk_len * (batch + 1)]
        if issparse(X_chunk):
            X_chunk = X_chunk.toarray()
        obs_tokens = obs_adata_output.iloc[batch * chunk_len:chunk_len * (batch + 1)].copy()
        tokenized, tokenized_connect_comp, tokenized_rna_type, tokenized_deviation_bin, tokenized_exp = tokenize_data(
            x=np.asarray(X_chunk),
            gene_connect_comp=adata_output.var["homo_connect_id"].values,
            # gene_connect_comp=adata_output.var["homo_connect_id_old"].values,
            gene_id=adata_output.var["gene_id"].values,
            gene_type_id=adata_output.var[gene_type_col].values,
            deviation_bin=adata_output.obsm["deviation_bin"][batch * chunk_len:chunk_len * (batch + 1)],
            mean_matrix=mean_matrix,
            max_seq_len=MAX_LENGTH,
            aux_token_len=AUX_TOKEN,
        )

        available_columns = []
        for col in ['brain_region', 'brain_region_main', 'x', 'y', 'assay', 'specie', 'idx', "original_index",
                    "cell_label", "density_token"]:
            if col in obs_tokens.columns:
                available_columns.append(col)

        obs_tokens = obs_tokens[available_columns].reset_index(drop=True)
        if anno:
            permutation = np.random.permutation(obs_tokens.shape[0])
            obs_tokens = obs_tokens.iloc[permutation].reset_index(drop=True)
            tokenized = tokenized[permutation]
            tokenized_connect_comp = tokenized_connect_comp[permutation]
            tokenized_rna_type = tokenized_rna_type[permutation]
            tokenized_deviation_bin = tokenized_deviation_bin[permutation]
            tokenized_exp = tokenized_exp[permutation]

        _write_job_bundle(
            output_dir=os.path.join(output_path, f"tokens-{batch:04d}"),
            obs_df=obs_tokens,
            x=tokenized,
            x_connect_comp=tokenized_connect_comp,
            x_rna_type=tokenized_rna_type,
            x_neighbor_gene_distribution=tokenized_deviation_bin,
            x_exp=tokenized_exp,
            connect_comp_lookup=connect_comp_lookup,
            rna_type_lookup=rna_type_lookup,
            rng=rng,
        )
    return output_path

def tokenize_adata_in_memory(
    adata,
    gene_dict_path: str,
    species: str,
    assay: str,
    use_hvg: bool = True,
    n_hvg: int = 1000,
    use_dev_abs: bool = False,
    min_genes: int = 3,
    min_cells: int = 3,
    cell_density: bool = True,
    gene_niche: bool = True,
    spatial_imputation: bool = False,

) -> dict:
    """
    Tokenize an AnnData object entirely in memory (no disk I/O).

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData. Will be copied internally.
    gene_dict_path : str
        Path to gene_dict.h5ad.
    species : str
        Species identifier.
    assay : str
        Assay identifier.
    use_hvg : bool, default=True
        Whether to use HVG selection.
    n_hvg : int, default=1000
        Number of HVGs.
    use_dev_abs : bool, default=False
        Whether to use abs deviation mode.
    min_genes : int, default=3
        Minimum genes per cell.
    min_cells : int, default=3
        Minimum cells per gene.
    cell_density : bool, default=True
        Whether to compute density token.
    gene_niche : bool, default=True
        Whether to compute deviation / niche token.
    spatial_imputation : bool, default=False
        Whether to perform spatial expression imputation (only for spatial data).

    Returns
    -------
    token_dict : dict
        Keys:
            - real_indices
            - attention_mask
            - connect_comp
            - rna_type
            - neighbor_gene_dist
            - exp
            - cell_raw_index
    """
    PADDING_TOKEN = 1

    adata = adata.copy()
    normalized_assay = normalize_assay_name(assay)
    gene_dict, mean_matrix = load_gene_dict_and_mean(gene_dict_path, normalized_assay)

    # ====== Assay-specific settings ======
    if normalized_assay == "snrna":
        cell_density = False
        gene_niche = False

    if normalized_assay == "stereo" and spatial_imputation:
        adata = spatial_expression_imputation(
            adata,
            spatial_key="spatial",
            n_neighbors=50,
        )

    # ====== QC ======
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # ====== HVG ======
    if use_hvg and adata.n_vars > n_hvg:
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat_v3")
        adata = adata[:, tmp.var.highly_variable].copy()

    # ====== Labels / index ======
    adata.obs["cell_label"] = 0
    adata.obs["original_index"] = adata.obs.index

    # ====== Ensembl check ======
    is_ensembl = adata.var.index.str.startswith("ENS")
    if not is_ensembl.all():
        raise ValueError(
            "adata must contain Ensembl IDs in var.index. "
            "Please convert gene names to Ensembl IDs first."
        )

    # ====== Standardize obs and align genes ======
    adata_output, mean_matrix = standardize_adata_obs(
        adata,
        gene_dict,
        mean_matrix,
        species,
        normalized_assay,
        cell_density,
    )

    # ====== Required brain region columns ======
    adata_output.obs["brain_region"] = adata_output.obs["slice"]
    if use_dev_abs:
        adata_output.obs["brain_region_main"] = adata_output.obs["slice"]
    else:
        adata_output.obs["brain_region_main"] = adata_output.obs["cell_label"]

    # ====== Deviation / niche ======
    if gene_niche:
        if use_dev_abs:
            adata_output = compute_deviation_bin_rapid_v2(
                adata_output,
                n_neighbors=50,
                n_bins=5,
                use_abs=True,
                store_neighbor_gene_distribution=False,
                zero_threshold=1e-4,
            )
        else:
            adata_output = compute_deviation_bin_rapid(
                adata_output,
                n_neighbors=50,
                n_bins=5,
                store_neighbor_gene_distribution=False,
            )
    else:
        adata_output.obsm["deviation_bin"] = np.zeros(
            (adata_output.shape[0], adata_output.shape[1]),
            dtype=np.int8,
        )
        adata_output.obs["density_token"] = np.zeros(
            adata_output.shape[0],
            dtype=np.int8,
        )

    # ====== Tokenize in chunks ======
    obs_df = adata_output.obs.reset_index()
    gene_type_col = _gene_type_column(adata_output.var)

    gene_connect_comp = adata_output.var["homo_connect_id"].values
    # gene_connect_comp = adata_output.var["homo_connect_id_old"].values
    gene_id = adata_output.var["gene_id"].values
    gene_type_id = adata_output.var[gene_type_col].values
    deviation_bin_all = adata_output.obsm["deviation_bin"]

    n_cells = adata_output.shape[0]
    chunk_size = 10_000
    n_chunks = math.ceil(n_cells / chunk_size)

    all_real_indices = []
    all_attention_mask = []
    all_connect_comp = []
    all_rna_type = []
    all_neighbor_gene_dist = []
    all_exp = []
    all_cell_raw_index = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_cells)

        X_chunk = adata_output.X[start:end]
        if issparse(X_chunk):
            X_chunk = X_chunk.toarray()

        obs_chunk = obs_df.iloc[start:end]

        tokenized, tok_connect, tok_rna, tok_dev, tok_exp = tokenize_data(
            x=X_chunk,
            gene_connect_comp=gene_connect_comp,
            gene_id=gene_id,
            gene_type_id=gene_type_id,
            deviation_bin=deviation_bin_all[start:end],
            mean_matrix=mean_matrix,
            max_seq_len=MAX_LENGTH,
            aux_token_len=AUX_TOKEN,
        )

        tokenized, tok_connect, tok_rna, tok_dev, tok_exp = _prepend_prefix_tokens(
            obs_chunk,
            tokenized,
            tok_connect,
            tok_rna,
            tok_dev,
            tok_exp,
        )

        # Inference path: no masking, only convert 0 -> padding token and build attention mask
        real_indices = tokenized.astype(np.int32, copy=True)
        real_indices[real_indices == 0] = PADDING_TOKEN
        attention_mask = (real_indices == PADDING_TOKEN)

        all_real_indices.append(real_indices)
        all_attention_mask.append(attention_mask)
        all_connect_comp.append(tok_connect.astype(np.int32))
        all_rna_type.append(tok_rna.astype(np.int32))
        all_neighbor_gene_dist.append(tok_dev.astype(np.int32))
        all_exp.append(tok_exp.astype(np.float32))
        all_cell_raw_index.append(obs_chunk["original_index"].to_numpy())

    return {
        "real_indices": np.concatenate(all_real_indices, axis=0),
        "attention_mask": np.concatenate(all_attention_mask, axis=0),
        "connect_comp": np.concatenate(all_connect_comp, axis=0),
        "rna_type": np.concatenate(all_rna_type, axis=0),
        "neighbor_gene_dist": np.concatenate(all_neighbor_gene_dist, axis=0),
        "exp": np.concatenate(all_exp, axis=0),
        "cell_raw_index": np.concatenate(all_cell_raw_index, axis=0),
    }


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


def _stack_array_column(values):
    array = values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)
    if isinstance(array, np.ndarray) and array.ndim == 2 and array.dtype != object:
        return array
    return np.vstack(array)


def _prepend_prefix_tokens(obs_df, x, x_connect_comp, x_rna_type, x_neighbor_gene_distribution, x_exp):
    x = x[:, :config_train["single_context_length"]].astype(np.int32, copy=False)
    x_connect_comp = x_connect_comp[:, :config_train["single_context_length"]].astype(np.int32, copy=False)
    x_rna_type = x_rna_type[:, :config_train["single_context_length"]].astype(np.int32, copy=False)
    x_neighbor_gene_distribution = x_neighbor_gene_distribution[:, :config_train["single_context_length"]].astype(
        np.int32, copy=False)
    x_exp = x_exp[:, :config_train["single_context_length"]].astype(np.float32, copy=False)

    if obs_df is None:
        return x, x_connect_comp, x_rna_type, x_neighbor_gene_distribution, x_exp

    if "density_token" in obs_df.columns:
        density_token = obs_df["density_token"].to_numpy(dtype=np.int32).reshape(-1, 1)
        zero_int = np.zeros((x.shape[0], 1), dtype=np.int32)
        zero_float = np.zeros((x.shape[0], 1), dtype=np.float32)
        x = np.concatenate((density_token, x), axis=1)
        x_connect_comp = np.concatenate((zero_int, x_connect_comp), axis=1)
        x_rna_type = np.concatenate((zero_int, x_rna_type), axis=1)
        x_neighbor_gene_distribution = np.concatenate((zero_int, x_neighbor_gene_distribution), axis=1)
        x_exp = np.concatenate((zero_float, x_exp), axis=1)

    if config_train["assay"] and "assay" in obs_df.columns:
        assay = obs_df["assay"].to_numpy(dtype=np.int32).reshape(-1, 1)
        zero_int = np.zeros((x.shape[0], 1), dtype=np.int32)
        zero_float = np.zeros((x.shape[0], 1), dtype=np.float32)
        x = np.concatenate((assay, x), axis=1)
        x_connect_comp = np.concatenate((zero_int, x_connect_comp), axis=1)
        x_rna_type = np.concatenate((zero_int, x_rna_type), axis=1)
        x_neighbor_gene_distribution = np.concatenate((zero_int, x_neighbor_gene_distribution), axis=1)
        x_exp = np.concatenate((zero_float, x_exp), axis=1)

    if config_train["species"] and "species" in obs_df.columns:
        species = obs_df["species"].to_numpy(dtype=np.int32).reshape(-1, 1)
        zero_int = np.zeros((x.shape[0], 1), dtype=np.int32)
        zero_float = np.zeros((x.shape[0], 1), dtype=np.float32)
        x = np.concatenate((species, x), axis=1)
        x_connect_comp = np.concatenate((zero_int, x_connect_comp), axis=1)
        x_rna_type = np.concatenate((zero_int, x_rna_type), axis=1)
        x_neighbor_gene_distribution = np.concatenate((zero_int, x_neighbor_gene_distribution), axis=1)
        x_exp = np.concatenate((zero_float, x_exp), axis=1)

    return x, x_connect_comp, x_rna_type, x_neighbor_gene_distribution, x_exp


def _build_feature_lookup(real_indices, feature_values, n_aux, n_tokens):
    max_token = int(real_indices.max()) if real_indices.size else 0
    lookup_size = max(max_token + 1, n_tokens + n_aux + 1)
    lookup = np.zeros(lookup_size, dtype=np.int32)
    valid = (real_indices > n_aux) & (real_indices < lookup_size)
    if np.any(valid):
        lookup[real_indices[valid]] = feature_values[valid]
    return lookup


def _build_global_feature_lookups(var_frame, n_aux, n_tokens):
    gene_type_col = _gene_type_column(var_frame)
    gene_ids = np.asarray(var_frame["gene_id"], dtype=np.int32) + n_aux
    connect_comp_ids = np.asarray(var_frame["homo_connect_id"], dtype=np.int32) + 1
    # connect_comp_ids = np.asarray(var_frame["homo_connect_id_old"], dtype=np.int32) + 1
    rna_type_ids = np.asarray(var_frame[gene_type_col], dtype=np.int32) + 1
    max_gene_token = int(gene_ids.max()) if gene_ids.size else 0
    lookup_size = max(max_gene_token + 1, n_tokens + n_aux + 1)
    connect_comp_lookup = np.zeros(lookup_size, dtype=np.int32)
    rna_type_lookup = np.zeros(lookup_size, dtype=np.int32)
    if gene_ids.size:
        connect_comp_lookup[gene_ids] = connect_comp_ids
        rna_type_lookup[gene_ids] = rna_type_ids
    return connect_comp_lookup, rna_type_lookup


def _new_rng(seed=None):
    if seed is None:
        seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
    return np.random.default_rng(seed)


def _mask_indices_numpy(indices, p, n_tokens, n_aux, rng):
    padding_token = 1
    real_indices = indices.astype(np.int32, copy=True)
    real_indices[real_indices == 0] = padding_token

    candidate_mask = real_indices > n_aux
    keep_mask = np.ones(real_indices.shape, dtype=np.int32)
    keep_mask[candidate_mask] = (rng.random(np.count_nonzero(candidate_mask)) >= p).astype(np.int32)

    masked_indices = real_indices * keep_mask
    masked_indices[real_indices == padding_token] = padding_token
    keep_mask[real_indices == padding_token] = padding_token

    replace_with_random = (masked_indices == 0) & (rng.random(masked_indices.shape) < 0.1)
    if replace_with_random.any():
        random_tokens = rng.integers(n_aux + 1, n_tokens + n_aux, size=masked_indices.shape, dtype=np.int32)
        masked_indices[replace_with_random] = random_tokens[replace_with_random]

    replace_with_original = (masked_indices == 0) & (rng.random(masked_indices.shape) < 0.1)
    if replace_with_original.any():
        masked_indices[replace_with_original] = real_indices[replace_with_original]

    attention_mask = masked_indices == padding_token
    return real_indices, masked_indices, keep_mask.astype(np.int32), attention_mask.astype(
        bool), replace_with_random.astype(bool), replace_with_original.astype(bool)


def _apply_synchronized_feature_mask(feature_values, mask, masked_indices, replace_with_random, replace_with_original,
                                     feature_lookup):
    masked_feature_values = feature_values.astype(np.int32, copy=True)
    masked_positions = mask == 0
    if not masked_positions.any():
        return masked_feature_values

    masked_feature_values[masked_positions] = 0
    if replace_with_random.any():
        random_gene_tokens = masked_indices[replace_with_random]
        random_feature_values = np.zeros(random_gene_tokens.shape, dtype=np.int32)
        valid_random = (random_gene_tokens >= 0) & (random_gene_tokens < feature_lookup.shape[0])
        random_feature_values[valid_random] = feature_lookup[random_gene_tokens[valid_random]]
        masked_feature_values[replace_with_random] = random_feature_values
    if replace_with_original.any():
        masked_feature_values[replace_with_original] = feature_values[replace_with_original]
    return masked_feature_values


def _dump_batched_array(values, batch_size, path):
    joblib.numpy_pickle.dump(batches(values, batch_size), path)


def _write_job_bundle(
        output_dir,
        obs_df,
        x,
        x_connect_comp,
        x_rna_type,
        x_neighbor_gene_distribution,
        x_exp,
        connect_comp_lookup=None,
        rna_type_lookup=None,
        rng=None,
):
    if rng is None:
        rng = _new_rng()

    x, x_connect_comp, x_rna_type, x_neighbor_gene_distribution, x_exp = _prepend_prefix_tokens(
        obs_df,
        x,
        x_connect_comp,
        x_rna_type,
        x_neighbor_gene_distribution,
        x_exp,
    )

    real_indices, masked_indices, mask, attention_mask, replace_with_random, replace_with_original = _mask_indices_numpy(
        x,
        p=config_train["masking_p"],
        n_tokens=config_train["n_tokens"],
        n_aux=config_train["n_aux"],
        rng=rng,
    )

    if connect_comp_lookup is None:
        connect_comp_lookup = _build_feature_lookup(real_indices, x_connect_comp, config_train["n_aux"],
                                                    config_train["n_tokens"])
    if rna_type_lookup is None:
        rna_type_lookup = _build_feature_lookup(real_indices, x_rna_type, config_train["n_aux"],
                                                config_train["n_tokens"])

    masked_connect_comp = _apply_synchronized_feature_mask(
        x_connect_comp,
        mask,
        masked_indices,
        replace_with_random,
        replace_with_original,
        connect_comp_lookup,
    )
    masked_rna_type = _apply_synchronized_feature_mask(
        x_rna_type,
        mask,
        masked_indices,
        replace_with_random,
        replace_with_original,
        rna_type_lookup,
    )

    os.makedirs(output_dir, exist_ok=True)
    batch_size = config_train["batch_size"]
    _dump_batched_array(masked_indices.astype(np.int32), batch_size,
                        os.path.join(output_dir, f'masked_indices_{batch_size}.job'))
    _dump_batched_array(mask.astype(np.int32), batch_size, os.path.join(output_dir, f'mask_{batch_size}.job'))
    _dump_batched_array(real_indices.astype(np.int32), batch_size,
                        os.path.join(output_dir, f'real_indices_{batch_size}.job'))
    _dump_batched_array(attention_mask.astype(bool), batch_size,
                        os.path.join(output_dir, f'attention_mask_{batch_size}.job'))
    _dump_batched_array(masked_connect_comp.astype(np.int32), batch_size,
                        os.path.join(output_dir, f'connect_comp_{batch_size}.job'))
    _dump_batched_array(masked_rna_type.astype(np.int32), batch_size,
                        os.path.join(output_dir, f'rna_type_{batch_size}.job'))

    if obs_df is not None and "original_index" in obs_df.columns:
        cell_raw_index = obs_df["original_index"].to_numpy()
    else:
        cell_raw_index = np.arange(real_indices.shape[0])
    _dump_batched_array(cell_raw_index, batch_size, os.path.join(output_dir, f'cell_raw_index_{batch_size}.job'))

    if obs_df is not None and "cell_label" in obs_df.columns:
        cell_labels = obs_df["cell_label"].to_numpy()
    else:
        cell_labels = np.zeros(real_indices.shape[0], dtype=np.int64)
    _dump_batched_array(cell_labels, batch_size, os.path.join(output_dir, f'cell_labels_{batch_size}.job'))
    _dump_batched_array(
        x_neighbor_gene_distribution.astype(np.int32),
        batch_size,
        os.path.join(output_dir, f'neighbor_gene_distribution_{batch_size}.job'),
    )
    _dump_batched_array(x_exp.astype(np.float32), batch_size, os.path.join(output_dir, f'exp_{batch_size}.job'))


def do_masking(adata, p, n_tokens, rng=None):
    if rng is None:
        rng = _new_rng()

    x = np.asarray(adata.obsm["X"], dtype=np.int32)
    real_indices, masked_indices, mask, attention_mask, replace_with_random, replace_with_original = _mask_indices_numpy(
        x,
        p=p,
        n_tokens=n_tokens,
        n_aux=config_train["n_aux"],
        rng=rng,
    )
    adata.obsm["X"] = real_indices
    adata.obsm["masked_indices"] = masked_indices
    adata.obsm["mask"] = mask
    adata.obsm["attention_mask"] = attention_mask

    if "X_connect_comp" in adata.obsm:
        connect_comp = np.asarray(adata.obsm["X_connect_comp"], dtype=np.int32)
        connect_comp_lookup = _build_feature_lookup(real_indices, connect_comp, config_train["n_aux"], n_tokens)
        adata.obsm["X_connect_comp"] = _apply_synchronized_feature_mask(
            connect_comp,
            mask,
            masked_indices,
            replace_with_random,
            replace_with_original,
            connect_comp_lookup,
        )

    if "X_rna_type" in adata.obsm:
        rna_type = np.asarray(adata.obsm["X_rna_type"], dtype=np.int32)
        rna_type_lookup = _build_feature_lookup(real_indices, rna_type, config_train["n_aux"], n_tokens)
        adata.obsm["X_rna_type"] = _apply_synchronized_feature_mask(
            rna_type,
            mask,
            masked_indices,
            replace_with_random,
            replace_with_original,
            rna_type_lookup,
        )

    return adata


def process_parquet(input_file, output_path):
    """
    Reads a token parquet file and writes batched .job outputs.
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
        "species": False,
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
    X = _stack_array_column(table["X"]).astype(np.int32, copy=False)
    X_connect_comp = _stack_array_column(table[data_connect_comp_key]).astype(np.int32, copy=False)
    X_rna_type = _stack_array_column(table[data_rna_type_key]).astype(np.int32, copy=False)
    X_neighbor_gene_distribution = _stack_array_column(table[data_neighbor_gene_distribution_key]).astype(np.int32,
                                                                                                          copy=False)
    X_exp = _stack_array_column(table[data_exp_key]).astype(np.float32, copy=False)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prefix = os.path.basename(input_file).replace(".parquet", "")
    _write_job_bundle(
        output_dir=os.path.join(output_path, prefix),
        obs_df=obs_df,
        x=X,
        x_connect_comp=X_connect_comp,
        x_rna_type=X_rna_type,
        x_neighbor_gene_distribution=X_neighbor_gene_distribution,
        x_exp=X_exp,
        rng=_new_rng(),
    )


def get_gene_mean_path(prior_dir: str, assay: str, use_metacell: bool = False):
    return None
