"""
Fast in-memory tokenizer for BrainBeacon inference.

Reuses all computation functions from tokenizer.py but eliminates
disk I/O (joblib serialization). Returns numpy arrays directly in memory.

Usage:
    from brainbeacon.tokenizer_fast import tokenize_adata_inmemory
    token_dict = tokenize_adata_inmemory(adata, gene_dict_path, specie="human", assay="merfish")
"""

import math
import numpy as np
import scanpy as sc
from scipy.sparse import issparse

from brainbeacon.tokenizer import (
    load_gene_dict_and_mean,
    normalize_assay_name,
    standardize_adata_obs,
    compute_deviation_bin_rapid_v2,
    compute_deviation_bin_rapid,
    tokenize_data,
    _prepend_prefix_tokens,
    _gene_type_column,
    spatial_expression_imputation,
    config_train,
    MAX_LENGTH,
    AUX_TOKEN,
)


PADDING_TOKEN = 1


def tokenize_adata_inmemory(
    adata,
    gene_dict_path: str,
    specie: str,
    assay: str,
    use_hvg: bool = True,
    n_hvg: int = 1000,
    use_dev_abs: bool = False,
    min_genes: int = 3,
    min_cells: int = 3,
    cell_density: bool = True,
    gene_niche: bool = True,
) -> dict:
    """Tokenize an AnnData object entirely in memory (no disk I/O).

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData. Will be copied internally; the original is not modified.
    gene_dict_path : str
        Path to gene_dict.h5ad.
    specie, assay : str
        Species and assay identifiers.
    use_hvg : bool
        Whether to select highly variable genes.
    n_hvg : int
        Number of HVGs.
    use_dev_abs : bool
        Whether to use abs deviation bins.

    Returns
    -------
    dict with keys:
        real_indices      : np.int32  (N_cells, seq_len)
        attention_mask    : np.bool_  (N_cells, seq_len)
        connect_comp      : np.int32  (N_cells, seq_len)
        rna_type          : np.int32  (N_cells, seq_len)
        neighbor_gene_dist: np.int32  (N_cells, seq_len)
        exp               : np.float32(N_cells, seq_len)
        cell_raw_index    : np.ndarray(N_cells,)
    """
    adata = adata.copy()
    normalized_assay = normalize_assay_name(assay)
    gene_dict, mean_matrix = load_gene_dict_and_mean(gene_dict_path, normalized_assay)

    # -- spatial imputation for stereo --
    if normalized_assay == "snrna":
        cell_density = False
        gene_niche = False
    if normalized_assay == "stereo":
        adata = spatial_expression_imputation(
            adata, spatial_key="spatial", n_neighbors=50,
        )

    # -- QC --
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # -- HVG --
    if use_hvg:
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat_v3")
        adata = adata[:, tmp.var.highly_variable].copy()

    # -- labels / index --
    adata.obs["cell_label"] = 0
    adata.obs["original_index"] = adata.obs.index

    # -- Ensembl check --
    is_ensembl = adata.var.index.str.startswith("ENS")
    if not is_ensembl.all():
        raise ValueError(
            "adata must contain Ensembl IDs in var.index. "
            "Please convert gene names to Ensembl IDs first."
        )

    # -- standardize obs, align genes --
    adata_output, mean_matrix = standardize_adata_obs(
        adata, gene_dict, mean_matrix, specie, normalized_assay, cell_density,
    )

    # -- brain_region columns (required by deviation) --
    adata_output.obs["brain_region"] = adata_output.obs["slice"]
    if use_dev_abs:
        adata_output.obs["brain_region_main"] = adata_output.obs["slice"]
    else:
        adata_output.obs["brain_region_main"] = adata_output.obs["cell_label"]

    # -- gene niche / deviation bins --
    if gene_niche:
        if use_dev_abs:
            adata_output = compute_deviation_bin_rapid_v2(
                adata_output, n_neighbors=50, n_bins=5,
                use_abs=True, store_neighbor_gene_distribution=False,
                zero_threshold=1e-4,
            )
        else:
            adata_output = compute_deviation_bin_rapid(
                adata_output, n_neighbors=50, n_bins=5,
                store_neighbor_gene_distribution=False,
            )
    else:
        adata_output.obsm["deviation_bin"] = np.zeros(
            (adata_output.shape[0], adata_output.shape[1]), dtype=np.int8,
        )
        adata_output.obs["density_token"] = np.zeros(adata_output.shape[0], dtype=np.int8)

    # -- tokenize in chunks (same as tokenization_h5ad but no disk writes) --
    obs_df = adata_output.obs.reset_index().rename(columns={"index": "idx"})
    obs_df["idx"] = obs_df["idx"].astype("i8")
    gene_type_col = _gene_type_column(adata_output.var)

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

        obs_chunk = obs_df.iloc[start:end].copy()

        tokenized, tok_connect, tok_rna, tok_dev, tok_exp = tokenize_data(
            x=np.asarray(X_chunk),
            gene_connect_comp=adata_output.var["homo_connect_id"].values,
            gene_id=adata_output.var["gene_id"].values,
            gene_type_id=adata_output.var[gene_type_col].values,
            deviation_bin=adata_output.obsm["deviation_bin"][start:end],
            mean_matrix=mean_matrix,
            max_seq_len=MAX_LENGTH,
            aux_token_len=AUX_TOKEN,
        )

        # prepend prefix tokens (specie / assay / density)
        tokenized, tok_connect, tok_rna, tok_dev, tok_exp = _prepend_prefix_tokens(
            obs_chunk, tokenized, tok_connect, tok_rna, tok_dev, tok_exp,
        )

        # For inference (masking_p=0): just replace 0 with padding and build attention mask
        real_indices = tokenized.astype(np.int32, copy=True)
        real_indices[real_indices == 0] = PADDING_TOKEN
        attention_mask = (real_indices == PADDING_TOKEN)

        all_real_indices.append(real_indices)
        all_attention_mask.append(attention_mask)
        all_connect_comp.append(tok_connect.astype(np.int32))
        all_rna_type.append(tok_rna.astype(np.int32))
        all_neighbor_gene_dist.append(tok_dev.astype(np.int32))
        all_exp.append(tok_exp.astype(np.float32))

        # cell raw index
        if "original_index" in obs_chunk.columns:
            all_cell_raw_index.append(obs_chunk["original_index"].to_numpy())
        else:
            all_cell_raw_index.append(np.arange(start, end))

    return {
        "real_indices": np.concatenate(all_real_indices, axis=0),
        "attention_mask": np.concatenate(all_attention_mask, axis=0),
        "connect_comp": np.concatenate(all_connect_comp, axis=0),
        "rna_type": np.concatenate(all_rna_type, axis=0),
        "neighbor_gene_dist": np.concatenate(all_neighbor_gene_dist, axis=0),
        "exp": np.concatenate(all_exp, axis=0),
        "cell_raw_index": np.concatenate(all_cell_raw_index, axis=0),
    }
