import os
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np
from pybiomart import Server
from sklearn.preprocessing import LabelEncoder

def map_to_human_orthologs(
    adata: ad.AnnData,
    save_path: str = None,
    source_species: str = "mouse",  # e.g., "mouse", "macaque", "marmoset"
    verbose: bool = True
) -> ad.AnnData:
    if verbose:
        print(f"Converting {source_species} genes to human orthologs...")

    # Define species mapping
    species_map = {
        "mouse": "mmusculus_gene_ensembl",
        "macaque": "mfascicularis_gene_ensembl",
        "marmoset": "cjacchus_gene_ensembl"
    }

    # --- 检查是否支持该物种 ---
    if source_species not in species_map:
        raise ValueError(f"Unsupported species '{source_species}'. Supported: {list(species_map.keys())}")

    # Connect to BioMart
    server = Server(host='http://www.ensembl.org')
    ds_source = server['ENSEMBL_MART_ENSEMBL'][species_map[source_species]]

    # Query gene mapping to human
    df_map = ds_source.query(attributes=[
        'ensembl_gene_id',
        'hsapiens_homolog_ensembl_gene',
        'hsapiens_homolog_orthology_type'
    ])
    df_map.columns = ['source_id', 'human_id', 'type']
    df_map = df_map[df_map['type'] == 'ortholog_one2one'].dropna(subset=['human_id'])
    mapping = dict(zip(df_map['source_id'], df_map['human_id']))

    # Filter genes to those with human orthologs
    keep = [g for g in adata.var_names if g in mapping]
    adata = adata[:, keep].copy()

    # --- Store original gene info ---
    adata.var[f'gene_symbol_{source_species}'] = adata.var['gene_symbol']
    adata.var[f'ensembl_id_{source_species}'] = adata.var_names

    # --- Replace var_names with human ensembl IDs ---
    adata.var_names = [mapping[g] for g in adata.var_names]
    adata.var['ensembl_id'] = adata.var_names
    adata.var_names.name = "ensembl_id"

    # --- Query human gene symbols ---
    ds_human = server['ENSEMBL_MART_ENSEMBL']['hsapiens_gene_ensembl']
    human_symbols = ds_human.query(attributes=['ensembl_gene_id', 'external_gene_name'])
    human_symbols.columns = ['ensembl_id', 'gene_symbol']
    human_symbols.dropna(inplace=True)
    symbol_dict = dict(zip(human_symbols['ensembl_id'], human_symbols['gene_symbol']))
    adata.var['gene_symbol'] = adata.var['ensembl_id'].map(symbol_dict)

    # Save the converted AnnData
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        adata.write_h5ad(save_path)
        if verbose:
            print(f"Converted adata saved to: {save_path}")

    return adata