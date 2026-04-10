from __future__ import annotations

import os
import torch
import anndata as ad
from ..model import OmicsFormer
from abc import ABC, abstractmethod
from typing import List, Union
from .experimental import symbol_to_ensembl
import json
import warnings
import scanpy as sc
from importlib.resources import files
from brainbeacon.configs.config import resolve_path, DEFAULT_PATHS
from brainbeacon.configs.stage1_config import config_train

def extract_input_embeddings(
        bb_pretrain_path: str,
        esm_embedding_path: str,
        gene_dict_path: str,
        n_aux: int
):
    """
    Extract combined input embeddings (gene_id + homo + rna_type + esm) from BrainBeacon checkpoint.
    """
    if bb_pretrain_path is None:
        raise ValueError("`stage1_ckpt_path` must not be None when `mask_type` != 'hidden'.")
    print(f"[INFO] Loading stage1 checkpoint for input embedding init: {bb_pretrain_path}")

    state = torch.load(bb_pretrain_path, map_location="cpu")["model_state_dict"]
    gene_dict = sc.read_h5ad(gene_dict_path)
    gene_var = gene_dict.var

    # 1. gene_id embedding
    gene_id_emb = state["embedding.basic_embedding.weight"][n_aux:].detach().cpu()

    # 2. homo embedding
    homo_emb = state["embedding.homo_connect_embedding.weight"].detach().cpu()
    homo_id = torch.tensor(gene_var["homo_connect_id"].values + 1)
    homo_emb_gene = homo_emb[homo_id]

    # 3. rna_type embedding
    rna_emb = state["embedding.rna_type_embedding.weight"].detach().cpu()
    rna_id = torch.tensor(gene_var["Gene_type_id"].values + 1)
    rna_emb_gene = rna_emb[rna_id]

    # 4. esm embedding (projected)
    esm_raw = torch.load(esm_embedding_path, map_location="cpu")
    esm_proj_w = state["esm_embedding_projection.weight"]
    esm_proj_b = state["esm_embedding_projection.bias"]
    esm_proj = (esm_raw[n_aux:] @ esm_proj_w.T + esm_proj_b).detach().cpu()

    # 5. sum as combined
    combined_emb = gene_id_emb + homo_emb_gene + rna_emb_gene + esm_proj
    return combined_emb

def load_pretrain(
        config: dict,
        stage1_ckpt_path,
        stage2_ckpt_path: str = None,
        use_pretrain: bool = True,
):

    """Load gene list from model_raw h5ad file"""
    gene_dict_path = DEFAULT_PATHS['GENE_DICT_PATH']
    esm_embedding_path = DEFAULT_PATHS['ESM_EMBED_PATH']
    gene_dict = sc.read_h5ad(gene_dict_path)
    config['gene_list'] = gene_dict.var.index.tolist()
    gene_list_size = len(config['gene_list'])
    if 'head_type' not in config:
        config['out_dim'] = gene_list_size
    print(f"[INFO] Gene list size: {gene_list_size}")

    if config['mask_type'] == "hidden":
        config['gene_emb'] = None
    else:
        config['gene_emb'] = extract_input_embeddings(
            bb_pretrain_path=stage1_ckpt_path,
            esm_embedding_path=esm_embedding_path,
            gene_dict_path=gene_dict_path,
            n_aux=config['n_aux']
        )

    model = OmicsFormer(**config)
    if use_pretrain and stage2_ckpt_path is not None:
        print(f"[INFO] Loading stage2 pretrained checkpoint: {stage2_ckpt_path}")
        state = torch.load(stage2_ckpt_path, map_location="cpu")
        pretrained_model_dict = state['model_state_dict'] if 'model_state_dict' in state else state
        model_dict = model.state_dict()

        pretrained_dict = {
            k: v
            for k, v in pretrained_model_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped_keys = sorted(model_dict.keys() - pretrained_dict.keys())
        print(f"[INFO] Skipped parameters ({len(skipped_keys)}): {skipped_keys}")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print("[INFO] use_pretrain=False, initialize OmicsFormer from config only.")

    del config['gene_emb']
    return model


def build_model_from_config(
        config: dict,
):
    model = OmicsFormer(**config)
    return model


class Pipeline(ABC):
    def __init__(
            self,
            config: dict | None = None,
            stage1_ckpt_path: str = None,
            stage2_ckpt_path: str = None,
            use_pretrain: bool = True,
    ):
        # Load pretrain model_raw
        # self.model_raw = load_pretrain(pretrain_prefix, overwrite_config, pretrain_directory)
        if stage1_ckpt_path is not None:
            self.model = load_pretrain(
                config=config,
                stage1_ckpt_path=stage1_ckpt_path,
                stage2_ckpt_path=stage2_ckpt_path,
                use_pretrain=use_pretrain
            )
        else:
            print("[INFO] stage1_ckpt_path is None. The model will be initialized with random weights.")
            # Only build model_raw from configs, without loading weights
            self.model = load_pretrain(config=config)
        self.gene_list = None
        self.fitted = False
        self.eval_dict = {}

    def common_preprocess(self, adata, hvg, covariate_fields, ensembl_auto_conversion):
        if covariate_fields:
            for i in covariate_fields:
                assert i in ['slice_cov', 'dataset_cov', 'platform_cov'], \
                    'Currently does not support customized covariate other than "slice_cov", "dataset_cov" and "platform_cov"'
        adata = adata.copy()
        if not adata.var.index.isin(self.model.gene_set).any():
            if ensembl_auto_conversion:
                print('Automatically converting gene symbols to ensembl ids...')
                adata.var.index = symbol_to_ensembl(adata.var.index.tolist())
                if (adata.var.index == '0').all():
                    raise ValueError(
                        'None of AnnData.var.index found in pre-trained gene set.')
                adata.var_names_make_unique()
            else:
                raise ValueError(
                    'None of AnnData.var.index found in pre-trained gene set. '
                    'In case the input gene names are gene symbols, please enable `ensembl_auto_conversion`, '
                    'or manually convert gene symbols to ensembl ids in the input dataset.'
                )

        if self.fitted:
            return adata[:, adata.var.index.isin(self.gene_list)]
        else:
            if hvg > 0:
                if hvg < adata.shape[1]:
                    sc.pp.highly_variable_genes(adata, n_top_genes=hvg, subset=True, flavor='seurat_v3')
                else:
                    warnings.warn('HVG number is larger than number of valid genes.')
            adata = adata[:, [x for x in adata.var.index.tolist() if x in self.model.gene_set]]
            self.gene_list = adata.var.index.tolist()
            return adata

    @abstractmethod
    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None, # A field in adata.obs for representing train-test split
            train_split: str = None, # A specific split where labels can be utilized for training
            valid_split: str = None, # A specific split where labels can be utilized for validation
            covariate_fields: List[str] = None, # A list of fields in adata.obs that contain cellular covariates
            label_fields: List[str] = None, # A list of fields in adata.obs that contain cell labels
            batch_gene_list: dict = None,  # A dictionary that contains batch and gene list pairs
            ensembl_auto_conversion: bool = True, # A bool value indicating whether the function automativally convert symbols to ensembl id
            device: Union[str, torch.device] = 'cpu'
            ):
        # Fine-tune the model_raw on an anndata object
        pass

    @abstractmethod
    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_gene_list: dict = None,
                ensembl_auto_conversion: bool = True,
                device: Union[str, torch.device] = 'cpu'
                ):
        # Inference on an anndata object
        pass

    @abstractmethod
    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = 'test',
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_gene_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu'
              ):
        # Inference on an anndata object and automatically evaluate
        pass
