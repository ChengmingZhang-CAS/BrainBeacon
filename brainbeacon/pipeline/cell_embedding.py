from __future__ import annotations

import os
import re
import time
import json
import fcntl
import hashlib
import torch
import joblib
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, List
from collections import OrderedDict
from contextlib import nullcontext

from brainbeacon.brain_beacon import BrainBeacon


def normalize_brainbeacon_model_config(model_config: dict) -> dict:
    """Normalize legacy and current BrainBeacon config keys for inference."""
    normalized = dict(model_config)

    if "use_esm_emb" not in normalized and "use_esm_embedding" in normalized:
        normalized["use_esm_emb"] = bool(normalized["use_esm_embedding"])
    if "use_esm_embedding" not in normalized and "use_esm_emb" in normalized:
        normalized["use_esm_embedding"] = bool(normalized["use_esm_emb"])

    if "use_gene_id_emb" not in normalized and "gene_id" in normalized:
        normalized["use_gene_id_emb"] = bool(normalized["gene_id"])
    if "gene_id" not in normalized and "use_gene_id_emb" in normalized:
        normalized["gene_id"] = bool(normalized["use_gene_id_emb"])

    normalized.setdefault("neighbor_enhance", True)
    normalized.setdefault("use_gene_id_emb", True)
    normalized.setdefault("use_homo_emb", True)
    normalized.setdefault("use_rna_type_emb", True)
    normalized.setdefault("use_esm_emb", True)
    normalized.setdefault("use_esm_embedding", bool(normalized["use_esm_emb"]))
    normalized.setdefault("use_pos_emb", True)
    normalized.setdefault("use_density_emb", True)
    normalized.setdefault("density_token_idx", 2)
    normalized.setdefault("token_batch_size", int(normalized.get("batch_size", 16)))
    normalized.setdefault("data_loader_batch_size", 1)
    normalized.setdefault("dataloader_num_workers", 4)
    normalized.setdefault("pin_memory", True)
    normalized.setdefault("persistent_workers", True)
    normalized.setdefault("prefetch_factor", 2)
    normalized.setdefault("joblib_cache_size", 2)
    normalized.setdefault("inference_amp", True)
    normalized.setdefault("amp_dtype", "float16")

    return normalized


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    normalized = str(dtype_name).strip().lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {dtype_name!r}")


def identity_collate(batch):
    if len(batch) != 1:
        raise ValueError(
            "DataLoader batch_size must remain 1 for pre-batched token joblib bundles. "
            f"Got {len(batch)} samples in one loader batch."
        )
    return batch[0]


def masked_mean_pooling(transformer_output, mask):
    mask = mask.unsqueeze(-1)
    masked_output = transformer_output * mask

    valid_length = mask.sum(dim=1, keepdim=False)
    valid_length = torch.clamp(valid_length, min=1)
    mean_pooled = masked_output.sum(dim=1, keepdim=False) / valid_length  # (b, d)
    return mean_pooled

def masked_weighted_pooling_fixL(transformer_output, mask, rank_weight_mode="softmax", weight_decay=0.998, temperature=300.0):
    """
    Args:
        transformer_output: (B, L, D)
        mask: (B, L)
        rank_weight_mode: "none", "exp", "linear", or "softmax"
        weight_decay: only used if rank_weight_mode == "exp"
        temperature: only used if rank_weight_mode == "softmax"
    """
    mask = mask.unsqueeze(-1).float()  # (B, L, 1)
    B, L, D = transformer_output.shape

    if rank_weight_mode == "exp":
        rank_weights = torch.tensor([weight_decay ** i for i in range(L)], device=transformer_output.device)
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    elif rank_weight_mode == "linear":
        rank_weights = 1.0 - torch.arange(L, device=transformer_output.device).float() / L
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    elif rank_weight_mode == "softmax":
        rank_scores = -torch.arange(L, device=transformer_output.device).float()
        rank_weights = torch.softmax(rank_scores / temperature, dim=0) * L  # scaled softmax
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    else:
        rank_weights = None

    if rank_weights is not None:
        weighted_mask = mask * rank_weights  # (B, L, 1)
        masked_output = transformer_output * weighted_mask
        valid_length = weighted_mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
    else:
        masked_output = transformer_output * mask
        valid_length = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

    mean_pooled = masked_output.sum(dim=1) / valid_length  # (B, D)
    return mean_pooled

def masked_weighted_pooling(
    transformer_output,
    mask,
    expr_weights=None,
    weight_mode="expr",
    weight_decay=0.998,
    temperature=300.0,
):
    """
    Generalized pooling with rank-based or expression-based weighting.

    Args:
        transformer_output: (B, L, D)
        mask: (B, L)
        exp: (B, L), required for expression-based modes
        weight_mode: one of ["none", "linear", "expdecay", "softmax", "expression"]
    Returns:
        (B, D)
    """
    mask = mask.float().unsqueeze(-1)  # (B, L, 1)
    B, L, D = transformer_output.shape

    # === Expression-only ===
    if weight_mode == "expression":
        if expr_weights is None:
            raise ValueError("expression must be provided when weight_mode='expression'")
        weights = expr_weights.unsqueeze(-1) * mask
        weighted_output = transformer_output * weights
        weight_sum = weights.sum(dim=1).clamp(min=1e-6)
        return weighted_output.sum(dim=1) / weight_sum

    # === Plain average ===
    if weight_mode == "none":
        weighted_output = transformer_output * mask
        weight_sum = mask.sum(dim=1).clamp(min=1e-6)
        return weighted_output.sum(dim=1) / weight_sum

    # === Rank-based ===
    valid_lengths = mask.squeeze(-1).sum(dim=1).long()
    rank_weights_list = []
    for i in range(B):
        l_i = valid_lengths[i].item()
        if weight_mode == "expdecay":
            weights = torch.tensor([weight_decay ** r for r in range(l_i)], device=transformer_output.device)
        elif weight_mode == "linear":
            weights = 1.0 - torch.arange(l_i, device=transformer_output.device).float() / l_i
        elif weight_mode == "softmax":
            scores = -torch.arange(l_i, device=transformer_output.device).float()
            weights = torch.softmax(scores / temperature, dim=0) * l_i
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
        padded = torch.zeros(L, device=transformer_output.device)
        padded[:l_i] = weights
        rank_weights_list.append(padded)

    rank_weights = torch.stack(rank_weights_list).unsqueeze(-1)  # (B, L, 1)
    weighted_mask = mask * rank_weights
    masked_output = transformer_output * weighted_mask
    weight_sum = weighted_mask.sum(dim=1).clamp(min=1e-6)
    return masked_output.sum(dim=1) / weight_sum


class BrainBeaconCellCluster(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = normalize_brainbeacon_model_config(model_config)
        self.pretrain_model = BrainBeacon(
            dim_model=self.model_config["dim_model"],
            nheads=self.model_config['nheads'],
            dim_feedforward=self.model_config['dim_feedforward'],
            nlayers=self.model_config['nlayers'],
            dropout=self.model_config['dropout'],
            n_tokens=self.model_config["n_tokens"],
            n_connect_comp=self.model_config["n_connect_comp"],
            n_aux=self.model_config["n_aux"],
            n_rna_type=self.model_config['n_rna_type'],
            n_neighbor=self.model_config['num_neighbors'],
            esm_embedding_dim=self.model_config['ems_embedding_dim'],
            total_context_length=self.model_config['context_length'] * self.model_config['num_neighbors'],
            neighbor_enhance=self.model_config["neighbor_enhance"],
            use_gene_id_emb=self.model_config["use_gene_id_emb"],
            use_homo_emb=self.model_config["use_homo_emb"],
            use_rna_type_emb=self.model_config["use_rna_type_emb"],
            use_esm_emb=self.model_config["use_esm_emb"],
            use_pos_emb=self.model_config["use_pos_emb"],
            use_density_emb=self.model_config["use_density_emb"],
            density_token_idx=self.model_config["density_token_idx"],
        )

    def forward(
        self,
        x_gene_id,
        x_connect_id,
        x_rna_type,
        attention_mask,
        esm_embedding,
        neighbor_gene_distribution,
        sequence_mask=None
    ):
        del sequence_mask
        return self.pretrain_model.encode(
            x_gene_id,
            x_connect_id,
            x_rna_type,
            attention_mask,
            esm_embedding,
            neighbor_gene_distribution,
        )


class ZeroshotJoblibDataset(Dataset):
    def __init__(
            self,
            real_indices_files,
            attention_mask_files,
            connect_comp_files,
            rna_type_files,
            file_prefix_list,
            cell_raw_index_files,
            neighbor_gene_distribution_files,
            exp_files,
            cache_size=2,
    ):
        self.real_indices_files = real_indices_files
        self.attention_mask_files = attention_mask_files
        self.connect_comp_files = connect_comp_files
        self.rna_type_files = rna_type_files
        self.file_prefix_list = file_prefix_list
        self.cell_raw_index_files = cell_raw_index_files
        self.neighbor_gene_distribution_files = neighbor_gene_distribution_files
        self.exp_files = exp_files
        self.cache_size = max(1, int(cache_size))
        self._bundle_cache = OrderedDict()
        self.file_lengths = [len(joblib.load(f)) for f in self.real_indices_files]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        """Total number of samples across all files"""
        return self.total_length

    def _find_file_idx(self, idx):
        """Find the file corresponding to the global index"""
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side='right')
        if file_idx > 0:
            idx = idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, idx

    def _load_bundle(self, file_idx):
        cached = self._bundle_cache.get(file_idx)
        if cached is not None:
            self._bundle_cache.move_to_end(file_idx)
            return cached

        bundle = (
            joblib.load(self.real_indices_files[file_idx]),
            joblib.load(self.attention_mask_files[file_idx]),
            joblib.load(self.connect_comp_files[file_idx]),
            joblib.load(self.rna_type_files[file_idx]),
            joblib.load(self.cell_raw_index_files[file_idx]),
            joblib.load(self.neighbor_gene_distribution_files[file_idx]),
            joblib.load(self.exp_files[file_idx]),
        )
        self._bundle_cache[file_idx] = bundle
        while len(self._bundle_cache) > self.cache_size:
            self._bundle_cache.popitem(last=False)
        return bundle

    def __getitem__(self, idx):
        """Load a sample based on the global index"""
        file_idx, sample_idx = self._find_file_idx(idx)
        try:
            (
                real_indices_bundle,
                attention_mask_bundle,
                connect_comp_bundle,
                rna_type_bundle,
                cell_raw_index_bundle,
                neighbor_gene_distribution_bundle,
                exp_bundle,
            ) = self._load_bundle(file_idx)
            real_indices = real_indices_bundle[sample_idx]
            attention_mask = attention_mask_bundle[sample_idx]
            connect_comp = connect_comp_bundle[sample_idx]
            rna_type = rna_type_bundle[sample_idx]
            neighbor_gene_distribution = neighbor_gene_distribution_bundle[sample_idx]
            exp = exp_bundle[sample_idx]
            # ensure cell_raw_idx is a list of strings
            cell_raw_idx = cell_raw_index_bundle[sample_idx]
            if isinstance(cell_raw_idx, np.ndarray):
                cell_raw_idx = cell_raw_idx.tolist() if cell_raw_idx.ndim == 1 else [str(x) for x in cell_raw_idx]
            elif isinstance(cell_raw_idx, (list, tuple)):
                cell_raw_idx = [str(x) for x in cell_raw_idx]
            else:
                cell_raw_idx = [str(cell_raw_idx)]

            if real_indices is None or attention_mask is None or connect_comp is None or rna_type is None:
                print(self.file_prefix_list[idx])
                print(real_indices, attention_mask, connect_comp, rna_type)

            return (
                torch.as_tensor(real_indices[:, :1000], dtype=torch.long),
                torch.as_tensor(attention_mask[:, :1000], dtype=torch.bool),
                torch.as_tensor(connect_comp[:, :1000], dtype=torch.long),
                torch.as_tensor(rna_type[:, :1000], dtype=torch.long),
                cell_raw_idx,
                torch.as_tensor(neighbor_gene_distribution[:, :1000], dtype=torch.float),
                torch.as_tensor(exp[:, :1000], dtype=torch.float)
            )
        except Exception as e:
            print(f"Error in ZeroshotJoblibDataset.__getitem__: {e}")
            print(
                f"Index: {idx}, file: {self.file_prefix_list[file_idx] if file_idx < len(self.file_prefix_list) else 'index_out_of_range'}, sample: {sample_idx}")

            if idx + 1 >= self.total_length:
                # create empty tensors for safety
                empty_tensor = torch.zeros((1, 1000), dtype=torch.long)
                empty_bool_tensor = torch.zeros((1, 1000), dtype=torch.bool)
                return (
                    empty_tensor,
                    empty_bool_tensor,
                    empty_tensor,
                    empty_tensor,
                    ["unknown"],
                    empty_tensor.float(),
                    empty_tensor.float(),
                )
            else:
                return self.__getitem__(idx + 1)

class CellEmbeddingPipeline:
    def __init__(self, pretrain_ckpt: str, model_config: dict, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the pipeline with model_raw and device settings.
        """
        self.device = torch.device(device)
        self.model_config = normalize_brainbeacon_model_config(model_config)
        self.model = None
        self.esm_embedding_map = None
        self.pretrain_ckpt: str = pretrain_ckpt
        self.initialize_model()
        self.initialize_esm_embedding_map()

    def initialize_model(self):
        """
        Initialize the model_raw and compute its size.
        """
        self.model = BrainBeaconCellCluster(self.model_config).to(self.device)
        if self.pretrain_ckpt:
            try:
                ckpt = torch.load(self.pretrain_ckpt, map_location=self.device)
                self.model.pretrain_model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded pretrain_model checkpoint: {self.pretrain_ckpt}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise

    def initialize_esm_embedding_map(self):
        if not self.model_config.get("use_esm_emb", True):
            self.esm_embedding_map = None
            print("ESM embedding is disabled. Skipping ESM embedding map load.")
            return

        esm_embedding_path = self.model_config.get("esm_embedding_path")
        if not esm_embedding_path:
            raise ValueError("Missing `esm_embedding_path` while `use_esm_emb=True`.")

        esm_map_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.esm_embedding_map = torch.load(esm_embedding_path, map_location=esm_map_device)
        if self.esm_embedding_map.device != esm_map_device:
            self.esm_embedding_map = self.esm_embedding_map.to(esm_map_device)
        print(f"Loaded ESM embedding map to {esm_map_device}: {esm_embedding_path}")

    def load_dataset(self, data_paths: List[str], cache_size: int = 2):
        """
        Load the dataset from the given paths.
        """
        real_indices_files_list = []
        attention_mask_files_list = []
        connect_comp_files_list = []
        rna_type_files_list = []
        cell_raw_index_list = []
        neighbor_gene_distribution_files_list = []
        file_prefix_list = []
        exp_files_list = []
        self.data_paths = data_paths

        token_dirs = []
        for prefix in sorted(os.listdir(data_paths)):
            dir_path = os.path.join(data_paths, prefix)
            if not os.path.isdir(dir_path):
                continue
            if not prefix.startswith("tokens-"):
                continue
            if not any(name.startswith("real_indices_") and name.endswith(".job") for name in os.listdir(dir_path)):
                continue
            token_dirs.append(dir_path)

        if not token_dirs:
            raise FileNotFoundError(
                f"No token joblib bundles were found under {data_paths}. "
                "Expected directories like tokens-0000 containing *.job files."
            )

        for dir_path in token_dirs:
            file_prefix_list.append(dir_path)
            for file in sorted(os.listdir(dir_path)):
                file_path = os.path.join(dir_path, file)
                # print(f"Data paths: {file_path}")
                if 'real_indices_' in file:
                    real_indices_files_list.append(file_path)
                elif 'attention_mask_' in file:
                    attention_mask_files_list.append(file_path)
                elif 'connect_comp_' in file:
                    connect_comp_files_list.append(file_path)
                elif 'rna_type_' in file:
                    rna_type_files_list.append(file_path)
                elif "cell_raw_index" in file:
                    cell_raw_index_list.append(file_path)
                elif 'neighbor_gene_distribution_' in file:
                    neighbor_gene_distribution_files_list.append(file_path)
                elif 'exp_' in file:
                    exp_files_list.append(file_path)

        dataset = ZeroshotJoblibDataset(
            real_indices_files_list,
            attention_mask_files_list,
            connect_comp_files_list,
            rna_type_files_list,
            file_prefix_list,
            cell_raw_index_list,
            neighbor_gene_distribution_files_list,
            exp_files_list,
            cache_size=cache_size,
        )
        return dataset

    def infer(
        self,
        dataloader,
        config_train: dict,
        return_attention: bool = False,
        attention_layers: Union[str, List[int]] = "last",
        attention_reduce: str = "mean_head",
        attention_dtype: str = "float16"
    ):
        """
        Run inference on new data using the pretrained model_raw.

        Args:
            dataloader: DataLoader for input data.
            config_train: Configuration dictionary.
            return_attention: If True, also return attention weights from all encoder layers.
            attention_layers: Which layers to collect attention from. Options:
                - "last": only the last layer (default, most memory efficient)
                - "all": all layers
                - [0, 5, -1]: specific layer indices (negative indexing supported)
            attention_reduce: How to reduce attention weights to save memory. Options:
                - "none": keep full (batch, nheads, seq, seq) - WARNING: very large!
                - "mean_head": average over heads -> (batch, seq, seq) (default)
                - "max_head": max over heads -> (batch, seq, seq)
            attention_dtype: Data type for stored attention. Options:
                - "float16": half precision (default, saves 50% memory)
                - "float32": full precision

        Returns:
            If return_attention=False: list of (cell_idx, embedding) tuples.
            If return_attention=True: tuple of (indexed_embeddings, attention_weights_list).
                attention_weights_list: list of dicts, each containing:
                    - 'cell_idx': cell indices for this batch
                    - 'real_indices': gene token indices, shape (batch, seq_len)
                    - 'attention': list of (layer_idx, weights) tuples
                      weights shape depends on attention_reduce setting
        """
        self.model.eval()
        indexed_embeddings = []
        attention_weights_list = []
        use_esm = self.esm_embedding_map is not None and self.model.pretrain_model.use_esm_emb
        pin_memory = bool(config_train.get("pin_memory", True))
        non_blocking = pin_memory and torch.device(self.device).type == "cuda"
        amp_enabled = bool(config_train.get("inference_amp", True)) and torch.device(self.device).type == "cuda"
        amp_dtype = _resolve_amp_dtype(config_train.get("amp_dtype", "float16"))

        # Enable attention hooks only for target layers (saves memory)
        if return_attention:
            # Pass target_layers directly to enable_attention_hooks
            # This avoids registering hooks for layers we don't need
            if attention_layers == "all":
                hook_target = None  # None means all layers
            else:
                hook_target = attention_layers  # "last" or list of indices

            self.model.pretrain_model.enable_attention_hooks(target_layers=hook_target)

        with torch.no_grad():
            for real_indices, attention_mask, connect_comp, rna_type, cell_raw_idx, neighbor_gene_distribution, exp in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
                real_indices = real_indices.to(self.device, non_blocking=non_blocking)
                attention_mask = attention_mask.to(self.device, non_blocking=non_blocking)
                connect_comp = connect_comp.to(self.device, non_blocking=non_blocking)
                rna_type = rna_type.to(self.device, non_blocking=non_blocking)
                neighbor_gene_distribution = neighbor_gene_distribution.long().to(self.device, non_blocking=non_blocking)
                exp = exp.float().to(self.device, non_blocking=non_blocking)

                if use_esm:
                    real_indices_view = real_indices.view(-1).long()
                    esm_embedding = torch.index_select(self.esm_embedding_map, dim=0, index=real_indices_view)
                    esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])
                else:
                    esm_embedding = None

                sequence_mask = (real_indices != 1).float()
                autocast_context = (
                    torch.autocast(device_type=torch.device(self.device).type, dtype=amp_dtype)
                    if amp_enabled else nullcontext()
                )
                with autocast_context:
                    output = self.model(
                        real_indices,
                        connect_comp,
                        rna_type,
                        attention_mask,
                        esm_embedding,
                        neighbor_gene_distribution,
                        sequence_mask,
                    )
                output = output.float()
                pool_skip_tokens = config_train.get("pool_skip_tokens", 2)
                weight_mode = config_train.get("weight_mode", "expression")

                if weight_mode == "expression":
                    cd_weight = config_train.get("cd_weight", 0.02)
                    expr_mode = config_train.get("expr_mode", None)
                    aux = torch.zeros((exp.shape[0], 2), device=exp.device, dtype=exp.dtype)
                    cd = torch.full((exp.shape[0], 1), cd_weight, device=exp.device, dtype=exp.dtype)
                    gene_expr = exp[:, 3:]  # actual gene tokens
                    if expr_mode == "log1pnorm":
                        gene_expr = torch.log1p(gene_expr) / torch.log(torch.tensor(2.0, device=gene_expr.device))
                    gene_expr = gene_expr / gene_expr.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    exp_features = torch.cat([aux, cd, gene_expr], dim=1)
                    expr_weights = exp_features[:, pool_skip_tokens:]
                else:
                    expr_weights = None
                output = masked_weighted_pooling(
                    output[:, pool_skip_tokens:, :],
                    sequence_mask[:, pool_skip_tokens:],
                    expr_weights=expr_weights,
                    weight_mode=weight_mode,
                    weight_decay=config_train.get("weight_decay", 0.998),
                    temperature=config_train.get("temperature", 300)
                )
                output = output.detach().cpu()

                assert len(cell_raw_idx) == output.shape[0], "Batch size mismatch"
                # Collect indexed embeddings
                indexed_embeddings.extend(zip(cell_raw_idx, output))

                # Collect attention weights if requested
                if return_attention:
                    attn_weights = self.model.pretrain_model.get_attention_weights()
                    processed_attn = []
                    for layer_idx, w in attn_weights:
                        # Reduce over heads to save memory
                        if attention_reduce == "mean_head":
                            w = w.mean(dim=1)  # (batch, seq, seq)
                        elif attention_reduce == "max_head":
                            w = w.max(dim=1).values  # (batch, seq, seq)
                        # Convert dtype
                        if attention_dtype == "float16":
                            w = w.half()
                        processed_attn.append((layer_idx, w.cpu()))
                    attention_weights_list.append({
                        'cell_idx': cell_raw_idx,
                        'real_indices': real_indices.cpu(),
                        'attention': processed_attn
                    })
                    self.model.pretrain_model.clear_attention_weights()

        # Cleanup attention hooks
        if return_attention:
            self.model.pretrain_model.disable_attention_hooks()
            return indexed_embeddings, attention_weights_list

        return indexed_embeddings

    def run(
        self,
        data_paths: List[str],
        config_train: dict,
        return_attention: bool = False,
        attention_layers: Union[str, List[int]] = "last",
        attention_reduce: str = "mean_head",
        attention_dtype: str = "float16"
    ):
        """
        Main method to run the entire inference pipeline.

        Args:
            data_paths: Path to tokenized data directory.
            config_train: Configuration dictionary.
            return_attention: If True, also return attention weights.
            attention_layers: Which layers to collect. Options:
                - "last": only last layer (default, most efficient)
                - "all": all layers (WARNING: large memory usage)
                - [0, -1]: specific layer indices
            attention_reduce: How to reduce attention. Options:
                - "mean_head": average over heads (default)
                - "max_head": max over heads
                - "none": keep all heads (WARNING: large memory)
            attention_dtype: "float16" (default) or "float32"

        Returns:
            If return_attention=False: list of (cell_idx, embedding) tuples.
            If return_attention=True: tuple of (indexed_embeddings, attention_weights_list).

        Example usage with attention weights:
        ```python
        pipeline = CellEmbeddingPipeline(pretrain_ckpt, model_config, device)

        # Without attention (default)
        embeddings = pipeline.run(data_paths, config_train)

        # With attention (memory-efficient defaults: last layer, mean over heads, float16)
        embeddings, attention_list = pipeline.run(
            data_paths, config_train,
            return_attention=True
        )
        # Each attention tensor shape: (batch, seq_len, seq_len) in float16
        # Memory per sample: 1000*1000*2 bytes = 2MB (vs 1GB with all layers/heads)

        # Get specific layers with full heads (more memory)
        embeddings, attention_list = pipeline.run(
            data_paths, config_train,
            return_attention=True,
            attention_layers=[0, -1],  # first and last layer
            attention_reduce="none",   # keep all heads
            attention_dtype="float32"
        )

        # Process attention with gene token info
        for batch_attn in attention_list:
            cell_indices = batch_attn['cell_idx']
            gene_tokens = batch_attn['real_indices']  # (batch, seq_len) gene token IDs
            for layer_idx, weights in batch_attn['attention']:
                print(f"Layer {layer_idx}: {weights.shape}, genes: {gene_tokens.shape}")
        ```
        """
        cache_size = int(config_train.get("joblib_cache_size", 2))
        dataset = self.load_dataset(data_paths, cache_size=cache_size)
        data_loader_batch_size = int(config_train.get("data_loader_batch_size", 1))
        if data_loader_batch_size != 1:
            raise ValueError(
                "data_loader_batch_size must remain 1 for pre-batched token joblib bundles. "
                f"Got {data_loader_batch_size}."
            )

        num_workers = max(0, int(config_train.get("dataloader_num_workers", 4)))
        pin_memory = bool(config_train.get("pin_memory", True))
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": data_loader_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": identity_collate,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = bool(config_train.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = max(2, int(config_train.get("prefetch_factor", 2)))
        data_loader = DataLoader(**loader_kwargs)

        return self.infer(
            data_loader,
            config_train,
            return_attention=return_attention,
            attention_layers=attention_layers,
            attention_reduce=attention_reduce,
            attention_dtype=attention_dtype
        )


def run_tokenization(
    adata_path,
    bb_token_dir,
    gene_dict_path,
    specie,
    assay,
    use_hvg=True,
    n_hvg=1000,
    force_tokenize=True,
    use_dev_abs=False,
    min_genes=0,
    min_cells=3,
    token_batch_size=None,
):
    """
    Tokenize input AnnData into BrainBeacon joblib bundles.
    """
    import brainbeacon.utils as bb_utils
    from brainbeacon.tokenizer import tokenization_h5ad

    if not os.path.exists(bb_token_dir):
        os.makedirs(bb_token_dir)

    lock_path = os.path.join(bb_token_dir, ".tokenization.lock")
    index_path = os.path.join(bb_token_dir, "cache_index.json")

    def _list_token_dirs(base_dir):
        if not os.path.isdir(base_dir):
            return []
        token_dirs = []
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if not os.path.isdir(item_path):
                continue
            if not item.startswith("tokens-"):
                continue
            try:
                item_names = os.listdir(item_path)
            except FileNotFoundError:
                continue
            if not any(name.startswith("real_indices_") and name.endswith(".job") for name in item_names):
                continue
            token_dirs.append(item_path)
        return token_dirs

    def _extract_batch_sizes(token_dirs):
        batch_sizes = set()
        for token_dir in token_dirs:
            try:
                token_dir_items = os.listdir(token_dir)
            except FileNotFoundError:
                continue
            for file_name in token_dir_items:
                match = re.fullmatch(r"real_indices_(\d+)\.job", file_name)
                if match:
                    batch_sizes.add(int(match.group(1)))
        return batch_sizes

    def _metadata_path(cache_dir):
        return os.path.join(cache_dir, "tokenization_meta.json")

    def _complete_path(cache_dir):
        return os.path.join(cache_dir, ".complete")

    def _load_metadata(cache_dir):
        metadata_path = _metadata_path(cache_dir)
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_metadata(cache_dir, metadata):
        metadata_path = _metadata_path(cache_dir)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def _load_index():
        if not os.path.exists(index_path):
            return {}
        with open(index_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}

    def _write_index(index_data):
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(index_data, handle, indent=2, sort_keys=True)

    def _cache_is_ready(cache_dir, expected_metadata):
        if not os.path.isdir(cache_dir):
            return False
        if not os.path.exists(_complete_path(cache_dir)):
            return False
        cache_metadata = _load_metadata(cache_dir)
        if cache_metadata != expected_metadata:
            return False
        token_dirs = _list_token_dirs(cache_dir)
        if not token_dirs:
            return False
        batch_sizes = _extract_batch_sizes(token_dirs)
        return batch_sizes == {expected_metadata["token_batch_size"]}

    def _build_versioned_cache_name(base_name):
        return f"{base_name}__{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.time_ns()}"

    requested_batch_size = int(token_batch_size or bb_utils.config_train["batch_size"])
    requested_metadata = {
        "specie": str(specie),
        "assay": str(assay),
        "use_hvg": bool(use_hvg),
        "n_hvg": int(n_hvg),
        "use_dev_abs": bool(use_dev_abs),
        "min_genes": int(min_genes),
        "min_cells": int(min_cells),
        "token_batch_size": requested_batch_size,
    }
    metadata_json = json.dumps(requested_metadata, sort_keys=True, separators=(",", ":"))
    cache_hash = hashlib.sha1(metadata_json.encode("utf-8")).hexdigest()[:12]
    canonical_cache_name = f"cache_{cache_hash}"

    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        print(f"Waiting for tokenization lock: {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            index_data = _load_index()
            cache_name = index_data.get(cache_hash, canonical_cache_name)
            cache_dir = os.path.join(bb_token_dir, cache_name)

            if not force_tokenize and _cache_is_ready(cache_dir, requested_metadata):
                token_dir_count = len(_list_token_dirs(cache_dir))
                print(
                    f"Tokenized joblib bundles found ({token_dir_count} dirs, batch_size={requested_batch_size}) "
                    f"at {cache_dir}. Skipping tokenization."
                )
                return cache_dir

            if not force_tokenize and cache_name != canonical_cache_name:
                canonical_cache_dir = os.path.join(bb_token_dir, canonical_cache_name)
                if _cache_is_ready(canonical_cache_dir, requested_metadata):
                    index_data[cache_hash] = canonical_cache_name
                    _write_index(index_data)
                    token_dir_count = len(_list_token_dirs(canonical_cache_dir))
                    print(
                        f"Tokenized joblib bundles found ({token_dir_count} dirs, batch_size={requested_batch_size}) "
                        f"at {canonical_cache_dir}. Skipping tokenization."
                    )
                    return canonical_cache_dir

            if force_tokenize or os.path.exists(os.path.join(bb_token_dir, canonical_cache_name)):
                target_cache_name = _build_versioned_cache_name(canonical_cache_name)
            else:
                target_cache_name = canonical_cache_name
            target_cache_dir = os.path.join(bb_token_dir, target_cache_name)
            staging_cache_dir = os.path.join(bb_token_dir, f".tmp_{target_cache_name}_{os.getpid()}")

            start = time.time()
            print(
                "Running tokenization into "
                f"{target_cache_dir} with token_batch_size={requested_batch_size}, "
                f"min_genes={int(min_genes)}, min_cells={int(min_cells)}..."
            )
            original_batch_size = int(bb_utils.config_train["batch_size"])
            bb_utils.config_train["batch_size"] = requested_batch_size
            try:
                bb_utils.tokenization_h5ad(
                    adata_path,
                    gene_dict_path,
                    specie=specie,
                    assay=assay,
                    output_path=staging_cache_dir,
                    use_hvg=use_hvg,
                    n_hvg=n_hvg,
                    use_dev_abs=use_dev_abs,
                    min_genes=int(min_genes),
                    min_cells=int(min_cells),
                )
            finally:
                bb_utils.config_train["batch_size"] = original_batch_size

            token_dirs = _list_token_dirs(staging_cache_dir)
            if not token_dirs:
                raise RuntimeError(
                    f"Tokenization completed, but no token joblib bundles were found under {staging_cache_dir}."
                )

            _write_metadata(staging_cache_dir, requested_metadata)
            with open(_complete_path(staging_cache_dir), "w", encoding="utf-8") as handle:
                handle.write("complete\n")
            if os.path.exists(target_cache_dir):
                original_target_cache_dir = target_cache_dir
                target_cache_name = _build_versioned_cache_name(canonical_cache_name)
                target_cache_dir = os.path.join(bb_token_dir, target_cache_name)
                print(
                    f"Target cache directory already exists ({original_target_cache_dir}). "
                    f"Publishing to a new versioned directory instead: {target_cache_dir}"
                )
            os.replace(staging_cache_dir, target_cache_dir)
            index_data[cache_hash] = target_cache_name
            _write_index(index_data)

            end = time.time()
            print(f"Preprocessing time: {(end - start)/60:.2f} minutes")
            return target_cache_dir
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def run_bb_inference(
    adata,
    token_data_path,
    config_train,
    pretrain_ckpt,
    device,
    save_path=None,
    pipeline: CellEmbeddingPipeline | None = None,
):
    time0 = time.time()
    config_train = normalize_brainbeacon_model_config(config_train)
    owns_pipeline = pipeline is None
    if pipeline is None:
        pipeline = CellEmbeddingPipeline(pretrain_ckpt=pretrain_ckpt, model_config=config_train, device=device)

    # Generate embeddings
    pred = pipeline.run(data_paths=token_data_path, config_train=config_train)

    # Extract index and embeddings from pred
    def _normalize_pred_index(idx):
        if isinstance(idx, np.ndarray):
            if idx.ndim == 0:
                return str(idx.item())
            if idx.size != 1:
                raise ValueError(f"Expected a single cell index, but got array with shape {idx.shape}.")
            return str(idx.reshape(-1)[0])
        if isinstance(idx, (list, tuple)):
            if len(idx) != 1:
                raise ValueError(f"Expected a single cell index, but got {len(idx)} values: {idx!r}")
            return str(idx[0])
        return str(idx)

    pred_indices, pred_embeddings = zip(*[(_normalize_pred_index(idx), emb.numpy()) for idx, emb in pred])
    pred_indices = np.array(pred_indices)
    pred_embeddings = np.array(pred_embeddings)

    # get obs_names from adata
    obs_names = np.array(adata.obs_names)  # Convert to NumPy array for fast operations

    # Require exact order match before saving or returning embeddings
    if np.array_equal(pred_indices, obs_names):
        print("obs_names and pred_indices are in the same order.")
        ordered_embeddings = pred_embeddings  # Direct assignment if order matches
    else:
        if len(pred_indices) != len(obs_names):
            raise ValueError(
                "Embedding order check failed: the number of predicted embeddings does not match "
                f"adata.obs_names ({len(pred_indices)} vs {len(obs_names)}). "
                "This usually means tokenization filtered out cells before inference. "
                "For inference, use min_genes=0 and regenerate the token cache. "
                "Aborting without saving."
            )

        mismatch_positions = np.flatnonzero(pred_indices != obs_names)
        first_mismatch = int(mismatch_positions[0]) if mismatch_positions.size > 0 else -1
        pred_value = pred_indices[first_mismatch] if first_mismatch >= 0 else "unknown"
        obs_value = obs_names[first_mismatch] if first_mismatch >= 0 else "unknown"
        raise ValueError(
            "Embedding order check failed: predicted cell indices do not match adata.obs_names order. "
            f"First mismatch at position {first_mismatch}: pred={pred_value}, obs={obs_value}. "
            "Aborting without saving."
        )

    if save_path is not None:
        np.savez_compressed(save_path, embeddings=ordered_embeddings)
        print(f"Embeddings saved to {save_path}")
    time1 = time.time()
    print("Time cost: ", (time1 - time0) / 60)

    del pred, pred_indices, pred_embeddings
    if owns_pipeline:
        del pipeline
        torch.cuda.empty_cache()
    return ordered_embeddings


def run_bbcellformer_recon(
    adata,
    bb_embedding_path,            # path to .npz embedding file
    bb_pretrain_path,             # path to BB encoder backbone weights
    cellformer_version,          # prefix like 'cellformer', used to find .yaml/.pt
    cellformer_directory,        # path to folder with pretrained CellFormer model_raw files
    device,
    cellformer_pretrain_path=None,  # Not used here, but required by the pipeline
    use_batch=True,
    use_spatial=True,
    do_fit=True,
    fit_epochs=500,  # can be set in the pipeline
    slice_sample=False,  # NEW
    enc_mod="flowformer",
    path_dict: dict = None,
    mask_type="hidden",  # 'hidden' or 'input'
    output_attentions=False,  # whether to return attention weights
    save_embedding_path=None,  # Optional now
    save_model_path=None,  # optional: save .pt model_raw weights
):
    from brainbeacon.bbcellformer.pipeline.reconstruction import ReconstructPipeline

    # Load AnnData file
    data = adata.copy()
    data.obs_names_make_unique()
    # set train/valid split
    np.random.seed(42)
    data.obs['valid_split'] = 'train'
    if 'slice' not in data.obs.columns:
        data.obs['slice'] = 'default'
    for batch_id in data.obs['slice'].unique():
        idx = data.obs['slice'] == batch_id
        cell_idx = np.where(idx)[0]
        n_valid = max(1, int(len(cell_idx) * 0.1))  # Ensure at least one cell is selected for validation
        valid_cells = np.random.choice(cell_idx, n_valid, replace=False)
        data.obs.iloc[valid_cells, data.obs.columns.get_loc('valid_split')] = 'valid'

    # load brainbeacon embeddings
    data.obsm['bb_emb'] = np.load(bb_embedding_path)['embeddings']

    # Add batch info if enabled
    if use_batch:
        data.obs['batch'] = data.obs['slice']

    if use_spatial and "spatial" in data.obsm.keys():
        all_coords = []

        for batch_id in data.obs['batch'].unique():
            idx = data.obs['batch'] == batch_id
            spatial = data.obsm['spatial'][idx]
            spatial = np.asarray(spatial)  # assure spatial is a NumPy array
            spatial_min = spatial.min(axis=0)
            spatial_max = spatial.max(axis=0)
            normalized = (spatial - spatial_min) / (spatial_max - spatial_min + 1e-8)

            data.obs.loc[idx, 'x_FOV_px'] = normalized[:, 0]
            data.obs.loc[idx, 'y_FOV_px'] = normalized[:, 1]

    # Initialize CellPLM embedding pipeline
    overwrite_config = {
        "name": f"bb_{enc_mod}",
        "enc_mod": enc_mod,
        'objective': 'nb',
        'mask_node_rate': 0.95,
        'mask_feature_rate': 0.25,
        'max_batch_size': 2000,
        'mask_type': mask_type,
        # 'mask_type': 'input',
    }
    # clear GPU memory before re-initializing the pipeline
    torch.cuda.empty_cache()
    pipeline = ReconstructPipeline(
        pretrain_prefix=cellformer_version,
        overwrite_config=overwrite_config,
        pretrain_directory=cellformer_directory,
        bb_pretrain_path=bb_pretrain_path,
        cellformer_pretrain_path=cellformer_pretrain_path,
        path_dict=path_dict,
        use_pretrain=True)
    if do_fit:
        # Only sample one slice if requested
        if slice_sample:
            # np.random.seed(42)
            rng = np.random.RandomState(None)  # 使用局部随机性，每次运行都不一样
            chosen_slice = rng.choice(data.obs['slice'].unique())
            fit_data = data[data.obs['slice'] == chosen_slice].copy()
            print(f"Training only on slice: {chosen_slice} ({fit_data.n_obs} cells)")
            MAX_CELLS = 20000
            if fit_data.n_obs > MAX_CELLS:
                print(f"[Warning] Too many cells in slice ({fit_data.n_obs}), subsampling to {MAX_CELLS}")
                sampled_indices = np.random.choice(fit_data.n_obs, MAX_CELLS, replace=False)
                fit_data = fit_data[sampled_indices].copy()
                print("fit data shape:", fit_data.shape)

        else:
            fit_data = data.copy()
        pipeline.fit(
            fit_data,  # AnnData object
            train_config={'epochs': fit_epochs},
            split_field='valid_split',
            train_split='train',
            valid_split='valid',
            device=device
        )
    inference_config = {
        'lr': 5e-4,
        'wd': 1e-6,
        'scheduler': 'plat',
        'epochs': 100,
        'max_eval_batch_size': 1000,
        'patience': 5,
        'workers': 0,
    }
    result = pipeline.predict(
        data,
        inference_config=inference_config,
        output_attentions=output_attentions,
        device=device
    )
    pred = result['pred']
    latent = result['latent']
    attention = result.get('attention', None)

    target_genes = pipeline.target_genes  # this was set inside predict()
    data = data[:, target_genes].copy()  # now data.var.index == target_genes

    data.obsm['X_emb'] = latent.cpu().numpy()  # Store embeddings in AnnData object
    data.obsm['X_pred'] = pred.cpu().numpy()  # Store predicted gene
    if output_attentions and attention is not None:
        data.uns['attention'] = attention.cpu().numpy()

    if save_model_path is not None:
        torch.save(pipeline.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    if save_embedding_path is not None:
        np.savez_compressed(save_embedding_path, embeddings=data.obsm['X_emb'])
        print(f"Embeddings saved to {save_embedding_path}")

    return data

def run_bbcellformer_pipeline(
    adata_path: str,
    specie: str,
    assay: str,
    gene_dict_path: str,
    bb_ckpt_path: str,
    cellplm_ckpt_path: str,
    output_dir: str,
    output_prefix: str,
    path_dict: dict = None,
    config_train: dict = None,
    config_update: dict = None,
    n_hvg: int = 1000,
    cd_weight: float = 0.02,
    use_hvg: bool = True,
    use_batch: bool = True,
    use_spatial: bool = True,
    weight_mode: str = "expression",
    force_tokenize: bool = True,
    use_dev_abs: bool = False,
    do_fit: bool = True,
    fit_epochs: int = 100,
    slice_sample=False,  # select one slice for training
    enc_mod="flowformer",
    mask_type="hidden",  # 'hidden' or 'input'
    output_attentions=False,  # whether to return attention weights
    save_model: bool = True,  # whether to save the model_raw
    save_model_path: str = None,
    save_embedding_path: str = None,
    device=None,
    seed: int = 42,
    deterministic: bool = True
):
    """Run the end-to-end BrainBeacon + CellFormer pipeline and return an updated AnnData.

    This function performs:
    1) Load AnnData from ``adata_path`` and set ``adata.obs["platform"] = assay``.
    2) Tokenization (BrainBeacon tokenizer) and save token files under ``output_dir``.
    3) BrainBeacon inference to produce cell embeddings (saved as ``*_bb_embeddings.npz``).
    4) CellFormer reconstruction / fitting and save final embeddings and model (optional).

    Parameters
    ----------
    adata_path : str
        Path to the input AnnData (``.h5ad``).
    specie : str
        Species name used in tokenization (e.g., ``"human"``, ``"mouse"``).
    assay : str
        Platform / assay name. Will be stored to ``adata.obs["platform"]``.
    gene_dict_path : str
        Path to the BrainBeacon gene dictionary (``.h5ad``).
    bb_ckpt_path : str
        Path to BrainBeacon pretrained checkpoint.
    cellplm_ckpt_path : str
        Path to CellPLM/CellFormer pretrained checkpoint.
    output_dir : str
        Output directory for intermediate files and results.
    output_prefix : str
        Prefix used to name output files.

    path_dict : dict, optional
        Optional path configuration passed to downstream reconstruction.
    config_train : dict
        Training/inference configuration. Must be provided.
        This function will update it with internal defaults (e.g., ``weight_mode``, ``cd_weight``).
    config_update : dict, optional
        Optional overrides merged into ``config_train`` after defaults are set.

    n_hvg : int, default 1000
        Number of HVGs to use if ``use_hvg=True``.
    cd_weight : float, default 0.02
        Cell-density token weight used by expression-weighted pooling.
    use_hvg : bool, default True
        Whether to perform HVG selection in tokenization.
    use_batch : bool, default True
        Whether to enable batch-related options in CellFormer reconstruction.
    use_spatial : bool, default True
        Whether to enable spatial options in CellFormer reconstruction.
    weight_mode : str, default "expression"
        Pooling mode used for embedding aggregation (e.g., ``"expression"``).
    force_tokenize : bool, default True
        If True, redo tokenization and overwrite intermediate outputs.
        Note: this flag also controls whether to skip BB inference when cached files exist.
    use_dev_abs : bool, default False
        Whether to use alternative dev/abs settings in tokenization (project-specific).

    do_fit : bool, default True
        Whether to fit/fine-tune CellFormer reconstruction.
    fit_epochs : int, default 100
        Number of epochs for fitting when ``do_fit=True``.
    slice_sample : bool, optional
        If True, select one slice for training (project-specific behavior).
    enc_mod : str, default "flowformer"
        Encoder module variant used by CellFormer.
    mask_type : str, default "hidden"
        Masking strategy, ``"hidden"`` or ``"input"``.
    output_attentions : bool, default False
        Whether to return/record attention weights during reconstruction.

    save_model : bool, default True
        Whether to save the trained/fitted CellFormer model.
    save_model_path : str, optional
        Path to save the model checkpoint. If None, a default path is used.
    save_embedding_path : str, optional
        Path to save final embeddings. If None, a default path is used.

    device : torch.device or str, optional
        Device to run on. If None, uses CUDA if available, else CPU.
    seed : int, default 42
        Random seed.
    deterministic : bool, default True
        Whether to enforce deterministic behavior (when supported).

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData object returned by ``run_bbcellformer_recon``.
        Intermediate files (tokenization outputs, BB embeddings, final embeddings/model)
        are saved under ``output_dir`` with the given ``output_prefix``.
    """
    import scanpy as sc

    from brainbeacon.tokenizer import set_seed

    # ====== 1. Setup ======
    os.makedirs(output_dir, exist_ok=True)
    if seed is not None:
        set_seed(seed, deterministic=deterministic)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_train is None:
        raise ValueError("`config_train` must be provided.")

    config_train.update({
        "weight_mode": weight_mode,
        "cd_weight": cd_weight,
        "masking_p": 0,
        "batch_size": 64,
        "expr_mode": None,
        # "use_gene_id_emb": True,
        # "use_homo_emb": True,
        # "use_rna_type_emb": True,
        # "use_esm_emb": True,
    })
    if config_update:
        config_train.update(config_update)

    # ====== 2. Load AnnData ======
    adata = sc.read_h5ad(adata_path)
    adata.obs["platform"] = assay

    # ====== 3. Tokenization ======
    bb_token_dir = os.path.join(output_dir, f"{output_prefix}_bb_token_dir")
    token_data_path = run_tokenization(
        adata_path=adata_path,
        bb_token_dir=bb_token_dir,
        gene_dict_path=gene_dict_path,
        specie=specie,
        assay=assay,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        force_tokenize=force_tokenize,
        use_dev_abs=use_dev_abs,
    )

    # ====== 4. BrainBeacon Inference ======
    bb_embedding_path = os.path.join(output_dir, f"{output_prefix}_bb_embeddings.npz")
    if os.path.exists(bb_embedding_path) and not force_tokenize:
        print(f"Skipping BB inference. Found existing file: {bb_embedding_path}")
    else:
        start_time = time.time()
        print(f"[BB inference] Start...")
        bb_emb = run_bb_inference(
            adata=adata,
            token_data_path=token_data_path,
            config_train=config_train,
            pretrain_ckpt=bb_ckpt_path,
            device=device,
            save_path=bb_embedding_path
        )
        end_time = time.time()
        print(f"BB inference complete. Saved to: {bb_embedding_path}")
        print(f"[BB inference] Time cost: {(end_time - start_time):.2f} sec")
    # adata.obsm["bb_emb"] = bb_emb

    # ====== 5. CellFormer Reconstruction ======
    if save_embedding_path is None:
        save_embedding_path = os.path.join(output_dir, f"{output_prefix}_embeddings.npz")
    if save_model and save_model_path is None:
        save_model_path = os.path.join(output_dir, f"{output_prefix}_cellformer.pt")

    adata = run_bbcellformer_recon(
        adata=adata,
        bb_embedding_path=bb_embedding_path,
        bb_pretrain_path=bb_ckpt_path,
        cellformer_version="cellformer",
        path_dict = path_dict,
        cellformer_directory=os.path.dirname(cellplm_ckpt_path),
        device=device,
        cellformer_pretrain_path=cellplm_ckpt_path,
        use_batch=use_batch,
        use_spatial=use_spatial,
        do_fit=do_fit,
        slice_sample=slice_sample,
        fit_epochs=fit_epochs,
        enc_mod=enc_mod,
        mask_type=mask_type,  # 'hidden' or 'input'
        output_attentions=output_attentions,  # whether to return attention weights
        save_embedding_path=save_embedding_path,
        save_model_path=save_model_path,
    )

    return adata
