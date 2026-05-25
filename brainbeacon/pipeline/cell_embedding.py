import os
import time
import torch
import joblib
import shutil
import torch.nn as nn
import numpy as np
import scanpy as sc
import copy

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, List

from brainbeacon.brain_beacon import BrainBeacon
from brainbeacon.tokenizer import set_seed
from brainbeacon.tokenizer import tokenize_adata_in_memory
from brainbeacon.configs.config import resolve_path
from brainbeacon.configs.stage1_config import stage1_config as default_config1
from brainbeacon.configs.stage2_config import stage2_config as default_config2
from brainbeacon.bbcellformer.pipeline.reconstruction import ReconstructPipeline


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

    if "use_cell_density" not in normalized and "use_density_emb" in normalized:
        normalized["use_cell_density"] = bool(normalized["use_density_emb"])
    if "use_density_emb" not in normalized and "use_cell_density" in normalized:
        normalized["use_density_emb"] = bool(normalized["use_cell_density"])

    if "use_gene_deviation" not in normalized and "neighbor_enhance" in normalized:
        normalized["use_gene_deviation"] = bool(normalized["neighbor_enhance"])
    if "neighbor_enhance" not in normalized and "use_gene_deviation" in normalized:
        normalized["neighbor_enhance"] = bool(normalized["use_gene_deviation"])

    normalized.setdefault("use_gene_id_emb", True)
    normalized.setdefault("use_homo_emb", True)
    normalized.setdefault("use_rna_type_emb", True)
    normalized.setdefault("use_esm_emb", True)
    normalized.setdefault("use_esm_embedding", bool(normalized["use_esm_emb"]))
    normalized.setdefault("use_pos_emb", True)
    normalized.setdefault("use_cell_density", True)
    normalized.setdefault("use_density_emb", bool(normalized["use_cell_density"]))
    normalized.setdefault("use_gene_deviation", True)
    normalized.setdefault("neighbor_enhance", bool(normalized["use_gene_deviation"]))
    normalized.setdefault("density_token_idx", 2)

    return normalized


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
            # n_neighbor=self.model_config['num_neighbors'],
            esm_embedding_dim=self.model_config['esm_emb_dim'],
            # total_context_length=self.model_config['context_length'] * self.model_config['num_neighbors'],
            use_gene_id_emb=self.model_config["use_gene_id_emb"],
            use_homo_emb=self.model_config["use_homo_emb"],
            use_rna_type_emb=self.model_config["use_rna_type_emb"],
            use_esm_emb=self.model_config["use_esm_emb"],
            use_pos_emb=self.model_config["use_pos_emb"],
            neighbor_enhance=self.model_config["use_gene_deviation"],
            use_density_emb=self.model_config["use_cell_density"],
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
            exp_files
    ):
        self.real_indices_files = real_indices_files
        self.attention_mask_files = attention_mask_files
        self.connect_comp_files = connect_comp_files
        self.rna_type_files = rna_type_files
        self.file_prefix_list = file_prefix_list
        self.cell_raw_index_files = cell_raw_index_files
        self.neighbor_gene_distribution_files = neighbor_gene_distribution_files
        self.exp_files = exp_files
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

    def __getitem__(self, idx):
        """Load a sample based on the global index"""
        file_idx, sample_idx = self._find_file_idx(idx)
        # Load the specific file (consider caching for better performance)
        try:
            real_indices_file = self.real_indices_files[file_idx]
            attention_mask_file = self.attention_mask_files[file_idx]
            connect_comp_file = self.connect_comp_files[file_idx]
            rna_type_file = self.rna_type_files[file_idx]
            cell_raw_index_file = self.cell_raw_index_files[file_idx]
            neighbor_gene_distribution_file = self.neighbor_gene_distribution_files[file_idx]
            exp_file = self.exp_files[file_idx]
            real_indices = joblib.load(real_indices_file)[sample_idx]
            attention_mask = joblib.load(attention_mask_file)[sample_idx]
            connect_comp = joblib.load(connect_comp_file)[sample_idx]
            rna_type = joblib.load(rna_type_file)[sample_idx]
            neighbor_gene_distribution = joblib.load(neighbor_gene_distribution_file)[sample_idx]
            exp = joblib.load(exp_file)[sample_idx]
            # ensure cell_raw_idx is a list of strings
            cell_raw_idx = joblib.load(cell_raw_index_file)[sample_idx]
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
                    empty_tensor,
                    empty_tensor,
                )
            else:
                return self.__getitem__(idx + 1)

class CellEmbeddingPipeline:
    def __init__(self, pretrain_ckpt: str, model_config: dict, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the pipeline with model_raw and device settings.
        """
        self.device = device
        self.model_config = normalize_brainbeacon_model_config(model_config)
        self.model = None
        self.pretrain_ckpt: str = pretrain_ckpt
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the model and load checkpoint weights when available.
        """
        self.model = BrainBeaconCellCluster(self.model_config).to(self.device)

        if not self.pretrain_ckpt:
            return

        strict_load = self.model_config.get("strict_load", False)

        try:
            ckpt = torch.load(self.pretrain_ckpt, map_location=self.device)
            pretrained_dict = ckpt["model_state_dict"]

            if strict_load:
                self.model.pretrain_model.load_state_dict(pretrained_dict)
                print(f"Loaded pretrain_model checkpoint (strict): {self.pretrain_ckpt}")

                # Apply homo-mean override to checkpoint-loaded token embeddings.
                self.apply_homo_mean_after_load()
                return

            model_dict = self.model.pretrain_model.state_dict()
            compatible_dict = {}
            skipped = []

            for k, v in pretrained_dict.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    compatible_dict[k] = v
                elif k in model_dict:
                    skipped.append(
                        f"{k}: ckpt {tuple(v.shape)} vs model {tuple(model_dict[k].shape)}"
                    )

            if skipped:
                print(f"[Warning] Skipped {len(skipped)} shape-mismatched params:")
                for s in skipped:
                    print(f"  {s}")

            model_dict.update(compatible_dict)
            self.model.pretrain_model.load_state_dict(model_dict)
            print(f"Loaded pretrain_model checkpoint (compatible): {self.pretrain_ckpt}")

            # Apply homo-mean override to checkpoint-loaded token embeddings.
            self.apply_homo_mean_after_load()

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise

    def _get_homo_mean_targets(self):
        """Return enabled homo-mean targets."""
        targets = self.model_config.get("homo_mean_targets", [])
        if targets is None:
            return []
        if isinstance(targets, str):
            return [x.strip() for x in targets.split(",") if x.strip()]
        return list(targets)

    def _get_gene_token_homo_ids(self, matrix_n_tokens=None):
        """Build valid gene token IDs and homo group IDs from gene_dict."""
        cache_key = "_cached_gene_token_homo_ids"
        if hasattr(self, cache_key):
            token_ids, homo_ids, n_homo = getattr(self, cache_key)
            if matrix_n_tokens is not None and token_ids.max().item() >= matrix_n_tokens:
                raise ValueError(
                    f"Token matrix row mismatch: max token_id={token_ids.max().item()}, "
                    f"but matrix has {matrix_n_tokens} rows."
                )
            return token_ids, homo_ids, n_homo

        gene_dict_path = self.model_config.get("gene_dict_path", None)
        if gene_dict_path is None:
            raise ValueError("homo_mean_targets requires gene_dict_path in model_config.")

        gene_dict = sc.read_h5ad(gene_dict_path)
        gvar = gene_dict.var

        for col in ["gene_id", "homo_connect_id"]:
            if col not in gvar.columns:
                raise KeyError(f"{col} not found in gene_dict.var.")

        n_aux = int(self.model_config["n_aux"])
        n_tokens = int(self.model_config["n_tokens"])
        n_homo = int(self.model_config["n_connect_comp"])

        gene_ids = gvar["gene_id"].astype(int).values
        homo_ids = gvar["homo_connect_id"].astype(int).values + 1
        token_ids = gene_ids + n_aux

        valid = (
                (gene_ids >= 0)
                & (gene_ids < n_tokens)
                & (token_ids >= 0)
                & (matrix_n_tokens is None or token_ids < matrix_n_tokens)
                & (homo_ids > 0)
                & (homo_ids <= n_homo)
        )

        if valid.sum() == 0:
            raise ValueError("No valid gene tokens found for homo-mean override.")

        token_ids = torch.as_tensor(token_ids[valid], dtype=torch.long)
        homo_ids = torch.as_tensor(homo_ids[valid], dtype=torch.long)

        setattr(self, cache_key, (token_ids, homo_ids, n_homo))
        return token_ids, homo_ids, n_homo

    @torch.no_grad()
    def _apply_homo_mean_to_token_matrix(self, matrix, name):
        """Replace token-level vectors by homo-group mean vectors."""
        if matrix.dim() != 2:
            raise ValueError(f"{name} must be a 2D matrix, got shape {tuple(matrix.shape)}.")

        token_ids, homo_ids, n_homo = self._get_gene_token_homo_ids(
            matrix_n_tokens=matrix.shape[0]
        )

        device = matrix.device
        dtype = matrix.dtype
        token_ids = token_ids.to(device=device)
        homo_ids = homo_ids.to(device=device)

        dim = matrix.shape[1]
        group_sum = torch.zeros(n_homo + 1, dim, device=device, dtype=dtype)
        group_count = torch.zeros(n_homo + 1, 1, device=device, dtype=dtype)

        group_sum.index_add_(0, homo_ids, matrix[token_ids])
        group_count.index_add_(
            0,
            homo_ids,
            torch.ones(homo_ids.shape[0], 1, device=device, dtype=dtype),
        )

        group_mean = group_sum / group_count.clamp(min=1.0)
        matrix[token_ids] = group_mean[homo_ids]

        print(
            f"[INFO] Applied homo-mean override to {name}: "
            f"{token_ids.numel()} tokens, {torch.unique(homo_ids).numel()} homo groups."
        )

    @torch.no_grad()
    def apply_homo_mean_after_load(self):
        """Apply homo-mean override to checkpoint-loaded gene ID embedding."""
        targets = self._get_homo_mean_targets()
        if len(targets) == 0:
            return

        unsupported = set(targets) - {"gene_id", "esm"}
        if unsupported:
            raise ValueError(
                f"Unsupported homo_mean_targets: {unsupported}. "
                f"Supported targets are ['gene_id', 'esm']."
            )

        if "gene_id" in targets:
            gene_id_matrix = self.model.pretrain_model.embedding.basic_embedding.weight
            self._apply_homo_mean_to_token_matrix(gene_id_matrix, "gene_id_embedding")

    @torch.no_grad()
    def apply_homo_mean_to_esm_map(self, esm_embedding_map):
        """Apply homo-mean override to ESM embedding map when requested."""
        targets = self._get_homo_mean_targets()
        if "esm" not in targets:
            return esm_embedding_map

        self._apply_homo_mean_to_token_matrix(esm_embedding_map, "esm_embedding_map")
        return esm_embedding_map

    def load_dataset(self, data_paths: List[str]):
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
            exp_files_list
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
        # Switch to evaluation mode
        self.model.eval()
        # Load ESM embedding map
        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        esm_embedding_map = self.apply_homo_mean_to_esm_map(esm_embedding_map)

        indexed_embeddings = []
        attention_weights_list = []

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
                real_indices = real_indices[0]
                attention_mask = attention_mask[0]
                connect_comp = connect_comp[0]
                rna_type = rna_type[0]
                real_indices_view = real_indices.view(-1).long()
                neighbor_gene_distribution = neighbor_gene_distribution[0].long()
                exp = exp[0].float()

                esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)               
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])
                sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices))
                real_indices, attention_mask, connect_comp, rna_type, esm_embedding, neighbor_gene_distribution = (
                    real_indices.to(self.device), attention_mask.to(self.device), connect_comp.to(self.device),
                    rna_type.to(self.device), esm_embedding.to(self.device), neighbor_gene_distribution.to(self.device)
                )
                output = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask)
                output = output.detach().cpu()
                # output = masked_mean_pooling(output[:, pool_skip_tokens:, :], sequence_mask[:, pool_skip_tokens:])
                pool_skip_tokens = config_train.get("pool_skip_tokens", 2)
                weight_mode = config_train.get("weight_mode", "expression")

                if weight_mode == "expression":
                    cd_weight = config_train.get("cd_weight", 0.02)
                    expr_mode = config_train.get("expr_mode", None)
                    aux = torch.zeros((exp.shape[0], 2), device=exp.device)  # species + platform
                    cd = torch.full((exp.shape[0], 1), cd_weight, device=exp.device)  # cell_density
                    gene_expr = exp[:, 3:]  # actual gene tokens
                    if expr_mode == "log1pnorm":
                        # gene_expr = torch.log1p(gene_expr)
                        gene_expr = torch.log1p(gene_expr) / torch.log(torch.tensor(2.0, device=gene_expr.device))
                    gene_expr = gene_expr / gene_expr.sum(dim=1, keepdim=True).clamp(min=1e-6)
                    exp = torch.cat([aux, cd, gene_expr], dim=1)  # shape (B, L)
                    expr_weights = exp[:, pool_skip_tokens:]
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
        dataset = self.load_dataset(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=0)

        return self.infer(
            data_loader,
            config_train,
            return_attention=return_attention,
            attention_layers=attention_layers,
            attention_reduce=attention_reduce,
            attention_dtype=attention_dtype
        )

class InMemoryTokenDataset(Dataset):
    """
    In-memory token dataset for fast Stage1 inference.

    Expected token_dict keys
    ------------------------
    - real_indices
    - attention_mask
    - connect_comp
    - rna_type
    - neighbor_gene_dist
    - exp
    - cell_raw_index
    """

    def __init__(self, token_dict: dict, max_len: int = 1000):
        # Pre-convert arrays to torch tensors once in __init__
        self.real_indices = torch.from_numpy(
            token_dict["real_indices"][:, :max_len].astype(np.int64, copy=False)
        )
        self.attention_mask = torch.from_numpy(
            token_dict["attention_mask"][:, :max_len].astype(np.bool_, copy=False)
        )
        self.connect_comp = torch.from_numpy(
            token_dict["connect_comp"][:, :max_len].astype(np.int64, copy=False)
        )
        self.rna_type = torch.from_numpy(
            token_dict["rna_type"][:, :max_len].astype(np.int64, copy=False)
        )
        self.neighbor_gene_dist = torch.from_numpy(
            token_dict["neighbor_gene_dist"][:, :max_len].astype(np.float32, copy=False)
        )
        self.exp = torch.from_numpy(
            token_dict["exp"][:, :max_len].astype(np.float32, copy=False)
        )

        # Keep raw index as a numpy array / python list, no per-sample str conversion here
        self.cell_raw_index = np.asarray(token_dict["cell_raw_index"])

    def __len__(self):
        return self.real_indices.shape[0]

    def __getitem__(self, idx):
        return (
            self.real_indices[idx],
            self.attention_mask[idx],
            self.connect_comp[idx],
            self.rna_type[idx],
            self.cell_raw_index[idx],
            self.neighbor_gene_dist[idx],
            self.exp[idx],
        )

def run_bb_inference_fast(
    adata,
    gene_dict_path: str,
    config_train: dict,
    pretrain_ckpt: str,
    device,
    species: str,
    assay: str,
    batch_size: int = 64,
    num_workers: int = 4,
    use_hvg: bool = True,
    n_hvg: int = 1000,
    use_dev_abs: bool = False,
    save_path: str = None,
):
    """
    Fast in-memory BrainBeacon inference.

    This path removes intermediate joblib token I/O and directly runs:
    adata -> in-memory tokenization -> dataloader -> BrainBeacon inference.

    Returns
    -------
    ordered_embeddings : np.ndarray
        Stage1 embeddings aligned to adata.obs_names.
    """

    time0 = time.time()
    config_train = normalize_brainbeacon_model_config(config_train)

    # ====== 1. In-memory tokenization ======
    print("[Stage1 memory] Tokenizing in memory...")
    t_tok = time.time()
    use_cell_density = config_train.get("use_cell_density", True)
    use_gene_deviation = config_train.get("use_gene_deviation", True)
    use_mean_norm = config_train.get("use_mean_norm", True)
    hvg_log1p = config_train.get("hvg_log1p", False)

    token_dict = tokenize_adata_in_memory(
        adata,
        gene_dict_path,
        species=species,
        assay=assay,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        use_dev_abs=use_dev_abs,
        cell_density=use_cell_density,
        gene_niche=use_gene_deviation,
        use_mean_norm=use_mean_norm,
        hvg_log1p=hvg_log1p,
    )
    print(
        f"[Stage1 memory] Tokenization done: "
        f"{token_dict['real_indices'].shape[0]} cells, "
        f"{time.time() - t_tok:.1f}s"
    )

    # ====== 2. Build dataset and dataloader ======
    dataset = InMemoryTokenDataset(token_dict, max_len=config_train['context_length'])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ====== 3. Model inference ======
    print(f"[Stage1 memory] Running inference (batch_size={batch_size})...")
    esm_emb_path = resolve_path('ESM_EMBED_PATH')
    config_train["esm_emb_path"] = esm_emb_path
    pipeline = CellEmbeddingPipeline(
        pretrain_ckpt=pretrain_ckpt,
        model_config=config_train,
        device=device,
    )
    pipeline.model.eval()

    # esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location="cpu")
    esm_embedding_map = torch.load(esm_emb_path, map_location="cpu").to(device)
    esm_embedding_map = pipeline.apply_homo_mean_to_esm_map(esm_embedding_map)  #

    pool_skip_tokens = config_train.get("pool_skip_tokens", 2)
    weight_mode = config_train.get("weight_mode", "expression")
    cd_weight = config_train.get("cd_weight", 0.02)
    if not config_train.get("use_cell_density", True):
        cd_weight = 0.0
    expr_mode = config_train.get("expr_mode", None)

    all_indices = []
    all_embeddings = []

    # with torch.no_grad():
    with torch.inference_mode():
        for real_indices, attention_mask, connect_comp, rna_type, cell_raw_idx, neighbor_gene_distribution, exp in tqdm(
                dataloader,
                desc="[Stage1 memory] Inference",
        ):
            real_indices = real_indices.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            connect_comp = connect_comp.to(device, non_blocking=True)
            rna_type = rna_type.to(device, non_blocking=True)
            neighbor_gene_distribution = neighbor_gene_distribution.to(device, non_blocking=True).long()
            exp = exp.to(device, non_blocking=True)

            real_indices_view = real_indices.view(-1).long()
            esm_embedding = torch.index_select(
                esm_embedding_map,
                dim=0,
                index=real_indices_view,
            )
            esm_embedding = esm_embedding.view(
                real_indices.shape[0],
                real_indices.shape[1],
                esm_embedding.shape[-1],
            )

            # better than torch.where(...)
            sequence_mask = (~attention_mask).long()

            output = pipeline.model(
                real_indices,
                connect_comp,
                rna_type,
                attention_mask,
                esm_embedding,
                neighbor_gene_distribution,
                sequence_mask,
            )

            # ====== Pooling on GPU ======
            if weight_mode == "expression":
                aux = torch.zeros((exp.shape[0], 2), device=exp.device)
                cd = torch.full((exp.shape[0], 1), cd_weight, device=exp.device)
                gene_expr = exp[:, 3:]

                if expr_mode == "log1pnorm":
                    gene_expr = torch.log1p(gene_expr) / torch.log(
                        torch.tensor(2.0, device=gene_expr.device)
                    )

                gene_expr = gene_expr / gene_expr.sum(dim=1, keepdim=True).clamp(min=1e-6)
                exp_full = torch.cat([aux, cd, gene_expr], dim=1)
                expr_weights = exp_full[:, pool_skip_tokens:]
            else:
                expr_weights = None

            output = masked_weighted_pooling(
                output[:, pool_skip_tokens:, :],
                sequence_mask[:, pool_skip_tokens:],
                expr_weights=expr_weights,
                weight_mode=weight_mode,
                weight_decay=config_train.get("weight_decay", 0.998),
                temperature=config_train.get("temperature", 300),
            )

            output = output.detach().cpu()

            all_indices.extend(cell_raw_idx)
            all_embeddings.append(output.numpy())

    pred_indices = np.array(all_indices)
    pred_embeddings = np.concatenate(all_embeddings, axis=0)

    # ====== 4. Order check ======
    obs_names = np.array(adata.obs_names)

    if np.array_equal(pred_indices, obs_names):
        print("obs_names and pred_indices are in the same order.")
        ordered_embeddings = pred_embeddings
    elif len(pred_indices) == len(obs_names):
        idx_map = {name: i for i, name in enumerate(pred_indices)}
        try:
            order = [idx_map[name] for name in obs_names]
        except KeyError as e:
            raise ValueError(
                f"Embedding order check failed: missing cell id {e} in predicted outputs."
            ) from e
        ordered_embeddings = pred_embeddings[order]
    else:
        raise ValueError(
            "Embedding order check failed: the number of predicted embeddings does not match "
            f"adata.obs_names ({len(pred_indices)} vs {len(obs_names)})."
        )

    # ====== 5. Save ======
    if save_path is not None:
        np.savez_compressed(save_path, embeddings=ordered_embeddings)
        print(f"[Stage1 memory] Embeddings saved to {save_path}")

    elapsed = time.time() - time0
    print(f"[Stage1 memory] Total time: {elapsed / 60:.2f} min")

    del pipeline, esm_embedding_map
    torch.cuda.empty_cache()

    return ordered_embeddings

def run_tokenization(
    adata,
    gene_dict_path,
    bb_token_dir,
    species,
    assay,
    use_hvg=True,
    n_hvg=1000,
    force_tokenize=True,
    use_dev_abs=False
):
    """
    Tokenize input AnnData into BrainBeacon joblib bundles.
    """
    from brainbeacon.tokenizer import tokenization_h5ad

    if not os.path.exists(bb_token_dir):
        os.makedirs(bb_token_dir)

    def _list_token_dirs(base_dir):
        token_dirs = []
        for item in sorted(os.listdir(base_dir)):
            item_path = os.path.join(base_dir, item)
            if not os.path.isdir(item_path):
                continue
            if not item.startswith("tokens-"):
                continue
            if not any(name.startswith("real_indices_") and name.endswith(".job") for name in os.listdir(item_path)):
                continue
            token_dirs.append(item_path)
        return token_dirs

    existing_token_dirs = _list_token_dirs(bb_token_dir)

    if existing_token_dirs and not force_tokenize:
        print(f"Tokenized joblib bundles found ({len(existing_token_dirs)} dirs). Skipping tokenization.")
        return bb_token_dir

    if force_tokenize:
        print("Forcing re-tokenization: clearing existing token folders...")
        for item in os.listdir(bb_token_dir):
            item_path = os.path.join(bb_token_dir, item)
            if item.startswith("tokens-") and os.path.isdir(item_path):
                shutil.rmtree(item_path)

    start = time.time()
    print("No existing tokenized bundles found. Running tokenization...")
    tokenization_h5ad(
        adata,
        gene_dict_path,
        species=species,
        assay=assay,
        output_path=bb_token_dir,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        use_dev_abs=use_dev_abs,
    )

    token_dirs = _list_token_dirs(bb_token_dir)
    if not token_dirs:
        raise RuntimeError(
            f"Tokenization completed, but no token joblib bundles were found under {bb_token_dir}."
        )

    end = time.time()
    print(f"Preprocessing time: {(end - start)/60:.2f} minutes")
    return bb_token_dir


def run_stage1_inference(
    adata,
    token_data_path,
    config_train,
    pretrain_ckpt,
    device,
    save_path=None
):
    time0 = time.time()
    config_train = normalize_brainbeacon_model_config(config_train)
    config_train["batch_size"] = 1  # Use batch size of 1 for inference
    pipeline = CellEmbeddingPipeline(pretrain_ckpt=pretrain_ckpt, model_config=config_train, device=device)

    # Generate embeddings
    pred = pipeline.run(data_paths=token_data_path, config_train=config_train)

    # Extract index and embeddings from pred
    pred_indices, pred_embeddings = zip(*[(str(idx[0]), emb.numpy()) for idx, emb in pred])
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

    del pipeline, pred, pred_indices, pred_embeddings
    torch.cuda.empty_cache()
    return ordered_embeddings

def run_stage1_pipeline(
    adata,
    gene_dict_path: str,
    stage1_ckpt_path: str,
    config_train: dict,
    species: str,
    assay: str,
    token_output_dir: str | None = None,
    save_embedding_path: str | None = None,
    stage1_mode: str = "disk",
    device=None,
):
    """
    Run Stage1 pipeline for BrainBeacon.

    Supported modes
    ---------------
    - "disk":
        Tokenize to disk with joblib bundles, then run disk-based inference.
    - "memory":
        Run in-memory fast inference without writing token bundles to disk.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object.
    gene_dict_path : str
        Path to gene dictionary used by Stage1 tokenization.
    stage1_ckpt_path : str
        Path to pretrained BrainBeacon checkpoint.
    config_train : dict
        Stage1 config dictionary.
    species : str
        Species identifier.
    assay : str
        Assay / platform identifier.
    token_output_dir : str or None, optional
        Directory for saving tokenized joblib bundles in disk mode.
    save_embedding_path : str or None, optional
        Path for saving Stage1 embeddings.
    stage1_mode : str, default="disk"
        Stage1 execution mode. One of {"disk", "memory"}.
    device : torch.device or str or None, optional
        Device used for inference.

    Returns
    -------
    adata : anndata.AnnData
        AnnData with `adata.obsm["bb_emb"]`.
    """
    if adata is None:
        raise ValueError("`adata` must be provided for Stage1 pipeline.")
    if gene_dict_path is None:
        raise ValueError("`gene_dict_path` must be provided for Stage1 pipeline.")
    if stage1_ckpt_path is None:
        raise ValueError("`stage1_ckpt_path` must be provided for Stage1 pipeline.")
    if config_train is None:
        raise ValueError("`config_train` must be provided for Stage1 pipeline.")

    if stage1_mode not in {"disk", "memory"}:
        raise ValueError(
            f"Unsupported `stage1_mode`: {stage1_mode}. "
            "Currently supported modes are {'disk', 'memory'}."
        )

    # config_train = normalize_brainbeacon_model_config(config_train)

    force_tokenize = config_train.get("force_tokenize", True)
    use_hvg = config_train.get("use_hvg", True)
    n_hvg = config_train.get("n_hvg", 5000)
    use_dev_abs = config_train.get("use_dev_abs", True)

    # ====== Disk mode ======
    if stage1_mode == "disk":
        if token_output_dir is None:
            raise ValueError("`token_output_dir` must be provided when `stage1_mode='disk'`.")

        stage1_start_time = time.time()

        print("====== Stage1: Tokenization (disk) ======")
        token_data_path = run_tokenization(
            adata=adata,
            gene_dict_path=gene_dict_path,
            bb_token_dir=token_output_dir,
            species=species,
            assay=assay,
            force_tokenize=force_tokenize,
            use_hvg=use_hvg,
            n_hvg=n_hvg,
            use_dev_abs=use_dev_abs,
        )

        print("====== Stage1: Inference (disk) ======")
        if save_embedding_path is not None and os.path.exists(save_embedding_path) and not force_tokenize:
            print(f"Skipping stage1 inference. Found existing file: {save_embedding_path}")
            bb_emb = np.load(save_embedding_path)["embeddings"]
        else:
            bb_emb = run_stage1_inference(
                adata=adata,
                config_train=config_train,
                pretrain_ckpt=stage1_ckpt_path,
                token_data_path=token_data_path,
                save_path=save_embedding_path,
                device=device,
            )

            if save_embedding_path is not None:
                print(f"Stage1 inference complete. Saved to: {save_embedding_path}")
            else:
                print("Stage1 inference complete.")

        adata.obsm["bb_emb"] = bb_emb
        stage1_end_time = time.time()
        print(f"[Stage1 total - disk] Time cost: {(stage1_end_time - stage1_start_time):.2f} sec")
        return adata

    # ====== Memory mode ======
    if stage1_mode == "memory":
        stage1_start_time = time.time()

        print("====== Stage1: Inference (memory) ======")
        if save_embedding_path is not None and os.path.exists(save_embedding_path) and not force_tokenize:
            print(f"Skipping stage1 inference. Found existing file: {save_embedding_path}")
            bb_emb = np.load(save_embedding_path)["embeddings"]
        else:
            bb_emb = run_bb_inference_fast(
                adata=adata,
                gene_dict_path=gene_dict_path,
                config_train=config_train,
                pretrain_ckpt=stage1_ckpt_path,
                device=device,
                species=species,
                assay=assay,
                batch_size=config_train.get("batch_size", 64),
                num_workers=config_train.get("num_workers", 4),
                use_hvg=use_hvg,
                n_hvg=n_hvg,
                use_dev_abs=use_dev_abs,
                save_path=save_embedding_path,
            )

            if save_embedding_path is not None:
                print(f"Stage1 inference complete. Saved to: {save_embedding_path}")
            else:
                print("Stage1 inference complete.")

        adata.obsm["bb_emb"] = bb_emb
        stage1_end_time = time.time()
        print(f"[Stage1 total - memory] Time cost: {(stage1_end_time - stage1_start_time):.2f} sec")
        return adata


class DictEncoder:
    def __init__(self, mapping):
        self.mapping = mapping

    def transform(self, values):
        return np.asarray([self.mapping[v] for v in values], dtype=np.int64)


class IdentityEncoder:
    def transform(self, values):
        return np.asarray(values, dtype=np.int64)


def build_covariate_encoders(config):
    cov_fields = config.get("covariate_fields")
    if not cov_fields:
        return None
    return {field: IdentityEncoder() for field in cov_fields}


def run_stage2_pipeline(
        adata,
        stage1_ckpt_path,
        stage2_ckpt_path=None,
        config=None,
        do_fit=True,
        fit_epochs=500,
        use_single_slice=False,
        use_coord=True,
        save_model_path=None,
        save_embedding_path=None,
        device=None,
):
    if config is None:
        raise ValueError("`config` must be provided for Stage2 pipeline.")
    # Load AnnData file
    data = adata.copy()
    data.obs_names_make_unique()
    # set train/valid split
    data.obs['valid_split'] = 'train'
    if 'slice' not in data.obs.columns:
        data.obs['slice'] = 'default'
    for slice_id in data.obs['slice'].unique():
        idx = data.obs['slice'] == slice_id
        cell_idx = np.where(idx)[0]
        n_valid = max(1, int(len(cell_idx) * 0.1))  # Ensure at least one cell is selected for validation
        valid_cells = np.random.choice(cell_idx, n_valid, replace=False)
        data.obs.iloc[valid_cells, data.obs.columns.get_loc('valid_split')] = 'valid'

    # Use slice as batch/group identifier for Stage2
    data.obs['batch'] = data.obs['slice']

    if use_coord and "spatial" in data.obsm.keys():
        for batch_id in data.obs['batch'].unique():
            idx = data.obs['batch'] == batch_id
            spatial = data.obsm['spatial'][idx]
            spatial = np.asarray(spatial)  # assure spatial is a NumPy array
            spatial_min = spatial.min(axis=0)
            spatial_max = spatial.max(axis=0)
            normalized = (spatial - spatial_min) / (spatial_max - spatial_min + 1e-8)

            data.obs.loc[idx, 'x_FOV_px'] = normalized[:, 0]
            data.obs.loc[idx, 'y_FOV_px'] = normalized[:, 1]

    platform_is_snrna = "platform" in data.obs.columns and data.obs["platform"].astype(str).str.lower().eq("snrna").all()
    if "spatial" not in data.obsm or platform_is_snrna:
        config = dict(config)
        config.update({"sampling_mode": "random", "use_patch": False, "eval_use_patch": False})
        print("[INFO] Non-spatial or snRNA-seq data detected. Disable Stage2 spatial patch mode.")

    # clear GPU memory before re-initializing the pipeline
    torch.cuda.empty_cache()
    pipeline = ReconstructPipeline(
        config=config,
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        use_pretrain=config.get("use_pretrain", True),
    )
    PLATFORM_MAP = {
        "merfish": 0,
        "xenium": 1,
        "starmap": 2,
        "slideseqv2": 3,
        "stereo": 4,
        "snrna": 5,  # map snRNA to merfish platform
    }

    if do_fit:
        # Only sample one slice if requested
        if use_single_slice:
            # np.random.seed(42)
            rng = np.random.RandomState(None)  # use local random
            chosen_slice = rng.choice(data.obs['slice'].unique())
            fit_data = data[data.obs['slice'] == chosen_slice].copy()
            print(f"Training only on slice: {chosen_slice} ({fit_data.n_obs} cells)")
        else:
            fit_data = data.copy()

        MAX_CELLS = config["max_train_cells"]
        if fit_data.n_obs > MAX_CELLS:
            print(f"[Warning] Too many cells in slice ({fit_data.n_obs}), subsampling to {MAX_CELLS}")
            sampled_indices = np.random.choice(fit_data.n_obs, MAX_CELLS, replace=False)
            fit_data = fit_data[sampled_indices].copy()
            print("fit data shape:", fit_data.shape)

        covariate_fields = config.get("covariate_fields", None)
        if covariate_fields is not None:
            if "platform_cov" in covariate_fields and "platform_cov" not in fit_data.obs.columns:
                if "platform" in fit_data.obs.columns:
                    fit_data.obs["platform_cov"] = fit_data.obs["platform"].map(PLATFORM_MAP).astype(np.int64)
            available_covariates = [c for c in covariate_fields if c in fit_data.obs.columns]
            missing_covariates = [c for c in covariate_fields if c not in fit_data.obs.columns]

            if len(missing_covariates) > 0:
                print(f"[Warning] Missing covariate fields in obs, drop them: {missing_covariates}")
            for c in available_covariates:
                fit_data.obs[c] = fit_data.obs[c].astype(np.int64)

            covariate_fields = available_covariates if len(available_covariates) > 0 else None
            config["covariate_fields"] = covariate_fields
            print(f"Using covariate fields: {covariate_fields}")

        covariate_encoders = build_covariate_encoders(config)
        train_config = {
            'epochs': fit_epochs,
            'max_batch_size': config['max_batch_size'],
            'use_patch': config.get("use_patch", True),
            'knn_k': config.get("knn_k", 10),
        }
        pipeline.fit(
            fit_data,  # AnnData object
            config_override=train_config,
            split_field='valid_split',
            train_split='train',
            valid_split='valid',
            covariate_fields=covariate_fields,
            covariate_encoders=covariate_encoders,
            device=device
        )
        if config.get("fit_only", False):
            if save_model_path is not None:
                torch.save(pipeline.model.state_dict(), save_model_path)
                print(f"Model saved to {save_model_path}")
            print("[INFO] fit_only=True. Skip Stage2 prediction.")
            torch.cuda.empty_cache()
            return data

    inference_config = {
        'max_eval_batch_size': config['max_eval_batch_size'],
        'use_patch': config.get("eval_use_patch", True),
        'knn_k': config.get("eval_knn_k", 5),
        'center_ratio': config.get("eval_center_ratio", 0.5),
    }
    result = pipeline.predict(
        data,
        inference_config=inference_config,
        # output_attentions=output_attentions,
        device=device
    )
    pred = result['pred']
    latent = result['latent']
    # attention = result.get('attention', None)

    target_genes = pipeline.target_genes  # this was set inside predict()
    data = data[:, target_genes].copy()  # now data.var.index == target_genes

    data.obsm['X_emb'] = latent.cpu().numpy()  # Store embeddings in AnnData object
    data.obsm['X_pred'] = pred.cpu().numpy()  # Store predicted gene
    # if output_attentions and attention is not None:
    #     data.uns['attention'] = attention.cpu().numpy()

    if save_model_path is not None:
        torch.save(pipeline.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    if save_embedding_path is not None:
        np.savez_compressed(save_embedding_path, embeddings=data.obsm['X_emb'])
        print(f"Embeddings saved to {save_embedding_path}")

    return data


def merge_config(default_config: dict, new_config: dict | None = None) -> dict:
    """
    Merge a user-provided config into the default config.
    """
    config = copy.deepcopy(default_config)
    if new_config is not None:
        config.update(new_config)
    return config

def resolve_pipeline_paths(
    output_dir: str,
    output_prefix: str,
    save_model: bool = True,
    save_model_path: str | None = None,
):
    """
    Resolve default output paths for the full BrainBeacon pipeline.
    """
    if output_dir is None:
        raise ValueError("`output_dir` must be provided.")
    if output_prefix is None:
        raise ValueError("`output_prefix` must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    stage1_token_dir = os.path.join(output_dir, f"{output_prefix}_tokens")
    stage1_embedding_path = os.path.join(output_dir, f"{output_prefix}_stage1_embeddings.npz")
    stage2_embedding_path = os.path.join(output_dir, f"{output_prefix}_stage2_embeddings.npz")

    if save_model:
        if save_model_path is None:
            stage2_model_path = os.path.join(output_dir, f"{output_prefix}_cellformer.pt")
        else:
            stage2_model_path = save_model_path
    else:
        stage2_model_path = None

    return {
        "stage1_token_dir": stage1_token_dir,
        "stage1_embedding_path": stage1_embedding_path,
        "stage2_embedding_path": stage2_embedding_path,
        "stage2_model_path": stage2_model_path,
    }

def run_brainbeacon_pipeline(
    adata=None,
    adata_path: str | None = None,
    species: str = None,
    assay: str = None,
    gene_dict_path: str = None,
    stage1_ckpt_path: str | None = None,
    stage2_ckpt_path: str | None = None,
    output_dir: str | None = None,
    output_prefix: str | None = None,
    stage1_config: dict | None = None,
    stage2_config: dict | None = None,
    do_fit: bool = True,
    fit_epochs: int = 100,
    use_single_slice: bool = False,
    stage1_mode: str = "memory",
    device=None,
    seed: int = 42,
    deterministic: bool = True,
    save_model: bool = True,
    save_model_path: str | None = None,
):
    """
    Run the full two-stage BrainBeacon pipeline.

    This is the top-level user-facing function for the BrainBeacon workflow.

    Stage 1 (intra-cell modeling)
    -----------------------------
    1. Tokenize gene expression into sequence-like model inputs.
    2. Run pretrained BrainBeacon inference to generate cell-level embeddings.

    Stage 2 (inter-cell modeling)
    -----------------------------
    1. Perform CellFormer-based spatial / inter-cell modeling.
    2. Optionally fit or fine-tune the stage 2 model.
    3. Generate refined latent representations.

    Parameters
    ----------
    adata : anndata.AnnData, optional
        Input AnnData object. If None, `adata_path` must be provided.
    adata_path : str or None, optional
        Path to the input AnnData file.
    species : str or None, optional
        Species identifier used for tokenization and metadata assignment.
    assay : str or None, optional
        Assay / platform name stored in `adata.obs["platform"]`.
    gene_dict_path : str or None, optional
        Path to the gene dictionary used by stage1 tokenization.
    stage1_ckpt_path : str or None, optional
        Path to the pretrained stage1 BrainBeacon checkpoint.
    stage2_ckpt_path : str or None, optional
        Path to the pretrained stage2 CellFormer checkpoint.
    output_dir : str or None, optional
        Directory used to save intermediate outputs and final artifacts.
    output_prefix : str or None, optional
        Prefix used to name output files.
    stage1_config : dict or None, optional
        Flat override dictionary for stage1 settings.
    stage2_config : dict or None, optional
        Flat override dictionary for stage2 settings.
    do_fit : bool, default=True
        Whether to fit / fine-tune the stage2 model in this run.
    fit_epochs : int, default=100
        Number of epochs used when `do_fit=True`.
    use_single_slice : bool, default=False
        Whether to use only a single slice for training the stage2 model.
    stage1_mode : str, default="disk"
        Stage1 execution mode. One of {"disk", "memory"}. "disk" mode uses joblib bundles for tokenization and inference.
    device : torch.device or str or None, optional
        Device used for computation.
    seed : int, default=42
        Random seed for reproducibility.
    deterministic : bool, default=True
        Whether to enforce deterministic behavior when possible.
    save_model : bool, default=True
        Whether to save the stage2 model after running stage2.
    save_model_path : str or None, optional
        Path used to save the stage2 model. If None and `save_model=True`,
        a default path under `output_dir` will be used.

    Returns
    -------
    adata : anndata.AnnData
        AnnData object after completing the full BrainBeacon pipeline.
    """
    # ====== Setup ======
    if seed is not None:
        set_seed(seed, deterministic=deterministic)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    if adata is None and adata_path is None:
        raise ValueError("One of `adata` or `adata_path` must be provided.")
    if adata is not None and adata_path is not None:
        raise ValueError("Only one of `adata` or `adata_path` should be provided.")
    if species is None:
        raise ValueError("`species` must be provided.")
    if assay is None:
        raise ValueError("`assay` must be provided.")
    if gene_dict_path is None:
        raise ValueError("`gene_dict_path` must be provided.")
    if stage1_ckpt_path is None:
        raise ValueError("`stage1_ckpt_path` must be provided.")
    if output_dir is None:
        raise ValueError("`output_dir` must be provided.")
    if output_prefix is None:
        raise ValueError("`output_prefix` must be provided.")

    cfg1 = merge_config(default_config=default_config1, new_config=stage1_config)
    cfg2 = merge_config(default_config=default_config2, new_config=stage2_config)
    cfg2["n_aux"] = cfg1["n_aux"]
    cfg2["bb_emb_dim"] = cfg1["dim_model"]
    cfg2["gene_dict_path"] = gene_dict_path

    # ====== Resolve output paths ======
    path_dict = resolve_pipeline_paths(
        output_dir=output_dir,
        output_prefix=output_prefix,
        save_model=save_model,
        save_model_path=save_model_path,
    )

    # ====== Load adata ======
    if adata is None:
        adata = sc.read_h5ad(adata_path)
    else:
        adata = adata.copy()

    adata.obs["species"] = species
    adata.obs["platform"] = assay

    # ====== Stage1 Pipeline ======
    adata = run_stage1_pipeline(
        adata=adata,
        gene_dict_path=gene_dict_path,
        stage1_ckpt_path=stage1_ckpt_path,
        config_train=cfg1,
        species=species,
        assay=assay,
        token_output_dir=path_dict["stage1_token_dir"],
        save_embedding_path=path_dict["stage1_embedding_path"],
        stage1_mode=stage1_mode,
        device=device,
    )

    # ====== Stage2 Pipeline ======
    adata = run_stage2_pipeline(
        adata=adata,
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        config=cfg2,
        do_fit=do_fit,
        fit_epochs=fit_epochs,
        use_single_slice=use_single_slice,
        save_model_path=path_dict["stage2_model_path"],
        save_embedding_path=path_dict["stage2_embedding_path"],
        device=device,
    )

    return adata

def train_stage2_on_multi_adata(
    dataset_info_list: list[dict],
    gene_dict_path: str,
    stage1_ckpt_path: str,
    stage2_ckpt_path: str | None = None,
    output_dir: str | None = None,
    output_prefix: str = "brainbeacon",
    stage1_config: dict | None = None,
    stage2_config: dict | None = None,
    num_global_epochs: int = 100,
    per_dataset_epochs: int = 50,
    shuffle_each_epoch: bool = True,
    use_single_slice: bool = False,
    stage1_mode: str = "memory",
    fit_only: bool = True,
    device=None,
    seed: int = 42,
    deterministic: bool = True,
    save_model: bool = True,
    save_all_epochs: bool = False,
) -> str | None:
    """
    Sequentially train / finetune Stage2 across multiple AnnData datasets.

    This function supports two input styles for each dataset item in
    `dataset_info_list`:

    Style 1 (legacy path-based):
    {
        "data_name": str,
        "data_dir": str,
        "adata_name": str,
        "species": str,
        "assay": str,
    }

    Style 2 (in-memory adata-based):
    {
        "data_name": str,
        "adata": AnnData,
        "species": str,
        "assay": str,
    }

    Parameters
    ----------
    dataset_info_list : list[dict]
        A list of dataset descriptions. Each item must provide either:
        - (`data_dir` and `adata_name`), or
        - `adata`
    gene_dict_path : str
        Path to gene dictionary.
    stage1_ckpt_path : str
        Path to pretrained Stage1 checkpoint.
    stage2_ckpt_path : str or None, optional
        Initial Stage2 checkpoint path. If None, Stage2 starts without a pretrained ckpt.
    output_dir : str
        Root output directory.
    output_prefix : str, default="brainbeacon"
        Prefix for saved outputs.
    stage1_config : dict or None
        Override config for Stage1.
    stage2_config : dict or None
        Override config for Stage2.
    num_global_epochs : int, default=100
        Number of outer epochs over all datasets.
    per_dataset_epochs : int, default=50
        Stage2 fit epochs for each dataset in each outer epoch.
    shuffle_each_epoch : bool, default=True
        Whether to shuffle dataset order every outer epoch.
    use_single_slice : bool, default=False
        Whether to use a single slice in Stage2 fitting.
    stage1_mode : str, default="memory"
        One of {"disk", "memory"}.
    fit_only : bool, default=True
        Whether to run Stage2 in fit-only mode (no prediction, just update the model).
    device : torch.device or str or None
        Compute device.
    seed : int, default=42
        Random seed.
    deterministic : bool, default=True
        Whether to enforce deterministic behavior.
    save_model : bool, default=True
        Whether to save updated Stage2 checkpoints.
    save_all_epochs : bool, default=False
        Whether to save one checkpoint per epoch per dataset.

    Returns
    -------
    final_stage2_ckpt_path : str or None
        Final Stage2 checkpoint path after sequential training.
    """
    if dataset_info_list is None or len(dataset_info_list) == 0:
        raise ValueError("`dataset_info_list` must be a non-empty list.")
    if gene_dict_path is None:
        raise ValueError("`gene_dict_path` must be provided.")
    if stage1_ckpt_path is None:
        raise ValueError("`stage1_ckpt_path` must be provided.")
    if output_dir is None:
        raise ValueError("`output_dir` must be provided.")

    os.makedirs(output_dir, exist_ok=True)

    current_stage2_ckpt_path = stage2_ckpt_path

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    stage2_config = {} if stage2_config is None else copy.deepcopy(stage2_config)
    stage2_config["fit_only"] = fit_only
    for global_epoch in range(num_global_epochs):
        print(f"\n========== Global Epoch {global_epoch + 1}/{num_global_epochs} ==========")

        datasets_this_epoch = list(dataset_info_list)
        if shuffle_each_epoch:
            rng = np.random.RandomState(seed + global_epoch if seed is not None else None)
            rng.shuffle(datasets_this_epoch)

        for ds_idx, ds in enumerate(datasets_this_epoch):
            if "data_name" not in ds:
                raise ValueError("Each dataset dict must contain `data_name`.")
            if "species" not in ds:
                raise ValueError(f"Dataset `{ds.get('data_name', 'unknown')}` missing `species`.")
            if "assay" not in ds:
                raise ValueError(f"Dataset `{ds.get('data_name', 'unknown')}` missing `assay`.")

            data_name = ds["data_name"]
            species = ds["species"]
            assay = ds["assay"]

            print(f"\n--- Training on {data_name} ({ds_idx + 1}/{len(datasets_this_epoch)}) ---")

            # Support both:
            # 1) in-memory AnnData: ds["adata"]
            # 2) path-based input: ds["data_dir"] + ds["adata_name"]
            adata = ds.get("adata", None)
            data_dir = ds.get("data_dir", None)
            adata_name = ds.get("adata_name", None)

            if adata is not None:
                adata_path = None
                print(f"[Input mode] Using in-memory adata for {data_name}")
            else:
                if data_dir is None or adata_name is None:
                    raise ValueError(
                        f"Dataset `{data_name}` must provide either `adata` "
                        f"or both `data_dir` and `adata_name`."
                    )
                adata_path = os.path.join(data_dir, adata_name)
                print(f"[Input mode] Using adata_path: {adata_path}")

            dataset_output_dir = os.path.join(output_dir, data_name)
            os.makedirs(dataset_output_dir, exist_ok=True)

            if save_model:
                if save_all_epochs:
                    current_save_model_path = os.path.join(
                        ckpt_dir,
                        f"{output_prefix}_epoch{global_epoch + 1:02d}_{data_name}.pt"
                    )
                else:
                    current_save_model_path = os.path.join(
                        ckpt_dir,
                        f"{output_prefix}_latest.pt"
                    )
            else:
                current_save_model_path = None

            run_brainbeacon_pipeline(
                adata=adata,
                adata_path=adata_path,
                species=species,
                assay=assay,
                gene_dict_path=gene_dict_path,
                stage1_ckpt_path=stage1_ckpt_path,
                stage2_ckpt_path=current_stage2_ckpt_path,
                output_dir=dataset_output_dir,
                output_prefix=f"{output_prefix}_{data_name}",
                stage1_config=stage1_config,
                stage2_config=stage2_config,
                do_fit=True,
                fit_epochs=per_dataset_epochs,
                use_single_slice=use_single_slice,
                stage1_mode=stage1_mode,
                device=device,
                seed=seed,
                deterministic=deterministic,
                save_model=save_model,
                save_model_path=current_save_model_path,
            )
            import gc
            del adata
            torch.cuda.empty_cache()
            gc.collect()

            if save_model:
                current_stage2_ckpt_path = current_save_model_path
                print(f"[Stage2 updated] Current checkpoint: {current_stage2_ckpt_path}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_stage2_ckpt_path = current_stage2_ckpt_path

    if save_model and current_stage2_ckpt_path is not None:
        if not os.path.exists(current_stage2_ckpt_path):
            raise FileNotFoundError(
                f"Current Stage2 checkpoint not found after training: {current_stage2_ckpt_path}"
            )

        final_stage2_ckpt_path = os.path.join(
            ckpt_dir,
            f"{output_prefix}_final.pt"
        )

        shutil.copy2(current_stage2_ckpt_path, final_stage2_ckpt_path)
        print(f"[Stage2 final] Copied completed checkpoint to: {final_stage2_ckpt_path}")

    return final_stage2_ckpt_path