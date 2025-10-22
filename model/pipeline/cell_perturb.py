import os
import time
import torch
import joblib
import shutil
import torch.nn as nn
import scanpy as sc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, List

from config.config_train_cdniche import config_train
from model.brain_beacon import BrainBeacon
from model.utils import tokenization_h5ad, process_parquet, set_seed
from model.bbcellformer.pipeline.reconstruction import ReconstructPipeline
from model.bbcellformer.pipeline.perturb import PerturbationReconstructionPipeline
from config.config_cdniche import GENE_DICT_PATH
from config.config_train_cdniche import config_train

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
        self.model_config = model_config
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
            use_gene_id_emb=self.model_config.get("use_gene_id_emb", True),  # Added: configurable switch
            use_homo_emb=self.model_config.get("use_homo_emb", True),  # new
            use_rna_type_emb=self.model_config.get("use_rna_type_emb", True),  # new
            use_esm_emb=self.model_config.get("use_esm_emb", True)  # new
        )

    def forward(self, x_gene_id, x_connect_id, x_rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask):
        token_embedding = self.pretrain_model.embedding(x_gene_id, x_connect_id, x_rna_type)
        token_embedding += self.pretrain_model.esm_embedding_projection(esm_embedding)
        if self.model_config['neighbor_enhance']:
            neighbor_embedding = self.pretrain_model.neighbor_projection(neighbor_gene_distribution)
            token_embedding += neighbor_embedding
        pos = self.pretrain_model.pos.to(token_embedding.device)
        pos_embedding = self.pretrain_model.positional_embedding(pos)  # batch x (n_tokens) x dim_model
        embeddings = self.pretrain_model.dropout(token_embedding + pos_embedding)
        transformer_output = self.pretrain_model.encoder(embeddings, src_key_padding_mask=attention_mask)
        return transformer_output


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
            # 确保 cell_raw_idx 是字符串类型
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

            # 如果遇到错误，返回一个空数据样本，而不是递归调用 (避免无限递归)
            if idx + 1 >= self.total_length:
                # 创建一个安全的默认返回值
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
                    empty_tensor
                )
            else:
                return self.__getitem__(idx + 1)


class CellEmbeddingPipeline:
    def __init__(self, pretrain_ckpt: str, model_config: dict, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the pipeline with model_raw and device settings.
        """
        self.device = device
        self.model_config = model_config
        self.model = None
        self.pretrain_ckpt: str = pretrain_ckpt
        self.initialize_model()

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

        for prefix in sorted(os.listdir(data_paths)):
            if prefix.endswith(".parquet"):
                continue
            file_prefix_list.append(os.path.join(data_paths, prefix))
            for file in os.listdir(os.path.join(data_paths, prefix)):
                file_path = os.path.join(data_paths, prefix, file)
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

    def infer(self, dataloader, config_train: dict):
        """
        Run inference on new data using the pretrained model_raw.
        """
        # Switch to evaluation mode
        self.model.eval()
        # Load ESM embedding map
        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        indexed_embeddings = []
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

        return indexed_embeddings

    def run(self, data_paths: List[str], config_train: dict):
        """
        Main method to run the entire training pipeline.
        """
        dataset = self.load_dataset(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        
        pred = self.infer(data_loader, config_train)
        return pred


def run_tokenization(
    adata_path,
    bb_token_dir,
    gene_dict_path,
    mean_path,
    specie,
    assay,
    use_hvg=True,
    n_hvg=1000,
    force_tokenize=True
):
    if not os.path.exists(bb_token_dir):
        os.makedirs(bb_token_dir)
    # Check if both .parquet files and corresponding tokens-* directories exist
    existing_parquets = [f for f in os.listdir(bb_token_dir) if f.endswith(".parquet")]
    existing_dirs = [d for d in os.listdir(bb_token_dir) if d.startswith("tokens-") and os.path.isdir(os.path.join(bb_token_dir, d))]

    # If all files exist and not forcing, skip
    if existing_parquets and len(existing_parquets) == len(existing_dirs) and not force_tokenize:
        print(
            f"Tokenized data found ({len(existing_parquets)} .parquet, {len(existing_dirs)} dirs). Skipping tokenization.")
        parquet_path = bb_token_dir
    else:
        if force_tokenize:
            print("Forcing re-tokenization: clearing existing .parquet files and token folders...")
            for item in os.listdir(bb_token_dir):
                item_path = os.path.join(bb_token_dir, item)
                if item.endswith(".parquet") or (item.startswith("tokens-") and os.path.isdir(item_path)):
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
        # Run tokenization if no existing Parquet files are found
        start = time.time()
        print("No existing tokenized files found. Running tokenization...")
        print("config_train['batch_size']",config_train["batch_size"])
        parquet_path = tokenization_h5ad(
            adata_path, gene_dict_path, mean_path,
            specie=specie, assay=assay,
            output_path=bb_token_dir,
            use_hvg=use_hvg, n_hvg=n_hvg, cell_density=False, gene_niche=False,
        )
        # Process all Parquet files
        for path in os.listdir(parquet_path):
            if path.endswith(".parquet"):
                parquet_file = os.path.join(parquet_path, path)
                # print(f"Processing file: {parquet_file}")

                if not os.path.exists(parquet_file):
                    print(f"Warning: {parquet_file} does not exist. Skipping...")
                    continue

                process_parquet(parquet_file, bb_token_dir)

        end = time.time()
        print(f"Preprocessing time: {(end - start)/60:.2f} minutes")
    return parquet_path


def run_bb_inference(
    adata,
    parquet_path,
    config_train,
    pretrain_ckpt,
    device,
    save_path=None
):
    time0 = time.time()
    config_train["batch_size"] = 1  # Use batch size of 1 for inference
    pipeline = CellEmbeddingPipeline(pretrain_ckpt=pretrain_ckpt, model_config=config_train, device=device)

    # Generate embeddings
    pred = pipeline.run(data_paths=parquet_path, config_train=config_train)

    # Extract index and embeddings from pred
    pred_indices, pred_embeddings = zip(*[(str(idx[0]), emb.numpy()) for idx, emb in pred])
    pred_indices = np.array(pred_indices)
    pred_embeddings = np.array(pred_embeddings)

    # Get embedding dimension
    embedding_dim = pred_embeddings.shape[1] if pred_embeddings.size > 0 else 0
    # get obs_names from adata
    obs_names = np.array(adata.obs_names)  # Convert to NumPy array for fast operations

    # Optimize assignment if orders match
    if np.array_equal(pred_indices, obs_names):
        print("obs_names and pred_indices are in the same order.")
        ordered_embeddings = pred_embeddings  # Direct assignment if order matches
    else:
        print("warning: The order of obs_names and pred_indices do not match. Reordering embeddings...")
        print("obs_names equal:", len(obs_names) == len(pred_indices))
        # Initialize embeddings with zeros
        ordered_embeddings = np.zeros((len(obs_names), embedding_dim))

        # Use NumPy for efficient lookup
        match_mask = np.isin(obs_names, pred_indices)
        matched_obs = obs_names[match_mask]

        sorted_idx = np.argsort(pred_indices)  # Sort pred_indices for binary search
        embedding_lookup = np.searchsorted(pred_indices[sorted_idx], matched_obs)  # Find positions

        ordered_embeddings[match_mask] = pred_embeddings[sorted_idx][embedding_lookup]  # Assign values

    if save_path is not None:
        np.savez_compressed(save_path, embeddings=ordered_embeddings)
        print(f"Embeddings saved to {save_path}")
    time1 = time.time()
    print("Time cost: ", (time1 - time0) / 60)

    del pipeline, pred, pred_indices, pred_embeddings
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
    use_spatial=False,
    do_fit=True,
    fit_epochs=100,  # can be set in the pipeline
    slice_sample=False,  # NEW
    save_embedding_path=None,  # Optional now
    save_model_path=None,  # optional: save .pt model_raw weights
    # New perturbation-aware parameters
    perturb_flag=None,  # Tensor, shape (B,), 0 or 1 indicating if sample is perturbed
    perturb_gene_id=None,  # Tensor, shape (B,), perturbed gene ID
    bbemb=None,  # callable, bbemb(gene_id_tensor) returns gene embedding from BrainBeacon
    use_perturbation=False,  # whether to use perturbation-aware modeling
    # New gene embedding parameters
    gene_embeddings=None,  # numpy array of gene embeddings
    symbol_to_emb_idx=None,  # dict mapping gene symbols to embedding indices
    condition_to_id=None,  # dict mapping condition names to IDs
    case_insensitive_mapping=None,  # dict mapping lowercase gene symbols to embedding indices
    # New parameters for model_raw loading
    output_dir=None,  # output directory for saving models
    output_prefix=None,  # output prefix for model_raw files
):
    # Load AnnData file
    data = adata.copy()
    data.obs_names_make_unique()
    # set train/valid split
    np.random.seed(42)
    data.obs['split'] = 'train'
    for batch_id in data.obs['slice'].unique():
        idx = data.obs['slice'] == batch_id
        cell_idx = np.where(idx)[0]
        n_valid = max(1, int(len(cell_idx) * 0.1))  # Ensure at least one cell is selected for validation
        valid_cells = np.random.choice(cell_idx, n_valid, replace=False)
        data.obs.iloc[valid_cells, data.obs.columns.get_loc('split')] = 'valid'

    # load brainbeacon embeddings
    data.obsm['bb_emb'] = np.load(bb_embedding_path)['embeddings']

    # Add perturbation information if provided
    covariate_fields = None
    if use_perturbation and perturb_flag is not None:
        # Convert tensor to numpy if needed
        if torch.is_tensor(perturb_flag):
            perturb_flag_np = perturb_flag.cpu().numpy()
        else:
            perturb_flag_np = perturb_flag
            
        if torch.is_tensor(perturb_gene_id):
            perturb_gene_id_np = perturb_gene_id.cpu().numpy()
        else:
            perturb_gene_id_np = perturb_gene_id
            
        # Add perturbation information to obs
        data.obs['perturb_flag'] = perturb_flag_np
        data.obs['perturb_gene_id'] = perturb_gene_id_np
        
        print(f"Added perturbation information: {len(perturb_flag_np)} samples, "
              f"{np.sum(perturb_flag_np)} perturbed samples")

    # Add batch info if enabled
    if use_batch:
        data.obs['batch'] = data.obs['condition']

    if use_spatial:
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
        'objective': 'imputation',
        'mask_node_rate': 0.95,
        'mask_feature_rate': 0.25,
        'max_batch_size': 5000,
        'mask_type': 'hidden',
        # 'mask_type': 'input',
    }
    
    # Add perturbation-aware configuration
    if use_perturbation:
        overwrite_config.update({
            'use_perturbation': True,
            'perturb_embedding_dim': 128,  # dimension for perturbation embedding
            'perturb_fusion_method': 'concat',  # 'concat', 'add', or 'attention'
        })
    
    # clear GPU memory before re-initializing the pipeline
    torch.cuda.empty_cache()
    
    # Choose pipeline based on perturbation usage
    if use_perturbation and perturb_flag is not None and perturb_gene_id is not None:
        print("Using perturbation-aware pipeline...")
        # Count unique perturbation conditions
        unique_conditions = len(np.unique(perturb_gene_id_np))
        print(f"Found {unique_conditions} unique perturbation conditions")
        
        # Initialize perturbation-aware pipeline
        pipeline = PerturbationReconstructionPipeline(
            gene_list=data.var.index.tolist(),
            enc_mod='transformer',
            enc_hid=1024,
            enc_layers=6,
            post_latent_dim=512,
            dec_mod='mlp',
            dec_hid=512,
            dec_layers=2,
            out_dim=len(data.var.index),
            batch_num=len(data.obs['batch'].unique()) if 'batch' in data.obs else 0,
            dataset_num=1,
            platform_num=1,
            mask_type='input',
            model_dropout=0.1,
            activation='gelu',
            norm='layernorm',
            enc_head=8,
            mask_node_rate=0.5,
            mask_feature_rate=0.25,
            drop_node_rate=0.0,
            max_batch_size=5000,
            pe_type=None,
            cat_pe=True,
            gene_emb=None,
            latent_mod='vae',
            use_perturbation=True,
            num_perturb_conditions=unique_conditions,
            device=device,
            # 新增：传递基因嵌入信息
            gene_embeddings=gene_embeddings,
            symbol_to_emb_idx=symbol_to_emb_idx,
            condition_to_id=condition_to_id,
            case_insensitive_mapping=case_insensitive_mapping,
            # 新增：传递预训练模型路径
            cellformer_pretrain_path=cellformer_pretrain_path,
        )
    else:
        print("Using standard pipeline...")
        pipeline = ReconstructPipeline(
            pretrain_prefix=cellformer_version,
            overwrite_config=overwrite_config,
            pretrain_directory=cellformer_directory,
            bb_pretrain_path=bb_pretrain_path,
            cellformer_pretrain_path=cellformer_pretrain_path,  # Not used here
            use_pretrain=True,
        )
    
    if do_fit:
        # Only sample one slice if requested
        if slice_sample:
            # np.random.seed(42)
            rng = np.random.RandomState(None)  # 使用局部随机性，每次运行都不一样
            chosen_slice = rng.choice(data.obs['slice'].unique())
            fit_data = data[data.obs['slice'] == chosen_slice].copy()
            print(f"Training only on slice: {chosen_slice} ({fit_data.n_obs} cells)")
        else:
            fit_data = data.copy() 
            
        # Fit pipeline based on type
        if use_perturbation and perturb_flag is not None and perturb_gene_id is not None:
            # Use perturbation-aware pipeline
            pipeline.fit(
                fit_data,
                covariate_fields=covariate_fields,
                use_perturbation=True,
                perturb_flag=perturb_flag_np,
                perturb_gene_id=perturb_gene_id_np,
                epochs=fit_epochs
            )
        else:
            # Use standard pipeline
            pipeline.fit(
                fit_data,  # AnnData object
                train_config={'epochs': fit_epochs},
                split_field='split',
                train_split='train',
                valid_split='valid',
                covariate_fields=covariate_fields,  # Pass perturbation info as covariates
                device=device
            )
    
    # Predict based on pipeline type
    if use_perturbation and perturb_flag is not None and perturb_gene_id is not None:
        # Use perturbation-aware pipeline
        # Load trained model_raw for prediction
        model_path = os.path.join(output_dir, f"{output_prefix}_cellformer.pt")
        data = pipeline.predict(
            data,
            covariate_fields=covariate_fields,
            use_perturbation=True,
            perturb_flag=perturb_flag_np,
            perturb_gene_id=perturb_gene_id_np,
            model_path=model_path
        )
        # Extract predictions from adata
        pred = torch.tensor(data.obsm['X_pred'])
        latent = torch.tensor(data.obsm['X_emb']) if 'X_emb' in data.obsm else torch.zeros((data.n_obs, 512))
    else:
        # Use standard pipeline
        pred, latent = pipeline.predict(data, device=device, covariate_fields=covariate_fields)
    
    # Handle target genes based on pipeline type
    if use_perturbation and perturb_flag is not None and perturb_gene_id is not None:
        # For perturbation pipeline, use all genes
        target_genes = data.var.index.tolist()
    else:
        # For standard pipeline, get target genes from pipeline
        target_genes = pipeline.target_genes  # this was set inside predict()
        data = data[:, target_genes].copy()  # now data.var.index == target_genes

    data.obsm['X_emb'] = latent.cpu().numpy()  # Store embeddings in AnnData object
    data.obsm['X_pred'] = pred.cpu().numpy()  # Store predicted gene

    if save_model_path is not None:
        torch.save(pipeline.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    if save_embedding_path is not None:
        np.savez_compressed(save_embedding_path, embeddings=data.obsm['X_emb'])
        print(f"Embeddings saved to {save_embedding_path}")

    return data

def perturb_bbcellformer_pipeline(
    adata_path: str,
    specie: str,
    assay: str,
    gene_dict_path: str,
    gene_mean_path: str,
    bb_ckpt_path: str,
    cellplm_ckpt_path: str,
    output_dir: str,
    output_prefix: str,
    config_train: dict = None,
    n_hvg: int = 1000,
    cd_weight: float = 0.02,
    use_hvg: bool = True,
    use_batch: bool = False,
    use_spatial: bool = False,
    weight_mode: str = "expression",
    force_tokenize: bool = True,
    do_fit: bool = True,
    fit_epochs: int = 100,
    slice_sample=False,  # select one slice for training
    save_model: bool = True,  # whether to save the model_raw
    save_model_path: str = None,
    save_embedding_path: str = None,
    device=None,
    # New perturbation-aware parameters
    use_perturbation: bool = False,
    perturb_flag: torch.Tensor = None,
    perturb_gene_id: torch.Tensor = None,
    # New gene embedding parameters
    gene_embeddings: np.ndarray = None,
    symbol_to_emb_idx: dict = None,
    condition_to_id: dict = None,
    case_insensitive_mapping: dict = None,
):
    # ====== 1. Setup ======
    os.makedirs(output_dir, exist_ok=True)
    set_seed(42)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_train is None:
        raise ValueError("`config_train` must be provided.")

    config_train.update({
        "weight_mode": weight_mode,
        "cd_weight": cd_weight,
        "masking_p": 0,
        "expr_mode": None,
        "use_gene_id_emb": True,
        "use_homo_emb": True,
        "use_rna_type_emb": True,
        "use_esm_emb": True,
    })

    # ====== 2. Load AnnData ======
    adata = sc.read_h5ad(adata_path)
    adata.obs["platform"] = assay

    # ====== 3. Tokenization ======
    bb_token_dir = os.path.join(output_dir, f"{output_prefix}_bb_token_dir")
    parquet_path = run_tokenization(
        adata_path=adata_path,
        bb_token_dir=bb_token_dir,
        gene_dict_path=gene_dict_path,
        mean_path=gene_mean_path,
        specie=specie,
        assay=assay,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        force_tokenize=force_tokenize,
    )

    # ====== 4. BrainBeacon Inference ======
    bb_embedding_path = os.path.join(output_dir, f"{output_prefix}_bb_embeddings.npz")
    if os.path.exists(bb_embedding_path) and not force_tokenize:
        print(f"Skipping BB inference. Found existing file: {bb_embedding_path}")
    else:
        bb_emb = run_bb_inference(
            adata=adata,
            parquet_path=parquet_path,
            config_train=config_train,
            pretrain_ckpt=bb_ckpt_path,
            device=device,
            save_path=bb_embedding_path
        )
        print(f"BB inference complete. Saved to: {bb_embedding_path}")
    # adata.obsm["bb_emb"] = bb_emb

    # ====== 5. CellFormer Reconstruction ======
    if save_model and save_embedding_path is None:
        save_embedding_path = os.path.join(output_dir, f"{output_prefix}_embeddings.npz")
    if save_model_path is None:
        save_model_path = os.path.join(output_dir, f"{output_prefix}_cellformer.pt")

    adata = run_bbcellformer_recon(
        adata=adata,
        bb_embedding_path=bb_embedding_path,
        bb_pretrain_path=bb_ckpt_path,
        cellformer_version="cellformer",
        cellformer_directory=os.path.dirname(cellplm_ckpt_path),
        device=device,
        cellformer_pretrain_path=cellplm_ckpt_path,
        use_batch=use_batch,
        use_spatial=use_spatial,
        do_fit=do_fit,
        slice_sample=slice_sample,
        fit_epochs=fit_epochs,
        save_embedding_path=save_embedding_path,
        save_model_path=save_model_path,
        # Pass perturbation parameters
        use_perturbation=use_perturbation,
        perturb_flag=perturb_flag,
        perturb_gene_id=perturb_gene_id,
        # Pass gene embedding parameters
        gene_embeddings=gene_embeddings,
        symbol_to_emb_idx=symbol_to_emb_idx,
        condition_to_id=condition_to_id,
        case_insensitive_mapping=case_insensitive_mapping,
        # Pass output parameters for model_raw loading
        output_dir=output_dir,
        output_prefix=output_prefix,
    )

    return adata
