import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy
import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union, List, Optional
from tqdm import tqdm
from anndata import AnnData
from typing import Optional, Dict
import matplotlib.pyplot as plt
import ot
from ot.unbalanced import sinkhorn_unbalanced
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
from model.brain_beacon import BrainBeacon
from model.brain_beacon import train_one_epoch
from config.config_cdniche import GENE_LOOKUP_DIR
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from scipy.stats import wasserstein_distance


def masked_mean_pooling(transformer_output, mask):
    mask = mask.unsqueeze(-1)
    masked_output = transformer_output * mask
    valid_length = mask.sum(dim=1, keepdim=False)
    valid_length = torch.clamp(valid_length, min=1)
    mean_pooled = masked_output.sum(dim=1, keepdim=False) / valid_length  # (b, d)
    return mean_pooled


def delete_gene_from_rank(real_indices, gene_ids_to_delete):
    """
    Remove specified gene indices from real_indices tensor.
    """
    mask = torch.ones_like(real_indices, dtype=torch.bool)
    for gid in gene_ids_to_delete:
        mask &= real_indices != gid
    return torch.where(mask, real_indices, torch.tensor(1, device=real_indices.device))  # 1 是 padding


def build_meta_from_index(real_indices, gene_lookup):
    B, T = real_indices.shape
    connect_comp = torch.zeros((B, T), dtype=torch.long)
    rna_type = torch.zeros((B, T), dtype=torch.long)
    for i in range(B):
        for j in range(T):
            idx = int(real_indices[i, j].item())
            if idx in gene_lookup:
                connect_comp[i, j] = gene_lookup[idx]["connect_comp_idx"]
                rna_type[i, j] = gene_lookup[idx]["rna_type_idx"]
    # Unmatched indices (e.g., auxiliary tokens) default to 0 → padding embedding
    attention_mask = real_indices == 1  # 1 is the padding token index
    return connect_comp, rna_type, attention_mask


def get_perturb_info(real_indices, gene_ids_to_perturb, perturb_type="delete"):
    """
    Extract perturbation metadata for each cell.

    Returns per-cell metadata:
        - is_perturbed: whether the cell contains perturbed genes
        - perturbed_gene_ranks: token positions of perturbed genes in the cell
        - n_perturbed_genes: number of perturbed genes in this cell
        - n_expressed_genes: number of non-padding genes in this cell
    """
    B, T = real_indices.shape
    info = []
    for i in range(B):
        cell_tokens = real_indices[i, 2:]  # Skip species and platform tokens
        cell_tokens_list = cell_tokens.tolist()

        perturbed_gene_ranks = [
            j for j, token in enumerate(cell_tokens_list) if token in gene_ids_to_perturb
        ]
        non_padding_count = (cell_tokens != 1).sum().item()  # 1 = padding token

        info.append({
            "is_perturbed": len(perturbed_gene_ranks) > 0,
            "perturbed_gene_ranks": perturbed_gene_ranks,
            "n_perturbed_genes": len(perturbed_gene_ranks),
            "n_expressed_genes": non_padding_count
        })

    return info


class InSilicoPerturberPipeline:
    def __init__(self,
                 pretrain_ckpt: str,
                 model_config: dict,
                 device: Union[str, torch.device] = 'cpu',
                 perturb_type: str = None,
                 genes_to_perturb: List[str] = None,
                 return_cell_results: bool = True,
                 return_gene_results: bool = True,
                 filter_perturbed_cells: bool = False,
                 max_rank: int = None,
                 sample_cells: Optional[int] = None,
                 sample_ratio: Optional[float] = None):
        self.device = device
        self.model_config = model_config
        self.model = None
        self.pretrain_ckpt = pretrain_ckpt
        self.perturb_type = perturb_type
        self.genes_to_perturb = genes_to_perturb
        self.return_cell_results = return_cell_results
        self.return_gene_results = return_gene_results
        self.filter_perturbed_cells = filter_perturbed_cells
        self.max_rank = max_rank
        self.sample_cells = sample_cells
        self.sample_ratio = sample_ratio

        self.gene_lookup = self._load_gene_lookup()
        self.gene_lookup_by_idx = self._build_reverse_lookup()
        self.gene_ids_to_perturb = self._get_gene_ids_to_perturb()
        self.initialize_model()

    def _load_gene_lookup(self):
        lookup_path = os.path.join(GENE_LOOKUP_DIR, "ensembl_to_all_idx.pkl")
        with open(lookup_path, "rb") as f:
            return pickle.load(f)

    def _build_reverse_lookup(self):
        rev = {}
        for ens_id, record in self.gene_lookup.items():
            rev[record["gene_idx"]] = {
                "connect_comp_idx": record["connect_comp_idx"],
                "rna_type_idx": record["rna_type_idx"]
            }
        return rev

    def _get_gene_ids_to_perturb(self):
        if self.genes_to_perturb is None:
            return []
        return [self.gene_lookup[gene]["gene_idx"] for gene in self.genes_to_perturb if gene in self.gene_lookup]

    def _get_ens_id_from_gene_idx(self, gid: int):
        for ens, val in self.gene_lookup.items():
            if val["gene_idx"] == gid:
                return ens
        return None

    def initialize_model(self):
        """
        Initialize the model and compute its size.
        """
        self.model = BrainBeacon(
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
            total_context_length=self.model_config['context_length'] * self.model_config['num_neighbors']
        ).to(self.device)

        # Compute model size
        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Model size: {:.3f}MB'.format(size_all_mb))

    def load_checkpoint(self):
        """
        Load a checkpoint if specified in the configuration.
        """
        if self.pretrain_ckpt:
            print(f"Loading checkpoint from {self.pretrain_ckpt}")
            ckpt = torch.load(self.pretrain_ckpt)
            self.model.load_state_dict(ckpt['model_state_dict'])
            return ckpt
        return None

    def load_dataset(self, data_path: str):
        """
        Load the dataset from the given paths.
        """
        masked_indices_files_list = []
        mask_files_list = []
        real_indices_files_list = []
        attention_mask_files_list = []
        connect_comp_files_list = []
        rna_type_files_list = []
        cell_ids_files_list = []
        cell_raw_index_list = []
        file_prefix_list = []
        self.data_path = data_path

        for prefix in os.listdir(data_path):
            if prefix.endswith(".parquet"):
                continue
            file_prefix_list.append(os.path.join(data_path, prefix))
            for file in os.listdir(os.path.join(data_path, prefix)):
                file_path = os.path.join(data_path, prefix, file)
                # print(f"Data paths: {file_path}")
                if 'masked_indices_' in file:
                    masked_indices_files_list.append(file_path)
                elif 'real_indices_' in file:
                    real_indices_files_list.append(file_path)
                elif 'attention_mask_' in file:
                    attention_mask_files_list.append(file_path)
                elif 'connect_comp_' in file:
                    connect_comp_files_list.append(file_path)
                elif 'rna_type_' in file:
                    rna_type_files_list.append(file_path)
                elif 'cell_ids_' in file:
                    cell_ids_files_list.append(file_path)
                elif 'mask_' in file:
                    mask_files_list.append(file_path)
                elif "cell_raw_index" in file:
                    cell_raw_index_list.append(file_path)
        print("masked_indices: ", masked_indices_files_list)
        dataset = FinetuneJoblibDataset(
            masked_indices_files_list,
            mask_files_list,
            real_indices_files_list,
            attention_mask_files_list,
            connect_comp_files_list,
            rna_type_files_list,
            cell_ids_files_list,
            file_prefix_list,
            cell_raw_index_list
        )
        return dataset

    def perturb_batch(self, real_indices):
        if self.perturb_type is None or self.genes_to_perturb is None:
            return real_indices

        target_gene_idxs = [self.gene_lookup[g]["gene_idx"] for g in self.genes_to_perturb if g in self.gene_lookup]
        if self.perturb_type == "delete":
            return delete_gene_from_rank(real_indices, target_gene_idxs)
        else:
            raise NotImplementedError(f"Perturbation type '{self.perturb_type}' not implemented yet.")

    def forward_pass(self, real_indices, esm_embedding_map):
        real_indices_view = real_indices.view(-1).long()
        esm_emb = torch.index_select(esm_embedding_map, 0, real_indices_view)
        esm_emb = esm_emb.view(real_indices.shape[0], real_indices.shape[1], -1)

        connect_comp, rna_type, attn_mask = build_meta_from_index(real_indices, self.gene_lookup_by_idx)
        real_indices, esm_emb, connect_comp, rna_type, attn_mask = [
            x.to(self.device) for x in [real_indices, esm_emb, connect_comp, rna_type, attn_mask]
        ]

        token_emb = self.model.embedding(real_indices, connect_comp, rna_type)
        token_emb += self.model.esm_embedding_projection(esm_emb)
        pos = self.model.pos.to(token_emb.device)
        emb = token_emb + self.model.positional_embedding(pos)

        out = self.model.encoder(emb, src_key_padding_mask=attn_mask)
        return out, attn_mask

    def infer(self, dataloader, config_train: dict):
        ckpt = self.load_checkpoint()
        if not ckpt:
            raise ValueError("Checkpoint file is missing.")

        self.model.eval()
        esm_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')

        with torch.no_grad():
            cell_results = [] if self.return_cell_results else None
            gene_results = [] if self.return_gene_results else None

            for real_indices, _, _, _, _, cell_ids in tqdm(dataloader):
                real_indices = real_indices[0]

                if self.sample_cells is not None and len(cell_ids) > self.sample_cells:
                    indices = torch.randperm(len(cell_ids))[:self.sample_cells]
                    real_indices = real_indices[indices].unsqueeze(0)
                    cell_ids = [cell_ids[i] for i in indices.tolist()]
                    print(f"[Sampling] Using {self.sample_cells} randomly selected cells out of {len(cell_ids)}")

                elif self.sample_ratio is not None and 0 < self.sample_ratio < 1:
                    sample_count = int(len(cell_ids) * self.sample_ratio)
                    if sample_count == 0:
                        print(
                            f"[Sampling] Skipping batch with {len(cell_ids)} cells due to too low sample_ratio={self.sample_ratio}")
                        continue
                    indices = torch.randperm(len(cell_ids))[:sample_count]
                    real_indices = real_indices[indices].unsqueeze(0)
                    cell_ids = [cell_ids[i] for i in indices.tolist()]
                    print(
                        f"[Sampling] Using {sample_count} ({self.sample_ratio * 100:.1f}%) randomly selected cells out of {len(cell_ids)}")

                real_indices = real_indices[0]

                out_orig, mask_orig = self.forward_pass(real_indices, esm_map)
                pert_info = get_perturb_info(real_indices, self.gene_ids_to_perturb)
                real_pert = self.perturb_batch(real_indices.clone())
                out_pert, mask_pert = self.forward_pass(real_pert, esm_map)

                emb_orig = masked_mean_pooling(out_orig[:, 2:, :].cpu(), (~mask_orig[:, 2:]).cpu())
                emb_pert = masked_mean_pooling(out_pert[:, 2:, :].cpu(), (~mask_pert[:, 2:]).cpu())
                cosine_cell = F.cosine_similarity(emb_orig, emb_pert, dim=1)

                emb_gene_orig = out_orig[:, 2:, :].cpu()
                emb_gene_pert = out_pert[:, 2:, :].cpu()
                cosine_gene = F.cosine_similarity(emb_gene_orig, emb_gene_pert, dim=-1)

                gene_idx_orig = real_indices[:, 2:].cpu()
                gene_idx_pert = real_pert[:, 2:].cpu()

                for i in range(len(cell_ids)):
                    info = pert_info[i]
                    cell_id = cell_ids[i][0] if isinstance(cell_ids[i], tuple) else cell_ids[i]

                    if self.return_cell_results:
                        if not self.filter_perturbed_cells or info["is_perturbed"]:
                            cell_results.append({
                                "cell_id": cell_id,
                                "emb_orig": emb_orig[i],
                                "emb_pert": emb_pert[i],
                                "cos_sim_cell": cosine_cell[i],  # cell-level similarity
                                "is_perturbed": info["is_perturbed"],
                                "perturbed_gene_ranks": info["perturbed_gene_ranks"],
                                "n_perturbed_genes": info["n_perturbed_genes"],
                                "n_expressed_genes": info["n_expressed_genes"],
                                "gene_idx_orig": gene_idx_orig[i],
                                "gene_idx_pert": gene_idx_pert[i]
                            })

                    if self.return_gene_results:
                        if not self.filter_perturbed_cells or info["is_perturbed"]:
                            for j in range(gene_idx_orig.shape[1]):
                                if self.max_rank is not None and j >= self.max_rank:
                                    break

                                gid = int(gene_idx_orig[i, j].item())
                                if gid == 1:
                                    continue

                                gene_results.append({
                                    "cell_id": cell_id,
                                    "gene_idx": gid,
                                    "ens_id": self._get_ens_id_from_gene_idx(gid),
                                    "rank": j,
                                    "n_expressed_genes": info["n_expressed_genes"],
                                    "is_perturbed": info["is_perturbed"],
                                    "is_perturbed_gene": j in info["perturbed_gene_ranks"],
                                    "cos_sim_gene": float(cosine_gene[i, j].item()),
                                    "cos_sim_cell": float(cosine_cell[i]),  # cell-level similarity
                                    "emb_orig": emb_gene_orig[i, j],
                                    "emb_pert": emb_gene_pert[i, j]
                                })

        output = {}
        if self.return_cell_results:
            output["cell_results"] = cell_results
        if self.return_gene_results:
            output["gene_results"] = gene_results
        return output

    def run(self, data_path: str, config_train: dict):  # data_paths → data_path
        dataset = self.load_dataset(data_path)
        loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4)
        return self.infer(loader, config_train)




def apply_gene_perturbation(
    adata,
    gene_list,
    mode="knockout",
    value=None,
    multiplier=2,  # default multiplier for overexpression
    target_obs_names=None,
    filter_by=None,
):
    """
    Apply gene perturbation (knockout or overexpression) to selected cells in AnnData.

    Parameters:
        adata : AnnData
            The input AnnData object.
        gene_list : list of str
            List of gene Ensembl IDs (must be in adata.var_names).
        mode : str
            Perturbation mode: "knockout" or "overexpress".
        value : float or None
            Fixed overexpression value (used if mode == "overexpress" and multiplier is None).
        multiplier : float or None
            Multiply the max expression of each gene to set overexpression value.
        target_obs_names : list or None
            Specific cells to perturb (based on adata.obs_names).
        filter_by : dict or None
            Alternative way to select cells by adata.obs fields (e.g., {"slice": "A", "cell_type": "B"}).

    Returns:
        perturbed_adata : AnnData
            A copy of the input AnnData with modified expression.
        perturbed_cells : pd.Index
            The obs_names of the cells that were perturbed.
    """

    perturbed_adata = adata.copy()

    # ===== Step 1: Select target cells =====
    if target_obs_names is not None:
        perturbed_cells = pd.Index(target_obs_names)
    elif filter_by is not None:
        mask = np.ones(len(adata), dtype=bool)
        for key, val in filter_by.items():
            mask &= (adata.obs[key] == val).values
        perturbed_cells = adata.obs_names[mask]
    else:
        # If neither provided, default to all cells
        perturbed_cells = adata.obs_names

    if len(perturbed_cells) == 0:
        raise ValueError("No cells selected for perturbation.")

    # ===== Step 2: Apply perturbation to each gene =====
    for gene_id in gene_list:
        # Convert gene symbol to gene_id if necessary
        if gene_id not in adata.var_names and "gene_symbol" in adata.var.columns:
            matches = adata.var.index[adata.var["gene_symbol"] == gene_id]
            if len(matches) == 0:
                raise ValueError(f"Gene symbol '{gene_id}' not found in adata.var['gene_symbol']")
            elif len(matches) > 1:
                raise ValueError(f"Multiple matches found for gene symbol '{gene_id}', please specify Ensembl ID")
            else:
                gene_id = matches[0]  # Replace symbol with corresponding Ensembl ID

        if gene_id not in adata.var_names:
            raise ValueError(f"Gene {gene_id} not found in adata.var_names")

        gene_idx = adata.var_names.get_loc(gene_id)
        cell_idx = adata.obs_names.get_indexer(perturbed_cells)

        if mode == "knockout":
            if scipy.sparse.issparse(perturbed_adata.X):
                perturbed_adata.X[cell_idx, gene_idx] = 0.0
            else:
                perturbed_adata.X[cell_idx, gene_idx] = 0.0

        elif mode == "overexpress":
            if value is not None:
                new_val = value
            elif multiplier is not None:
                # Use max value of this gene across all cells
                gene_expr = adata[:, gene_id].X
                max_val = gene_expr.max() if not scipy.sparse.issparse(gene_expr) else gene_expr.max()
                new_val = max_val * multiplier
            else:
                raise ValueError("Overexpression mode requires either `value` or `multiplier`.")

            if scipy.sparse.issparse(perturbed_adata.X):
                perturbed_adata.X[cell_idx, gene_idx] = new_val
            else:
                perturbed_adata.X[cell_idx, gene_idx] = new_val

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    return perturbed_adata, perturbed_cells

def inject_cells_into_niche(
    target_adata: AnnData,
    donor_adata: AnnData,
    target_filter: Optional[Dict[str, str]] = None,
    donor_filter: Optional[Dict[str, str]] = None,
    n_inject: Optional[int] = None,
    spatial_jitter_std: float = 1.0,
    random_state: int = 42,
) -> AnnData:
    """
    Inject donor cells into the spatial niche of target cells by perturbing coordinates.

    Parameters
    ----------
    target_adata : AnnData
        The original AnnData containing the target spatial niche.
    donor_adata : AnnData
        The AnnData providing cells to inject into the target spatial context.
    target_filter : dict, optional
        Dictionary to filter target cells, e.g., {"cell_type": "OL-WM"}.
    donor_filter : dict, optional
        Dictionary to filter donor cells, e.g., {"age_group": "young"}.
    n_inject : int, optional
        Number of donor cells to inject. If None, use all donor cells after filtering.
    spatial_jitter_std : float
        Scaling factor for coordinate noise added to donor cells.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    AnnData
        A new AnnData object with original and injected donor cells with modified spatial coordinates.
    """

    np.random.seed(random_state)

    # ===== 1. Filter target and donor cells =====
    target_mask = np.ones(len(target_adata), dtype=bool)
    if target_filter:
        for k, v in target_filter.items():
            target_mask &= (target_adata.obs[k] == v).values
    target_cells = target_adata[target_mask].copy()

    donor_mask = np.ones(len(donor_adata), dtype=bool)
    if donor_filter:
        for k, v in donor_filter.items():
            donor_mask &= (donor_adata.obs[k] == v).values
    donor_cells = donor_adata[donor_mask].copy()

    if len(donor_cells) == 0 or len(target_cells) == 0:
        raise ValueError("Filtered donor or target cells are empty.")

    # ===== 2. Sample donor cells if needed =====
    if n_inject is None:
        n_inject = len(donor_cells)
    else:
        n_inject = min(n_inject, len(donor_cells))
    donor_cells = donor_cells[np.random.choice(len(donor_cells), size=n_inject, replace=False)].copy()

    # ===== 3. Compute target spacing (median NN distance) =====
    target_coords = target_cells.obsm["spatial"]
    if len(target_coords) < 2:
        raise ValueError("Not enough target cells to compute nearest neighbor distances.")
    nbrs = NearestNeighbors(n_neighbors=2).fit(target_coords)
    dists, _ = nbrs.kneighbors(target_coords)
    median_nn_dist = np.median(dists[:, 1])  # skip self-distance

    # ===== 4. Assign donor cells to random target cells =====
    target_idx = np.random.choice(len(target_cells), size=n_inject, replace=True)
    base_positions = target_coords[target_idx]

    # ===== 5. Add spatial noise to donor coordinates =====
    noise = np.random.normal(loc=0.0, scale=median_nn_dist * spatial_jitter_std, size=(n_inject, 2))
    donor_cells.obsm["spatial"] = base_positions + noise

    # ===== 6. Add flag columns (no change to obs_names) =====
    donor_cells.obs["injected"] = True
    target_adata.obs["injected"] = False

    if "slice" in donor_cells.obs.columns:
        donor_cells.obs["injected_from_slice"] = donor_cells.obs["slice"]
    else:
        donor_cells.obs["injected_from_slice"] = "donor"
    target_adata.obs["injected_from_slice"] = None

    # ===== 7. Combine target and donor AnnData =====
    combined_adata = target_adata.concatenate(donor_cells, batch_key=None, index_unique=None)

    return combined_adata

def inject_cells_randomly(
    target_adata: ad.AnnData,
    donor_adata: ad.AnnData,
    celltype: str | None = None,
    spatial_key: str = "spatial",
    n_inject: int | None = None,
    random_state: int = 0,
) -> ad.AnnData:
    """
    In target_adata, randomly inject cells from donor_adata.

    - target_adata: AnnData, usually the old slice
    - donor_adata: AnnData, usually the young slice
    - celltype: None or str, specify cell type to inject from donor
    - spatial_key: spatial coordinates key in obsm
    - n_inject: number of cells to inject; if None, use all available donor cells
    - random_state: random seed for reproducibility
    """
    rng = np.random.default_rng(random_state)

    # ---- 1) donor pool ----
    if celltype is None:
        donor_pool = donor_adata.copy()
    else:
        donor_pool = donor_adata[donor_adata.obs["cell_type"] == celltype].copy()

    if donor_pool.n_obs < 1:
        raise ValueError(f"No cells in (celltype={celltype})")

    if n_inject is None:
        n_use = donor_pool.n_obs
    else:
        n_use = int(n_inject)

    idx = rng.integers(0, donor_pool.n_obs, size=n_use)
    donor_sel = donor_pool[idx].copy()
    coords = target_adata.obsm[spatial_key]
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    rand_x = rng.uniform(x_min, x_max, n_use)
    rand_y = rng.uniform(y_min, y_max, n_use)
    donor_sel.obsm[spatial_key] = np.vstack([rand_x, rand_y]).T

    donor_sel.obs["injected"] = True
    if celltype is None:
        donor_sel.obs["injected_celltype"] = donor_sel.obs["cell_type"].astype(str).values
    else:
        donor_sel.obs["injected_celltype"] = celltype

    donor_sel.obs_names = [f"{sid}_inj{i}" for i, sid in enumerate(donor_sel.obs_names.astype(str))]
    target_out = target_adata.copy()
    target_out.obs["injected"] = False

    combined = ad.concat([target_out, donor_sel], axis=0, join="outer", merge="same")
    return combined

def inject_cells_theory(
    target_adata: ad.AnnData,
    donor_adata: ad.AnnData,
    celltype: str,
    spatial_key: str = "spatial",
    random_state: int = 1,
) -> ad.AnnData:
    """
    theory replacement style cell injection:
    
    - replace target celltype cells with donor celltype cells
    - donor cells are randomly sampled with replacement to match target celltype count
    - coordinates of donor cells are set to target celltype cells
    """
    rng = np.random.default_rng(random_state)

    # ---- 1) donor pool ----
    donor_pool = donor_adata[donor_adata.obs["cell_type"] == celltype].copy()
    if donor_pool.n_obs < 1:
        raise ValueError(f"donor 中没有 celltype={celltype} 的细胞")

    target_rest = target_adata[target_adata.obs["cell_type"] != celltype].copy()
    target_to_replace = target_adata[target_adata.obs["cell_type"] == celltype].copy()
    n_target = target_to_replace.n_obs

    idx = rng.integers(0, donor_pool.n_obs, size=n_target)
    donor_sel = donor_pool[idx].copy()
    donor_sel.obsm[spatial_key] = target_to_replace.obsm[spatial_key].copy()
    donor_sel.obs["injected"] = True
    donor_sel.obs["injected_celltype"] = celltype
    donor_sel.obs_names = [f"{sid}_rep{i}" for i, sid in enumerate(target_to_replace.obs_names.astype(str))]

    target_rest.obs["injected"] = False

    combined = ad.concat([target_rest, donor_sel], axis=0, join="outer", merge="same")
    return combined

def analyze_embedding_similarity_change(
    adata_ori_result, adata_perturb_result,
    target_slice_young, target_slice_old, target_celltype,
    embedding_key="X_emb"
):
    """
    Compute cosine similarity and Euclidean distance between old and young cell embeddings
    before and after perturbation.
    """
    # Extract embeddings
    emb_young = adata_ori_result.obsm[embedding_key][
        (adata_ori_result.obs["slice"] == target_slice_young) &
        (adata_ori_result.obs["cell_type"] == target_celltype)
    ]
    emb_old_ori = adata_ori_result.obsm[embedding_key][
        (adata_ori_result.obs["slice"] == target_slice_old) &
        (adata_ori_result.obs["cell_type"] == target_celltype)
    ]
    emb_old_perturb = adata_perturb_result.obsm[embedding_key][
        (adata_perturb_result.obs["slice"] == target_slice_old) &
        (adata_perturb_result.obs["cell_type"] == target_celltype)
    ]

    # Mean embedding of young cells
    mean_young_emb = emb_young.mean(axis=0)

    # Cosine similarity
    sim_ori = cosine_similarity(emb_old_ori, mean_young_emb.reshape(1, -1)).mean()
    sim_perturb = cosine_similarity(emb_old_perturb, mean_young_emb.reshape(1, -1)).mean()

    # Euclidean distance
    dist_ori = euclidean_distances(emb_old_ori, mean_young_emb.reshape(1, -1)).mean()
    dist_perturb = euclidean_distances(emb_old_perturb, mean_young_emb.reshape(1, -1)).mean()

    return {
        "similarity_before": sim_ori,
        "similarity_after": sim_perturb,
        "delta_similarity": sim_perturb - sim_ori,
        "euclidean_before": dist_ori,
        "euclidean_after": dist_perturb,
        "delta_euclidean": dist_ori - dist_perturb  # positive: closer after
    }

def analyze_gene_reconstruction_change(
    adata_ori_result,
    adata_perturb_result,
    target_obs_names=None,
    filter_by=None,
    top_n=100,
    sort_abs=True,
    recon_key="X_pred"
):
    """
    Compare reconstructed gene expression between original and perturbed AnnData objects.
    """
    # ===== Step 1: Select target obs_names =====
    if target_obs_names is not None:
        selected_obs_names = pd.Index(target_obs_names)
    elif filter_by is not None:
        mask = np.ones(len(adata_perturb_result), dtype=bool)
        for key, val in filter_by.items():
            mask &= (adata_perturb_result.obs[key] == val).values
        selected_obs_names = adata_perturb_result.obs_names[mask]
    else:
        raise ValueError("You must specify either `target_obs_names` or `filter_by`.")

    # ===== Step 2: Ensure intersection with both adatas =====
    selected_obs_names = selected_obs_names[
        selected_obs_names.isin(adata_ori_result.obs_names) &
        selected_obs_names.isin(adata_perturb_result.obs_names)
    ]

    if len(selected_obs_names) == 0:
        raise ValueError("No matching obs_names found in both adatas after filtering.")

    # ===== Step 3: Get reconstructed expression =====
    obs_idx = adata_ori_result.obs_names.get_indexer(selected_obs_names)
    X_ori = adata_ori_result.obsm[recon_key][obs_idx]
    X_perturb = adata_perturb_result.obsm[recon_key][obs_idx]

    # ===== Step 4: Compute gene-wise mean and delta =====
    mean_ori = X_ori.mean(axis=0)
    mean_perturb = X_perturb.mean(axis=0)
    delta = mean_perturb - mean_ori

    # ===== Step 5: Construct result DataFrame =====
    df = pd.DataFrame({
        "gene_id": adata_ori_result.var_names,
        "gene_symbol": adata_ori_result.var["gene_symbol"].values,
        "ori_mean_expr": mean_ori,
        "perturb_mean_expr": mean_perturb,
        "delta_expr": delta,
        "abs_delta": np.abs(delta)
    })

    df_sorted = df.sort_values("abs_delta", ascending=False).head(top_n) if sort_abs \
        else df.sort_values("delta_expr", ascending=False).head(top_n)

    return df_sorted

def analyze_gene_reconstruction_change(
    adata_ori_result,
    adata_perturb_result,
    target_obs_names=None,
    filter_by=None,
    top_n=100,
    sort_abs=True,
    recon_key="X_pred"
):
    """
    Compare reconstructed gene expression between original and perturbed AnnData objects.
    """
    # ===== Step 1: Select target obs_names =====
    if target_obs_names is not None:
        selected_obs_names = pd.Index(target_obs_names)
    elif filter_by is not None:
        mask = np.ones(len(adata_perturb_result), dtype=bool)
        for key, val in filter_by.items():
            mask &= (adata_perturb_result.obs[key] == val).values
        selected_obs_names = adata_perturb_result.obs_names[mask]
    else:
        raise ValueError("You must specify either `target_obs_names` or `filter_by`.")

    # ===== Step 2: Ensure intersection with both adatas =====
    selected_obs_names = selected_obs_names[
        selected_obs_names.isin(adata_ori_result.obs_names) &
        selected_obs_names.isin(adata_perturb_result.obs_names)
    ]

    if len(selected_obs_names) == 0:
        raise ValueError("No matching obs_names found in both adatas after filtering.")

    # ===== Step 3: Get reconstructed expression =====
    obs_idx = adata_ori_result.obs_names.get_indexer(selected_obs_names)
    X_ori = adata_ori_result.obsm[recon_key][obs_idx]
    X_perturb = adata_perturb_result.obsm[recon_key][obs_idx]

    # ===== Step 4: Compute gene-wise mean and delta =====
    mean_ori = X_ori.mean(axis=0)
    mean_perturb = X_perturb.mean(axis=0)
    delta = mean_perturb - mean_ori

    # ===== Step 5: Construct result DataFrame =====
    df = pd.DataFrame({
        "gene_id": adata_ori_result.var_names,
        "gene_symbol": adata_ori_result.var["gene_symbol"].values,
        "ori_mean_expr": mean_ori,
        "perturb_mean_expr": mean_perturb,
        "delta_expr": delta,
        "abs_delta": np.abs(delta)
    })

    df_sorted = df.sort_values("abs_delta", ascending=False).head(top_n) if sort_abs \
        else df.sort_values("delta_expr", ascending=False).head(top_n)

    return df_sorted

def analyze_embedding_similarity_change_ot(
    adata_ori_result, adata_perturb_result,
    target_slice_young, target_slice_old, target_celltype,
    embedding_key="X_emb",
    sinkhorn_reg=1,
    uot_lambda=10 
):
    """
    Use POT Unbalanced Sinkhorn algorithm to align embeddings of old cells to young cells,
    and compute cosine similarity before and after perturbation.
    """

    # Subset
    adata_y = adata_ori_result[(adata_ori_result.obs["slice"] == target_slice_young) &
                               (adata_ori_result.obs["cell_type"] == target_celltype)].copy()
    adata_o = adata_ori_result[(adata_ori_result.obs["slice"] == target_slice_old) &
                               (adata_ori_result.obs["cell_type"] == target_celltype)].copy()
    adata_op = adata_perturb_result[(adata_perturb_result.obs["slice"] == target_slice_old) &
                                    (adata_perturb_result.obs["cell_type"] == target_celltype)].copy()

    assert adata_o.shape[0] == adata_op.shape[0], "Perturbed and original old cell counts don't match"

    # Embeddings
    Z_y = np.asarray(adata_y.obsm[embedding_key], dtype=np.float64)
    Z_o = np.asarray(adata_o.obsm[embedding_key], dtype=np.float64)
    Z_op = np.asarray(adata_op.obsm[embedding_key], dtype=np.float64)

    n_y, n_o = Z_y.shape[0], Z_o.shape[0]
    a = np.ones((n_y,)) / n_y
    b = np.ones((n_o,)) / n_o

    def get_projected(Z_target):
        M = ot.dist(Z_y, Z_target, metric='euclidean') ** 2
        T = sinkhorn_unbalanced(a, b, M, reg=sinkhorn_reg, reg_m=uot_lambda)
        T_norm = T / T.sum(axis=0, keepdims=True)
        return T_norm.T @ Z_y  # shape (n_o, d)

    Z_y_proj_o = get_projected(Z_o)
    Z_y_proj_op = get_projected(Z_op)

    Z_o = torch.tensor(Z_o, dtype=torch.float32)
    Z_op = torch.tensor(Z_op, dtype=torch.float32)
    Z_y_proj_o = torch.tensor(Z_y_proj_o, dtype=torch.float32)
    Z_y_proj_op = torch.tensor(Z_y_proj_op, dtype=torch.float32)

    # Cosine similarity
    sim_before = F.cosine_similarity(Z_o, Z_y_proj_o, dim=1)
    sim_after = F.cosine_similarity(Z_op, Z_y_proj_op, dim=1)
    sim_change = sim_after - sim_before

    # Euclidean distance
    dist_before = torch.norm(Z_o - Z_y_proj_o, dim=1)
    dist_after = torch.norm(Z_op - Z_y_proj_op, dim=1)
    dist_change = dist_after - dist_before

    return {
        "mean_similarity_before": sim_before.mean().item(),
        "mean_similarity_after": sim_after.mean().item(),
        "mean_similarity_change": sim_change.mean().item(),
        "all_similarity_change": sim_change.detach().cpu().numpy(),
        "mean_distance_before": dist_before.mean().item(),
        "mean_distance_after": dist_after.mean().item(),
        "mean_distance_change": dist_change.mean().item(),
        "all_distance_change": dist_change.detach().cpu().numpy()
    }
    

def analyze_embedding_similarity_change_similarity_niche(
    adata_ori_result, adata_perturb_result,
    target_slice_young, target_slice_old, target_celltype,
    embedding_key="X_emb", gamma=1.0
):
    """
    Compare old vs young before/after perturbation on a niche, excluding a target cell type.
    Returns cosine similarity, Euclidean distance, EMD (Wasserstein), and MMD.
    """
    # --- slice masks (exclude target_celltype as in your code) ---
    mask_young = (
        (adata_ori_result.obs["slice"] == target_slice_young) &
        (adata_ori_result.obs["cell_type"] != target_celltype)
    )
    mask_old_ori = (
        (adata_ori_result.obs["slice"] == target_slice_old) &
        (adata_ori_result.obs["cell_type"] != target_celltype)
    )
    mask_old_pert = (
        (adata_perturb_result.obs["slice"] == target_slice_old) &
        (adata_perturb_result.obs["cell_type"] != target_celltype)
    )

    # --- extract embeddings ---
    emb_young = np.asarray(adata_ori_result.obsm[embedding_key][mask_young])
    emb_old_ori = np.asarray(adata_ori_result.obsm[embedding_key][mask_old_ori])
    emb_old_perturb = np.asarray(adata_perturb_result.obsm[embedding_key][mask_old_pert])

    # --- basic checks ---
    if emb_young.size == 0:
        raise ValueError("No young cells found with the given filters.")
    if emb_old_ori.size == 0:
        raise ValueError("No old (original) cells found with the given filters.")
    if emb_old_perturb.size == 0:
        raise ValueError("No old (perturbed) cells found with the given filters.")
    if emb_young.ndim != 2 or emb_old_ori.ndim != 2 or emb_old_perturb.ndim != 2:
        raise ValueError("Embeddings must be 2-D arrays (cells x dims).")

    # --- handle NaNs robustly (or switch to filtering if preferred) ---
    emb_young = np.nan_to_num(emb_young, nan=0.0)
    emb_old_ori = np.nan_to_num(emb_old_ori, nan=0.0)
    emb_old_perturb = np.nan_to_num(emb_old_perturb, nan=0.0)

    # --- mean vector of young ---
    mean_young_emb = emb_young.mean(axis=0, keepdims=True)

    # --- cosine similarity (higher = closer to young) ---
    sim_ori = cosine_similarity(emb_old_ori, mean_young_emb).mean()
    sim_perturb = cosine_similarity(emb_old_perturb, mean_young_emb).mean()

    # --- Euclidean distance (lower = closer to young) ---
    dist_ori = euclidean_distances(emb_old_ori, mean_young_emb).mean()
    dist_perturb = euclidean_distances(emb_old_perturb, mean_young_emb).mean()

    # --- EMD / Wasserstein (1D per dimension, averaged) ---
    d = emb_young.shape[1]
    emd_ori = np.mean([wasserstein_distance(emb_old_ori[:, i], emb_young[:, i]) for i in range(d)])
    emd_perturb = np.mean([wasserstein_distance(emb_old_perturb[:, i], emb_young[:, i]) for i in range(d)])

    # --- MMD with RBF kernel (lower = closer) ---
    def compute_mmd(X, Y, gamma_):
        Kxx = rbf_kernel(X, X, gamma=gamma_).mean()
        Kyy = rbf_kernel(Y, Y, gamma=gamma_).mean()
        Kxy = rbf_kernel(X, Y, gamma=gamma_).mean()
        return Kxx + Kyy - 2 * Kxy

    mmd_ori = compute_mmd(emb_old_ori, emb_young, gamma)
    mmd_perturb = compute_mmd(emb_old_perturb, emb_young, gamma)

    return {
        "cosine": (float(sim_ori), float(sim_perturb)),
        "euclidean": (float(dist_ori), float(dist_perturb)),
        "emd": (float(emd_ori), float(emd_perturb)),
        "mmd": (float(mmd_ori), float(mmd_perturb)),
    }
    

def analyze_embedding_similarity_change(
    adata_ori_result, adata_perturb_result,
    target_slice_young, target_slice_old, target_celltype,
    embedding_key="X_emb", gamma=1.0
):
    """
    Compute cosine similarity, Euclidean distance, Wasserstein distance (EMD) and MMD 
    between old and young cell embeddings before and after perturbation.
    """
    # Extract embeddings
    emb_young = adata_ori_result.obsm[embedding_key][
        (adata_ori_result.obs["slice"] == target_slice_young) &
        (adata_ori_result.obs["cell_type"] == target_celltype)
    ]
    emb_old_ori = adata_ori_result.obsm[embedding_key][
        (adata_ori_result.obs["slice"] == target_slice_old) &
        (adata_ori_result.obs["cell_type"] == target_celltype)
    ]
    emb_old_perturb = adata_perturb_result.obsm[embedding_key][
        (adata_perturb_result.obs["slice"] == target_slice_old) &
        (adata_perturb_result.obs["cell_type"] == target_celltype)
    ]

    # Mean embedding of young cells
    mean_young_emb = emb_young.mean(axis=0)

    # ---- Cosine similarity ----
    sim_ori = cosine_similarity(emb_old_ori, mean_young_emb.reshape(1, -1))
    sim_perturb = cosine_similarity(emb_old_perturb, mean_young_emb.reshape(1, -1))

    # ---- Euclidean distance ----
    dist_ori = euclidean_distances(emb_old_ori, mean_young_emb.reshape(1, -1))
    dist_perturb = euclidean_distances(emb_old_perturb, mean_young_emb.reshape(1, -1))

    # ---- Wasserstein (1D per dimension, averaged) ----
    emd_ori = np.mean([wasserstein_distance(emb_old_ori[:, i], emb_young[:, i]) 
                       for i in range(emb_young.shape[1])])
    emd_perturb = np.mean([wasserstein_distance(emb_old_perturb[:, i], emb_young[:, i]) 
                           for i in range(emb_young.shape[1])])

    # ---- MMD (with RBF kernel) ----
    def compute_mmd(X, Y, gamma):
        Kxx = rbf_kernel(X, X, gamma=gamma).mean()
        Kyy = rbf_kernel(Y, Y, gamma=gamma).mean()
        Kxy = rbf_kernel(X, Y, gamma=gamma).mean()
        return Kxx + Kyy - 2 * Kxy

    mmd_ori = compute_mmd(emb_old_ori, emb_young, gamma)
    mmd_perturb = compute_mmd(emb_old_perturb, emb_young, gamma)

    return {
        "cosine": (sim_ori.mean(), sim_perturb.mean()),
        "euclidean": (dist_ori.mean(), dist_perturb.mean()),
        "emd": (emd_ori, emd_perturb),
        "mmd": (mmd_ori, mmd_perturb),
    }
    
def compute_delta_cosine(adata_ori, adata_perturb, slice_young, slice_old, emb_key="X_emb", celltype_key="cell_type"):
    """
    计算 ΔCosine (Perturb vs Baseline)，返回 DataFrame
    """
    # ---- baseline embedding ----
    X_ori = adata_ori.obsm[emb_key]
    if hasattr(X_ori, "toarray"):
        X_ori = X_ori.toarray()

    sl_ori = adata_ori.obs["slice"].astype(str).values
    young_mask = sl_ori == slice_young
    old_mask   = sl_ori == slice_old

    # normalize
    X_unit_ori = normalize(X_ori, norm="l2", axis=1)
    young_centroid = normalize(X_unit_ori[young_mask].mean(axis=0, keepdims=True), norm="l2", axis=1)[0]

    def cos_to_young(X):
        X_unit = normalize(X, norm="l2", axis=1)
        return X_unit @ young_centroid

    cos_y_old_baseline = cos_to_young(X_ori[old_mask])
    ct_old_baseline = adata_ori.obs[celltype_key].values[old_mask]

    # ---- perturb embedding ----
    Xp = adata_perturb.obsm[emb_key]
    if hasattr(Xp, "toarray"):
        Xp = Xp.toarray()
    sl_p = adata_perturb.obs["slice"].astype(str).values
    pert_mask = sl_p == slice_old

    cos_y_after = cos_to_young(Xp[pert_mask])
    ct_after = adata_perturb.obs[celltype_key].values[pert_mask]

    # ---- ΔCosine per cell ----
    recs = []
    for ctype in np.unique(ct_after):
        if ctype.lower() in ["other", "others"]:
            continue
        vals_after = cos_y_after[ct_after == ctype]
        vals_base  = cos_y_old_baseline[ct_old_baseline == ctype]
        if len(vals_after)==0 or len(vals_base)==0:
            continue
        base_mean = np.mean(vals_base)
        for v in vals_after - base_mean:
            recs.append({"perturb_injected": ctype, "celltype": ctype, "delta_cos": v})

    return pd.DataFrame(recs)

def plot_cosine_to_centroids_with_perturb(
    adata_ori,
    adata_perturb,
    slice_young,
    slice_old,
    celltype_key="cell_type",
    emb_key="X_emb",
    title="Cell state positioning relative to young and old centroids",
    agg_by_celltype=False,
    save_path=None
):

    # ====== embedding ======
    X_ori = adata_ori.obsm[emb_key]
    if hasattr(X_ori, "toarray"):
        X_ori = X_ori.toarray()
    X_perturb = adata_perturb.obsm[emb_key]
    if hasattr(X_perturb, "toarray"):
        X_perturb = X_perturb.toarray()

    # ====== Young / Old / Perturb ======
    sl_ori = adata_ori.obs["slice"].astype(str).values
    young_mask = sl_ori == slice_young
    old_mask   = sl_ori == slice_old
    sl_perturb = adata_perturb.obs["slice"].astype(str).values
    pert_mask = sl_perturb == slice_old

    X_unit_ori = normalize(X_ori, norm="l2", axis=1)
    young_centroid = normalize(X_unit_ori[young_mask].mean(axis=0, keepdims=True), norm="l2", axis=1)[0]
    old_centroid   = normalize(X_unit_ori[old_mask].mean(axis=0, keepdims=True),   norm="l2", axis=1)[0]
    def cos_coords(X):
        X_unit = normalize(X, norm="l2", axis=1)
        cos_y = X_unit @ young_centroid
        cos_o = X_unit @ old_centroid
        return cos_y, cos_o

    cos_y_y, cos_o_y = cos_coords(X_ori[young_mask])
    cos_y_o, cos_o_o = cos_coords(X_ori[old_mask])
    cos_y_p, cos_o_p = cos_coords(X_perturb[pert_mask])

    # ====== 作图 ======
    plt.figure(figsize=(6.5, 6))

    if not agg_by_celltype:

        plt.scatter(cos_y_y, cos_o_y, s=20, c="#076a3aff", alpha=0.7, label="Young cells")
        plt.scatter(cos_y_o, cos_o_o, s=20, c="#073f6aff", alpha=0.7, label="Old cells")
        plt.scatter(cos_y_p, cos_o_p, s=20, c="#791f1fff", alpha=0.7, label="Perturb cells")

        plt.scatter(np.mean(cos_y_y), np.mean(cos_o_y), c="#076a3aff", s=200, marker="*", edgecolor="white", linewidth=1, label="Young mean")
        plt.scatter(np.mean(cos_y_o), np.mean(cos_o_o), c="#073f6aff", s=200, marker="*", edgecolor="white", linewidth=1, label="Old mean")
        plt.scatter(np.mean(cos_y_p), np.mean(cos_o_p), c="#791f1fff", s=200, marker="*", edgecolor="white", linewidth=1, label="Perturb mean")

    else:
        for cos_y, cos_o, label in [(cos_y_y, cos_o_y, "Young"), (cos_y_o, cos_o_o, "Old"), (cos_y_p, cos_o_p, "Perturb")]:
            ct = adata_ori.obs[celltype_key].values[young_mask] if label=="Young" else (
                 adata_ori.obs[celltype_key].values[old_mask] if label=="Old" else adata_perturb.obs[celltype_key].values[pert_mask])
            df = pd.DataFrame({"celltype": ct, "cos_y": cos_y, "cos_o": cos_o})
            df_mean = df.groupby("celltype")[["cos_y", "cos_o"]].mean().reset_index()
            plt.scatter(df_mean["cos_y"], df_mean["cos_o"], s=50, alpha=0.9, label=label)

    lim_min = min(cos_y_y.min(), cos_o_y.min(), cos_y_o.min(), cos_o_o.min(), cos_y_p.min(), cos_o_p.min())
    lim_max = max(cos_y_y.max(), cos_o_y.max(), cos_y_o.max(), cos_o_o.max(), cos_y_p.max(), cos_o_p.max())
    pad = 0.02
    plt.plot([lim_min-pad, lim_max+pad], [lim_min-pad, lim_max+pad], ls="--", c="gray", lw=1)
    plt.xlim(lim_min-pad, lim_max+pad)
    plt.ylim(lim_min-pad, lim_max+pad)

    plt.xlabel("Cosine similarity to young centroid")
    plt.ylabel("Cosine similarity to old centroid")
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.2, ls="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()