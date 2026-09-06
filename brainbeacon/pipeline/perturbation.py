import os
import json
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy

import torch
import pickle
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Union, List
from tqdm import tqdm
from anndata import AnnData
from typing import Optional, Dict
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from brainbeacon.brain_beacon import BrainBeacon
from brainbeacon.configs.config import resolve_path
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, rbf_kernel
from scipy.stats import wasserstein_distance
from matplotlib.colors import LinearSegmentedColormap

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
        gene_lookup_dir = resolve_path("GENE_LOOKUP_DIR")
        lookup_path = os.path.join(gene_lookup_dir, "ensembl_to_all_idx.pkl")
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
        Initialize the brainbeacon and compute its size.
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

        # Compute brainbeacon size
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


import numpy as np
import pandas as pd
import scipy.sparse


def compute_mmd(X, Y, gamma=1.0):
    Kxx = rbf_kernel(X, X, gamma=gamma).mean()
    Kyy = rbf_kernel(Y, Y, gamma=gamma).mean()
    Kxy = rbf_kernel(X, Y, gamma=gamma).mean()
    return Kxx + Kyy - 2 * Kxy


def apply_gene_perturbation(
    adata,
    gene_list,
    mode="knockout",
    value=None,
    multiplier=2,
    target_obs_names=None,
    filter_by=None,
    perturb_percent=None,
    random_state=0,
    knockout_only_expressed=True,
    knockout_expr_threshold="auto",
    knockout_var_cutoff=1000,
    knockout_high_var_threshold=5.0,
    verbose=True,
):
    """
    Apply gene perturbation to selected cells in an AnnData.

    KO logic:
        If knockout_only_expressed=True, KO only applies to cells where the
        target gene expression is above an effective threshold.

        If knockout_expr_threshold == "auto":
            - adata.n_vars < knockout_var_cutoff:
                effective threshold = 0.0
            - adata.n_vars >= knockout_var_cutoff:
                effective threshold = knockout_high_var_threshold

        Default:
            - panel data / Xenium-like: expr > 0
            - whole-transcriptome / Stereo-seq-like: expr > 5

    Returns
    -------
    perturbed_adata : AnnData
        A copy of adata with modified .X.

    perturbed_cells : pandas.Index
        obs_names of actually perturbed cells.
    """

    perturbed_adata = adata.copy()

    # ============================================================
    # Step 1: Select candidate cells
    # ============================================================
    if target_obs_names is not None:
        candidate_cells = pd.Index(target_obs_names)

        missing = candidate_cells.difference(adata.obs_names)
        if len(missing) > 0:
            raise ValueError(
                f"{len(missing)} target_obs_names are not found in adata.obs_names."
            )

    elif filter_by is not None:
        mask = np.ones(len(adata), dtype=bool)

        for key, val in filter_by.items():
            if key not in adata.obs.columns:
                raise KeyError(f"{key} not found in adata.obs")

            if val is None:
                continue

            mask &= adata.obs[key].astype(str).values == str(val)

        candidate_cells = adata.obs_names[mask]

    else:
        candidate_cells = adata.obs_names

    if len(candidate_cells) == 0:
        raise ValueError("No candidate cells selected for perturbation.")

    original_candidate_cells = pd.Index(candidate_cells)

    # ============================================================
    # Step 2: Resolve genes
    # ============================================================
    resolved_gene_ids = []

    for gene in gene_list:
        gene_id = gene

        if gene_id not in adata.var_names and "gene_symbol" in adata.var.columns:
            matches = adata.var.index[
                adata.var["gene_symbol"].astype(str).values == str(gene_id)
            ]

            if len(matches) == 0:
                raise ValueError(
                    f"Gene symbol '{gene_id}' not found in adata.var['gene_symbol']."
                )

            elif len(matches) > 1:
                raise ValueError(
                    f"Multiple matches found for gene symbol '{gene_id}'. "
                    f"Please specify Ensembl ID. Matches: {list(matches[:10])}"
                )

            else:
                gene_id = matches[0]

        if gene_id not in adata.var_names:
            raise ValueError(f"Gene {gene_id} not found in adata.var_names.")

        resolved_gene_ids.append(gene_id)

    # ============================================================
    # Step 3: Determine KO threshold
    # ============================================================
    effective_ko_threshold = None

    if mode == "knockout":
        if knockout_expr_threshold == "auto":
            if adata.n_vars < knockout_var_cutoff:
                effective_ko_threshold = 0.0
            else:
                effective_ko_threshold = float(knockout_high_var_threshold)
        else:
            effective_ko_threshold = float(knockout_expr_threshold)

    # ============================================================
    # Step 4: For KO, keep only truly perturbable cells
    # ============================================================
    if mode == "knockout" and knockout_only_expressed:
        candidate_idx = adata.obs_names.get_indexer(candidate_cells)

        if np.any(candidate_idx < 0):
            raise ValueError("Some candidate cells are not found in adata.obs_names.")

        expressed_any_gene = np.zeros(len(candidate_cells), dtype=bool)

        for gene_id in resolved_gene_ids:
            gene_idx = adata.var_names.get_loc(gene_id)
            expr = adata.X[candidate_idx, gene_idx]

            if scipy.sparse.issparse(expr):
                expr = expr.toarray().ravel()
            else:
                expr = np.asarray(expr).ravel()

            expressed_any_gene |= expr > effective_ko_threshold

        candidate_cells = pd.Index(candidate_cells[expressed_any_gene])

        if len(candidate_cells) == 0:
            raise ValueError(
                "No truly perturbable cells found for knockout. "
                f"All selected cells have expression <= {effective_ko_threshold} "
                f"for genes: {resolved_gene_ids}"
            )

    # ============================================================
    # Step 5: Subsample candidate cells by perturb_percent
    # ============================================================
    if perturb_percent is None:
        perturbed_cells = pd.Index(candidate_cells)

    else:
        if perturb_percent <= 0:
            raise ValueError("perturb_percent must be > 0.")

        if perturb_percent <= 1:
            frac = perturb_percent
        elif perturb_percent <= 100:
            frac = perturb_percent / 100.0
        else:
            raise ValueError(
                "perturb_percent should be in (0, 1] as fraction "
                "or in (1, 100] as percentage."
            )

        n_candidate = len(candidate_cells)
        n_perturb = int(np.ceil(n_candidate * frac))
        n_perturb = max(1, min(n_perturb, n_candidate))

        rng = np.random.default_rng(random_state)
        selected = rng.choice(
            candidate_cells.values,
            size=n_perturb,
            replace=False,
        )
        perturbed_cells = pd.Index(selected)

    if len(perturbed_cells) == 0:
        raise ValueError("No cells selected after perturb_percent subsampling.")

    cell_idx = adata.obs_names.get_indexer(perturbed_cells)

    if np.any(cell_idx < 0):
        raise ValueError("Some perturbed cells are not found in adata.obs_names.")

    # ============================================================
    # Step 6: Apply perturbation
    # ============================================================
    for gene_id in resolved_gene_ids:
        gene_idx = adata.var_names.get_loc(gene_id)

        if mode == "knockout":
            perturbed_adata.X[cell_idx, gene_idx] = 0.0

        elif mode == "overexpress":
            if value is not None:
                new_val = value

            elif multiplier is not None:
                gene_expr = adata[:, gene_id].X

                if scipy.sparse.issparse(gene_expr):
                    max_val = gene_expr.max()
                else:
                    max_val = np.asarray(gene_expr).max()

                new_val = max_val * multiplier

            else:
                raise ValueError(
                    "Overexpression mode requires either `value` or `multiplier`."
                )

            perturbed_adata.X[cell_idx, gene_idx] = new_val

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    # ============================================================
    # Step 7: Print summary
    # ============================================================
    if verbose:
        print("========== apply_gene_perturbation ==========")
        print(f"mode: {mode}")
        print(f"genes input: {gene_list}")
        print(f"genes resolved: {resolved_gene_ids}")
        print(f"adata.n_vars: {adata.n_vars}")
        print(f"candidate cells before perturb expression filter: {len(original_candidate_cells)}")

        if mode == "knockout":
            print(f"knockout_only_expressed: {knockout_only_expressed}")
            print(f"knockout_expr_threshold: {knockout_expr_threshold}")
            print(f"knockout_var_cutoff: {knockout_var_cutoff}")
            print(f"knockout_high_var_threshold: {knockout_high_var_threshold}")
            print(f"effective KO threshold: {effective_ko_threshold}")

            if knockout_only_expressed:
                print(
                    "candidate cells after KO expression filter: "
                    f"{len(candidate_cells)}"
                )

        print(f"perturbed cells: {len(perturbed_cells)}")

        if perturb_percent is not None:
            print(f"perturb_percent: {perturb_percent}")
            print(f"actual fraction: {len(perturbed_cells) / len(candidate_cells):.4f}")

        print(f"random_state: {random_state}")

        if filter_by is not None:
            print(f"filter_by: {filter_by}")

        print("=============================================")

    return perturbed_adata, perturbed_cells



def _dense_matrix(x) -> np.ndarray:
    return x.toarray() if scipy.sparse.issparse(x) else np.asarray(x)


def _safe_vector_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() == 0:
        return float("nan")
    a = a[ok]
    b = b[ok]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _mmd_rbf_median(
    x: np.ndarray,
    y: np.ndarray,
    max_cells: int = 500,
    random_state: int = 42,
) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    if len(x) > max_cells:
        x = x[rng.choice(len(x), size=max_cells, replace=False)]
    if len(y) > max_cells:
        y = y[rng.choice(len(y), size=max_cells, replace=False)]
    z = np.vstack([x, y])
    d = euclidean_distances(z, z, squared=True)
    positive = d[d > 0]
    gamma = 1.0 / (float(np.median(positive)) + 1e-8) if positive.size else 1.0
    return float(
        rbf_kernel(x, x, gamma=gamma).mean()
        + rbf_kernel(y, y, gamma=gamma).mean()
        - 2.0 * rbf_kernel(x, y, gamma=gamma).mean()
    )


def align_public_perturbation_pair(
    base: AnnData,
    virt: AnnData,
    align_vars: bool = False,
) -> tuple[AnnData, AnnData]:
    """Align baseline and virtual perturbation AnnData by shared obs, optionally by shared genes."""
    common_obs = base.obs_names.intersection(virt.obs_names)
    if len(common_obs) == 0:
        raise ValueError("No shared obs_names between baseline and virtual AnnData.")
    if not align_vars:
        return base[common_obs].copy(), virt[common_obs].copy()
    common_vars = base.var_names.intersection(virt.var_names)
    if len(common_vars) == 0:
        raise ValueError("No shared var_names between baseline and virtual AnnData.")
    return base[common_obs, common_vars].copy(), virt[common_obs, common_vars].copy()


def public_perturbation_condition_masks(
    adata_obj: AnnData,
    condition_key: str = "condition",
    control_label: str = "control",
    real_label: str = "real_perturb",
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for control and real perturbation cells."""
    if condition_key not in adata_obj.obs.columns:
        raise ValueError(f"AnnData lacks obs['{condition_key}'].")
    cond = adata_obj.obs[condition_key].astype(str).to_numpy()
    control = cond == control_label
    real = cond == real_label
    if not control.any() or not real.any():
        found = sorted(pd.unique(cond).astype(str))
        raise ValueError(
            f"Need both {control_label!r} and {real_label!r} in obs['{condition_key}']; found {found}."
        )
    return control, real


def compute_public_perturbation_embedding_metrics(
    base: AnnData,
    virt: AnnData,
    emb_key: str = "X_emb",
    condition_key: str = "condition",
    control_label: str = "control",
    real_label: str = "real_perturb",
    max_mmd_cells: int = 500,
    random_state: int = 42,
    compute_realness: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute embedding movement metrics for public perturbation validation.

    The virtual perturbation is read from the control cells in ``virt`` and compared with the
    same control cells in ``base`` against the real perturbation cells in ``base``.
    """
    base, virt = align_public_perturbation_pair(base, virt, align_vars=False)
    control, real = public_perturbation_condition_masks(
        base,
        condition_key=condition_key,
        control_label=control_label,
        real_label=real_label,
    )
    if emb_key not in base.obsm or emb_key not in virt.obsm:
        raise ValueError(f"Both AnnData objects must contain obsm['{emb_key}'].")

    x_base = np.asarray(base.obsm[emb_key], dtype=np.float64)
    x_virt = np.asarray(virt.obsm[emb_key], dtype=np.float64)
    xb_c = np.nan_to_num(x_base[control], nan=0.0, posinf=0.0, neginf=0.0)
    xv_c = np.nan_to_num(x_virt[control], nan=0.0, posinf=0.0, neginf=0.0)
    xb_r = np.nan_to_num(x_base[real], nan=0.0, posinf=0.0, neginf=0.0)

    real_centroid = xb_r.mean(axis=0, keepdims=True)
    cos_before = cosine_similarity(xb_c, real_centroid).reshape(-1)
    cos_after = cosine_similarity(xv_c, real_centroid).reshape(-1)
    eu_before = euclidean_distances(xb_c, real_centroid).reshape(-1)
    eu_after = euclidean_distances(xv_c, real_centroid).reshape(-1)
    cos_delta = cos_after - cos_before
    eu_delta = eu_after - eu_before

    real_shift = xb_r.mean(axis=0) - xb_c.mean(axis=0)
    virtual_shift = xv_c.mean(axis=0) - xb_c.mean(axis=0)
    real_shift_norm = float(np.linalg.norm(real_shift))
    virtual_shift_norm = float(np.linalg.norm(virtual_shift))
    denom = float(np.dot(real_shift, real_shift))

    metrics = {
        "n_control_aligned": int(control.sum()),
        "n_real_aligned": int(real.sum()),
        "cosine_before": float(np.mean(cos_before)),
        "cosine_after_virtual": float(np.mean(cos_after)),
        "delta_cosine": float(np.mean(cos_delta)),
        "median_delta_cosine": float(np.median(cos_delta)),
        "fraction_cells_cosine_improved": float(np.mean(cos_delta > 0)),
        "euclidean_before": float(np.mean(eu_before)),
        "euclidean_after_virtual": float(np.mean(eu_after)),
        "delta_euclidean": float(np.mean(eu_delta)),
        "median_delta_euclidean": float(np.median(eu_delta)),
        "fraction_cells_euclidean_improved": float(np.mean(eu_delta < 0)),
        "mmd_control_vs_real": _mmd_rbf_median(xb_c, xb_r, max_cells=max_mmd_cells, random_state=random_state),
        "mmd_virtual_vs_real": _mmd_rbf_median(xv_c, xb_r, max_cells=max_mmd_cells, random_state=random_state),
        "embedding_shift_cosine": _safe_vector_cosine(real_shift, virtual_shift),
        "embedding_real_shift_norm": real_shift_norm,
        "embedding_virtual_shift_norm": virtual_shift_norm,
        "embedding_projection_fraction": float(np.dot(virtual_shift, real_shift) / denom) if denom > 0 else float("nan"),
        "embedding_shift_norm_ratio": virtual_shift_norm / real_shift_norm if real_shift_norm > 0 else float("nan"),
    }

    k = min(20, xb_r.shape[0])
    if k >= 1:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(xb_r)
        base_d = nn.kneighbors(xb_c, return_distance=True)[0].mean(axis=1)
        virt_d = nn.kneighbors(xv_c, return_distance=True)[0].mean(axis=1)
        metrics.update(
            {
                "knn_real_distance_before": float(np.mean(base_d)),
                "knn_real_distance_after_virtual": float(np.mean(virt_d)),
                "knn_real_distance_delta": float(np.mean(virt_d) - np.mean(base_d)),
            }
        )

    if compute_realness:
        y = np.concatenate([np.zeros(xb_c.shape[0]), np.ones(xb_r.shape[0])])
        x_train = np.vstack([xb_c, xb_r])
        if len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 3:
            try:
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
                )
                clf.fit(x_train, y)
                train_prob = clf.predict_proba(x_train)[:, 1]
                p_base = clf.predict_proba(xb_c)[:, 1]
                p_virt = clf.predict_proba(xv_c)[:, 1]
                p_real = clf.predict_proba(xb_r)[:, 1]
                metrics.update(
                    {
                        "realness_train_auc": float(roc_auc_score(y, train_prob)),
                        "realness_prob_control": float(np.mean(p_base)),
                        "realness_prob_virtual": float(np.mean(p_virt)),
                        "realness_prob_real": float(np.mean(p_real)),
                        "realness_prob_delta": float(np.mean(p_virt) - np.mean(p_base)),
                    }
                )
            except Exception as exc:
                metrics["realness_error"] = str(exc)

    per_cell = pd.DataFrame(
        {
            "cell": base.obs_names[control],
            "cosine_before": cos_before,
            "cosine_after": cos_after,
            "delta_cosine": cos_delta,
            "euclidean_before": eu_before,
            "euclidean_after": eu_after,
            "delta_euclidean": eu_delta,
        }
    )
    return metrics, per_cell


def _public_symbol_list(adata_obj: AnnData) -> list[str]:
    if "gene_symbol" in adata_obj.var.columns:
        return [str(x) for x in adata_obj.var["gene_symbol"].values]
    return [str(x) for x in adata_obj.var_names]


def _public_matrix_for_genes(adata_obj: AnnData, layer: str, genes: list[str]) -> np.ndarray:
    idx = [int(adata_obj.var_names.get_loc(g)) for g in genes]
    if layer == "raw":
        return np.asarray(_dense_matrix(adata_obj.X[:, idx]), dtype=np.float64)
    if layer == "decoder":
        if "X_pred" not in adata_obj.obsm:
            raise ValueError("Decoder signature metrics require obsm['X_pred'].")
        return np.asarray(adata_obj.obsm["X_pred"], dtype=np.float64)[:, idx]
    raise ValueError(f"Unsupported layer: {layer}")


def public_signature_metrics(
    real_sig: np.ndarray,
    virtual_sig: np.ndarray,
    gene_names: list[str],
    symbols: list[str],
    top_n: int = 50,
) -> dict:
    real_sig = np.asarray(real_sig, dtype=np.float64)
    virtual_sig = np.asarray(virtual_sig, dtype=np.float64)
    valid = np.isfinite(real_sig) & np.isfinite(virtual_sig)
    real_sig = real_sig[valid]
    virtual_sig = virtual_sig[valid]
    genes = np.asarray(gene_names, dtype=object)[valid]
    syms = np.asarray(symbols, dtype=object)[valid]
    out = {
        "n_signature_genes": int(real_sig.size),
        "signature_cosine": _safe_vector_cosine(real_sig, virtual_sig),
        "signature_real_norm": float(np.linalg.norm(real_sig)),
        "signature_virtual_norm": float(np.linalg.norm(virtual_sig)),
    }
    real_norm2 = float(np.dot(real_sig, real_sig))
    out["signature_projection_fraction"] = (
        float(np.dot(virtual_sig, real_sig) / real_norm2) if real_norm2 > 0 else float("nan")
    )
    out["signature_norm_ratio"] = (
        out["signature_virtual_norm"] / out["signature_real_norm"]
        if out["signature_real_norm"] > 0
        else float("nan")
    )
    if real_sig.size == 0:
        return out

    n_top = min(top_n, real_sig.size)
    top_real = np.argsort(np.abs(real_sig))[-n_top:]
    top_virtual = np.argsort(np.abs(virtual_sig))[-n_top:]
    real_set = set(map(int, top_real))
    virtual_set = set(map(int, top_virtual))
    union = real_set | virtual_set
    product = real_sig[top_real] * virtual_sig[top_real]
    out["top_abs_jaccard"] = float(len(real_set & virtual_set) / len(union)) if union else float("nan")
    out["top_real_direction_agreement"] = float(np.mean(product > 0))
    out["top_real_nonzero_virtual_fraction"] = float(np.mean(np.abs(virtual_sig[top_real]) > 1e-12))
    top_rows = []
    for i in top_real[np.argsort(-np.abs(real_sig[top_real]))[:10]]:
        top_rows.append(
            {
                "gene": str(genes[i]),
                "symbol": str(syms[i]),
                "real_delta": float(real_sig[i]),
                "virtual_delta": float(virtual_sig[i]),
                "same_direction": bool(real_sig[i] * virtual_sig[i] > 0),
            }
        )
    out["top_real_genes_json"] = json.dumps(top_rows, ensure_ascii=True)
    return out


def score_delta_against_public_signature(
    base_control: np.ndarray,
    real_cells: np.ndarray,
    virtual_control: np.ndarray,
    real_sig: np.ndarray,
) -> dict:
    real_sig = np.asarray(real_sig, dtype=np.float64)
    denom = np.linalg.norm(real_sig)
    if denom <= 0:
        return {"signature_score_delta": float("nan"), "signature_score_fraction": float("nan")}
    direction = real_sig / denom
    base_score = np.asarray(base_control, dtype=np.float64) @ direction
    real_score = np.asarray(real_cells, dtype=np.float64) @ direction
    virtual_score = np.asarray(virtual_control, dtype=np.float64) @ direction
    true_delta = float(np.nanmean(real_score) - np.nanmean(base_score))
    virtual_delta = float(np.nanmean(virtual_score) - np.nanmean(base_score))
    return {
        "signature_score_delta": virtual_delta,
        "signature_score_true_delta": true_delta,
        "signature_score_fraction": virtual_delta / true_delta if abs(true_delta) > 1e-12 else float("nan"),
    }


def expression_signature_metrics_variant(
    base: AnnData,
    virt: AnnData,
    layer: str,
    prefix: str,
    top_n: int = 50,
    exclude_genes: set[str] | None = None,
    clamp_values: dict[str, float] | None = None,
    condition_key: str = "condition",
    control_label: str = "control",
    real_label: str = "real_perturb",
) -> dict:
    """Compute real-vs-control and virtual-vs-control signature agreement for one expression layer."""
    base, virt = align_public_perturbation_pair(base, virt, align_vars=True)
    control, real = public_perturbation_condition_masks(
        base,
        condition_key=condition_key,
        control_label=control_label,
        real_label=real_label,
    )
    common_genes = list(base.var_names.intersection(virt.var_names))
    if exclude_genes:
        common_genes = [g for g in common_genes if g not in exclude_genes]
    if not common_genes:
        return {f"{prefix}_n_signature_genes": 0}

    xb = _public_matrix_for_genes(base, layer, common_genes)
    xv = _public_matrix_for_genes(virt, layer, common_genes)
    if clamp_values:
        xv = np.array(xv, copy=True)
        gene_to_idx = {g: i for i, g in enumerate(common_genes)}
        for gene_id, value in clamp_values.items():
            if gene_id in gene_to_idx:
                xv[control, gene_to_idx[gene_id]] = float(value)

    xb_c = xb[control]
    xb_r = xb[real]
    xv_c = xv[control]
    real_sig = xb_r.mean(axis=0) - xb_c.mean(axis=0)
    virtual_sig = xv_c.mean(axis=0) - xb_c.mean(axis=0)
    symbol_lookup = dict(zip(list(base.var_names), _public_symbol_list(base)))
    symbols = [symbol_lookup.get(g, g) for g in common_genes]
    out = public_signature_metrics(real_sig, virtual_sig, common_genes, symbols, top_n=top_n)
    out.update(score_delta_against_public_signature(xb_c, xb_r, xv_c, real_sig))
    return {f"{prefix}_{k}": v for k, v in out.items()}


def compute_public_perturbation_metrics(
    base_result: AnnData,
    virt_result: AnnData,
    base_input: AnnData | None = None,
    virt_input: AnnData | None = None,
    paper_direct_gene_ids: list[str] | set[str] | None = None,
    control_direct_gene_ids: list[str] | set[str] | None = None,
    clamp_values: dict[str, float] | None = None,
    top_n: int = 50,
    emb_key: str = "X_emb",
    condition_key: str = "condition",
    control_label: str = "control",
    real_label: str = "real_perturb",
) -> tuple[dict, pd.DataFrame]:
    """
    Compute the standard public perturbation metric bundle used by BrainBeacon analyses.

    Returns a metrics dictionary plus per-control-cell embedding movement rows.
    """
    paper_direct = set(paper_direct_gene_ids or [])
    control_direct = set(control_direct_gene_ids or [])
    all_direct = paper_direct | control_direct
    if clamp_values is None:
        clamp_values = {g: 0.0 for g in control_direct}

    metrics, per_cell = compute_public_perturbation_embedding_metrics(
        base_result,
        virt_result,
        emb_key=emb_key,
        condition_key=condition_key,
        control_label=control_label,
        real_label=real_label,
    )
    metrics.update(
        expression_signature_metrics_variant(
            base_result,
            virt_result,
            "decoder",
            "decoder_include_direct",
            top_n=top_n,
            condition_key=condition_key,
            control_label=control_label,
            real_label=real_label,
        )
    )
    metrics.update(
        expression_signature_metrics_variant(
            base_result,
            virt_result,
            "decoder",
            "decoder_exclude_paper_direct",
            top_n=top_n,
            exclude_genes=paper_direct,
            condition_key=condition_key,
            control_label=control_label,
            real_label=real_label,
        )
    )
    metrics.update(
        expression_signature_metrics_variant(
            base_result,
            virt_result,
            "decoder",
            "decoder_exclude_all_direct",
            top_n=top_n,
            exclude_genes=all_direct,
            condition_key=condition_key,
            control_label=control_label,
            real_label=real_label,
        )
    )
    metrics.update(
        expression_signature_metrics_variant(
            base_result,
            virt_result,
            "decoder",
            "decoder_clamp_control_direct",
            top_n=top_n,
            clamp_values=clamp_values,
            condition_key=condition_key,
            control_label=control_label,
            real_label=real_label,
        )
    )
    if base_input is not None and virt_input is not None:
        metrics.update(
            expression_signature_metrics_variant(
                base_input,
                virt_input,
                "raw",
                "raw_include_direct",
                top_n=top_n,
                condition_key=condition_key,
                control_label=control_label,
                real_label=real_label,
            )
        )
        metrics.update(
            expression_signature_metrics_variant(
                base_input,
                virt_input,
                "raw",
                "raw_exclude_all_direct",
                top_n=top_n,
                exclude_genes=all_direct,
                condition_key=condition_key,
                control_label=control_label,
                real_label=real_label,
            )
        )
    return metrics, per_cell


def plot_spatial_delta_cosine_after_perturbation(
    adata_ori,
    adata_perturb,
    slice_perturb="Hippocampus_Y_2_1",
    slice_reference="Hippocampus_O_2_1",
    target_celltype="OL-WM",
    highlight_cells=None,
    emb_key="X_emb",
    slice_key="slice",
    celltype_key="cell_type",
    reference_mode="global_reference",  # "global_reference" or "celltype_reference"
    spot_size=1,
    cmap="RdBu_r",
    use_soft_cmap=False,
    vlim=None,
    percentile_low=2,
    percentile_high=98,
    title=None,
    save_path=None,
    dpi=500,
    target_marker="*",
    target_size=90,
    target_edgecolor="white",
    target_linewidth=0.9,
    show=True,
    verbose=True,
):
    """
    Plot spatial delta cosine similarity after perturbation.

    This function can be used for both OE and KO.

    Delta definition:
        delta = cosine(cell after perturbation, reference centroid)
              - cosine(cell before perturbation, reference centroid)

    Positive delta:
        the plotted cells become more similar to the reference after perturbation.

    Parameters
    ----------
    adata_ori:
        Original AnnData before perturbation.

    adata_perturb:
        AnnData after perturbation inference.

    slice_perturb:
        The slice to plot and perturb.

        For OE young -> old:
            slice_perturb    = young slice
            slice_reference  = old reference slice

        For KO old -> young:
            slice_perturb    = old slice
            slice_reference  = young reference slice

    slice_reference:
        The reference slice used to build the centroid.

    target_celltype:
        Cell type used for summary statistics.

    highlight_cells:
        Cells to highlight as stars.

        For OE:
            pass perturbed_cells_final.

        For KO:
            pass perturbed_cells_final from the updated apply_gene_perturbation(),
            where KO only returns truly affected cells.

    reference_mode:
        "global_reference":
            use all cells in slice_reference as one reference centroid.

        "celltype_reference":
            for each plotted cell, use the reference centroid of the same cell type.

    Color logic:
        If vlim is None:
            vmin = percentile_low(delta)
            vmax = percentile_high(delta)
        Else:
            use manually specified vlim=(vmin, vmax)

    Returns
    -------
    adata_plot:
        AnnData containing only plotted cells, with delta metrics in .obs.
    """

    # ============================================================
    # Basic checks
    # ============================================================
    for key in [slice_key, celltype_key]:
        if key not in adata_ori.obs:
            raise KeyError(f"{key} not found in adata_ori.obs")
        if key not in adata_perturb.obs:
            raise KeyError(f"{key} not found in adata_perturb.obs")

    if emb_key not in adata_ori.obsm:
        raise KeyError(f"{emb_key} not found in adata_ori.obsm")

    if emb_key not in adata_perturb.obsm:
        raise KeyError(f"{emb_key} not found in adata_perturb.obsm")

    if "spatial" not in adata_ori.obsm:
        raise KeyError("'spatial' not found in adata_ori.obsm")

    # ============================================================
    # Build masks
    # ============================================================
    plot_mask = adata_ori.obs[slice_key].astype(str).values == str(slice_perturb)
    ref_mask = adata_ori.obs[slice_key].astype(str).values == str(slice_reference)

    if plot_mask.sum() == 0:
        raise ValueError(f"No cells found in plotted slice: {slice_perturb}")

    if ref_mask.sum() == 0:
        raise ValueError(f"No cells found in reference slice: {slice_reference}")

    selected_plot_index = adata_ori.obs_names[plot_mask]

    adata_plot = adata_ori[selected_plot_index, :].copy()

    # ============================================================
    # Align plotted cells between original and perturbed AnnData
    # ============================================================
    missing = selected_plot_index.difference(adata_perturb.obs_names)

    if len(missing) > 0:
        raise ValueError(f"{len(missing)} plotted cells are missing in adata_perturb.")

    X_before = adata_ori[selected_plot_index, :].obsm[emb_key]
    X_after = adata_perturb[selected_plot_index, :].obsm[emb_key]

    if hasattr(X_before, "toarray"):
        X_before = X_before.toarray()

    if hasattr(X_after, "toarray"):
        X_after = X_after.toarray()

    X_before = np.nan_to_num(np.asarray(X_before), nan=0.0)
    X_after = np.nan_to_num(np.asarray(X_after), nan=0.0)

    # ============================================================
    # Compute delta cosine
    # ============================================================
    cos_before = np.full(X_before.shape[0], np.nan)
    cos_after = np.full(X_before.shape[0], np.nan)
    delta_cos = np.full(X_before.shape[0], np.nan)

    if reference_mode == "global_reference":
        X_ref = adata_ori[ref_mask, :].obsm[emb_key]

        if hasattr(X_ref, "toarray"):
            X_ref = X_ref.toarray()

        X_ref = np.nan_to_num(np.asarray(X_ref), nan=0.0)
        ref_centroid = X_ref.mean(axis=0, keepdims=True)

        cos_before = cosine_similarity(X_before, ref_centroid).ravel()
        cos_after = cosine_similarity(X_after, ref_centroid).ravel()
        delta_cos = cos_after - cos_before

    elif reference_mode == "celltype_reference":
        plot_celltypes = adata_ori.obs.loc[
            selected_plot_index,
            celltype_key
        ].astype(str).values

        all_celltypes = adata_ori.obs[celltype_key].astype(str).values

        for ct in np.unique(plot_celltypes):
            idx = plot_celltypes == ct
            ref_ct_mask = ref_mask & (all_celltypes == ct)

            if ref_ct_mask.sum() == 0:
                continue

            X_ref_ct = adata_ori[ref_ct_mask, :].obsm[emb_key]

            if hasattr(X_ref_ct, "toarray"):
                X_ref_ct = X_ref_ct.toarray()

            X_ref_ct = np.nan_to_num(np.asarray(X_ref_ct), nan=0.0)
            ref_ct_centroid = X_ref_ct.mean(axis=0, keepdims=True)

            cos_before[idx] = cosine_similarity(
                X_before[idx],
                ref_ct_centroid
            ).ravel()

            cos_after[idx] = cosine_similarity(
                X_after[idx],
                ref_ct_centroid
            ).ravel()

            delta_cos[idx] = cos_after[idx] - cos_before[idx]

    else:
        raise ValueError(
            "reference_mode must be 'global_reference' or 'celltype_reference'."
        )

    # ============================================================
    # Write values back to adata_plot.obs
    # ============================================================
    before_key = f"cos_to_reference_before_{reference_mode}"
    after_key = f"cos_to_reference_after_{reference_mode}"
    delta_key = f"delta_cos_to_reference_{reference_mode}"

    adata_plot.obs[before_key] = cos_before
    adata_plot.obs[after_key] = cos_after
    adata_plot.obs[delta_key] = delta_cos

    adata_plot.obs["is_target_celltype"] = (
        adata_plot.obs[celltype_key].astype(str).values == str(target_celltype)
    )

    # ============================================================
    # Highlight actual perturbed cells
    # ============================================================
    if highlight_cells is None:
        highlight_cells = []

    highlight_cells = pd.Index(highlight_cells).astype(str)

    adata_plot.obs["is_highlight_cell"] = (
        adata_plot.obs_names.astype(str).isin(highlight_cells)
    )

    # ============================================================
    # Color map
    # ============================================================
    if use_soft_cmap:
        cmap_to_use = LinearSegmentedColormap.from_list(
            "soft_blue_white_red",
            [
                "#5B9DB8",
                "#D8EEF3",
                "#F7F7F7",
                "#F0B39A",
                "#B94A48",
            ],
        )
    else:
        cmap_to_use = cmap

    # ============================================================
    # Color range
    # ============================================================
    valid_delta = delta_cos[np.isfinite(delta_cos)]

    if len(valid_delta) == 0:
        raise ValueError("All delta cosine values are NaN.")

    if vlim is None:
        vmin = np.nanpercentile(valid_delta, percentile_low)
        vmax = np.nanpercentile(valid_delta, percentile_high)

        if np.isnan(vmin) or np.isnan(vmax) or np.isclose(vmin, vmax):
            vmin = np.nanmin(valid_delta)
            vmax = np.nanmax(valid_delta)

        if np.isclose(vmin, vmax):
            vmin -= 1e-6
            vmax += 1e-6
    else:
        vmin, vmax = vlim

    if title is None:
        title = "Δ cosine similarity to reference after perturbation"

    # ============================================================
    # Spatial plot: all plotted cells as circles
    # ============================================================
    sc.pl.spatial(
        adata_plot,
        color=delta_key,
        spot_size=spot_size,
        cmap=cmap_to_use,
        vmin=vmin,
        vmax=vmax,
        title=title,
        show=False,
    )

    fig = plt.gcf()
    ax = fig.axes[0]

    # ============================================================
    # Overlay highlighted perturbation cells as stars
    # ============================================================
    highlight_mask = adata_plot.obs["is_highlight_cell"].values

    if np.sum(highlight_mask) > 0:
        coords = np.asarray(adata_plot.obsm["spatial"])

        x = coords[highlight_mask, 0]
        y = coords[highlight_mask, 1]

        c = adata_plot.obs.loc[
            highlight_mask,
            delta_key
        ].astype(float).values

        ax.scatter(
            x,
            y,
            c=c,
            cmap=cmap_to_use,
            vmin=vmin,
            vmax=vmax,
            marker=target_marker,
            s=target_size,
            edgecolors=target_edgecolor,
            linewidths=target_linewidth,
            zorder=20,
        )

    # ============================================================
    # Save or show
    # ============================================================
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    else:
        plt.close(fig)

    # ============================================================
    # Summary
    # ============================================================
    target_mask_plot = adata_plot.obs["is_target_celltype"].values
    highlight_mask = adata_plot.obs["is_highlight_cell"].values
    delta_plot = adata_plot.obs[delta_key].values.astype(float)

    if verbose:
        print(f"Cells Δ cosine to reference ({reference_mode}):")
        print(f"  plotted slice             = {slice_perturb}")
        print(f"  reference slice           = {slice_reference}")
        print(f"  target cell type          = {target_celltype}")
        print(f"  all plotted cells n       = {np.sum(np.isfinite(delta_plot))}")
        print(f"  target celltype n         = {np.sum(target_mask_plot)}")
        print(f"  non-target cells n        = {np.sum(~target_mask_plot)}")
        print(f"  highlighted cells n       = {np.sum(highlight_mask)}")
        print("--------------------------------------------------")
        print(f"  delta min                 = {np.nanmin(delta_plot):.6f}")
        print(f"  delta max                 = {np.nanmax(delta_plot):.6f}")
        print(
            "  delta p1/p2/p5/p50/p95/p98/p99 =",
            np.nanpercentile(delta_plot, [1, 2, 5, 50, 95, 98, 99]),
        )
        print("--------------------------------------------------")
        print(f"  color vmin / vmax         = {vmin:.6f} / {vmax:.6f}")
        print(f"  all cells mean delta      = {np.nanmean(delta_plot):.6f}")
        print(f"  all cells median delta    = {np.nanmedian(delta_plot):.6f}")
        print(f"  target mean delta         = {np.nanmean(delta_plot[target_mask_plot]):.6f}")
        print(f"  target median delta       = {np.nanmedian(delta_plot[target_mask_plot]):.6f}")
        print(f"  non-target mean delta     = {np.nanmean(delta_plot[~target_mask_plot]):.6f}")
        print(f"  non-target median delta   = {np.nanmedian(delta_plot[~target_mask_plot]):.6f}")

        if np.sum(highlight_mask) > 0:
            print(f"  highlighted mean delta    = {np.nanmean(delta_plot[highlight_mask]):.6f}")
            print(f"  highlighted median delta  = {np.nanmedian(delta_plot[highlight_mask]):.6f}")
        else:
            print("  highlighted mean delta    = NA")
            print("  highlighted median delta  = NA")

    return adata_plot

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
    Inject donor cells into the spatial niche of target cells by assigning donor coordinates
    near randomly selected target cells.

    Parameters
    ----------
    target_adata : AnnData
        Target AnnData containing spatial coordinates in ``.obsm["spatial"]`` (shape: n_cells x 2).
    donor_adata : AnnData
        Donor AnnData providing cells to inject.
    target_filter : dict[str, str] or None, default None
        Filter for selecting target cells from ``target_adata.obs`` (AND logic).
    donor_filter : dict[str, str] or None, default None
        Filter for selecting donor cells from ``donor_adata.obs`` (AND logic).
    n_inject : int or None, default None
        Number of donor cells to inject. If None, inject all filtered donor cells.
    spatial_jitter_std : float, default 1.0
        Noise scale for donor coordinates. Noise std is ``median_nn_distance(target) * spatial_jitter_std``.
    random_state : int, default 42
        Random seed.

    Returns
    -------
    AnnData
        Combined AnnData of target + injected donor cells. Adds ``obs["injected"]`` and
        ``obs["injected_from_slice"]`` to indicate injected cells and their origin.
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
#
# def analyze_embedding_similarity_change(
#     adata_ori_result, adata_perturb_result,
#     target_slice_young, target_slice_old, target_celltype,
#     embedding_key="X_emb"
# ):
#     """
#     Compute cosine similarity and Euclidean distance between old and young cell embeddings
#     before and after perturbation.
#     """
#     # Extract embeddings
#     emb_young = adata_ori_result.obsm[embedding_key][
#         (adata_ori_result.obs["slice"] == target_slice_young) &
#         (adata_ori_result.obs["cell_type"] == target_celltype)
#     ]
#     emb_old_ori = adata_ori_result.obsm[embedding_key][
#         (adata_ori_result.obs["slice"] == target_slice_old) &
#         (adata_ori_result.obs["cell_type"] == target_celltype)
#     ]
#     emb_old_perturb = adata_perturb_result.obsm[embedding_key][
#         (adata_perturb_result.obs["slice"] == target_slice_old) &
#         (adata_perturb_result.obs["cell_type"] == target_celltype)
#     ]
#
#     # Mean embedding of young cells
#     mean_young_emb = emb_young.mean(axis=0)
#
#     # Cosine similarity
#     sim_ori = cosine_similarity(emb_old_ori, mean_young_emb.reshape(1, -1)).mean()
#     sim_perturb = cosine_similarity(emb_old_perturb, mean_young_emb.reshape(1, -1)).mean()
#
#     # Euclidean distance
#     dist_ori = euclidean_distances(emb_old_ori, mean_young_emb.reshape(1, -1)).mean()
#     dist_perturb = euclidean_distances(emb_old_perturb, mean_young_emb.reshape(1, -1)).mean()
#
#     return {
#         "similarity_before": sim_ori,
#         "similarity_after": sim_perturb,
#         "delta_similarity": sim_perturb - sim_ori,
#         "euclidean_before": dist_ori,
#         "euclidean_after": dist_perturb,
#         "delta_euclidean": dist_ori - dist_perturb  # positive: closer after
#     }

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
        import ot
        from ot.unbalanced import sinkhorn_unbalanced
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
    embedding_key="X_emb", gamma=1.0,
    direction="KO"
):
    """
    Compare niche embeddings before/after perturbation, excluding target cell type.
    Returns cosine similarity, Euclidean distance, EMD (Wasserstein), and MMD.

    direction="KO": perturbed = old slice, reference = young niche.
    direction="OE": perturbed = young slice, reference = old niche.
    """
    if direction == "KO":
        mask_ref = (
            (adata_ori_result.obs["slice"] == target_slice_young) &
            (adata_ori_result.obs["cell_type"] != target_celltype)
        )
        mask_ori = (
            (adata_ori_result.obs["slice"] == target_slice_old) &
            (adata_ori_result.obs["cell_type"] != target_celltype)
        )
        mask_pert = (
            (adata_perturb_result.obs["slice"] == target_slice_old) &
            (adata_perturb_result.obs["cell_type"] != target_celltype)
        )
    else:
        mask_ref = (
            (adata_ori_result.obs["slice"] == target_slice_old) &
            (adata_ori_result.obs["cell_type"] != target_celltype)
        )
        mask_ori = (
            (adata_ori_result.obs["slice"] == target_slice_young) &
            (adata_ori_result.obs["cell_type"] != target_celltype)
        )
        mask_pert = (
            (adata_perturb_result.obs["slice"] == target_slice_young) &
            (adata_perturb_result.obs["cell_type"] != target_celltype)
        )

    emb_ref = np.asarray(adata_ori_result.obsm[embedding_key][mask_ref])
    emb_ori = np.asarray(adata_ori_result.obsm[embedding_key][mask_ori])
    emb_pert = np.asarray(adata_perturb_result.obsm[embedding_key][mask_pert])

    if emb_ref.size == 0:
        raise ValueError("No reference cells found with the given filters.")
    if emb_ori.size == 0:
        raise ValueError("No original cells found with the given filters.")
    if emb_pert.size == 0:
        raise ValueError("No perturbed cells found with the given filters.")

    emb_ref = np.nan_to_num(emb_ref, nan=0.0)
    emb_ori = np.nan_to_num(emb_ori, nan=0.0)
    emb_pert = np.nan_to_num(emb_pert, nan=0.0)

    mean_ref_emb = emb_ref.mean(axis=0, keepdims=True)

    sim_ori = cosine_similarity(emb_ori, mean_ref_emb).mean()
    sim_perturb = cosine_similarity(emb_pert, mean_ref_emb).mean()

    dist_ori = euclidean_distances(emb_ori, mean_ref_emb).mean()
    dist_perturb = euclidean_distances(emb_pert, mean_ref_emb).mean()

    d = emb_ref.shape[1]
    emd_ori = np.mean([wasserstein_distance(emb_ori[:, i], emb_ref[:, i]) for i in range(d)])
    emd_perturb = np.mean([wasserstein_distance(emb_pert[:, i], emb_ref[:, i]) for i in range(d)])

    mmd_ori = compute_mmd(emb_ori, emb_ref, gamma)
    mmd_perturb = compute_mmd(emb_pert, emb_ref, gamma)

    return {
        "cosine": (float(sim_ori), float(sim_perturb)),
        "euclidean": (float(dist_ori), float(dist_perturb)),
        "emd": (float(emd_ori), float(emd_perturb)),
        "mmd": (float(mmd_ori), float(mmd_perturb)),
    }


def analyze_embedding_similarity_change(
    adata_ori_result, adata_perturb_result,
    target_slice_young, target_slice_old, target_celltype,
    embedding_key="X_emb", gamma=1.0,
    direction="KO"
):
    """
    Compute cosine similarity, Euclidean distance, Wasserstein distance (EMD) and MMD
    between perturbed and reference cell embeddings.

    direction="KO": perturbed = old cells, reference = mean of young cells.
    direction="OE": perturbed = young cells, reference = mean of old cells.
    """
    if direction == "KO":
        emb_ref = adata_ori_result.obsm[embedding_key][
            (adata_ori_result.obs["slice"] == target_slice_young) &
            (adata_ori_result.obs["cell_type"] == target_celltype)
        ]
        emb_ori = adata_ori_result.obsm[embedding_key][
            (adata_ori_result.obs["slice"] == target_slice_old) &
            (adata_ori_result.obs["cell_type"] == target_celltype)
        ]
        emb_pert = adata_perturb_result.obsm[embedding_key][
            (adata_perturb_result.obs["slice"] == target_slice_old) &
            (adata_perturb_result.obs["cell_type"] == target_celltype)
        ]
    else:
        emb_ref = adata_ori_result.obsm[embedding_key][
            (adata_ori_result.obs["slice"] == target_slice_old) &
            (adata_ori_result.obs["cell_type"] == target_celltype)
        ]
        emb_ori = adata_ori_result.obsm[embedding_key][
            (adata_ori_result.obs["slice"] == target_slice_young) &
            (adata_ori_result.obs["cell_type"] == target_celltype)
        ]
        emb_pert = adata_perturb_result.obsm[embedding_key][
            (adata_perturb_result.obs["slice"] == target_slice_young) &
            (adata_perturb_result.obs["cell_type"] == target_celltype)
        ]

    mean_ref_emb = emb_ref.mean(axis=0)

    sim_ori = cosine_similarity(emb_ori, mean_ref_emb.reshape(1, -1))
    sim_perturb = cosine_similarity(emb_pert, mean_ref_emb.reshape(1, -1))

    dist_ori = euclidean_distances(emb_ori, mean_ref_emb.reshape(1, -1))
    dist_perturb = euclidean_distances(emb_pert, mean_ref_emb.reshape(1, -1))

    emd_ori = np.mean([wasserstein_distance(emb_ori[:, i], emb_ref[:, i])
                       for i in range(emb_ref.shape[1])])
    emd_perturb = np.mean([wasserstein_distance(emb_pert[:, i], emb_ref[:, i])
                           for i in range(emb_ref.shape[1])])

    mmd_ori = compute_mmd(emb_ori, emb_ref, gamma)
    mmd_perturb = compute_mmd(emb_pert, emb_ref, gamma)

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
    target_celltype=None,
    highlight_cells=None,
    celltype_key="cell_type",
    slice_key="slice",
    emb_key="X_emb",
    title="Cell state positioning relative to young and old centroids",
    agg_by_celltype=False,
    exclude_celltype=False,
    direction="KO",
    perturb_scope="slice",  # "slice" or "highlight"
    save_path=None,
    show=True,
):
    """
    Scatter plot of cosine similarity to young vs old centroids.

    X-axis:
        cosine similarity to young centroid.

    Y-axis:
        cosine similarity to old centroid.

    direction:
        "KO":
            perturbation is applied to old slice.
        "OE":
            perturbation is applied to young slice.

    perturb_scope:
        "slice":
            plot all cells in the perturbed slice.
            KO: all old slice cells after perturbation.
            OE: all young slice cells after perturbation.

        "highlight":
            plot only actual perturbed cells passed by highlight_cells.
            For KO, this should be perturbed_cells_final from updated apply_gene_perturbation().
            For OE, this should also be perturbed_cells_final.

    exclude_celltype:
        If True, remove target_celltype from young, old, and perturbed groups.
        This is mainly for niche-level plot.
    """

    # ============================================================
    # Basic checks
    # ============================================================
    for adata_name, adata in [("adata_ori", adata_ori), ("adata_perturb", adata_perturb)]:
        if emb_key not in adata.obsm:
            raise KeyError(f"{emb_key} not found in {adata_name}.obsm")
        if slice_key not in adata.obs:
            raise KeyError(f"{slice_key} not found in {adata_name}.obs")
        if celltype_key not in adata.obs:
            raise KeyError(f"{celltype_key} not found in {adata_name}.obs")

    if direction not in ["KO", "OE"]:
        raise ValueError("direction must be 'KO' or 'OE'.")

    if perturb_scope not in ["slice", "highlight"]:
        raise ValueError("perturb_scope must be 'slice' or 'highlight'.")

    if perturb_scope == "highlight" and highlight_cells is None:
        raise ValueError("When perturb_scope='highlight', highlight_cells must be provided.")

    # ============================================================
    # Extract embeddings
    # ============================================================
    X_ori = adata_ori.obsm[emb_key]
    X_perturb = adata_perturb.obsm[emb_key]

    if hasattr(X_ori, "toarray"):
        X_ori = X_ori.toarray()
    if hasattr(X_perturb, "toarray"):
        X_perturb = X_perturb.toarray()

    X_ori = np.nan_to_num(np.asarray(X_ori), nan=0.0)
    X_perturb = np.nan_to_num(np.asarray(X_perturb), nan=0.0)

    # ============================================================
    # Build base masks
    # ============================================================
    sl_ori = adata_ori.obs[slice_key].astype(str).values
    sl_perturb = adata_perturb.obs[slice_key].astype(str).values

    ct_ori = adata_ori.obs[celltype_key].astype(str).values
    ct_perturb = adata_perturb.obs[celltype_key].astype(str).values

    mask_young = sl_ori == str(slice_young)
    mask_old_ori = sl_ori == str(slice_old)

    if direction == "OE":
        mask_pert = sl_perturb == str(slice_young)
        perturb_label = "Perturbed young slice"
        perturb_mean_label = "Perturbed young mean"
        perturb_color = "#791f1fff"
    else:
        mask_pert = sl_perturb == str(slice_old)
        perturb_label = "Perturbed old slice"
        perturb_mean_label = "Perturbed old mean"
        perturb_color = "#6f8f94ff"

    # ============================================================
    # Restrict perturbed group to actual KO/OE cells if requested
    # ============================================================
    if perturb_scope == "highlight":
        highlight_cells = pd.Index(highlight_cells).astype(str)

        mask_highlight = adata_perturb.obs_names.astype(str).isin(highlight_cells)

        mask_pert = mask_pert & mask_highlight

        perturb_label = "Actual perturbed cells"
        perturb_mean_label = "Actual perturbed mean"

    # ============================================================
    # Optionally exclude target cell type
    # ============================================================
    if exclude_celltype and target_celltype is not None:
        mask_young = mask_young & (ct_ori != str(target_celltype))
        mask_old_ori = mask_old_ori & (ct_ori != str(target_celltype))
        mask_pert = mask_pert & (ct_perturb != str(target_celltype))

    # ============================================================
    # Check mask sizes
    # ============================================================
    if mask_young.sum() == 0:
        raise ValueError("No young cells selected.")
    if mask_old_ori.sum() == 0:
        raise ValueError("No old cells selected.")
    if mask_pert.sum() == 0:
        raise ValueError(
            "No perturbed cells selected. "
            "Check direction, perturb_scope, highlight_cells, and exclude_celltype."
        )

    # ============================================================
    # Compute young / old centroids from original embeddings
    # ============================================================
    X_unit_ori = normalize(X_ori, norm="l2", axis=1)

    young_centroid = normalize(
        X_unit_ori[mask_young].mean(axis=0, keepdims=True),
        norm="l2",
        axis=1,
    )[0]

    old_centroid = normalize(
        X_unit_ori[mask_old_ori].mean(axis=0, keepdims=True),
        norm="l2",
        axis=1,
    )[0]

    def cos_coords(X):
        X_unit = normalize(X, norm="l2", axis=1)
        cos_y = X_unit @ young_centroid
        cos_o = X_unit @ old_centroid
        return cos_y, cos_o

    cos_y_y, cos_o_y = cos_coords(X_ori[mask_young])
    cos_y_o, cos_o_o = cos_coords(X_ori[mask_old_ori])
    cos_y_p, cos_o_p = cos_coords(X_perturb[mask_pert])

    # ============================================================
    # Plot
    # ============================================================
    plt.figure(figsize=(6.5, 6))

    if not agg_by_celltype:
        plt.scatter(
            cos_y_y,
            cos_o_y,
            s=35,
            c="#076a3aff",
            alpha=0.7,
            edgecolor="none",
            label="Young cells",
        )

        plt.scatter(
            cos_y_o,
            cos_o_o,
            s=35,
            c="#073f6aff",
            alpha=0.7,
            edgecolor="none",
            label="Old cells",
        )

        plt.scatter(
            cos_y_p,
            cos_o_p,
            s=45 if perturb_scope == "highlight" else 35,
            c=perturb_color,
            alpha=0.9,
            edgecolor="none",
            label=perturb_label,
        )

        plt.scatter(
            np.mean(cos_y_y),
            np.mean(cos_o_y),
            c="#076a3aff",
            s=200,
            marker="*",
            edgecolor="white",
            linewidth=1,
            label="Young mean",
        )

        plt.scatter(
            np.mean(cos_y_o),
            np.mean(cos_o_o),
            c="#073f6aff",
            s=200,
            marker="*",
            edgecolor="white",
            linewidth=1,
            label="Old mean",
        )

        plt.scatter(
            np.mean(cos_y_p),
            np.mean(cos_o_p),
            c=perturb_color,
            s=220,
            marker="*",
            edgecolor="white",
            linewidth=1,
            label=perturb_mean_label,
        )

    else:
        for cos_y, cos_o, label, mask, adata in [
            (cos_y_y, cos_o_y, "Young", mask_young, adata_ori),
            (cos_y_o, cos_o_o, "Old", mask_old_ori, adata_ori),
            (cos_y_p, cos_o_p, perturb_label, mask_pert, adata_perturb),
        ]:
            ct = adata.obs[celltype_key].values[mask]
            df = pd.DataFrame(
                {
                    "celltype": ct,
                    "cos_y": cos_y,
                    "cos_o": cos_o,
                }
            )

            df_mean = (
                df.groupby("celltype")[["cos_y", "cos_o"]]
                .mean()
                .reset_index()
            )

            plt.scatter(
                df_mean["cos_y"],
                df_mean["cos_o"],
                s=50,
                alpha=0.9,
                label=label,
            )

    # ============================================================
    # Axis range
    # ============================================================
    all_vals = np.concatenate(
        [
            cos_y_y,
            cos_o_y,
            cos_y_o,
            cos_o_o,
            cos_y_p,
            cos_o_p,
        ]
    )

    lim_min = np.nanmin(all_vals)
    lim_max = np.nanmax(all_vals)
    pad = 0.02

    plt.plot(
        [lim_min - pad, lim_max + pad],
        [lim_min - pad, lim_max + pad],
        ls="--",
        c="gray",
        lw=1,
    )

    plt.xlim(lim_min - pad, lim_max + pad)
    plt.ylim(lim_min - pad, lim_max + pad)

    plt.xlabel("Cosine similarity to young centroid")
    plt.ylabel("Cosine similarity to old centroid")
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(alpha=0.2, ls="--")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if show:
        plt.show()
    else:
        plt.close()

    # ============================================================
    # Summary
    # ============================================================
    print("========== plot_cosine_to_centroids_with_perturb ==========")
    print(f"direction              = {direction}")
    print(f"perturb_scope          = {perturb_scope}")
    print(f"exclude_celltype       = {exclude_celltype}")
    print(f"target_celltype        = {target_celltype}")
    print(f"young cells n          = {mask_young.sum()}")
    print(f"old cells n            = {mask_old_ori.sum()}")
    print(f"perturbed plotted n    = {mask_pert.sum()}")
    print("----------------------------------------------------------")
    print(f"young mean coord       = ({np.mean(cos_y_y):.6f}, {np.mean(cos_o_y):.6f})")
    print(f"old mean coord         = ({np.mean(cos_y_o):.6f}, {np.mean(cos_o_o):.6f})")
    print(f"perturbed mean coord   = ({np.mean(cos_y_p):.6f}, {np.mean(cos_o_p):.6f})")
    print("==========================================================")

    return {
        "cos_young_young": cos_y_y,
        "cos_old_young": cos_o_y,
        "cos_young_old": cos_y_o,
        "cos_old_old": cos_o_o,
        "cos_young_perturbed": cos_y_p,
        "cos_old_perturbed": cos_o_p,
        "mask_young": mask_young,
        "mask_old": mask_old_ori,
        "mask_perturbed": mask_pert,
    }


def plot_global_euclidean_shift(
    adata_ori,
    adata_perturb,
    emb_key="X_emb",
    title="Perturbation-Induced Changes in Euclidean Distance to Global Reference",
    cmap="RdBu_r",
    save_path=None
):
    X_ori = adata_ori.obsm[emb_key]
    if hasattr(X_ori, "toarray"):
        X_ori = X_ori.toarray()

    X_perturb = adata_perturb.obsm[emb_key]
    if hasattr(X_perturb, "toarray"):
        X_perturb = X_perturb.toarray()

    global_centroid = X_ori.mean(axis=0, keepdims=True)

    dist_before = np.linalg.norm(X_ori - global_centroid, axis=1)
    dist_after = np.linalg.norm(X_perturb - global_centroid, axis=1)

    plt.figure(figsize=(7, 6))
    sns.kdeplot(
        x=dist_before,
        y=dist_after,
        fill=True,
        cmap=cmap,
        bw_adjust=0.8,
        thresh=0.02,
        levels=60
    )

    min_val = min(dist_before.min(), dist_after.min())
    max_val = max(dist_before.max(), dist_after.max())
    plt.plot([min_val, max_val], [min_val, max_val], '--', color='gray', lw=1.2)
    plt.xlabel("Euclidean Distance to Global Centroid (Before Perturbation)", fontsize=12)
    plt.ylabel("Euclidean Distance to Global Centroid (After Perturbation)", fontsize=12)
    plt.title(title, fontsize=13, pad=10)
    plt.grid(alpha=0.2, lw=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)
        plt.close()
    else:
        plt.show()


# ============================================================
# Backward-compatible aliases (do NOT remove — external callers depend on these names)
# ============================================================

def analyze_embedding_similarity_change_OE(*args, **kwargs):
    kwargs["direction"] = "OE"
    return analyze_embedding_similarity_change(*args, **kwargs)


def analyze_embedding_similarity_change_similarity_niche_OE(*args, **kwargs):
    kwargs["direction"] = "OE"
    return analyze_embedding_similarity_change_similarity_niche(*args, **kwargs)


def plot_cosine_to_centroids_with_perturb_OE(*args, **kwargs):
    kwargs["direction"] = "OE"
    return plot_cosine_to_centroids_with_perturb(*args, **kwargs)


def plot_cosine_to_centroids_with_perturb_old(*args, **kwargs):
    return plot_cosine_to_centroids_with_perturb(*args, **kwargs)


def plot_cosine_to_centroids_non_target(adata_ori, adata_perturb, slice_young, slice_old, target_cell, **kwargs):
    return plot_cosine_to_centroids_with_perturb(
        adata_ori, adata_perturb, slice_young, slice_old,
        target_celltype=target_cell, exclude_celltype=True, **kwargs)
