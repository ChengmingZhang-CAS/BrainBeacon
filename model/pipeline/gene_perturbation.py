import os
import torch
import joblib
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, List, Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm
from config.config_train_cdniche import config_train
from model.brain_beacon import BrainBeacon
from config.config import GENE_LOOKUP_DIR
import pickle

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
            total_context_length=self.model_config['context_length'] * self.model_config['num_neighbors']
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

            if idx + 1 >= self.total_length:
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


class InSilicoPerturberPipeline:
    def __init__(
        self,
        pretrain_ckpt: str,
        model_config: dict,
        device: Union[str, torch.device] = 'cpu',
        perturb_type: str = "delete",
        genes_to_perturb: Optional[List[str]] = None,
        keep_unperturbed: bool = False,
    ):
        """
        Initialize the perturbation pipeline with model_raw and perturbation settings.
        """
        self.device = device
        self.model_config = model_config
        self.model = None
        self.pretrain_ckpt: str = pretrain_ckpt

        # --- Perturbation settings ---
        self.perturb_type = perturb_type
        self.genes_to_perturb_names = genes_to_perturb if genes_to_perturb is not None else []
        self.keep_unperturbed = keep_unperturbed

        # --- Load gene lookup table ---
        lookup_path = os.path.join(GENE_LOOKUP_DIR, "ensembl_to_gene_idx.pkl")
        with open(lookup_path, "rb") as f:
            self.ensembl_to_gene_idx = pickle.load(f)
        self.idx_to_ensembl = {v: k for k, v in self.ensembl_to_gene_idx.items()}

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

    def apply_perturbation(self, real_indices: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Apply in silico perturbation to the input_ids (real_indices) and attention_mask.
        """
        if not self.genes_to_perturb_names:
            return real_indices, attention_mask

        padding_token = 1
        perturbed_real_indices = real_indices.clone()
        perturbed_attention_mask = attention_mask.clone()

        # Only support single gene for now
        gene_name = self.genes_to_perturb_names[0]
        gene_idx = self.ensembl_to_gene_idx.get(gene_name, None)
        if gene_idx is None:
            return real_indices, attention_mask

        # Mask target gene by replacing it with PAD
        mask_to_delete = (perturbed_real_indices == gene_idx)
        # if mask_to_delete.sum() > 0:
        #     print(f"[Perturbation] Gene '{gene_name}' (token idx={gene_idx}) removed from {mask_to_delete.sum().item()} positions.")
        perturbed_real_indices = torch.where(mask_to_delete, torch.full_like(perturbed_real_indices, padding_token),
                                             perturbed_real_indices)
        perturbed_attention_mask = torch.where(mask_to_delete, torch.ones_like(perturbed_attention_mask),
                                               perturbed_attention_mask)

        return perturbed_real_indices, perturbed_attention_mask

    def token_idx_to_gene_name(self, token_idx: int) -> str:
        return self.idx_to_ensembl.get(token_idx, "UNKNOWN")

    def infer(self, dataloader, config_train: dict):
        """
        Run inference on new data using the pretrained model_raw.
        Forward once with original inputs (baseline) and once with perturbed inputs.
        """
        self.model.eval()
        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')

        cell_indices = []
        perturbed_flags = []
        baseline_outputs = []
        perturbed_outputs = []
        baseline_gene_name_lists = []
        perturbed_gene_name_lists = []

        with torch.no_grad():
            for batch_idx, (
                    real_indices, attention_mask, connect_comp, rna_type, cell_raw_idx,
                    neighbor_gene_distribution, exp
            ) in enumerate(tqdm(dataloader, desc="Processing batches", total=len(dataloader))):

                real_indices = real_indices[0]
                attention_mask = attention_mask[0]
                connect_comp = connect_comp[0]
                rna_type = rna_type[0]
                real_indices_view = real_indices.view(-1).long()
                neighbor_gene_distribution = neighbor_gene_distribution[0].long()
                exp = exp[0].float()

                # Load ESM embeddings
                esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], -1)

                # Move to device
                real_indices = real_indices.to(self.device)
                attention_mask = attention_mask.to(self.device)
                connect_comp = connect_comp.to(self.device)
                rna_type = rna_type.to(self.device)
                esm_embedding = esm_embedding.to(self.device)
                neighbor_gene_distribution = neighbor_gene_distribution.to(self.device)

                pool_skip_tokens = config_train.get("pool_skip_tokens", 3)

                # Forward baseline
                sequence_mask = (real_indices != 1).long()
                baseline_output = self.model(
                    real_indices, connect_comp, rna_type, attention_mask,
                    esm_embedding, neighbor_gene_distribution, sequence_mask
                )  # (B, L, H)
                baseline_output = baseline_output[:, pool_skip_tokens:, :].detach().cpu()

                # Apply perturbation
                perturbed_real_indices, perturbed_attention_mask = self.apply_perturbation(
                    real_indices, attention_mask
                )

                # Forward perturbed
                sequence_mask = (perturbed_real_indices != 1).long()
                perturbed_output = self.model(
                    perturbed_real_indices, connect_comp, rna_type, perturbed_attention_mask,
                    esm_embedding, neighbor_gene_distribution, sequence_mask
                )
                perturbed_output = perturbed_output[:, pool_skip_tokens:, :].detach().cpu()

                # Compute mask and perturb flag
                mask = (perturbed_real_indices[:, pool_skip_tokens:] != 1).cpu()
                perturbed_flags_batch = (real_indices != perturbed_real_indices).any(dim=1).tolist()

                for i in range(mask.shape[0]):
                    is_perturbed = perturbed_flags_batch[i]

                    if not is_perturbed and not self.keep_unperturbed:
                        continue

                    perturbed_flags.append(is_perturbed)
                    cell_indices.append(cell_raw_idx[i][0])

                    baseline_outputs.append(baseline_output[i][mask[i]])
                    perturbed_outputs.append(perturbed_output[i][mask[i]])

                    baseline_names = [
                        self.idx_to_ensembl.get(idx.item(), "UNKNOWN")
                        for idx, keep in zip(real_indices[i, pool_skip_tokens:], mask[i]) if keep
                    ]
                    perturbed_names = [
                        self.idx_to_ensembl.get(idx.item(), "UNKNOWN")
                        for idx, keep in zip(perturbed_real_indices[i, pool_skip_tokens:], mask[i]) if keep
                    ]

                    baseline_gene_name_lists.append(baseline_names)
                    perturbed_gene_name_lists.append(perturbed_names)

        results = {
            "cell_raw_index": cell_indices,
            "perturbed_flag": perturbed_flags,
            "baseline_embedding": baseline_outputs,
            "perturbed_embedding": perturbed_outputs,
            "baseline_gene_names": baseline_gene_name_lists,
            "perturbed_gene_names": perturbed_gene_name_lists,
        }

        return results

    def run(self, data_paths: List[str], config_train: dict):
        """
        Main method to run the entire training pipeline.
        """
        dataset = self.load_dataset(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        
        pred = self.infer(data_loader, config_train)
        return pred

def analyze_gene_impact(results: dict, metric: str = "cosine", topk: int = 10):
    """
    Analyze perturbation impact on gene-level embeddings for each perturbed cell.

    Args:
        results (dict): Output from the perturbation pipeline.
        metric (str): Similarity metric to use: 'cosine' or 'l2'.
        topk (int): Number of top-changed genes to return per cell.

    Returns:
        dict with keys:
            - 'perturbed_cell_names': cell_raw_index of perturbed cells
            - 'similarity_score': average similarity (cosine or l2) per cell
            - 'top_changed_genes': list of top-k most changed genes per cell
    """
    assert metric in {"cosine", "l2"}, "metric must be either 'cosine' or 'l2'"

    perturbed_flags = results["perturbed_flag"]
    baseline_embeddings = results["baseline_embedding"]
    perturbed_embeddings = results["perturbed_embedding"]
    gene_names = results["baseline_gene_names"]
    cell_ids = results["cell_raw_index"]

    perturbed_cell_names = []
    similarity_score = []
    top_changed_genes = []

    for i, flag in enumerate(perturbed_flags):
        if not flag:
            continue

        base_emb = baseline_embeddings[i].numpy()
        pert_emb = perturbed_embeddings[i].numpy()
        genes = gene_names[i]

        if base_emb.shape != pert_emb.shape:
            continue

        if metric == "cosine":
            sims = np.array([
                cosine_similarity(base_emb[j][None], pert_emb[j][None])[0][0]
                for j in range(base_emb.shape[0])
            ])
            score = np.mean(sims)
            changes = 1 - sims  # higher change = 1 - sim
        else:  # 'l2'
            dists = np.linalg.norm(base_emb - pert_emb, axis=1)
            score = np.mean(dists)
            changes = dists

        topk_idx = np.argsort(-changes)[:topk]
        top_genes = [genes[j] for j in topk_idx]

        perturbed_cell_names.append(cell_ids[i])
        similarity_score.append(score)
        top_changed_genes.append(top_genes)

    return {
        "perturbed_cell_names": perturbed_cell_names,
        "similarity_score": similarity_score,
        "top_changed_genes": top_changed_genes
    }


def plot_top_changed_gene_frequency(
    impact_result: dict,
    topn: int = 20,
    output_path: str = None,
    figsize=(10, 6),
    show_percentage: bool = True
):
    """
    Plot the frequency (or percentage) of genes appearing in top-k changed genes across cells,
    with text annotation on bars.

    Args:
        impact_result (dict): Output from `analyze_gene_impact`.
        topn (int): Number of most frequent genes to show.
        output_path (str or None): Path to save the plot.
        figsize (tuple): Figure size.
        show_percentage (bool): Whether to convert frequency to percentage.
    """
    top_genes_all = [gene for gene_list in impact_result["top_changed_genes"] for gene in gene_list]
    gene_counter = Counter(top_genes_all)
    most_common = gene_counter.most_common(topn)

    genes, freqs_raw = zip(*most_common)

    if show_percentage:
        num_cells = len(impact_result["perturbed_cell_names"])
        freqs = [f / num_cells * 100 for f in freqs_raw]
        labels = [f"{v:.1f}%" for v in freqs]
        xlabel = "Percentage of Cells (%)"
    else:
        freqs = freqs_raw
        labels = [str(v) for v in freqs]
        xlabel = "Frequency (appeared in Top-k)"

    plt.figure(figsize=figsize)
    bars = plt.barh(genes[::-1], freqs[::-1], color='slateblue')
    plt.xlabel(xlabel)
    plt.title(f"Top {topn} Most Frequently Changed Genes")

    # Add value labels to bars
    for bar, label in zip(bars, labels[::-1]):
        plt.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            label, va='center', fontsize=10
        )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


def get_gene_change_matrix(results: dict, metric: str = "cosine", topk_union: int = None) -> pd.DataFrame:
    """
    Construct gene x cell matrix of change scores (1 - cosine or L2).

    Args:
        results (dict): Output from the perturbation pipeline.
        metric (str): Similarity metric: 'cosine' or 'l2'.
        topk_union (int or None): If set, restrict to union of top-k genes per cell.

    Returns:
        pd.DataFrame: rows=gene names, cols=cell ids, values=change score
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import pandas as pd

    assert metric in {"cosine", "l2"}

    all_gene_scores = {}

    for i, flag in enumerate(results["perturbed_flag"]):
        if not flag:
            continue

        cell_name = results["cell_raw_index"][i]
        base_emb = results["baseline_embedding"][i].numpy()
        pert_emb = results["perturbed_embedding"][i].numpy()
        gene_list = results["baseline_gene_names"][i]

        if base_emb.shape != pert_emb.shape:
            continue

        if metric == "cosine":
            sims = np.array([
                cosine_similarity(base_emb[j][None], pert_emb[j][None])[0][0]
                for j in range(base_emb.shape[0])
            ])
            changes = 1 - sims
        else:
            changes = np.linalg.norm(base_emb - pert_emb, axis=1)

        gene_scores = dict(zip(gene_list, changes))

        if topk_union is not None:
            gene_scores = dict(sorted(gene_scores.items(), key=lambda x: -x[1])[:topk_union])

        all_gene_scores[cell_name] = gene_scores

    df = pd.DataFrame.from_dict(all_gene_scores, orient='index').T
    return df

def plot_gene_change_heatmap(df, output_path=None, zscore=True, cluster=True):
    """
    Plot heatmap of gene × cell change matrix.

    Args:
        df (pd.DataFrame): Gene x Cell matrix of change scores
        output_path (str): Optional path to save figure
        zscore (bool): Whether to apply z-score normalization per gene
        cluster (bool): Whether to use clustermap (hierarchical clustering)
    """
    if zscore:
        df = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1) + 1e-6, axis=0)

    plt.figure(figsize=(max(10, df.shape[1] // 3), max(6, df.shape[0] // 3)))

    if cluster:
        df = df.fillna(0)
        sns.clustermap(df, cmap="RdBu_r", figsize=(10, 10), xticklabels=False)
    else:
        sns.heatmap(df, cmap="RdBu_r")

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_top_perturbed_genes_spatial(
    adata: sc.AnnData,
    impact_result: dict,
    symbol_dict: dict,
    output_dir: str,
    topn: int = 4,
    layer_key: str = "counts",
    embedding: str = "spatial",
    log1p: bool = False
):
    """
    Plot spatial expression of top perturbation-response genes.

    Args:
        adata (AnnData): Full annotated dataset (with spatial and counts layer).
        impact_result (dict): Output of analyze_gene_impact.
        symbol_dict (dict): Mapping from Ensembl ID to gene symbol.
        output_dir (str): Where to save the final figure.
        topn (int): Number of top perturbed genes to plot.
        layer_key (str): Which layer to extract raw counts from.
        embedding (str): Spatial basis name (default: 'spatial').
        log1p (bool): Whether to apply log1p to expression matrix.
    """
    import scanpy as sc
    from collections import Counter
    import os
    import glob

    # 1. Count top genes from impact_result
    top_genes_all = [gene for gene_list in impact_result["top_changed_genes"] for gene in gene_list]
    gene_counter = Counter(top_genes_all)
    top_genes = [gene for gene, _ in gene_counter.most_common(topn)]
    print(f"[✓] Top {topn} perturbation-response genes:", top_genes)

    # 2. Map symbol to var index
    valid_genes = adata.var_names[adata.var["symbol"].isin(top_genes)]
    plot_data = adata[:, valid_genes].copy()
    plot_data.X = plot_data.layers[layer_key].copy()

    # 3. Optionally apply log1p
    if log1p:
        sc.pp.log1p(plot_data)

    # 4. Plot spatial expression
    sc.pl.embedding(
        plot_data,
        basis=embedding,
        color=valid_genes,
        use_raw=False,
        s=100,
        save="_top_perturbed_genes_spatial.png",
        title=[symbol_dict.get(g, g) for g in valid_genes]
    )

    # 5. Move plot to output dir
    perturb_fig = glob.glob("figures/*top_perturbed_genes_spatial.png")
    if perturb_fig:
        os.makedirs(output_dir, exist_ok=True)
        os.replace(perturb_fig[0], os.path.join(output_dir, "spatial_top_perturbed_genes.png"))


def run_go_enrichment(gene_list, outdir="enrichr_results", organism="Human", verbose=True, top_term=20):
    """
    Run GO enrichment on a list of genes using gseapy and generate barplot.

    Args:
        gene_list (list): List of gene symbols (e.g. ['IFITM3', 'CD44']).
        outdir (str): Output directory for enrichment results and plots.
        organism (str): Organism name, e.g. 'Human' or 'Mouse'.
        verbose (bool): Whether to print top enrichment terms.
        top_term (int): Number of top terms to plot.

    Returns:
        pd.DataFrame: Enrichment result table.
    """
    import os
    import gseapy as gp
    import pandas as pd

    os.makedirs(outdir, exist_ok=True)

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=["GO_Biological_Process_2021", "KEGG_2021_Human", "Reactome_2022", "MSigDB_Hallmark_2020"],
        organism=organism,
        outdir=outdir,
        cutoff=0.05,  # FDR cutoff
        no_plot=True
    )

    result_df = enr.results

    if verbose:
        print(result_df.head(10))

    # Save enrichment barplot
    if not result_df.empty:
        barplot_path = os.path.join(outdir, "enrichr_barplot.png")
        gp.barplot(
            df=result_df,
            title="GO/KEGG Enrichment",
            top_term=top_term,
            figsize=(6, 5),
            ofname=barplot_path
        )
        print(f"[✓] Barplot saved to {barplot_path}")
    else:
        print("[!] No enrichment terms passed the cutoff, no plot generated.")

    return result_df

