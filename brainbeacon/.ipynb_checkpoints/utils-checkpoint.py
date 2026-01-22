import os
import random
import numpy as np
import torch
import scanpy as sc
import anndata as ad
import pandas as pd
import math
import numba
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from sklearn.neighbors import NearestNeighbors
import pyarrow
from tqdm import tqdm
from brainbeacon.config.config import NUM_KNN_NEIGHBOR
from brainbeacon.config.config import specie_dict
from brainbeacon.config.config import technology_dict
from brainbeacon.config.config import MAX_LENGTH
from brainbeacon.config.config import AUX_TOKEN
from config.config_train import config_train
import joblib
import pyarrow.parquet as pq

config_train["single_context_length"] = config_train["context_length"]
config_train["total_context_length"] = config_train["context_length"] * config_train["num_neighbors"]

def set_seed(seed):
    """
    Sets the seed for all libraries used.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


# MEAN_PATH = "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST/tokenier_mean/merfish_gene_nonzero_means.npy"
# OUT_PATH = f'/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST/{SAMPLE_TAG}_{SPECIE}_parquet'


def sf_normalize(X):
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero devision error
    counts += counts == 0.
    # normalize to 10000. counts
    scaling_factor = 10000. / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(
        x: np.array,
        gene_connect_comp: np.array,
        rna_type_id: np.array,
        max_seq_len: int = -1,
        aux_tokens: int = 30
):
    scores_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    scores_connect_comp_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    scores_rna_type_id_final = np.empty((x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1]))
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        gene_connect_comp_sorted = gene_connect_comp[sorted_indices]
        rna_type_id_sorted = rna_type_id[sorted_indices]
        sorted_indices = sorted_indices + aux_tokens  # we reserve some tokens for padding etc (just in case)
        gene_connect_comp_sorted = gene_connect_comp_sorted + 1  # 0 for padding
        rna_type_id_sorted = rna_type_id_sorted + 1
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
            scores_connect_comp = np.zeros(max_seq_len, dtype=np.int32)
            scores_type_id = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
            scores_connect_comp = np.zeros_like(cell, dtype=np.int32)
            scores_type_id = np.zeros_like(cell, dtype=np.int32)
        scores[:len(sorted_indices)] = sorted_indices.astype(np.int32)
        scores_final[i, :] = scores
        scores_connect_comp[:len(gene_connect_comp_sorted)] = gene_connect_comp_sorted.astype(np.int32)
        scores_connect_comp_final[i, :] = scores_connect_comp
        scores_type_id[:len(rna_type_id_sorted)] = rna_type_id_sorted.astype(np.int32)
        scores_rna_type_id_final[i, :] = scores_type_id
    return scores_final, scores_connect_comp_final, scores_rna_type_id_final

def tokenize_data(
        x: np.array,
        gene_connect_comp: np.array,
        gene_type_id: np.array,
        median_counts_per_gene: np.array,
        max_seq_len: int,
        aux_token_len: int
):
    """Tokenize the input gene vector to a vector of 32-bit integers."""

    x = np.nan_to_num(x)  # is NaN values, fill with 0s
    x = sf_normalize(x)
    median_counts_per_gene += median_counts_per_gene == 0
    out = x / median_counts_per_gene.reshape((1, -1))
    scores_final, scores_connect_comp_final, scores_rna_type_id = _sub_tokenize_data(
        out, gene_connect_comp, gene_type_id, max_seq_len, aux_token_len
    )

    return scores_final.astype('i4'), scores_connect_comp_final.astype('i4'), scores_rna_type_id.astype('i4')

def convert_ndarray_to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def tokenization_h5ad(adata_path, gene_dict_path, mean_path, specie=None, assay=None, output_path=None):
    """
    Brainbeacon input tokenization
    Conver H5ad to Joblib
    """
    assert gene_dict_path, "Input `gene_dict_path` cannot be empty."
    gene_dict = sc.read_h5ad(gene_dict_path)
    print(f"path to process: {adata_path}")
    adata = sc.read_h5ad(adata_path)
    # tmp
    # adata = adata[:600]
    # 通常 Ensembl ID 的格式为 'ENSG' 或 'ENSMUSG' 开头的字符串
    is_ensembl = adata.var.index.str.startswith(('ENS'))

    if not is_ensembl.all():
        raise ValueError(
            "adata must contain Ensembl IDs in `var.index`. "
            "Please convert gene names to Ensembl IDs before proceeding."
        )
    print(f"before quality control adata shape: {adata.shape}")
    # if adata.shape[1] < 600:
    #     sc.pp.filter_cells(adata, min_genes=1)
    #     sc.pp.filter_genes(adata, min_cells=20)
    # else:
    #     sc.pp.filter_cells(adata, min_genes=100)
    #     sc.pp.filter_genes(adata, min_cells=20)   
    # print(f"after quality control adata shape: {adata.shape}")
        # 检查和补全 obs 中的列
    if 'slice' in adata.obs.columns:
        slice_column = adata.obs['slice']
    else:
        slice_column = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='slice')

    if 'cell_label' in adata.obs.columns:
        cell_label_column = adata.obs['cell_label']
    else:
        cell_label_column = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='cell_label')

    if 'region' in adata.obs.columns:
        region_column = adata.obs['region']
    else:
        region_column = pd.Series(['unknown'] * adata.shape[0], index=adata.obs.index, name='region')

    # 更新 obs
    adata.obs = pd.concat([slice_column, cell_label_column, region_column], axis=1)
    if isinstance(adata.obsm["spatial"], pd.DataFrame):
        adata.obs["x"] = adata.obsm["spatial"].to_numpy()[:, 0]
        adata.obs["y"] = adata.obsm["spatial"].to_numpy()[:, 1]
    elif isinstance(adata.obsm["spatial"], np.ndarray):
        adata.obs["x"] = adata.obsm["spatial"][:, 0]
        adata.obs["y"] = adata.obsm["spatial"][:, 1]
    else:
        raise TypeError(f"Unsupported type for adata.obsm['spatial']: {type(adata.obsm['spatial'])}")
    adata.uns = {}
    adata.obsm = {}
    adata_output = ad.concat([gene_dict, adata], join='outer', axis=0)
    # adata_output.reset_index(inplace=True)
    adata_output = adata_output[1:]
    adata_output = adata_output[:, gene_dict.var.index]
    adata_output.var = gene_dict.var.copy()
    adata_output = adata_output[(~pd.isna(adata_output.obs["x"])) & (~pd.isna(adata_output.obs["y"]))]
    adata_output.obs = adata_output.obs.reset_index(drop=True)
    adata_output.obs['modality'] = 'spatial'
    adata_output.obs['split'] = 'train'
    adata_output.obs['specie'] = specie
    adata_output.obs['assay'] = assay
    # adata_output.obs["cell_label"] = adata_output.obs["cell_label"].cat.add_categories("Unknown")
    # adata_output.obs["cell_label"] = adata_output.obs["cell_label"].fillna("Unknown")
    adata_output.obs.replace({'specie': specie_dict}, inplace=True)
    adata_output.obs.replace({'assay': technology_dict}, inplace=True)
    # 没有brain_region
    adata_output.obs['brain_region'] = "unknown"
    adata_output.obs['brain_region_main'] = adata_output.obs["cell_label"]

    for i in range(NUM_KNN_NEIGHBOR):
        adata_output.obs[f"knn_neighbor_{i}"] = None
        adata_output.obs[f"knn_neighbor_distance_{i}"] = None

    for idx in adata_output.obs["brain_region_main"].unique():
        # knn_sample = adata_output[adata_output.obs["brain_region_main"] == idx]
        knn_sample_obs = adata_output.obs.loc[adata_output.obs["brain_region_main"] == idx, :] # Beacase Large_MEM env error
        # knn_sample_obs = knn_sample.obs.copy()
        # knn_sample = adata_output[adata_output.obs["brain_region_main"] == idx]
        print(f"brain region {idx} knn x input: {knn_sample_obs.shape}")
        nbrs = NearestNeighbors(n_neighbors=NUM_KNN_NEIGHBOR, algorithm='ball_tree', n_jobs=6).fit(
            knn_sample_obs[["x", "y"]]
        )
        distances, indices = nbrs.kneighbors(knn_sample_obs[["x", "y"]])
        # Because of Large_MEM env error, use the following code to avoid memory error
        # 将索引转换为 NumPy 数组
        indices_array = knn_sample_obs.index.to_numpy()[indices]
        knn_sample_obs = knn_sample_obs.copy()  # 确保 knn_sample_obs 是副本
        for i in range(NUM_KNN_NEIGHBOR):
            knn_sample_obs[f"knn_neighbor_{i}"] = indices[:, i]
            knn_sample_obs[f"knn_neighbor_distance_{i}"] = distances[:, i]
        adata_output.obs[adata_output.obs["brain_region_main"] == idx] = knn_sample_obs
    mean_matrix = np.load(mean_path)
    obs_adata_output = adata_output.obs
    print('n_obs: ', obs_adata_output.shape[0])
    N_BATCHES = math.ceil(obs_adata_output.shape[0] / 10000)
    # N_BATCHES = 1
    print('N_BATCHES: ', N_BATCHES)
    batch_indices = np.array_split(obs_adata_output.index, N_BATCHES)
    chunk_len = len(batch_indices[0])
    print('chunk_len: ', chunk_len)
    obs_adata_output = obs_adata_output.reset_index().rename(columns={'index': 'idx'})
    obs_adata_output['idx'] = obs_adata_output['idx'].astype('i8')

    for batch in tqdm(range(N_BATCHES)):
        obs_tokens = obs_adata_output.iloc[batch * chunk_len:chunk_len * (batch + 1)].copy()
        tokenized, tokenized_connect_comp, tokenized_rna_type = tokenize_data(
            adata_output.X[batch * chunk_len:chunk_len * (batch + 1)],
            adata_output.var["homo_connect_id"].values,
            adata_output.var["Gene_type_id"].values,
            mean_matrix,
            MAX_LENGTH,
            AUX_TOKEN
        )
        tokenized_nb_list, tokenized_nb_connect_comp_list, tokenized_nb_rna_type_list = [], [], []
        for idx in range(NUM_KNN_NEIGHBOR):
            tokenized_nb, tokenized_nb_connect_comp, tokenized_nb_rna_type = tokenize_data(
                adata_output[obs_tokens[f"knn_neighbor_{idx}"].values].X,
                adata_output.var["homo_connect_id"].values,
                adata_output.var["Gene_type_id"].values,
                mean_matrix,
                MAX_LENGTH,
                AUX_TOKEN
            )
            tokenized_nb_list.append(tokenized_nb)
            tokenized_nb_connect_comp_list.append(tokenized_nb_connect_comp)
            tokenized_nb_rna_type_list.append(tokenized_nb_rna_type)

        obs_tokens = obs_tokens[['brain_region', 'brain_region_main', 'x', 'y', 'assay', 'specie', 'idx']]
        # concatenate dataframes
        obs_tokens['X'] = [tokenized[i, :] for i in range(tokenized.shape[0])]
        obs_tokens['X_connect_comp'] = [
            tokenized_connect_comp[i, :] for i in range(tokenized_connect_comp.shape[0])]
        obs_tokens['X_rna_type'] = [tokenized_rna_type[i, :] for i in range(tokenized_rna_type.shape[0])]
        for idx in range(NUM_KNN_NEIGHBOR):
            tk_nb, tk_nb_connect_comp, tk_nb_rna_type = \
                tokenized_nb_list[idx], tokenized_nb_connect_comp_list[idx], tokenized_nb_rna_type_list[idx]
            obs_tokens[f'X_neighbor_{idx}'] = [
                tk_nb[i, :] for i in range(tk_nb.shape[0])
            ]
            obs_tokens[f'X_neighbor_{idx}_connect_comp'] = [
                tk_nb_connect_comp[i, :] for i in range(tk_nb_connect_comp.shape[0])
            ]
            obs_tokens[f'X_neighbor_{idx}_rna_type'] = [
                tk_nb_rna_type[i, :] for i in range(tk_nb_rna_type.shape[0])
            ]
        # obs_tokens = obs_tokens.sample(frac=1)
        # 对所有列应用转换
        obs_tokens = obs_tokens.applymap(convert_ndarray_to_list)

        # 重新尝试创建 PyArrow Table
        total_table = pyarrow.Table.from_pandas(obs_tokens)
        assert output_path, "Output path must be provided."
        pq.write_table(
            total_table, os.path.join(output_path, f"tokens-{batch}.parquet"),
            row_group_size=1024
        )
    return output_path

def split_iter(a: list, n: int):
    """Pack a dataset (array of samples) into an array of batches"""
    q = len(a) // n - 1
    assert q > 0
    k, m = divmod(len(a), q)
    for i in range(q):
        yield a[i*n:(i+1)*n]

def batches(data, batch_size=36):
    iterator = split_iter(data, batch_size)
    return iterator

def do_masking(adata, p, n_tokens):
    padding_token = 1
    cls_token = 3
    indices = torch.as_tensor(adata.obsm["X"], dtype=torch.long)
    # 0 is originally the padding token, we change it to 1
    indices = torch.where(indices == 0, torch.as_tensor(padding_token, dtype=torch.long), indices)
    adata.obsm["X"] = indices.numpy()

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p)  # mask indices with probability p
    # mask sure the aux token will not be masked
    mask = torch.where(indices > config_train['n_aux'], mask, torch.ones_like(mask))

    masked_indices = indices * mask  # masked_indices
    # we just mask non-padding indices
    masked_indices = torch.where(indices != padding_token, masked_indices, indices)
    # in the model_raw we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation
    mask = torch.where(indices == padding_token, torch.as_tensor(padding_token, dtype=torch.long), mask)


    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices,
                                 indices)  # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.as_tensor(padding_token, dtype=torch.long),
                       mask)  # we change the mask so that it doesn't mask any CLS token

    # 80% of masked indices are masked
    # 10% of masked indices are a random token
    # 10% of masked indices are the real token

    random_tokens = torch.randint(10, n_tokens, size=masked_indices.shape, device=masked_indices.device)
    random_tokens = random_tokens * torch.bernoulli(torch.ones_like(random_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, random_tokens,
                                 masked_indices)  # put random tokens just in the previously masked tokens

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(torch.ones_like(same_tokens) * 0.1).type(torch.int64)

    masked_indices = torch.where(masked_indices == 0, same_tokens,
                                 masked_indices)  # put same tokens just in the previously masked tokens

    adata.obsm['masked_indices'] = masked_indices.numpy()
    adata.obsm['mask'] = mask.numpy()

    attention_mask = (masked_indices == padding_token)
    adata.obsm['attention_mask'] = attention_mask.type(torch.bool).numpy()

    return adata

def process_parquet(input_file, output_path):
    """
    Reads a .parquet file (instead of .h5ad), converts it to AnnData with
    the same 'X', 'X_connect_comp', 'X_rna_type', etc. keys, and then 
    applies the same logic (masking, neighbor enhancement, joblib splits, etc.).
    """
    print(f"Begin processing: {input_file}")
    # 1) Read parquet -> DataFrame
    df = pq.read_table(input_file).to_pandas()
    print(f"DataFrame shape from parquet = {df.shape}")
     # obs columns you might have, adapt as needed
    obs_cols = []
    for c in ["brain_region", "brain_region_main", "x", "y", "assay", "specie", "idx"]:
        if c in df.columns:
            obs_cols.append(c)

    # Create .X from the "X" column
    if "X" not in df.columns:
        raise ValueError("No 'X' column in parquet; cannot proceed.")

    X_stack = np.vstack(df["X"].values)  # shape => (n_cells, context_length)
    adata = ad.AnnData(
        X=X_stack,
        obs=df[obs_cols].copy() if obs_cols else None  # optional
    )

    # Set .obsm with the same keys used in the original code
    data_connect_comp_key = 'X_connect_comp'
    data_rna_type_key = 'X_rna_type'
    data_cell_ids_key = 'X_cell_ids'

    if data_connect_comp_key not in df.columns:
        raise ValueError(f"No '{data_connect_comp_key}' in parquet.")
    if data_rna_type_key not in df.columns:
        raise ValueError(f"No '{data_rna_type_key}' in parquet.")

    adata.obsm[data_connect_comp_key] = np.vstack(df[data_connect_comp_key].values)
    adata.obsm[data_rna_type_key] = np.vstack(df[data_rna_type_key].values)

    # Also handle neighbor columns if they exist
    # In the original code, we needed up to config_train["num_neighbors"] - 1 neighbors
    # We will store them here. Then we can do the same concatenation logic.
    for idx in range(config_train["num_neighbors"] - 1):
        neighbor_key = f'X_neighbor_{idx}'
        neighbor_connect_key = f'X_neighbor_{idx}_connect_comp'
        neighbor_rna_type_key = f'X_neighbor_{idx}_rna_type'
        # We only *store* them in adata.obsm for now, 
        # the "concatenation" logic is done below (like the h5ad version).
        if neighbor_key in df.columns and neighbor_connect_key in df.columns:
            adata.obsm[neighbor_key] = np.vstack(df[neighbor_key].values)
            adata.obsm[neighbor_connect_key] = np.vstack(df[neighbor_connect_key].values)
            adata.obsm[neighbor_rna_type_key] = np.vstack(df[neighbor_rna_type_key].values)

    data_key = 'X'
    X = adata.X.copy()
    X = torch.as_tensor(X, dtype=torch.float32)  # no need toarray() if it's dense
    X_connect_comp = torch.as_tensor(adata.obsm[data_connect_comp_key], dtype=torch.float32)
    X_rna_type = torch.as_tensor(adata.obsm[data_rna_type_key], dtype=torch.float32)

    # truncate single context length
    X = X[:, :config_train["single_context_length"]]
    X_connect_comp = X_connect_comp[:, :config_train["single_context_length"]]
    X_rna_type = X_rna_type[:, :config_train["single_context_length"]]

    # If config_train["assay"] is True and we have 'assay' in obs => add assay token
    if config_train["assay"] and 'assay' in adata.obs.columns:
        assay = torch.as_tensor(adata.obs['assay'], dtype=torch.float32).view(-1, 1)
        X = torch.cat((assay, X), dim=1)
        X_connect_comp = torch.cat((assay, X_connect_comp), dim=1)

    # If config_train["specie"] is True and we have 'specie' => add specie token
    if config_train["specie"] and 'specie' in adata.obs.columns:
        specie = torch.as_tensor(adata.obs['specie'], dtype=torch.float32).view(-1, 1)
        X = torch.cat((specie, X), dim=1)
        X_connect_comp = torch.cat((specie, X_connect_comp), dim=1)

    # We'll store cell_ids in the same shape
    cell_ids = torch.zeros_like(X, dtype=torch.int32)

    # Add neighbor tokens
    if config_train["neighbor_enhance"]:
        for idx in range(config_train["num_neighbors"] - 1):
            neighbor_key = f'X_neighbor_{idx}'
            neighbor_connect_key = f'X_neighbor_{idx}_connect_comp'
            neighbor_rna_type_key = f'X_neighbor_{idx}_rna_type'
            if neighbor_key in adata.obsm and neighbor_connect_key in adata.obsm:
                neighbor = torch.as_tensor(adata.obsm[neighbor_key], dtype=torch.float32)
                neighbor_connect_comp = torch.as_tensor(adata.obsm[neighbor_connect_key], dtype=torch.float32)
                neighbor_rna_type = torch.as_tensor(adata.obsm[neighbor_rna_type_key], dtype=torch.float32)

                # Truncate single_context_length
                neighbor = neighbor[:, :config_train["single_context_length"]]
                neighbor_connect_comp = neighbor_connect_comp[:, :config_train["single_context_length"]]
                neighbor_rna_type = neighbor_rna_type[:, :config_train["single_context_length"]]

                neighbor_cell_ids = torch.ones_like(neighbor, dtype=torch.int32) * (idx + 1)

                X = torch.cat((X, neighbor), dim=1)
                X_connect_comp = torch.cat((X_connect_comp, neighbor_connect_comp), dim=1)
                X_rna_type = torch.cat((X_rna_type, neighbor_rna_type), dim=1)
                cell_ids = torch.cat((cell_ids, neighbor_cell_ids), dim=1)
            else:
                print(f"neighbor_key {neighbor_key} or {neighbor_connect_key} not in adata.obsm")

    # Truncate total context length
    X = X[:, :config_train["total_context_length"]]
    X_connect_comp = X_connect_comp[:, :config_train["total_context_length"]]
    X_rna_type = X_rna_type[:, :config_train["total_context_length"]]
    cell_ids = cell_ids[:, :config_train["total_context_length"]]

    # Store back into adata.obsm
    adata.obsm[data_key] = X.numpy()
    adata.obsm[data_connect_comp_key] = X_connect_comp.numpy()
    adata.obsm[data_rna_type_key] = X_rna_type.numpy()
    adata.obsm[data_cell_ids_key] = cell_ids.numpy()

    # Masking
    adata = do_masking(adata, config_train["masking_p"], config_train["n_tokens"])

    print(f"masked_indices shape: {adata.obsm['masked_indices'].shape}")
    print(f"mask shape: {adata.obsm['mask'].shape}")
    print(f"X shape: {adata.obsm['X'].shape}")
    print(f"attention_mask shape: {adata.obsm['attention_mask'].shape}")

    # Save as new .h5ad (same logic as original code)

 
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    prefix = os.path.basename(input_file).replace(".parquet", "")
    output_file = f"{prefix}_pre.h5ad"
    print(f"Joblib output directory: {os.path.join(output_path, prefix)}")
    # adata.write_h5ad(os.path.join(output_path, output_file))
    # print(f"Saved processed .h5ad to: {os.path.join(output_path, output_file)}")

    # Also save joblib
    if not os.path.exists(os.path.join(output_path, prefix)):
        os.makedirs(os.path.join(output_path, prefix))

    masked_indices_batches = batches(adata.obsm["masked_indices"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(masked_indices_batches), os.path.join(
        output_path, prefix, f'masked_indices_{config_train["batch_size"]}.job'))
    mask_batches = batches(adata.obsm["mask"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(mask_batches), os.path.join(
        output_path, prefix, f'mask_{config_train["batch_size"]}.job'))
    real_indices_batches = batches(adata.obsm["X"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(real_indices_batches), os.path.join(
        output_path, prefix, f'real_indices_{config_train["batch_size"]}.job'))
    attention_mask_batches = batches(adata.obsm["attention_mask"], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(attention_mask_batches), os.path.join(
        output_path, prefix, f'attention_mask_{config_train["batch_size"]}.job'))
    connect_comp_batches = batches(adata.obsm[data_connect_comp_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(connect_comp_batches), os.path.join(
        output_path, prefix, f'connect_comp_{config_train["batch_size"]}.job'))
    rna_type_batches = batches(adata.obsm[data_rna_type_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(rna_type_batches), os.path.join(
        output_path, prefix, f'rna_type_{config_train["batch_size"]}.job'))
    cell_ids_batches = batches(adata.obsm[data_cell_ids_key], config_train["batch_size"])
    joblib.numpy_pickle.dump(list(cell_ids_batches), os.path.join(
        output_path, prefix, f'cell_ids_{config_train["batch_size"]}.job'))
    save_path = os.path.join(output_path, prefix, f'cell_ids_{config_train["batch_size"]}.job')
    print(f"Finished splitting and saving joblib files for: {input_file}")
    print("Saving to:", save_path)