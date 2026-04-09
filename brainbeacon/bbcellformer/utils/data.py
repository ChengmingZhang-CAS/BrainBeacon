import math
import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy
import torch
from torch.utils.data import Dataset
import json
import os
import warnings
from typing import List
import torch.nn.functional as F
from scipy.sparse import csr_matrix

SPATIAL_PLATFORM_LIST = ['merfish', 'xenium', 'starmap', 'slideseqv2', 'stereo']


def sparse_scipy_to_tensor(x: scipy.sparse.csr_matrix):
    return torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float().coalesce()


class TranscriptomicDataset(Dataset):
    def __init__(self, adata: ad.AnnData,
                 split_field: str = None,
                 covariate_fields: List[str] = None,
                 label_fields: List[str] = None,
                 batch_gene_list: dict = None,
                 covariate_encoders: dict = None,
                 label_encoders: dict = None,
                 order_required: bool = False,
                 partition_mode: str = "batch",  # "patch"
                 target_cells_per_patch: int = 2000,
                 halo_expand_ratio: float = 0.1):
        self.seq_list = []
        self.coord_list = []
        self.order_list = []
        self.batch_gene_list = batch_gene_list
        self.covariate_fields = covariate_fields
        self.label_fields = label_fields
        self.order_required = order_required
        self.gene_list = adata.var.index.tolist()
        self.partition_mode = partition_mode
        self.target_cells_per_patch = target_cells_per_patch
        self.halo_expand_ratio = halo_expand_ratio
        assert self.partition_mode in ["batch", "patch"], f"Unsupported partition_mode: {self.partition_mode}"

        if self.batch_gene_list:
            assert 'batch' in adata.obs, 'Batch specific gene list is set but batch labels are not found in AnnData.obs.'
            self.batch_gene_mask = {}
            g2id = dict(zip(self.gene_list, list(range(len(self.gene_list)))))
            for batch in batch_gene_list:
                idx = torch.LongTensor([g2id[g] for g in batch_gene_list[batch] if g in g2id])
                self.batch_gene_mask[batch] = torch.zeros(len(g2id)).bool()
                self.batch_gene_mask[batch][idx] = True
        else:
            self.batch_gene_mask = None

        if split_field:
            assert split_field in adata.obs, f'Split field `{split_field}` is specified but not found in AnnData.obs.'
            self.split_list = []
        else:
            self.split_list = None

        if not label_fields:
            label_fields = []
        if not covariate_fields:
            covariate_fields = []
        self.label_list = dict(zip(label_fields, [[] for _ in range(len(label_fields))]))
        self.covariate_list = dict(zip(covariate_fields, [[] for _ in range(len(covariate_fields))]))
        if not covariate_encoders:  # Fit LabelEncoder on covariates
            self.covariate_encoders = dict(
                zip(covariate_fields, [LabelEncoder().fit(adata.obs[c]) for c in covariate_fields]))
        else:  # Load pre-fit LabelEncoder
            self.covariate_encoders = covariate_encoders
        if not label_encoders:  # Fit LabelEncoder on labels
            self.label_encoders = dict(zip(label_fields, [LabelEncoder().fit(adata.obs[l]) for l in label_fields]))
        else:  # Load pre-fit LabelEncoder
            self.label_encoders = label_encoders
        covariates = dict(
            zip(covariate_fields,
                [self.covariate_encoders[c].transform(adata.obs[c]) for c in covariate_fields]))
        labels = dict(
            zip(label_fields,
                [self.label_encoders[l].transform(adata.obs[l]) for l in label_fields]))

        if 'batch' not in adata.obs:
            warnings.warn(
                'Batch labels not found in AnnData.obs. All cells are considered from the same sample by default.')
            batch_labels = np.zeros(adata.shape[0], dtype=np.int8)
            self.batch_list = [None]
        else:
            batch_le = LabelEncoder().fit(adata.obs['batch'])
            batch_labels = batch_le.transform(adata.obs['batch'])
            self.batch_list = []

        # use bb_emb if available
        if 'bb_emb' in adata.obsm:
            self.bb_emb_list = []
            for batch in range(batch_labels.max() + 1):
                bb_tensor = torch.tensor(adata.obsm['bb_emb'][batch_labels == batch], dtype=torch.float32)
                self.bb_emb_list.append(bb_tensor)
        else:
            self.bb_emb_list = None

        for batch in range(batch_labels.max() + 1):
            x = csr_matrix(adata[batch_labels == batch].X.astype(float))
            self.seq_list.append(sparse_scipy_to_tensor(x))

            for c in covariate_fields:
                self.covariate_list[c].append(torch.from_numpy(covariates[c][batch_labels == batch]))

            for l in label_fields:
                self.label_list[l].append(torch.from_numpy(labels[l][batch_labels == batch]))

            if 'batch' in adata.obs:
                self.batch_list.append(batch_le.classes_[batch])

            # if 'platform' in adata.obs and adata.obs['platform'][batch_labels == batch][0] in SPATIAL_PLATFORM_LIST:
            #     coord_x = torch.tensor(adata.obs['x_FOV_px'][batch_labels == batch])[:, None]
            #     coord_y = torch.tensor(adata.obs['y_FOV_px'][batch_labels == batch])[:, None]
            #     self.coord_list.append(torch.cat([coord_x, coord_y], 1))
            if 'platform' in adata.obs and adata.obs.loc[batch_labels == batch, 'platform'].iloc[0] in SPATIAL_PLATFORM_LIST:
                coord_x = torch.tensor(
                    adata.obs.loc[batch_labels == batch, 'x_FOV_px'].to_numpy(),
                    dtype=torch.float32,
                )[:, None]
                coord_y = torch.tensor(
                    adata.obs.loc[batch_labels == batch, 'y_FOV_px'].to_numpy(),
                    dtype=torch.float32,
                )[:, None]
                self.coord_list.append(torch.cat([coord_x, coord_y], 1))
            else:
                self.coord_list.append(torch.zeros(x.shape[0], 2) - 1)

            if split_field:
                self.split_list.append(adata.obs[split_field][batch_labels == batch])

            if order_required:
                self.order_list.append(torch.from_numpy((batch_labels == batch).nonzero()[0]))

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        return_dict = {'coord': self.coord_list[idx],
                       'x_seq': self.seq_list[idx]}
        for c in self.covariate_list:
            return_dict[c] = self.covariate_list[c][idx]
        for l in self.label_list:
            return_dict[l] = self.label_list[l][idx]

        if self.split_list:
            return_dict['split'] = self.split_list[idx]
        else:
            return_dict['split'] = None

        if self.batch_gene_mask:
            return_dict['gene_mask'] = self.batch_gene_mask[self.batch_list[idx]]

        return_dict['gene_list'] = self.gene_list

        # new add
        if self.bb_emb_list is not None:
            return_dict['bb_emb'] = self.bb_emb_list[idx]

        if self.order_required:
            return_dict['order_list'] = self.order_list[idx]
        return return_dict


class SCDataset(Dataset):
    def __init__(self, tensor_dir='/', gene_set=None):

        with open(f'{tensor_dir}/metadata.json') as f:
            self.batch_metadata = json.load(f)
        with open(f'{tensor_dir}/dataset_metadata.json') as f:
            self.dataset_metadata = json.load(f)
        if 'gene_list' in self.batch_metadata:
            del self.batch_metadata['gene_list']
        self.tensor_dir = tensor_dir
        self.isddp = False
        self.bid2did = dict(zip(self.batch_metadata['batch_id'], self.batch_metadata['dataset_id']))
        self.did2gene = dict(zip(self.dataset_metadata['id'], self.dataset_metadata['gene_list']))
        if gene_set:
            gene_mask = []
            for i in self.dataset_metadata['gene_list']:
                i = set(i)
                gene_mask.append(torch.tensor([j in i for j in gene_set]).bool())
            self.did2mask = dict(zip(self.dataset_metadata['id'], gene_mask))
        else:
            self.did2mask = None

    def __len__(self):
        return len(self.batch_metadata['batch_id'])

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, str(self.batch_metadata['batch_id'][idx]) + '.pt')
        seq = torch.load(tensor_path).coalesce()
        if self.batch_metadata['platform'][idx] in SPATIAL_PLATFORM_LIST:
            coord = torch.load(os.path.join(self.tensor_dir, str(self.batch_metadata['batch_id'][idx]) + '.coord.pt'))
        else:
            coord = torch.zeros([seq.shape[0], 2]).float() - 1
        batch_id = torch.zeros(seq.shape[0]).long() + int(self.batch_metadata['batch_id'][idx])
        dataset_id = torch.zeros(seq.shape[0]).long() + int(self.batch_metadata['dataset_id'][idx])
        gene_mask = self.get_gene_mask(self.batch_metadata['dataset_id'][idx]) if self.did2mask else torch.ones(
            [seq.shape[1]]).bool()
        return seq, coord, batch_id, dataset_id, gene_mask

    def get_gene_list(self, dataset_id):
        return self.did2gene[dataset_id]

    def get_gene_mask(self, dataset_id):
        assert self.did2mask, 'gene_set was not passed when created dataset.'
        return self.did2mask[dataset_id]

    def get_partition(self, rank):
        assert self.isddp, 'Dataset is not a ddp dataset. Please call ".to_ddp()" before querying partition.'
        return self._partition(self.partitions[rank])

    def _partition(self, idx):
        assert self.isddp, 'Dataset is not a ddp dataset.'
        return SCPartitionDataset(self.batch_metadata, self.tensor_dir, idx)

    def get_valid(self):
        assert self.isddp, 'Dataset is not a ddp dataset. Please call ".to_ddp()" before querying validation set.'
        assert len(self.val_idx) > 0, 'No available validation set.'
        return self._partition(self.val_idx)

    def to_ddp(self, n_partitions, max_batch_size=2000, val_num=0, val_idx=None):
        assert not self.isddp, 'Dataset is already ddp dataset.'

        if val_num > 0:
            if not val_idx:
                ids = np.random.permutation(len(self.batch_metadata['batch_id']))
                self.val_idx = ids[:val_num]
                self.train_idx = ids[val_num:]
            else:
                self.train_idx = np.array(
                    [i for i in range(len(self.batch_metadata['batch_id'])) if i not in set(val_idx)])
                self.val_idx = np.array(val_idx)
            self.partitions = balanced_partition(np.array(self.batch_metadata['batch_size'])[self.train_idx],
                                                 n_partitions,
                                                 max_batch_size)
            new_partitions = [[] for _ in range(n_partitions)]
            for i, p in enumerate(self.partitions):
                for j in p:
                    new_partitions[i].append(self.train_idx[j])
            self.partitions = new_partitions

        else:
            self.train_idx = np.arange(len(self.batch_metadata['batch_id']))
            self.val_idx = np.array([])
            self.partitions = balanced_partition(self.batch_metadata['batch_size'], n_partitions, max_batch_size)
        self.isddp = True


class SCPartitionDataset(Dataset):
    def __init__(self, batch_metadata, tensor_dir, idx, gene_set=None):
        self.batch_metadata = {}
        for k in batch_metadata:
            self.batch_metadata[k] = [batch_metadata[k][i] for i in idx]
        self.tensor_dir = tensor_dir
        with open(f'{tensor_dir}/dataset_metadata.json') as f:
            self.dataset_metadata = json.load(f)

        self.bid2did = dict(zip(self.batch_metadata['batch_id'], self.batch_metadata['dataset_id']))
        self.did2gene = dict(zip(self.dataset_metadata['id'], self.dataset_metadata['gene_list']))

        if gene_set:
            gene_mask = []
            for i in self.dataset_metadata['gene_list']:
                i = set(i)
                gene_mask.append(torch.tensor([j in i for j in gene_set]).bool())
            self.did2mask = dict(zip(self.dataset_metadata['id'], gene_mask))
        else:
            self.did2mask = None

    def __len__(self):
        return len(self.batch_metadata['batch_id'])  # //10

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, str(self.batch_metadata['batch_id'][idx]) + '.pt')
        seq = torch.load(tensor_path).coalesce()
        if self.batch_metadata['platform'][idx] in SPATIAL_PLATFORM_LIST:
            coord = torch.load(os.path.join(self.tensor_dir, str(self.batch_metadata['batch_id'][idx]) + '.coord.pt'))
        else:
            coord = torch.zeros([seq.shape[0], 2]).float() - 1
        if seq.shape[0] > 2000:
            randid = torch.randperm(seq.shape[0])
            coord = coord[randid[:2000]]
            seq = seq.index_select(0, randid[:2000]).coalesce()
        batch_id = torch.zeros([seq.shape[0]]).long() + int(self.batch_metadata['batch_id'][idx])
        dataset_id = torch.zeros([seq.shape[0]]).long() + int(self.batch_metadata['dataset_id'][idx])
        gene_mask = self.get_gene_mask(self.batch_metadata['dataset_id'][idx]) if self.did2mask else torch.ones(
            [seq.shape[1]]).bool()
        seq = [seq.indices(), seq.values(), torch.tensor(seq.shape)]
        return seq, coord, batch_id, dataset_id, gene_mask

    def get_gene_list(self, dataset_id):
        return self.did2gene[dataset_id]

    def get_gene_mask(self, dataset_id):
        assert self.did2mask, 'gene_set was not passed when created dataset.'
        return self.did2mask[dataset_id]


class XDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._num = self[list(self.keys())[0]].shape[0]

    # No longer required
    # def check(self):
    #     for k, v in self.items():
    #         assert isinstance(v, torch.Tensor), f'{k} is not a torch.Tensor'
    #         assert v.shape[0] == self._num, f'{k} contains {v.shape[0]} samples. Expected: f{self._num}'

    def size(self):
        warnings.warn("Deprecated function: Xdict.size().", DeprecationWarning)
        return self._num

    # Not usable for sparse data
    # def drop(self, ratio):
    #     drop_num = int(self._num * ratio)
    #     keep_idx = np.random.permutation(self._num)[drop_num:]
    #     for k, v in self.items():
    #         self[k] = v[keep_idx]
    #     return self


def clean_batches(data):
    # Remove batch with less than 1000 cells
    sc.pp.filter_cells(data, min_counts=5)
    remove_list = []
    for b in data.obs['batch'].value_counts().reset_index().iterrows():
        if b[1]['batch'] < 500:
            remove_list.append(b[1]['index'])
    data = data[~data.obs['batch'].isin(set(remove_list))]
    return data


def balanced_partition(data, n_partitions, max_batch_size=2000):
    # Sort batches
    if torch.is_tensor(data[0]):
        batch_sizes = [(i, len(batch)) for i, batch in enumerate(data)]
    else:
        batch_sizes = [(i, batch) for i, batch in enumerate(data)]
    batch_sizes.sort(key=lambda x: x[1], reverse=True)

    # inialize partitions
    partitions = [[] for _ in range(n_partitions)]

    # Fill partitions
    j = 0
    for (i, _) in batch_sizes:
        partitions[j].append(i)
        j = (j + 1) % n_partitions
    return partitions


def stratified_sample_genes_by_sparsity(data, boundaries=None, seed=10):
    df = data.to_df()
    zero_rates = 1 - df.astype(bool).sum(axis=0) / df.shape[0]
    if boundaries is None:
        # boundaries = [0, zero_rates.mean() - zero_rates.std(), zero_rates.mean(),
        #               min(zero_rates.mean() + zero_rates.std(), 1)]
        boundaries = [0, 0.75, 0.9, 0.95, 1]
    gene_group = pd.cut(zero_rates, boundaries, labels=False)
    # gene_df = pd.DataFrame({'zero_rates': zero_rates, 'gene_group': gene_group})
    zero_rates = zero_rates.groupby(gene_group, group_keys=False)
    samples = zero_rates.apply(lambda x: x.sample(min(len(x), 25), random_state=seed))
    return list(samples.index)


def data_setup(adata, return_sparse=True, device='cpu'):
    warnings.warn("`Data_setup` function is deprecated. Use `CellPLM.pipeline` instead.", DeprecationWarning)
    # Data Setup
    order = torch.arange(adata.shape[0], device=device)
    lb = LabelEncoder().fit(adata.obs['batch'])
    batch_labels = lb.transform(adata.obs['batch'])
    # print(lb.classes_)
    seq_list = [[], [], [], []] if return_sparse else []
    batch_list = []
    order_list = []
    dataset_list = []
    coord_list = []
    if adata.obs['cell_type'].dtype != int:
        labels = LabelEncoder().fit_transform(adata.obs['cell_type'])
    else:
        labels = adata.obs['cell_type'].values
        print(labels.mean())
    label_list = []
    dataset_label = LabelEncoder().fit_transform(adata.obs['Dataset'])
    for batch in range(batch_labels.max() + 1):
        if return_sparse:
            x = (adata.X[batch_labels == batch]).astype(float)
            x = list(map(torch.from_numpy, [x.indptr, x.indices, x.data])) + [torch.tensor(x.shape)]
            for i in range(4):
                seq_list[i].append(x[i].to(device))
        else:
            x = torch.from_numpy(adata.X[batch_labels == batch].todense()).float()
            seq_list.append(x.to(device))
        # x = torch.sparse_csr_tensor(x.indptr, x.indices, x.data, (x.shape[0], x.shape[1])).to_sparse().float()
        # seq_list.append(x)
        order_list.append(order[batch_labels == batch])
        dataset_list.append(torch.from_numpy(dataset_label[batch_labels == batch]).long().to(device))
        batch_list.append(torch.from_numpy(batch_labels[batch_labels == batch]).to(device))
        if adata.obs['platform'][batch_labels == batch][0] in SPATIAL_PLATFORM_LIST:
            coord_list.append(
                torch.from_numpy(adata.obs[['x_FOV_px', 'y_FOV_px']][batch_labels == batch].values).to(device))
        else:
            coord_list.append(torch.zeros(order_list[-1].shape[0], 2).to(device) - 1)
        label_list.append(torch.from_numpy(labels[batch_labels == batch].astype(int)).to(device))
    del order
    return seq_list, batch_list, batch_labels, order_list, dataset_list, coord_list, label_list


def make_spatial_patches(
    coord,
    target_cells_per_patch=2000,
    halo_ratio=0.2,
    min_center_ratio=0.2,
    min_center_cells=32,
):
    """
    Build spatial patches for one slice using:
    1) initial grid-based center partition
    2) merge tiny center patches at the center level
    3) recompute halo/full patch after merging

    Parameters
    ----------
    coord : torch.Tensor
        Cell coordinates with shape [n_cells, 2].
    target_cells_per_patch : int, default=2000
        Target number of center cells per patch.
    halo_ratio : float, default=0.2
        Relative padding ratio added to each side of the merged center bbox.
    min_center_ratio : float, default=0.2
        Minimum center size ratio relative to target_cells_per_patch.
        Example: if target_cells_per_patch=1000 and min_center_ratio=0.2,
        then patches with center size < 200 will be merged.
    min_center_cells : int, default=32
        Absolute lower bound for small center patches.

    Returns
    -------
    patch_list : list[dict]
        Each patch dict contains:
            - center_idx: global indices of center cells
            - full_idx: global indices of full patch cells (center + halo)
            - center_mask: boolean mask of center cells in full_idx
    grid_info : dict
        Patch layout information for debugging.
    """
    assert isinstance(coord, torch.Tensor)
    assert coord.ndim == 2 and coord.shape[1] == 2
    assert coord.shape[0] > 0
    assert target_cells_per_patch >= 1
    assert halo_ratio >= 0
    assert min_center_ratio >= 0
    assert min_center_cells >= 1

    coord = coord.detach().cpu()
    x = coord[:, 0]
    y = coord[:, 1]
    n_cells = coord.shape[0]

    # ------------------------------------------------------------------
    # 1) Determine grid size
    # ------------------------------------------------------------------
    n_patches_target = max(1, math.ceil(n_cells / target_cells_per_patch))

    x_min = x.min().item()
    x_max = x.max().item()
    y_min = y.min().item()
    y_max = y.max().item()

    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)
    aspect = x_range / y_range

    n_x = max(1, math.ceil(math.sqrt(n_patches_target * aspect)))
    n_y = max(1, math.ceil(n_patches_target / n_x))

    patch_w = x_range / n_x
    patch_h = y_range / n_y

    # ------------------------------------------------------------------
    # 2) Assign cells to initial center grids
    # ------------------------------------------------------------------
    grid_x = torch.floor((x - x_min) / patch_w).long()
    grid_y = torch.floor((y - y_min) / patch_h).long()

    grid_x = torch.clamp(grid_x, min=0, max=n_x - 1)
    grid_y = torch.clamp(grid_y, min=0, max=n_y - 1)

    linear_id = grid_x * n_y + grid_y

    sort_idx = torch.argsort(linear_id)
    linear_id_sorted = linear_id[sort_idx]

    unique_ids, counts = torch.unique_consecutive(linear_id_sorted, return_counts=True)
    starts = torch.cat([
        torch.tensor([0], dtype=torch.long),
        counts.cumsum(0)[:-1]
    ])

    # Store initial center groups by grid id
    groups = {}
    for uid, start, count in zip(unique_ids.tolist(), starts.tolist(), counts.tolist()):
        i = uid // n_y
        j = uid % n_y
        center_idx = sort_idx[start:start + count]

        groups[(i, j)] = {
            "grid_i": i,
            "grid_j": j,
            "center_idx": center_idx,
        }

    if len(groups) == 0:
        raise RuntimeError("No spatial patch was constructed.")

    # ------------------------------------------------------------------
    # 3) Merge tiny center patches at center level
    # ------------------------------------------------------------------
    min_required = max(min_center_cells, int(target_cells_per_patch * min_center_ratio))

    active_keys = set(groups.keys())
    sorted_keys = sorted(active_keys)

    def _group_size(key):
        return groups[key]["center_idx"].numel()

    def _group_centroid(key):
        idx = groups[key]["center_idx"]
        c = coord[idx].float().mean(dim=0)
        return c

    def _neighbor_keys(key):
        """Return nearby grid neighbors: 4-neighborhood first, then diagonals."""
        i, j = key
        cand_4 = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
        cand_diag = [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

        neigh_4 = [k for k in cand_4 if k in active_keys]
        neigh_diag = [k for k in cand_diag if k in active_keys]
        return neigh_4, neigh_diag

    # We process from small to large, but do not merge into already removed keys.
    changed = True
    while changed:
        changed = False

        current_keys = sorted(active_keys, key=lambda k: _group_size(k))
        for key in current_keys:
            if key not in active_keys:
                continue

            size = _group_size(key)
            if size >= min_required:
                continue

            neigh_4, neigh_diag = _neighbor_keys(key)
            candidates = neigh_4 if len(neigh_4) > 0 else neigh_diag

            # If there is no adjacent patch, keep it as is.
            if len(candidates) == 0:
                continue

            # Prefer non-small neighbors first
            large_candidates = [k for k in candidates if _group_size(k) >= min_required]
            use_candidates = large_candidates if len(large_candidates) > 0 else candidates

            # Among candidates, choose the one with the largest center size.
            # If tied, choose the nearest centroid.
            src_centroid = _group_centroid(key)

            def _score(k):
                size_score = _group_size(k)
                dist_score = torch.sum((_group_centroid(k) - src_centroid) ** 2).item()
                return (-size_score, dist_score)

            target_key = sorted(use_candidates, key=_score)[0]

            # Merge center_idx only; halo will be recomputed later.
            merged_idx = torch.cat(
                [groups[target_key]["center_idx"], groups[key]["center_idx"]],
                dim=0
            )
            groups[target_key]["center_idx"] = merged_idx

            # Remove small patch from active set
            active_keys.remove(key)
            changed = True

    # ------------------------------------------------------------------
    # 4) Build full patches from merged center patches
    # ------------------------------------------------------------------
    final_keys = sorted(active_keys)
    patch_list = []

    center_sizes = []
    full_sizes = []
    full_center_ratios = []

    for key in final_keys:
        center_idx = groups[key]["center_idx"]

        center_coord = coord[center_idx]
        cx0 = center_coord[:, 0].min().item()
        cx1 = center_coord[:, 0].max().item()
        cy0 = center_coord[:, 1].min().item()
        cy1 = center_coord[:, 1].max().item()

        # Use center bbox to define halo expansion
        bw = max(cx1 - cx0, 1e-8)
        bh = max(cy1 - cy0, 1e-8)

        pad_x = bw * halo_ratio
        pad_y = bh * halo_ratio

        fx0 = cx0 - pad_x
        fx1 = cx1 + pad_x
        fy0 = cy0 - pad_y
        fy1 = cy1 + pad_y

        full_mask = (x >= fx0) & (x <= fx1) & (y >= fy0) & (y <= fy1)
        full_idx = torch.nonzero(full_mask, as_tuple=False).squeeze(1).long()

        center_mask = torch.zeros(full_idx.shape[0], dtype=torch.bool)

        # Map global center index to local position in full_idx
        pos_map = {idx.item(): pos for pos, idx in enumerate(full_idx)}
        center_pos = [pos_map[idx.item()] for idx in center_idx]
        center_mask[torch.tensor(center_pos, dtype=torch.long)] = True

        patch_list.append({
            "center_idx": center_idx,
            "full_idx": full_idx,
            "center_mask": center_mask,
        })

        csz = center_idx.numel()
        fsz = full_idx.numel()
        center_sizes.append(csz)
        full_sizes.append(fsz)
        full_center_ratios.append(fsz / max(csz, 1))

    # ------------------------------------------------------------------
    # 5) Debug info
    # ------------------------------------------------------------------
    grid_info = {
        "n_cells": n_cells,
        "n_patches_target": n_patches_target,
        "n_patches_actual": len(patch_list),
        "n_x": n_x,
        "n_y": n_y,
        "patch_w": patch_w,
        "patch_h": patch_h,
        "halo_ratio": halo_ratio,
        "min_required_center_cells": min_required,
        "center_cells_mean": float(sum(center_sizes) / len(center_sizes)),
        "center_cells_min": int(min(center_sizes)),
        "center_cells_max": int(max(center_sizes)),
        "full_cells_mean": float(sum(full_sizes) / len(full_sizes)),
        "full_cells_min": int(min(full_sizes)),
        "full_cells_max": int(max(full_sizes)),
        "full_center_ratio_mean": float(sum(full_center_ratios) / len(full_center_ratios)),
        "full_center_ratio_min": float(min(full_center_ratios)),
        "full_center_ratio_max": float(max(full_center_ratios)),
    }

    return patch_list, grid_info


def make_spatial_neighbors(
    coord,
    max_full_size=2000,
    knn_k=10,
    center_ratio=None,
    truncate_context=False,
    min_context_per_center=1,
):
    """
    Build neighbor-based batches with the same API as make_spatial_patches.

    Parameters
    ----------
    coord : torch.Tensor
        Spatial coordinates, shape [N, d].
    max_full_size : int
        Maximum number of nodes in one inference subgraph (center + context).
    knn_k : int
        Number of candidate neighbors for each center.
    center_ratio : float or None
        If None, use conservative mode:
            n_center = max_full_size // (1 + knn_k)
        If provided, use efficiency mode:
            n_center = int(max_full_size * center_ratio)
    truncate_context : bool
        Whether to truncate context if union size exceeds max_full_size.
    min_context_per_center : int
        Minimum number of neighbors to preserve for each center when truncation is enabled.

    Returns
    -------
    neighbor_list : list of dict
        Each dict contains:
            - 'full_idx': local node indices of current subgraph
            - 'center_idx': local indices of centers inside full_idx
            - 'center_mask': bool mask of centers inside full_idx
    """
    device = coord.device
    N = coord.shape[0]

    if N == 0:
        return []
    if max_full_size <= 0:
        raise ValueError("max_full_size must be > 0")
    if knn_k < 0:
        raise ValueError("knn_k must be >= 0")
    if min_context_per_center < 0:
        raise ValueError("min_context_per_center must be >= 0")

    if center_ratio is None:
        n_center = max_full_size // max(1, 1 + knn_k)
        n_center = max(1, n_center)
    else:
        n_center = int(max_full_size * center_ratio)
        n_center = max(1, min(n_center, max_full_size))

    neighbor_list = []

    for start in range(0, N, n_center):
        end = min(start + n_center, N)
        center_idx = torch.arange(start, end, device=device)
        n_cur_center = center_idx.numel()

        # No room for context
        if n_cur_center >= max_full_size:
            center_idx = center_idx[:max_full_size]
            full_idx = center_idx
            center_local_idx = torch.arange(full_idx.numel(), device=device)
            center_mask = torch.ones(full_idx.numel(), dtype=torch.bool, device=device)
            neighbor_list.append(
                {
                    "full_idx": full_idx,
                    "center_idx": center_local_idx,
                    "center_mask": center_mask,
                }
            )
            continue

        # No neighbors needed / possible
        if knn_k == 0 or N == 1:
            full_idx = center_idx
            center_local_idx = torch.arange(full_idx.numel(), device=device)
            center_mask = torch.ones(full_idx.numel(), dtype=torch.bool, device=device)
            neighbor_list.append(
                {
                    "full_idx": full_idx,
                    "center_idx": center_local_idx,
                    "center_mask": center_mask,
                }
            )
            continue

        # kNN candidates for each center
        center_coord = coord[center_idx]                 # [C, d]
        dist = torch.cdist(center_coord, coord)         # [C, N]

        row_idx = torch.arange(n_cur_center, device=device)
        dist[row_idx, center_idx] = float("inf")        # exclude self

        k = min(knn_k, max(N - 1, 0))
        if k == 0:
            full_idx = center_idx
            center_local_idx = torch.arange(full_idx.numel(), device=device)
            center_mask = torch.ones(full_idx.numel(), dtype=torch.bool, device=device)
            neighbor_list.append(
                {
                    "full_idx": full_idx,
                    "center_idx": center_local_idx,
                    "center_mask": center_mask,
                }
            )
            continue

        knn_idx = torch.topk(
            dist,
            k=k,
            dim=1,
            largest=False,
        ).indices                                        # [C, k]

        # Remove accidental overlap with centers
        is_center = torch.isin(knn_idx, center_idx)
        knn_idx = torch.where(is_center, torch.full_like(knn_idx, -1), knn_idx)

        budget = max_full_size - n_cur_center
        if budget <= 0:
            full_idx = center_idx
        else:
            # ===== Fast path: direct union first =====
            all_neighbor_idx = knn_idx[knn_idx >= 0]
            if all_neighbor_idx.numel() > 0:
                all_neighbor_idx = torch.unique(all_neighbor_idx)
                is_center = torch.isin(all_neighbor_idx, center_idx)
                all_neighbor_idx = all_neighbor_idx[~is_center]

            if (not truncate_context) or (all_neighbor_idx.numel() <= budget):
                # No truncation needed
                if all_neighbor_idx.numel() > 0:
                    full_idx = torch.cat([center_idx, all_neighbor_idx], dim=0)
                else:
                    full_idx = center_idx
            else:
                # ===== Truncation path: fair allocation =====
                selected_neighbors = []
                selected_set = set(center_idx.tolist())

                # Step 1: guarantee minimum context per center
                base_take = min(min_context_per_center, k)
                for i in range(n_cur_center):
                    taken = 0
                    for j in range(k):
                        nid = knn_idx[i, j].item()
                        if nid < 0 or nid in selected_set:
                            continue
                        selected_neighbors.append(nid)
                        selected_set.add(nid)
                        taken += 1
                        if len(selected_neighbors) >= budget or taken >= base_take:
                            break
                    if len(selected_neighbors) >= budget:
                        break

                # Step 2: round-robin fill remaining budget
                if len(selected_neighbors) < budget:
                    for j in range(k):
                        for i in range(n_cur_center):
                            nid = knn_idx[i, j].item()
                            if nid < 0 or nid in selected_set:
                                continue
                            selected_neighbors.append(nid)
                            selected_set.add(nid)
                            if len(selected_neighbors) >= budget:
                                break
                        if len(selected_neighbors) >= budget:
                            break

                if len(selected_neighbors) > 0:
                    neighbor_idx = torch.tensor(
                        selected_neighbors,
                        device=device,
                        dtype=torch.long,
                    )
                    full_idx = torch.cat([center_idx, neighbor_idx], dim=0)
                else:
                    full_idx = center_idx

        # Build global -> local map
        idx_map = torch.full((N,), -1, device=device, dtype=torch.long)
        idx_map[full_idx] = torch.arange(full_idx.numel(), device=device)

        center_local_idx = idx_map[center_idx]
        center_local_idx = center_local_idx[center_local_idx >= 0]

        center_mask = torch.zeros(full_idx.numel(), dtype=torch.bool, device=device)
        center_mask[center_local_idx] = True

        neighbor_list.append(
            {
                "full_idx": full_idx,
                "center_idx": center_idx,  # original center_idx
                "center_mask": center_mask,
            }
        )

    return neighbor_list
