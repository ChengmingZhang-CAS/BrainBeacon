import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import scipy
from sklearn.preprocessing import LabelEncoder
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


class TranscriptomicDatasetPerturb(Dataset):
    def __init__(self, adata: ad.AnnData,
                 split_field: str = None,
                 covariate_fields: List[str] = None,
                 label_fields: List[str] = None,
                 batch_gene_list: dict = None,
                 covariate_encoders: dict = None,
                 label_encoders: dict = None,
                 order_required: bool = False,
                 use_perturbation: bool = False,
                 perturb_flag: np.ndarray = None,
                 perturb_gene_id: np.ndarray = None):
        self.seq_list = []
        self.coord_list = []
        self.order_list = []
        self.batch_gene_list = batch_gene_list
        self.covariate_fields = covariate_fields
        self.label_fields = label_fields
        self.order_required = order_required
        self.gene_list = adata.var.index.tolist()
        
        # Perturbation-specific attributes
        self.use_perturbation = use_perturbation
        if use_perturbation and perturb_flag is not None and perturb_gene_id is not None:
            self.perturb_flag_list = perturb_flag
            self.perturb_gene_id_list = perturb_gene_id
            print(f"Added perturbation information: {len(perturb_flag)} samples")
        else:
            self.perturb_flag_list = None
            self.perturb_gene_id_list = None

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

        # For perturbation tasks, we need to split data into smaller batches to avoid memory issues
        # Use max_batch_size to create artificial batches
        max_batch_size = 10000  # Reduced from 20000 to avoid memory issues
        
        if 'batch' not in adata.obs:
            warnings.warn(
                'Batch labels not found in AnnData.obs. Creating artificial batches for memory management.')
            # Create artificial batches based on max_batch_size
            n_batches = (adata.shape[0] + max_batch_size - 1) // max_batch_size
            batch_labels = np.array([i // max_batch_size for i in range(adata.shape[0])], dtype=np.int8)
            self.batch_list = [f'artificial_batch_{i}' for i in range(n_batches)]
        else:
            # Use original batch labels but ensure they don't exceed max_batch_size
            batch_le = LabelEncoder().fit(adata.obs['batch'])
            batch_labels = batch_le.transform(adata.obs['batch'])
            self.batch_list = []
            
            # Check if any batch is too large and split if necessary
            for batch_id in range(batch_labels.max() + 1):
                batch_size = np.sum(batch_labels == batch_id)
                if batch_size > max_batch_size:
                    # Split large batch into smaller ones
                    batch_indices = np.where(batch_labels == batch_id)[0]
                    n_sub_batches = (batch_size + max_batch_size - 1) // max_batch_size
                    for sub_batch in range(n_sub_batches):
                        start_idx = sub_batch * max_batch_size
                        end_idx = min((sub_batch + 1) * max_batch_size, batch_size)
                        batch_labels[batch_indices[start_idx:end_idx]] = len(self.batch_list)
                        self.batch_list.append(f'{batch_le.classes_[batch_id]}_sub_{sub_batch}')
                else:
                    self.batch_list.append(batch_le.classes_[batch_id])

        # use bb_emb if available
        if 'bb_emb' in adata.obsm:
            self.bb_emb_list = []
            for batch in range(batch_labels.max() + 1):
                bb_tensor = torch.tensor(adata.obsm['bb_emb'][batch_labels == batch], dtype=torch.float32)
                self.bb_emb_list.append(bb_tensor)
        else:
            self.bb_emb_list = None

        # Process each batch
        for batch in range(batch_labels.max() + 1):
            batch_mask = batch_labels == batch
            x = csr_matrix(adata[batch_mask].X.astype(float))
            self.seq_list.append(sparse_scipy_to_tensor(x))

            for c in covariate_fields:
                self.covariate_list[c].append(torch.from_numpy(covariates[c][batch_mask]))

            for l in label_fields:
                self.label_list[l].append(torch.from_numpy(labels[l][batch_mask]))

            if split_field:
                self.split_list.append(adata.obs[split_field][batch_mask])

            if order_required:
                self.order_list.append(torch.from_numpy((batch_mask).nonzero()[0]))

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, idx):
        # 安全检查：确保索引有效
        if idx >= len(self.seq_list):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.seq_list)} items")
        
        # 安全检查：确保序列不为空
        if len(self.seq_list[idx]) == 0:
            print(f"WARNING: Empty sequence at index {idx}, skipping")
            # 返回一个最小的有效样本
            return {
                'x_seq': torch.zeros((1, len(self.gene_list)), dtype=torch.float32),
                'input_mask': torch.ones(1, dtype=torch.bool),
                'gene_mask': torch.ones(len(self.gene_list), dtype=torch.bool),
                'batch': torch.zeros(1, dtype=torch.long),
                'gene_list': self.gene_list
            }
        
        return_dict = {}
        return_dict['x_seq'] = self.seq_list[idx]
        return_dict['input_mask'] = torch.ones(len(self.seq_list[idx]), dtype=torch.bool)
        
        # if self.coord_list is not None:
        #     return_dict['coord'] = self.coord_list[idx]
        # else:
        #     return_dict['coord'] = torch.zeros([len(self.seq_list[idx]), 2]).float() - 1

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
            
        # Add batch information
        if self.batch_list and idx < len(self.batch_list):
            return_dict['batch'] = torch.zeros(len(self.seq_list[idx]), dtype=torch.long) + idx
        else:
            return_dict['batch'] = torch.zeros(len(self.seq_list[idx]), dtype=torch.long)
                
        # Add gene mask - 确保长度匹配
        gene_mask_length = len(self.gene_list)
        if hasattr(self, 'seq_list') and len(self.seq_list[idx]) > 0:
            # 如果序列长度与基因列表长度不匹配，进行调整
            seq_length = self.seq_list[idx].shape[1] if hasattr(self.seq_list[idx], 'shape') else len(self.seq_list[idx])
            if seq_length != gene_mask_length:
                print(f"WARNING: Sequence length ({seq_length}) != gene list length ({gene_mask_length}) at index {idx}")
                gene_mask_length = min(seq_length, gene_mask_length)
        
        return_dict['gene_mask'] = torch.ones(gene_mask_length, dtype=torch.bool)
        
        # Add input mask for reconstruction
        return_dict['input_mask'] = torch.ones(len(self.seq_list[idx]), dtype=torch.bool)
            
        # Add perturbation information if available
        if self.use_perturbation and self.perturb_flag_list is not None and self.perturb_gene_id_list is not None:
            # Get the indices for this batch
            batch_start = sum(len(self.seq_list[i]) for i in range(idx))
            batch_end = batch_start + len(self.seq_list[idx])
            
            # 安全检查：确保索引在有效范围内
            if batch_start < len(self.perturb_flag_list) and batch_end <= len(self.perturb_flag_list):
                return_dict['perturb_flag'] = torch.tensor(self.perturb_flag_list[batch_start:batch_end], dtype=torch.long)
                return_dict['perturb_gene_id'] = torch.tensor(self.perturb_gene_id_list[batch_start:batch_end], dtype=torch.long)
            else:
                print(f"WARNING: Perturbation indices out of range at index {idx}")
                # 使用默认值
                return_dict['perturb_flag'] = torch.zeros(len(self.seq_list[idx]), dtype=torch.long)
                return_dict['perturb_gene_id'] = torch.zeros(len(self.seq_list[idx]), dtype=torch.long)
            
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
