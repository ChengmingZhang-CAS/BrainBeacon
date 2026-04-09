import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from scipy.stats import expon
from scipy.sparse import csr_matrix
from .sparse import simple_mask
import torch.distributions as td
import math

def sample_spatial_patch(coord, n_keep):
    """
    Sample a spatially contiguous patch by choosing one random center cell
    and keeping its nearest neighbors.
    """
    n_cells = coord.shape[0]
    if n_cells <= n_keep:
        return torch.arange(n_cells, device=coord.device)

    center_idx = torch.randint(0, n_cells, (1,), device=coord.device).item()
    center_coord = coord[center_idx:center_idx + 1]
    dist = torch.sum((coord - center_coord) ** 2, dim=1)
    cell_idx = torch.topk(dist, k=n_keep, largest=False).indices
    return cell_idx


def drop_nodes_raw(x_dict, drop_node_rate=0., max_batch_size=2000, inplace=True):
    if inplace == False:
        raise NotImplementedError('Only support inplace drop nodes')

    if drop_node_rate > 0:  # cell_idx is the index of the nodes that are not dropped
        n_keep = min(max_batch_size, int(x_dict['x_seq'].shape[0] * (1 - drop_node_rate)))
        cell_idx = torch.randperm(x_dict['x_seq'].shape[0], device=x_dict['x_seq'].device)[:n_keep]
        # cell_idx = torch.randperm(x_dict['x_seq'].shape[0], device=x_dict['x_seq'].device)[
        #            :min(max_batch_size, int(x_dict['x_seq'].shape[0] * (1 - drop_node_rate)))]

        x_dict['x_seq'] = x_dict['x_seq'].index_select(0, cell_idx)
        if 'bb_emb' in x_dict:  # new added
            x_dict['bb_emb'] = x_dict['bb_emb'][cell_idx]
        if 'batch' in x_dict:
            x_dict['batch'] = x_dict['batch'][cell_idx]
        if 'h' in x_dict:
            x_dict['h'] = x_dict['h'][cell_idx]
        if 'g' in x_dict:
            x_dict['g'] = x_dict['g'][cell_idx][:, cell_idx]
        if 'coord' in x_dict:
            x_dict['coord'] = x_dict['coord'][cell_idx]
        if 'label' in x_dict:
            x_dict['label'] = x_dict['label'][cell_idx]
        if 'lib_size' in x_dict:
            x_dict['lib_size'] = x_dict['lib_size'][cell_idx]
        if 'x_masked_seq' in x_dict:
            x_dict['x_masked_seq'] = x_dict['x_masked_seq'].index_select(0, cell_idx)
        if 'dataset' in x_dict:
            x_dict['dataset'] = x_dict['dataset'][cell_idx]
        if 'loss_mask' in x_dict:
            x_dict['loss_mask'] = x_dict['loss_mask'][cell_idx]



def drop_nodes(x_dict, drop_node_rate=0., max_batch_size=2000, inplace=True, cell_idx=None):
    """
    Subset node-aligned fields in x_dict.

    If cell_idx is provided, use it directly.
    Otherwise, randomly sample nodes according to drop_node_rate.
    """
    if inplace is False:
        raise NotImplementedError('Only support inplace drop nodes')

    if cell_idx is None:
        if drop_node_rate <= 0:
            return x_dict

        n_keep = min(max_batch_size, int(x_dict['x_seq'].shape[0] * (1 - drop_node_rate)))
        cell_idx = torch.randperm(
            x_dict['x_seq'].shape[0],
            device=x_dict['x_seq'].device
        )[:n_keep]

    x_dict['x_seq'] = x_dict['x_seq'].index_select(0, cell_idx)
    if 'bb_emb' in x_dict:
        x_dict['bb_emb'] = x_dict['bb_emb'][cell_idx]
    if 'batch' in x_dict:
        x_dict['batch'] = x_dict['batch'][cell_idx]
    if 'h' in x_dict:
        x_dict['h'] = x_dict['h'][cell_idx]
    if 'g' in x_dict:
        x_dict['g'] = x_dict['g'][cell_idx][:, cell_idx]
    if 'coord' in x_dict:
        x_dict['coord'] = x_dict['coord'][cell_idx]
    if 'label' in x_dict:
        x_dict['label'] = x_dict['label'][cell_idx]
    if 'lib_size' in x_dict:
        x_dict['lib_size'] = x_dict['lib_size'][cell_idx]
    if 'x_masked_seq' in x_dict:
        x_dict['x_masked_seq'] = x_dict['x_masked_seq'].index_select(0, cell_idx)
    if 'dataset' in x_dict:
        x_dict['dataset'] = x_dict['dataset'][cell_idx]
    if 'slice_cov' in x_dict:
        x_dict['slice_cov'] = x_dict['slice_cov'][cell_idx]
    if 'dataset_cov' in x_dict:
        x_dict['dataset_cov'] = x_dict['dataset_cov'][cell_idx]
    if 'platform_cov' in x_dict:
        x_dict['platform_cov'] = x_dict['platform_cov'][cell_idx]
    if 'platform' in x_dict:
        x_dict['platform'] = x_dict['platform'][cell_idx]
    if 'loss_mask' in x_dict:
        x_dict['loss_mask'] = x_dict['loss_mask'][cell_idx]

    return x_dict


def sample_knn_neighbors(center_idx, x_dict, knn_k=8, exclude_self=True):
    """
    Find k nearest spatial neighbors for each center node.

    Parameters
    ----------
    center_idx : torch.Tensor
        Global indices of center nodes, shape [n_center].
    x_dict : dict
        Must contain 'coord' with shape [N, d].
    knn_k : int
        Number of nearest neighbors to retrieve for each center.
    exclude_self : bool
        Whether to exclude the center node itself from its neighbor list.

    Returns
    -------
    neighbor_idx : torch.Tensor
        Flattened global neighbor indices.
    """
    coord = x_dict['coord']
    device = coord.device
    num_nodes = coord.shape[0]

    if center_idx.numel() == 0 or num_nodes == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    max_valid_k = max(num_nodes - 1, 0) if exclude_self else num_nodes
    knn_k = min(knn_k, max_valid_k)

    if knn_k == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    center_coord = coord[center_idx]  # [n_center, d]
    dist = torch.cdist(center_coord, coord)  # [n_center, N]

    if exclude_self:
        row_idx = torch.arange(center_idx.shape[0], device=device)
        dist[row_idx, center_idx] = float('inf')

    knn_idx = torch.topk(
        dist,
        k=knn_k,
        dim=1,
        largest=False,
    ).indices  # [n_center, knn_k]

    neighbor_idx = knn_idx.reshape(-1)
    return neighbor_idx


def build_spatial_cell_idx(center_idx, neighbor_idx, max_batch_size):
    """
    Build final selected node indices for spatial mode.

    Strategy
    --------
    1. Keep all unique center nodes first.
    2. Remove neighbors already included in center.
    3. Randomly sample neighbors to fill remaining slots.
    4. If center itself exceeds max_batch_size, randomly subsample center.
    """
    device = center_idx.device

    center_idx = torch.unique(center_idx)
    neighbor_idx = torch.unique(neighbor_idx)

    if neighbor_idx.numel() > 0:
        is_center = torch.isin(neighbor_idx, center_idx)
        neighbor_idx = neighbor_idx[~is_center]

    n_center = center_idx.numel()
    if n_center >= max_batch_size:
        perm = torch.randperm(n_center, device=device)
        return center_idx[perm[:max_batch_size]]

    n_remain = max_batch_size - n_center
    if neighbor_idx.numel() > 0:
        perm = torch.randperm(neighbor_idx.numel(), device=device)
        neighbor_idx = neighbor_idx[perm[:n_remain]]
        cell_idx = torch.cat([center_idx, neighbor_idx], dim=0)
    else:
        cell_idx = center_idx

    return cell_idx


def build_spatial_patch_mask(
    x_dict,
    center_ratio,
    max_batch_size,
    knn_k=8,
):
    """
    Build spatial patch and corresponding input_mask.

    Assumption
    ----------
    This function is only called when num_nodes > max_batch_size.

    Data flow
    ---------
    1. Randomly sample center nodes.
    2. Find kNN neighbors for centers.
    3. Build final cell_idx with center priority.
    4. Subset x_dict using drop_nodes(..., cell_idx=cell_idx).
    5. Build input_mask with center=1 and others=0.
    """
    num_nodes = x_dict['x_seq'].shape[0]
    device = x_dict['x_seq'].device

    n_center = int(center_ratio * max_batch_size)
    n_center = max(1, min(n_center, max_batch_size, num_nodes))

    # Randomly sample center nodes
    center_idx = torch.randperm(num_nodes, device=device)[:n_center]

    # Find neighbors
    neighbor_idx = sample_knn_neighbors(
        center_idx=center_idx,
        x_dict=x_dict,
        knn_k=knn_k,
        exclude_self=True,
    )

    # Build final selected indices
    cell_idx = build_spatial_cell_idx(
        center_idx=center_idx,
        neighbor_idx=neighbor_idx,
        max_batch_size=max_batch_size,
    )

    # Map global center indices to local indices after subsetting
    idx_map = torch.full((num_nodes,), -1, device=device, dtype=torch.long)
    idx_map[cell_idx] = torch.arange(cell_idx.numel(), device=device)
    center_local_idx = idx_map[center_idx]
    center_local_idx = center_local_idx[center_local_idx >= 0]

    # Subset x_dict
    drop_nodes(
        x_dict,
        drop_node_rate=0.,
        max_batch_size=max_batch_size,
        inplace=True,
        cell_idx=cell_idx,
    )

    # Only center nodes contribute to loss
    x_dict['input_mask'] = torch.zeros(
        (x_dict['x_seq'].shape[0], 1),
        device=device,
        dtype=torch.float32,
    )
    x_dict['input_mask'][center_local_idx] = 1.0

    return x_dict


class NullMaskBuilder(nn.Module):
    def __init__(self, drop_node_rate, max_batch_size=2000):
        super().__init__()
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size

    def apply_mask(self, x_dict):
        if self._drop_node_rate > 0 and self.training:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        # x_dict['mask'] = torch.arange(x_dict['h'].shape[0], device=x_dict['h'].device)
        x_dict['input_mask'] = torch.ones(*x_dict['x_seq'].shape, device=x_dict['x_seq'].device).int()
        return x_dict

class MaskBuilder(nn.Module):
    def __init__(self, mask_node_rate, mask_feature_rate, drop_node_rate=0, max_batch_size=2000, edge_mask=None, mask_beta=False):
        super().__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate
        self._edge_mask = edge_mask
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size
        if self._mask_node_rate > 0 and self._mask_feature_rate and mask_beta:
            alpha = 5
            beta = 4 / self._mask_feature_rate + 2 - alpha
            self.beta_dist = td.Beta(alpha, beta)

        self.mask_beta = mask_beta

    def update_mask_ratio(self, mask_node_rate, mask_feature_rate):
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

    # This function mask parts of the nodes, and only the masked nodes will be used in the loss function
    def apply_mask(self, x_dict):
        if self.training and self._drop_node_rate > 0:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        if self.training and self._mask_node_rate > 0:
            if 'x_masked_seq' in x_dict:
                x = x_dict['x_masked_seq']
            else:
                x = x_dict['x_seq']

            if self.mask_beta:
                mask_ratio = self.beta_dist.sample((x.shape[0],)).to(x.device)
                mask_ratio[mask_ratio > 0.9] = 0.9
                num_nodes = x.shape[0]
                perm = np.random.permutation(num_nodes)
                num_mask_nodes = int(self._mask_node_rate * num_nodes)
                keep_nodes = perm[num_mask_nodes:]
                mask = torch.rand(*x.shape, device=x.device) <= mask_ratio.unsqueeze(-1)
                mask[keep_nodes] = False
            else:
                num_nodes = x.shape[0]
                perm = np.random.permutation(num_nodes)
                num_mask_nodes = int(self._mask_node_rate * num_nodes)
                keep_nodes = perm[num_mask_nodes:]  # keep_nodes is the index of the nodes that are not masked
                mask = torch.rand(*x.shape, device=x.device) <= self._mask_feature_rate
                mask[keep_nodes] = False

            x = x.coalesce()
            masked_x_seq = simple_mask(x, mask)
            x_dict['masked_x_seq'] = masked_x_seq
            x_dict['input_mask'] = mask.int()
        else:
            x_dict['input_mask'] = torch.ones(*x_dict['x_seq'].shape, device=x_dict['x_seq'].device).int()
        return x_dict

class HiddenMaskBuilder(nn.Module):
    def __init__(self, mask_node_rate, mask_countsure_rate, drop_node_rate=0, max_batch_size=2000, sampling_mode="spatial",
                 center_ratio=0.5, knn_k=30, edge_mask=None):
        super().__init__()
        self._mask_node_rate = mask_node_rate
        self._mask_countsure_rate = mask_countsure_rate
        self._edge_mask = edge_mask
        self._drop_node_rate = drop_node_rate
        self._max_batch_size = max_batch_size
        self._sampling_mode = sampling_mode
        self._center_ratio = center_ratio
        self._knn_k = knn_k

    def update_mask_ratio(self, mask_node_rate, mask_feature_rate):
        self._mask_node_rate = mask_node_rate
        self._mask_feature_rate = mask_feature_rate

    # This function mask parts of the nodes, and only the masked nodes will be used in the loss function
    def apply_mask(self, x_dict):

        num_nodes = x_dict['x_seq'].shape[0]
        if self.training and self._sampling_mode == "spatial" and num_nodes > self._max_batch_size:
            return build_spatial_patch_mask(
                x_dict=x_dict,
                center_ratio=self._center_ratio,
                max_batch_size=self._max_batch_size,
                knn_k=self._knn_k,
            )

        if self._drop_node_rate > 0 and self.training:
            drop_nodes(x_dict, self._drop_node_rate, self._max_batch_size)
        # Spatial patch mode: center cells are supervised, edge cells are context only

        if self._mask_node_rate > 0 and self.training:
            num_nodes = x_dict['h'].shape[0]
            perm = np.random.permutation(num_nodes)
            num_mask_nodes = int(self._mask_node_rate * num_nodes)
            keep_nodes = perm[num_mask_nodes:] # keep_nodes is the index of the nodes that are not masked

            out_x = F.dropout(x_dict['h'], p=self._mask_countsure_rate) # mask the countsures of all nodes
            out_x[keep_nodes] = x_dict['h'][keep_nodes] # keep the countsures of the nodes that are not masked
            # x_dict['h'] = out_x
            x_dict['input_mask'] = torch.zeros(x_dict['h'].shape[0], device=x_dict['h'].device).unsqueeze(-1)
            x_dict['input_mask'][perm[: num_mask_nodes]] = 1.
        else:
            x_dict['input_mask'] = torch.ones(x_dict['h'].shape[0], device=x_dict['h'].device).unsqueeze(-1)
        return x_dict


class InputDropoutMaskBuilder(nn.Module):
    def __init__(self, input_drop_type="mar", valid_drop_rate=0.1, test_drop_rate=0.1, seed=10,
                 min_gene_counts=5):
        super().__init__()
        assert 0 <= valid_drop_rate < 1, "valid_drop_rate should be in [0, 1)"
        assert 0 < test_drop_rate < 1, "test_drop_rate should be in (0, 1)"
        assert 0 < valid_drop_rate + test_drop_rate < 1, "Total masking rate should be in (0, 1)"
        self._input_drop_type = input_drop_type
        self._valid_drop_rate = valid_drop_rate
        self._test_drop_rate = test_drop_rate
        self._min_gene_counts = min_gene_counts
        self._seed = seed
        if input_drop_type == "mcar":
            self.distr = "uniform"
        elif input_drop_type == "mar":
            self.distr = "exp"
        else:
            raise NotImplementedError(f"Expect mask_type in ['mar', 'mcar'], but found {self.mask_type}")

    def _get_probs(self, vec):
        return {
            "exp": expon.pdf(vec, 0, 20),
            "uniform": np.tile([1. / len(vec)], len(vec)),
        }.get(self.distr)
    
    def apply_mask(self, x_seq):
        counts = x_seq.to_dense()
        train_mask = np.ones(counts.shape, dtype=bool)
        valid_mask = np.zeros(counts.shape, dtype=bool)
        test_mask = np.zeros(counts.shape, dtype=bool)
        rng = np.random.default_rng(self._seed)

        for c in range(counts.shape[0]):
            # Retrieve indices of positive values
            ind_pos = torch.nonzero(counts[c], as_tuple=True)[0]
            cells_c_pos = counts[c, ind_pos]

            # Get masking probability of each value
            if len(cells_c_pos) > self._min_gene_counts:
                mask_prob = self._get_probs(cells_c_pos)
                mask_prob = mask_prob / sum(mask_prob)
                n_test = int(np.floor(len(cells_c_pos) * self._test_drop_rate))
                n_valid = int(np.floor(len(cells_c_pos) * self._valid_drop_rate))
                if n_test + n_valid >= len(cells_c_pos):
                    print(f"Too many genes masked for cell {c} ({n_test + n_valid}/{len(cells_c_pos)})")
                    n_test -= 1
                    n_valid -= 1

                idx_mask = np.ones(len(ind_pos), dtype=bool)
                test_idx = rng.choice(np.arange(len(ind_pos)), n_test, p=mask_prob, replace=False)
                train_mask[c, ind_pos[test_idx]] = False
                test_mask[c, ind_pos[test_idx]] = True
                if self._valid_drop_rate > 0:
                    idx_mask[test_idx] = False
                    masked_mask_prob = mask_prob[idx_mask] / sum(mask_prob[idx_mask])
                    valid_idx = rng.choice(np.arange(len(ind_pos))[idx_mask], n_valid, p=masked_mask_prob, replace=False)
                    train_mask[c, ind_pos[valid_idx]] = False
                    valid_mask[c, ind_pos[valid_idx]] = True

        return train_mask, valid_mask, test_mask
