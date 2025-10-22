import torch
import torch.nn as nn
import numpy as np
import scanpy as sc
import anndata as ad
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from tqdm import tqdm
from copy import deepcopy
from ..utils.eval import downstream_eval, aggregate_eval_results, imputation_eval
from ..utils.data import XDict, TranscriptomicDataset
from typing import List, Literal, Union
from .experimental import symbol_to_ensembl
from torch.utils.data import DataLoader
import warnings
from . import Pipeline, load_pretrain
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import scipy.sparse

ReconstructDefaultModelConfig = {
    'objective': 'imputation',
    'mask_node_rate': 0.95,
    'mask_feature_rate': 0.25,
    'max_batch_size': 5000,
    'mask_type': 'hidden',
}

ReconstructDefaultPipelineConfig = {
    'lr': 5e-4,
    'wd': 1e-6,
    'scheduler': 'plat',
    'epochs': 100,
    'max_eval_batch_size': 5000,
    'patience': 5,
    'workers': 0,
}

def inference(model, dataloader, split, device, batch_size, order_required=False, output_attentions=False):
    if order_required and split:
        warnings.warn('When cell order required to be preserved, dataset split will be ignored.')

    with torch.no_grad():
        model.eval()
        epoch_loss = []
        order_list = []
        latent = []
        pred = []
        attention_list = [] if output_attentions else None
        for i, data_dict in enumerate(dataloader):
            if not order_required and split:
                split_mask = np.array(data_dict['split']) == split
                if np.sum(split_mask) == 0:
                    continue
                idx = torch.tensor(np.where(split_mask)[0])
            else:
                idx = torch.arange(data_dict['x_seq'].shape[0])

            for j in range(0, len(idx), batch_size):
                if len(idx) - j < batch_size:
                    cur = idx[j:]
                else:
                    cur = idx[j:j + batch_size]
                input_dict = {}
                for k in data_dict:
                    if k == 'x_seq':
                        input_dict[k] = data_dict[k].index_select(0, cur).to(device)
                    elif k == 'gene_mask':
                        input_dict[k] = data_dict[k].to(device)
                    elif k not in ['gene_list', 'split']:
                        input_dict[k] = data_dict[k][cur].to(device)
                x_dict = XDict(input_dict)
                out_dict, loss = model(x_dict, data_dict['gene_list'], output_attentions=output_attentions)
                epoch_loss.append(loss.item())
                # pred.append(out_dict['pred'])
                # latent.append(out_dict['latent'])
                pred.append(out_dict['pred'].cpu())     # prevent OOM
                latent.append(out_dict['latent'].cpu()) # prevent OOM
                if output_attentions and 'attention' in out_dict:
                    # Extract per-layer attention, shape: List[Tensor(seq_len, seq_len)]
                    attn_per_batch = [attn[0].mean(0).cpu() for attn in out_dict['attention']]
                    # Stack into a single tensor: [n_layers, seq_len, seq_len]
                    attn_tensor = torch.stack(attn_per_batch, dim=0)
                    attention_list.append(attn_tensor)

                if order_required:
                    order_list.append(input_dict['order_list'])
        torch.cuda.empty_cache()
        pred = torch.cat(pred)
        latent = torch.cat(latent)

        if order_required:
            # order = torch.cat(order_list)
            # order.scatter_(0, order.clone(), torch.arange(order.shape[0]).to(order.device))
            order = torch.cat(order_list).cpu()  # cpu
            order.scatter_(0, order.clone(), torch.arange(order.shape[0]))  # not.to(device)
            pred = pred[order]
            latent = latent[order]

        if output_attentions and attention_list is not None:
            n_layers = attention_list[0].shape[0]
            total_cells = pred.shape[0]

            # Initialize attention tensor
            attention = torch.zeros((n_layers, total_cells, total_cells), dtype=attention_list[0].dtype)

            start = 0
            for attn_tensor in attention_list:
                _, n_cell, _ = attn_tensor.shape
                attention[:, start:start+n_cell, start:start+n_cell] = attn_tensor
                start += n_cell

            if order_required:
                attention = attention[:, order][:, :, order]
        else:
            attention = None

        # === Final result ===
        result = {
            'pred': pred,
            'latent': latent,
            'loss': sum(epoch_loss) / len(epoch_loss),
        }
        if attention is not None:
            result['attention'] = attention
        return result

class ReconstructPipeline(Pipeline):
    def __init__(self,
                 pretrain_prefix: str,
                 overwrite_config: dict = ReconstructDefaultModelConfig,
                 pretrain_directory: str = './ckpt',
                 bb_pretrain_path: str = None,
                 cellformer_pretrain_path: str = None,
                 use_pretrain: bool = True,
                 ):
        super().__init__(
            pretrain_prefix=pretrain_prefix,
            overwrite_config=overwrite_config,
            pretrain_directory=pretrain_directory,
            bb_pretrain_path=bb_pretrain_path,
            cellformer_pretrain_path=cellformer_pretrain_path,
            use_pretrained=use_pretrain
        )
        self.label_encoders = None

    def fit(self, adata: ad.AnnData,
            train_config: dict = None,
            split_field: str = None,
            train_split: str = 'train',
            valid_split: str = 'valid',
            covariate_fields: List[str] = None,
            label_fields: List[str] = None,
            batch_gene_list: dict = None,
            ensembl_auto_conversion: bool = True,
            device: Union[str, torch.device] = 'cpu',
            ):
        config = ReconstructDefaultPipelineConfig.copy()
        if train_config:
            config.update(train_config)
        torch.cuda.empty_cache()
        self.model.to(device)
        assert not self.fitted, 'Current pipeline is already fitted and does not support continual training. Please initialize a new pipeline.'
        if label_fields:
            warnings.warn('`label_fields` argument is ignored in ImputationPipeline.')
        # adata = ad.concat([query_data, reference_data], join='outer', label='ref', keys=[False, True])
        adata = self.common_preprocess(adata, 0, covariate_fields, ensembl_auto_conversion=False)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, split_field, covariate_fields, label_fields, batch_gene_list)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=config['workers'])
        optim = torch.optim.AdamW(self.model.parameters(), lr=config['lr'], weight_decay=config['wd'])

        if config['scheduler'] == 'plat':
            scheduler = ReduceLROnPlateau(optim, 'min', patience=config['patience'], factor=0.9)
        else:
            scheduler = None

        train_loss = []
        valid_loss = []
        final_epoch = -1
        best_dict = None

        for epoch in tqdm(range(config['epochs'])):
            self.model.train()
            epoch_loss = []

            if epoch < 5:
                for param_group in optim.param_groups:
                    param_group['lr'] = config['lr'] * (epoch + 1) / 5

            for i, data_dict in enumerate(dataloader):
                if split_field and np.sum(data_dict['split'] == train_split) == 0:
                    continue
                input_dict = data_dict.copy()
                del input_dict['gene_list'], input_dict['split']
                for k in input_dict:
                    input_dict[k] = input_dict[k].to(device)
                x_dict = XDict(input_dict)
                out_dict, loss = self.model(x_dict, data_dict['gene_list'])
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optim.step()
                epoch_loss.append(loss.item())

            train_loss.append(sum(epoch_loss) / len(epoch_loss))
            if config['scheduler'] == 'plat':
                scheduler.step(train_loss[-1])
            result_dict = inference(self.model, dataloader, valid_split, device,
                                    config['max_eval_batch_size'])
            valid_loss.append(result_dict['loss'])

            print(f'Epoch {epoch} | Train loss: {train_loss[-1]:.4f} | Valid loss: {valid_loss[-1]:.4f}')

            if min(valid_loss) == valid_loss[-1]:
                best_dict = deepcopy(self.model.state_dict())
                # final_epoch = epoch

            # if min(valid_loss) != min(valid_loss[-config['es']:]):
            #     print(f'Early stopped. Best validation performance achieved at epoch {final_epoch}.')
            #     break

        assert best_dict, 'Best state dict was not stored. Please report this issue on Github.'
        self.model.load_state_dict(best_dict)
        self.fitted = True
        return self

    def predict(self, adata: ad.AnnData,
                inference_config: dict = None,
                covariate_fields: List[str] = None,
                batch_gene_list: dict = None,
                ensembl_auto_conversion: bool = True,
                output_attentions: bool = False,
                device: Union[str, torch.device] = 'cpu',):

        self.model.to(device)
        config = ReconstructDefaultPipelineConfig.copy()
        if inference_config:
            config.update(inference_config)
        adata = self.common_preprocess(adata, 0, covariate_fields, ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, None, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        output = inference(self.model, dataloader, None, device,
                  config['max_eval_batch_size'], order_required=True, output_attentions=output_attentions)
        pred = output['pred']
        latent = output['latent']
        # Ensure target_genes is defined
        if inference_config is None:
            inference_config = {}
        if "target_genes" not in inference_config:
            inference_config["target_genes"] = adata.var.index.tolist()

        # Apply target gene mask
        target_genes = inference_config["target_genes"]
        target_mask = torch.tensor(
            [adata.var.index.get_loc(g) for g in target_genes]
        ).long().to(pred.device)
        pred = pred[:, target_mask]

        # Save target_genes for later use
        self.target_genes = target_genes

        result = {
            'pred': pred,
            'latent': latent,
        }
        if output_attentions:
            result['attention'] = output.get("attention", None)

        return result

    def score(self, adata: ad.AnnData,
              evaluation_config: dict = None,
              split_field: str = None,
              target_split: str = None,
              covariate_fields: List[str] = None,
              label_fields: List[str] = None,
              batch_gene_list: dict = None,
              ensembl_auto_conversion: bool = True,
              device: Union[str, torch.device] = 'cpu',
              ):
        self.model.to(device)
        config = ReconstructDefaultPipelineConfig.copy()
        if evaluation_config:
            config.update(evaluation_config)
        adata = self.common_preprocess(adata, 0, covariate_fields, ensembl_auto_conversion)
        print(f'After filtering, {adata.shape[1]} genes remain.')
        dataset = TranscriptomicDataset(adata, None, order_required=True)
        dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        pred = inference(self.model, dataloader, None, device,
                         config['max_eval_batch_size'], order_required=True)['pred']
        if 'target_genes' in evaluation_config:
            target_mask = torch.tensor(
                [adata.var.index.get_loc(g) for g in evaluation_config['target_genes']]).long().to(pred.device)
            pred = pred[:, target_mask]
        if len(label_fields) != 1:
            raise NotImplementedError(
                f'`label_fields` containing multiple labels (f{len(label_fields)}) is not implemented for imputation pipeline. Please raise an issue on Github for further support.')
        if scipy.sparse.issparse(adata.obsm[label_fields[0]]):
            labels = torch.from_numpy(adata.obsm[label_fields[0]].toarray()).to(pred.device)
        else:
            labels = torch.from_numpy(adata.obsm[label_fields[0]]).to(pred.device)
        assert labels.shape[1] == pred.shape[
            1], f'Inconsistent number of genes between prediction ({pred.shape[1]}) and labels ({labels.shape[1]}). Please check: (1) Correct target gene list is provided in evaluation_config["target_genes"]. (2) Correct ground-truth gene expressions are provided in .obsm[label_fields[0]].'
        if split_field and target_split:
            labels = labels[adata.obs[split_field]==target_split]
            pred = pred[adata.obs[split_field]==target_split]
        # size_factor = 1e4 / (labels.sum(1, keepdims=True) + torch.from_numpy(adata.X.sum(1)).to(pred.device))
        # size_factor[size_factor.isinf()] == 0
        # labels = size_factor * labels
        # pred = size_factor * pred
        return imputation_eval(torch.log1p(pred), torch.log1p(labels))






