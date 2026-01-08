import torch
import torch.nn as nn
import torch.nn.init as init
import pytorch_lightning as pl
from typing import List
from torch import optim
import numpy as np
import math

MASK_TOKEN = 0
CLS_TOKEN = 2


class GeneEmbedding(torch.nn.Module):
    def __init__(self, n_tokens, n_connect_comp, dim_model, n_aux):
        super(GeneEmbedding, self).__init__()
        self.basic_embedding = nn.Embedding(
            num_embeddings=n_tokens + n_aux, embedding_dim=dim_model, padding_idx=1
        )
        self.homo_connect_embedding = nn.Embedding(num_embeddings=n_connect_comp + 1, embedding_dim=dim_model)

    def forward(self, x_gene_id, x_connect_id):
        x_gene_emb = self.basic_embedding(x_gene_id)
        x_connect_emb = self.homo_connect_embedding(x_connect_id)
        return x_gene_emb + x_connect_emb


class BrainBeacon(pl.LightningModule):

    def __init__(self,
                 dim_model: int,
                 nheads: int,
                 dim_feedforward: int,
                 nlayers: int,
                 dropout: float,
                 batch_first: bool,
                 masking_p: float,
                 n_tokens: int,
                 n_connect_comp: int,
                 n_aux: int,
                 single_context_length: int,
                 total_context_length: int,
                 lr: float,
                 warmup: int,
                 batch_size: int,
                 max_epochs: int,
                 autoregressive: bool,
                 pool: str = None,
                 cls_classes: int = 164,
                 supervised_task: int = None,
                 learnable_pe: bool = True,
                 specie: bool = False,
                 assay: bool = False,
                 modality: bool = False,
                 contrastive: bool = False,
                 neighbor_enhance: bool = False,
                 num_neighbors: int = 0,
                 use_esm_embedding: bool = False,
                 esm_embedding_dim: int = 1280,
                 esm_embedding_map: dict = None
                 ):
        """
        Args:
            dim_model (int): Dimensionality of the model_raw
            nheads (int): Number of attention heads
            dim_feedforward (int): Dimensionality of MLPs of attention blocks
            batch_first (int): batch first dimension
            n_tokens (int): total number of tokens (WITHOUT auxiliar tokens)
            single_context_length (int): length of the context, who would have guessed...
            total_context_length (int): length of the context
            autoregressive (bool): if True, implements autoregressive training
            pool (str): could be None, 'cls' or 'mean'. CLS adds a token at the beginning, mean just averages all tokens. If not supervised task during training, is ignored
            cls_classes (int): number of classes to classify
            supervised_task (str): None, 'classification' or 'regression'
            learnable_pe (bool): if True, positional embeddings are learnable embeddings, otherwise are derived from trigonometric functions
        """
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            dropout=dropout,
            layer_norm_eps=1e-12
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=False
        )

        # As in HuggingFace
        self.classifier_head = nn.Linear(dim_model, n_tokens + n_aux, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens + n_aux))  # each token has its own bias
        self.classifier_head.bias = bias

        # As in HuggingFace
        self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(dim_model, cls_classes)

        # Token embedding learnable weights
        # self.embeddings = nn.Embedding(num_embeddings=n_tokens + 5, embedding_dim=dim_model, padding_idx=1)
        self.embeddings = GeneEmbedding(n_tokens, n_connect_comp, dim_model, n_aux)
        if use_esm_embedding:
            self.esm_embedding_projection = nn.Linear(esm_embedding_dim, dim_model)

        if pool == 'cls':
            total_context_length += 1

        if not learnable_pe:
            self.positional_embedding = PositionalEncoding(d_model=dim_model, max_seq_len=total_context_length)
        else:
            # uses learnable weights as positional embeddings
            self.positional_embedding = nn.Embedding(num_embeddings=total_context_length, embedding_dim=dim_model)
            self.dropout = nn.Dropout(p=dropout)
            self.pos = torch.arange(0, total_context_length, dtype=torch.long)

        # MLM loss
        self.loss = nn.CrossEntropyLoss()

        if supervised_task is not None:
            self.cls_loss = nn.CrossEntropyLoss()

        self.autoregressive = autoregressive

        self.save_hyperparameters()

        self.gc_freq = 5

        self.batch_train_losses = []

        self.initialize_weights()

    def forward(self, x, x_connect_comp, x_esm_embedding, attention_mask):
        # x -> size: batch x (context_length) x 1
        token_embedding = self.embeddings(x, x_connect_comp)  # batch x (n_tokens) x dim_model
        if self.hparams.use_esm_embedding:
            token_embedding += self.esm_embedding_projection(x_esm_embedding)

        if self.hparams.learnable_pe:
            pos = self.pos.to(token_embedding.device)
            pos_embedding = self.positional_embedding(pos)  # batch x (n_tokens) x dim_model
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        transformer_output = self.encoder(
            embeddings,
            is_causal=self.autoregressive,
            src_key_padding_mask=attention_mask
        )  # batch x (n_tokens) x dim_model

        # MLM prediction
        prediction = self.classifier_head(transformer_output)

        return {
            'mlm_prediction': prediction,
            'transformer_output': transformer_output
        }

    def training_step(self, batch, batch_idx, *args, **kwargs):
        with torch.no_grad():
            batch = do_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)

        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['X']
        attention_mask = batch['attention_mask']
        connect_comp = batch['X_connect_comp']
        esm_embedding = batch['esm_embedding']
        predictions = self.forward(masked_indices, connect_comp, esm_embedding, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        # we just evaluate on the masked tokens (mask = 0)
        real_indices = torch.where(mask == MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(
            torch.int64)

        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens + self.hparams.n_aux)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)

        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length

        if self.hparams.masking_p == 0.0:  # this case is uniquely for the fine-tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices)  # MLM loss

        self.log('train_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        with torch.no_grad():
            batch = do_masking(batch, self.hparams.masking_p, self.hparams.n_tokens)

        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['X']
        attention_mask = batch['attention_mask']
        connect_comp = batch['X_connect_comp']
        esm_embedding = batch['esm_embedding']
        predictions = self.forward(masked_indices, connect_comp, esm_embedding, attention_mask)
        mlm_predictions = predictions['mlm_prediction']

        real_indices = torch.where(mask == MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(
            torch.int64)

        mlm_predictions = mlm_predictions.view(-1, self.hparams.n_tokens + self.hparams.n_aux)
        real_indices = real_indices.view(-1)
        masked_indices = masked_indices.view(-1)

        # There's a corner case that returns NaN loss: when there are no masked tokens
        # however, likelihood of that is (1-p)^context_length

        if self.hparams.masking_p == 0.0:  # this case is uniquely for the fine tuning case (check _fine_tune_model)
            loss = torch.tensor(0.0, device=mlm_predictions.device)
        else:
            loss = self.loss(mlm_predictions, real_indices)  # MLM loss

        self.log('val_loss', loss, sync_dist=True, prog_bar=True, reduce_fx='mean')

        return loss

    def get_embeddings(self, batch, layers: List[int] = [11], function: str = "mean"):
        """
            This function gets representations to later load them in some script
            that computes a downstream task

            batch: batch who representation will be outputed
            layers (List[int]): list that contains the indices of the layers whose repr. will obtain
            function (str): "concat", "mean", "sum", "cls" or None to combine the hidden rep. obtained
        """

        #batch['X'] = batch['X'][:, :self.hparams.context_length]

        batch = do_masking(batch, 0.0, self.hparams.n_tokens + 5)
        masked_indices = batch['masked_indices']
        mask = batch['mask']
        real_indices = batch['X']
        attention_mask = batch['attention_mask']

        token_embedding = self.embeddings(masked_indices)  # batch x (n_tokens) x dim_model

        if self.hparams.learnable_pe:
            pos_embedding = self.positional_embedding(
                self.pos.to(token_embedding.device))  # batch x (n_tokens) x dim_model
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.positional_embedding(token_embedding)

        hidden_repr = []
        # embeddings = self.encoder(embeddings, is_causal=self.autoregressive, src_key_padding_mask=attention_mask)

        for i in range(len(self.encoder.layers)):
            layer = self.encoder.layers[i]
            embeddings = layer(embeddings, is_causal=self.autoregressive,
                               src_key_padding_mask=attention_mask)  # bs x seq_len x dim
            if i in layers:
                # drop the first three tokens since are auxiliar
                embeddings = embeddings[:, 3:, :]
                hidden_repr.append(embeddings)

        if function == "mean":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            hidden_repr = torch.mean(combined_tensor, dim=-1)  # bs x seq_len x dim
            hidden_repr = torch.mean(combined_tensor, dim=1).squeeze()  # bs x dim

        if function == "sum":
            combined_tensor = torch.stack(hidden_repr, dim=-1)
            hidden_repr = torch.sum(combined_tensor, dim=-1)  # bs x seq_len x dim

        if function == "concat":
            hidden_repr = torch.cat(hidden_repr, dim=2)

        return hidden_repr, batch['assay'], batch['specie'], batch['modality']

    def on_before_batch_transfer(self, batch, dataloader_idx: int):

        batch, _ = batch

        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        data_key = 'X'
        data_connect_comp_key = 'X_connect_comp'
        data_cell_ids_key = 'X_cell_ids'
        batch[data_key] = batch[data_key][:, :self.hparams.single_context_length]
        batch[data_connect_comp_key] = batch[data_connect_comp_key][:, :self.hparams.single_context_length]
        batch[data_cell_ids_key] = torch.zeros_like(
            batch[data_connect_comp_key], dtype=torch.int32, device=batch[data_key].device
        )
        if self.hparams.pool == 'cls':  # Add cls token at the beginning of the set
            x = batch[data_key]
            x_connect_comp = batch[data_connect_comp_key]
            x_cell_ids = batch[data_cell_ids_key]
            cls = torch.ones((x.shape[0], 1), dtype=torch.int32, device=x.device) * CLS_TOKEN  # CLS token is index 2
            x = torch.cat((cls, x), dim=1)  # add CLS
            x_connect_comp = torch.cat((cls, x_connect_comp), dim=1)
            cell_cls_ids = torch.zeros((x.shape[0], 1), dtype=torch.int32, device=x.device)
            cell_ids = torch.cat((cell_cls_ids, x_cell_ids), dim=1)
            batch[data_key] = x
            batch[data_connect_comp_key] = x_connect_comp
            batch[data_cell_ids_key] = cell_ids

        if self.hparams.modality:
            x = batch[data_key]
            x_connect_comp = batch[data_connect_comp_key]
            modality = batch['modality']
            x_cell_ids = batch[data_cell_ids_key]
            x = torch.cat((modality.reshape(-1, 1), x), dim=1)  # add modality token
            x_connect_comp = torch.cat((modality.reshape(-1, 1), x_connect_comp), dim=1)
            cell_modality_ids = torch.zeros((x.shape[0], 1), dtype=torch.int32, device=x.device)
            x_cell_ids = torch.cat((cell_modality_ids, x_cell_ids), dim=1)
            batch[data_key] = x
            batch[data_connect_comp_key] = x_connect_comp
            batch[data_cell_ids_key] = x_cell_ids

        if self.hparams.assay:
            x = batch[data_key]
            x_connect_comp = batch[data_connect_comp_key]
            assay = batch['assay']
            x_cell_ids = batch[data_cell_ids_key]
            x = torch.cat((assay.reshape(-1, 1), x), dim=1)  # add assay token
            x_connect_comp = torch.cat((assay.reshape(-1, 1), x_connect_comp), dim=1)
            cell_cls_ids = torch.zeros((x.shape[0], 1), dtype=torch.int32, device=x.device)
            x_cell_ids = torch.cat((cell_cls_ids, x_cell_ids), dim=1)
            batch[data_key] = x
            batch[data_connect_comp_key] = x_connect_comp
            batch[data_cell_ids_key] = x_cell_ids

        if self.hparams.specie:
            x = batch[data_key]
            x_connect_comp = batch[data_connect_comp_key]
            specie = batch['specie']
            x_cell_ids = batch[data_cell_ids_key]
            x = torch.cat((specie.reshape(-1, 1), x), dim=1)  # add organism token
            x_connect_comp = torch.cat((specie.reshape(-1, 1), x_connect_comp), dim=1)
            cell_cls_ids = torch.zeros((x.shape[0], 1), dtype=torch.int32, device=x.device)
            x_cell_ids = torch.cat((cell_cls_ids, x_cell_ids), dim=1)
            batch[data_key] = x
            batch[data_connect_comp_key] = x_connect_comp
            batch[data_cell_ids_key] = x_cell_ids

        if self.hparams.neighbor_enhance:
            x = batch[data_key]
            x_connect_comp = batch[data_connect_comp_key]
            x_cell_ids = batch[data_cell_ids_key]
            for idx in range(1, self.hparams.num_neighbors):
                x_neighbor = batch[f'X_neighbor_{idx}'][:, :self.hparams.single_context_length]
                x_connect_comp_neighbor = batch[f'X_neighbor_{idx}_connect_comp'][:, :self.hparams.single_context_length]
                x = torch.cat((x, x_neighbor), dim=1)
                x_connect_comp = torch.cat((x_connect_comp, x_connect_comp_neighbor), dim=1)
                cell_neighbor_ids = torch.ones_like(x_connect_comp_neighbor, dtype=torch.int32, device=x.device) * idx
                x_cell_ids = torch.cat((x_cell_ids, cell_neighbor_ids), dim=1)
            batch[data_key] = x
            batch[data_connect_comp_key] = x_connect_comp
            batch[data_cell_ids_key] = x_cell_ids

        if self.hparams.supervised_task:  # turn feature to predict into label
            if 'cell_type' in batch.keys():
                batch['label'] = batch['cell_type']
            if 'X_niche' in batch.keys():
                batch['label'] = batch['X_niche']

        if self.hparams.use_esm_embedding:
            batch_indices_view = batch['X'].view(-1)
            esm_embedding_map = self.hparams.esm_embedding_map.to(batch_indices_view.device)
            esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=batch_indices_view)
            esm_embedding = esm_embedding.view(batch['X'].shape[0], batch['X'].shape[1], esm_embedding.shape[-1])
            batch['esm_embedding'] = esm_embedding

        batch['X'] = batch['X'][:, :self.hparams.total_context_length]
        batch['X_connect_comp'] = batch['X_connect_comp'][:, :self.hparams.total_context_length]
        if self.hparams.use_esm_embedding:
            batch['esm_embedding'] = batch['esm_embedding'][:, :self.hparams.total_context_length]

        return batch

    def configure_optimizers(self):

        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_epochs=self.hparams.max_epochs
        )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.zeros_(m.bias)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_epochs):
        self.warmup = warmup
        self.max_num_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(1e-5, base_lr * lr_factor) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_epochs))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


def do_masking(batch, p, n_tokens):
    padding_token = 1
    cls_token = 3

    indices = batch['X']

    indices = torch.where(indices == 0, torch.tensor(padding_token),
                          indices)  # 0 is originally the padding token, we change it to 1
    batch['X'] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices), p)  # mask indices with probability p

    masked_indices = indices * mask  # masked_indices
    masked_indices = torch.where(indices != padding_token, masked_indices, indices)  # we just mask non-padding indices
    mask = torch.where(indices == padding_token, torch.tensor(padding_token),
                       mask)  # in the model_raw we evaluate the loss of mask position 0
    # so we make the mask of all PAD tokens to be 1 so that it's not taken into account in the loss computation

    # Notice for the following 2 lines that masked_indices has already not a single padding token masked
    masked_indices = torch.where(indices != cls_token, masked_indices,
                                 indices)  # same with CLS, no CLS token can be masked
    mask = torch.where(indices == cls_token, torch.tensor(padding_token),
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

    batch['masked_indices'] = masked_indices
    batch['mask'] = mask

    attention_mask = (masked_indices == padding_token)
    batch['attention_mask'] = attention_mask.type(torch.bool)

    return batch
