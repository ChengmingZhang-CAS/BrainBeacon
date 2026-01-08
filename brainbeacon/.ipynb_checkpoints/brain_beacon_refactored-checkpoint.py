import numpy as np
import torch
import torch.nn as nn
import joblib
import os

from torch.utils.data import Dataset
from tqdm import tqdm


# Constants
MASK_TOKEN = 0
CLS_TOKEN = 2


def save_checkpoint(epoch, global_step, model, optimizer, path):
    """Save model_raw and optimizer state to a checkpoint."""
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


class PretrainDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.adata_list = args
        self.cumsum_lengths = np.cumsum([adata.shape[0] for adata in self.adata_list])

    def __len__(self):
        return self.cumsum_lengths[-1]

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.cumsum_lengths, idx, side="right")
        if file_index == 0:
            row_idx = idx
        else:
            row_idx = idx - self.cumsum_lengths[file_index - 1]
        masked_indices = self.adata_list[file_index][row_idx].obsm['masked_indices'][0]
        mask = self.adata_list[file_index][row_idx].obsm['mask'][0]
        real_indices = self.adata_list[file_index][row_idx].obsm['X'][0]
        attention_mask = self.adata_list[file_index][row_idx].obsm['attention_mask'][0]
        connect_comp = self.adata_list[file_index][row_idx].obsm['X_connect_comp'][0]
        rna_type = self.adata_list[file_index][row_idx].obsm['X_rna_type'][0]
        cell_ids = self.adata_list[file_index][row_idx].obsm['X_cell_ids'][0]

        return [
            masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type, cell_ids
        ]


class PretrainJoblibDataset(Dataset):
    def __init__(
            self,
            masked_indices_files,
            mask_files,
            real_indices_files,
            attention_mask_files,
            connect_comp_files,
            rna_type_files,
            cell_ids_files,
            file_prefix_list
    ):
        self.masked_indices_files = masked_indices_files
        self.mask_files = mask_files
        self.real_indices_files = real_indices_files
        self.attention_mask_files = attention_mask_files
        self.connect_comp_files = connect_comp_files
        self.rna_type_files = rna_type_files
        self.cell_ids_files = cell_ids_files
        self.file_prefix_list = file_prefix_list
        # Load metadata (e.g., lengths) for all files
        print(f"begin to read files length: {len(self.masked_indices_files)}")
        self.file_lengths = [len(joblib.load(f)) for f in self.masked_indices_files]
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
            masked_indices_file = self.masked_indices_files[file_idx]
            mask_file = self.mask_files[file_idx]
            real_indices_file = self.real_indices_files[file_idx]
            attention_mask_file = self.attention_mask_files[file_idx]
            connect_comp_file = self.connect_comp_files[file_idx]
            rna_type_file = self.rna_type_files[file_idx]
            cell_ids_file = self.cell_ids_files[file_idx]

            masked_indices = joblib.load(masked_indices_file)[sample_idx]
            mask = joblib.load(mask_file)[sample_idx]
            real_indices = joblib.load(real_indices_file)[sample_idx]
            attention_mask = joblib.load(attention_mask_file)[sample_idx]
            connect_comp = joblib.load(connect_comp_file)[sample_idx]
            rna_type = joblib.load(rna_type_file)[sample_idx]
            cell_ids = joblib.load(cell_ids_file)[sample_idx]
            if masked_indices is None or mask is None or real_indices is None or attention_mask is None or connect_comp is None or rna_type is None or cell_ids is None:
                print(self.file_prefix_list[idx])
                print(masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type, cell_ids)
            return (
                torch.as_tensor(masked_indices, dtype=torch.int32),
                torch.as_tensor(mask, dtype=torch.int32),
                torch.as_tensor(real_indices, dtype=torch.int32),
                torch.as_tensor(attention_mask, dtype=torch.bool),
                torch.as_tensor(connect_comp, dtype=torch.int32),
                torch.as_tensor(rna_type, dtype=torch.int32),
                torch.as_tensor(cell_ids, dtype=torch.int32)
            )
        except Exception as e:
            print(f"Error: {e}")
            print(f"Index: {idx}, file: {self.file_prefix_list[idx]}, sample: {sample_idx}")
            return self.__getitem__(idx + 1)


class GeneEmbedding(nn.Module):
    def __init__(self, n_tokens, n_connect_comp, n_rna_type, n_neighbor, dim_model, n_aux):
        super(GeneEmbedding, self).__init__()
        self.basic_embedding = nn.Embedding(
            num_embeddings=n_tokens + n_aux, embedding_dim=dim_model, padding_idx=1
        )
        self.homo_connect_embedding = nn.Embedding(
            num_embeddings=n_connect_comp + 1, embedding_dim=dim_model
        )
        self.rna_type_embedding = nn.Embedding(
            num_embeddings=n_rna_type + 1, embedding_dim=dim_model
        )
        self.cell_ids_embedding = nn.Embedding(
            num_embeddings=n_neighbor, embedding_dim=dim_model
        )

    def forward(self, x_gene_id, x_connect_id, x_rna_type, x_cell_ids):
        x_gene_emb = self.basic_embedding(x_gene_id.long())
        x_connect_emb = self.homo_connect_embedding(x_connect_id.long())
        x_rna_emb = self.rna_type_embedding(x_rna_type.long())
        x_cell_emb = self.cell_ids_embedding(x_cell_ids.long())
        return x_gene_emb + x_connect_emb + x_rna_emb + x_cell_emb


class BrainBeacon(nn.Module):
    def __init__(
            self,
            dim_model,
            nheads,
            dim_feedforward,
            nlayers,
            dropout,
            n_tokens,
            n_connect_comp,
            n_aux,
            n_rna_type,
            n_neighbor,
            esm_embedding_dim,
            total_context_length
    ):
        super(BrainBeacon, self).__init__()
        self.embedding = GeneEmbedding(n_tokens, n_connect_comp, n_rna_type, n_neighbor, dim_model, n_aux)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=1e-12,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.loss = nn.CrossEntropyLoss()
        self.classifier_head = nn.Linear(dim_model, n_tokens + n_aux, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens + n_aux))  # each token has its own bias
        self.classifier_head.bias = bias
        self.esm_embedding_projection = nn.Linear(esm_embedding_dim, dim_model)

        self.positional_embedding = nn.Embedding(num_embeddings=total_context_length, embedding_dim=dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pos = torch.arange(0, total_context_length, dtype=torch.long)

        self.initialize_weights()

    def get_esm_embedding(self, x):
        x_view = x.view(-1).long()
        esm_embedding = torch.index_select(self.esm_embedding_map, dim=0, index=x_view)
        esm_embedding = esm_embedding.view(x.shape[0], x.shape[1], esm_embedding.shape[-1])
        return esm_embedding

    def initialize_weights(self):
        for m in self.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x_gene_id, x_connect_id, x_rna_type, x_cell_ids, attention_mask, esm_embedding):
        token_embedding = self.embedding(x_gene_id, x_connect_id, x_rna_type, x_cell_ids)
        token_embedding += self.esm_embedding_projection(esm_embedding)
        pos = self.pos.to(token_embedding.device)
        pos_embedding = self.positional_embedding(pos)  # batch x (n_tokens) x dim_model
        embeddings = self.dropout(token_embedding + pos_embedding)
        transformer_output = self.encoder(embeddings, src_key_padding_mask=attention_mask)
        prediction = self.classifier_head(transformer_output)
        return prediction


def train_one_epoch(model, dataloader, optimizer, criterion, device, rank, writer, esm_embedding_map, global_step, logger, logdir):
    model.train()
    total_loss = 0.0
    for masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type, cell_ids in tqdm(dataloader):
        masked_indices = masked_indices[0]
        mask = mask[0]
        real_indices = real_indices[0]
        attention_mask = attention_mask[0]
        connect_comp = connect_comp[0]
        rna_type = rna_type[0]
        cell_ids = cell_ids[0]

        real_indices_view = real_indices.view(-1).long()
        esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
        esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])
        masked_indices, attention_mask, connect_comp, rna_type, cell_ids, esm_embedding = masked_indices.to(device), \
            attention_mask.to(device), connect_comp.to(device), rna_type.to(device), cell_ids.to(device), \
            esm_embedding.to(device)

        mlm_predictions = model(masked_indices, connect_comp, rna_type, cell_ids, attention_mask, esm_embedding)
        real_indices = torch.where(mask == MASK_TOKEN, real_indices, torch.tensor(-100, dtype=torch.long)).type(
            torch.int64)
        real_indices = real_indices.to(device)

        loss = criterion(mlm_predictions.view(-1, mlm_predictions.shape[-1]), real_indices.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        global_step += 1

        # Logging and checkpointing
        if rank == 0:
            if global_step % 1000 == 0:
                avg_loss = total_loss / 1000
                print(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
                if writer:
                    writer.add_scalar("Loss/Step", avg_loss, global_step)
                if logger:
                    logger.info(f"Step {global_step}, Avg Loss: {avg_loss:.4f}")
                total_loss = 0.0
            if global_step % 10000 == 0:
                checkpoint_path = os.path.join(logdir, f"checkpoint_step_{global_step}.pt")
                save_checkpoint(
                    epoch=None,  # epoch can be None since we're saving by step
                    model=model.module,
                    optimizer=optimizer,
                    path=checkpoint_path,
                    global_step=global_step
                )
                print(f"Checkpoint saved at step {global_step} to {checkpoint_path}")
                if logger:
                    logger.info(f"Checkpoint saved at step {global_step} to {checkpoint_path}")
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_gene_id, x_connect_id, y in dataloader:
            x_gene_id, x_connect_id, y = x_gene_id.to(device), x_connect_id.to(device), y.to(device)
            outputs = model(x_gene_id, x_connect_id)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)
