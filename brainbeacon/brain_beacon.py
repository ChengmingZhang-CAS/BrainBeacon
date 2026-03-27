import numpy as np
import torch
import torch.nn as nn
import joblib
import os

from torch.utils.data import Dataset
from tqdm import tqdm
import time
from contextlib import nullcontext
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR

# Constants
MASK_TOKEN = 0
CLS_TOKEN = 2




def get_linear_warmup_scheduler(optimizer, num_warmup_steps):
    def lr_lambda(current_step):
        return min(1.0, float(current_step) / float(max(1, num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(epoch, global_step, model, optimizer, path):
    """Save model_raw and optimizer state to a checkpoint."""
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, path)


class PretrainJoblibDataset(Dataset):
    def __init__(
            self,
            masked_indices_files,
            mask_files,
            real_indices_files,
            attention_mask_files,
            connect_comp_files,
            rna_type_files,
            neighbor_gene_distribution_files,
            file_prefix_list,
            file_labels=None
    ):
        self.masked_indices_files = masked_indices_files
        self.mask_files = mask_files
        self.real_indices_files = real_indices_files
        self.attention_mask_files = attention_mask_files
        self.connect_comp_files = connect_comp_files
        self.rna_type_files = rna_type_files
        self.neighbor_gene_distribution_files = neighbor_gene_distribution_files
        self.file_prefix_list = file_prefix_list
        self.file_labels = file_labels
        # Load metadata (e.g., lengths) for all files
        print(f"begin to read files length: {len(self.masked_indices_files)}")
        self.file_lengths = [len(joblib.load(f)) for f in self.masked_indices_files]
        self.cumulative_lengths = np.cumsum(self.file_lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self):
        """Total number of samples across all files"""
        return self.total_length

    def get_label(self, idx):
        """Return (platform_id, species_id) for the sample at global index idx."""
        file_idx, _ = self._find_file_idx(idx)
        return self.file_labels[file_idx]

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
            neighbor_gene_distribution_file = self.neighbor_gene_distribution_files[file_idx]

            masked_indices = joblib.load(masked_indices_file)[sample_idx]
            mask = joblib.load(mask_file)[sample_idx]
            real_indices = joblib.load(real_indices_file)[sample_idx]
            attention_mask = joblib.load(attention_mask_file)[sample_idx]
            connect_comp = joblib.load(connect_comp_file)[sample_idx]
            rna_type = joblib.load(rna_type_file)[sample_idx]
            neighbor_gene_distribution = joblib.load(neighbor_gene_distribution_file)[sample_idx]
            if masked_indices is None or mask is None or real_indices is None or attention_mask is None or connect_comp is None or rna_type is None:
                print(self.file_prefix_list[idx])
                print(masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type)
            return (
                torch.as_tensor(masked_indices[:, :1000], dtype=torch.long),
                torch.as_tensor(mask[:, :1000], dtype=torch.long),
                torch.as_tensor(real_indices[:, :1000], dtype=torch.long),
                torch.as_tensor(attention_mask[:, :1000], dtype=torch.bool),
                torch.as_tensor(connect_comp[:, :1000], dtype=torch.long),
                torch.as_tensor(rna_type[:, :1000], dtype=torch.long),
                torch.as_tensor(neighbor_gene_distribution[:, :1000], dtype=torch.float),
                # torch.as_tensor(masked_indices, dtype=torch.long),
                # torch.as_tensor(mask, dtype=torch.long),
                # torch.as_tensor(real_indices, dtype=torch.long),
                # torch.as_tensor(attention_mask, dtype=torch.bool),
                # torch.as_tensor(connect_comp, dtype=torch.long),
                # torch.as_tensor(rna_type, dtype=torch.long),
                # torch.as_tensor(cell_ids, dtype=torch.long)
            )
        except Exception as e:
            print(f"Error: {e}")
            print(f"Index: {idx}, file: {self.file_prefix_list[idx]}, sample: {sample_idx}")
            return self.__getitem__(idx + 1)


class GeneEmbedding(nn.Module):
    def __init__(self, n_tokens, n_connect_comp, n_rna_type, n_neighbor, dim_model, n_aux,
                 use_gene_id_emb=True, use_homo_emb=True, use_rna_type_emb=True):
        super(GeneEmbedding, self).__init__()
        self.use_gene_id_emb = use_gene_id_emb
        self.use_homo_emb = use_homo_emb
        self.use_rna_type_emb = use_rna_type_emb
        self.basic_embedding = nn.Embedding(
            num_embeddings=n_tokens + n_aux, embedding_dim=dim_model, padding_idx=1
        )
        self.homo_connect_embedding = nn.Embedding(
            num_embeddings=n_connect_comp + 1, embedding_dim=dim_model
        )
        self.rna_type_embedding = nn.Embedding(
            num_embeddings=n_rna_type + 1, embedding_dim=dim_model
        )
        # self.cell_ids_embedding = nn.Embedding(
        #     num_embeddings=n_neighbor, embedding_dim=dim_model
        # )
        # self.cor_embedding = LearnableFourierPositionalEncoding(
        #     G=1, M=2, F_dim=dim_model, H_dim=dim_model, D=dim_model, gamma=1.0
        # )

    def forward(self, x_gene_id, x_connect_id, x_rna_type):
        # x_gene_emb = self.basic_embedding(x_gene_id.long())
        # x_connect_emb = self.homo_connect_embedding(x_connect_id.long())
        # x_rna_emb = self.rna_type_embedding(x_rna_type.long())

        # x_cell_emb = self.cell_ids_embedding(x_cell_ids.long())  # remove when no neighbor
        # x_cor_emb = self.cor_embedding(x_cell_cor).unsqueeze(1).repeat(1, 2, 1)
        # x_gene_emb[:, :2, :] = x_gene_emb[:, :2, :] + x_cor_emb
        if self.use_gene_id_emb:
            x_gene_emb = self.basic_embedding(x_gene_id.long())
        else:
            x_gene_emb = 0

        if self.use_homo_emb:
            x_connect_emb = self.homo_connect_embedding(x_connect_id.long())
        else:
            x_connect_emb = 0

        if self.use_rna_type_emb:
            x_rna_emb = self.rna_type_embedding(x_rna_type.long())
        else:
            x_rna_emb = 0

        return x_gene_emb + x_connect_emb + x_rna_emb


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
            total_context_length,
            use_gene_id_emb=True,
            use_homo_emb=True,
            use_rna_type_emb=True,
            use_esm_emb=True,  # add esm usage flag
            use_pos_emb=True,  # add positional embedding usage flag
            use_density_emb=True,  # add density token usage flag
            density_token_idx=2,  # density token position (default: 2 when specie=True, assay=True)
            neighbor_enhance=True,
    ):
        super(BrainBeacon, self).__init__()
        self.use_esm_emb = use_esm_emb
        self.use_pos_emb = use_pos_emb
        self.use_density_emb = use_density_emb
        self.density_token_idx = density_token_idx
        # self.embedding = GeneEmbedding(n_tokens, n_connect_comp, n_rna_type, n_neighbor, dim_model, n_aux)
        self.embedding = GeneEmbedding(
            n_tokens, n_connect_comp, n_rna_type, n_neighbor, dim_model, n_aux,
            use_gene_id_emb=use_gene_id_emb,
            use_homo_emb=use_homo_emb,
            use_rna_type_emb=use_rna_type_emb
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model, nhead=nheads, dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=1e-12,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        # Attention hooks
        self._attention_weights = []
        self._attention_hooks = []
        self.loss = nn.CrossEntropyLoss()
        self.classifier_head = nn.Linear(dim_model, n_tokens + n_aux, bias=False)
        bias = nn.Parameter(torch.zeros(n_tokens + n_aux))  # each token has its own bias
        self.classifier_head.bias = bias
        self.esm_embedding_projection = nn.Linear(esm_embedding_dim, dim_model)
        if neighbor_enhance:
            self.neighbor_projection = nn.Embedding(num_embeddings=6, embedding_dim=dim_model)
            # self.neighbor_layer_norm = nn.LayerNorm(dim_model)

        self.positional_embedding = nn.Embedding(num_embeddings=1000, embedding_dim=dim_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pos = torch.arange(0, 1000, dtype=torch.long)
        self.neighbor_enhance = neighbor_enhance

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

    def enable_attention_hooks(self, target_layers=None):
        """Register forward hooks to capture attention weights from encoder layers.

        Args:
            target_layers: Which layers to capture attention from. Options:
                - None: all layers (default, but uses more memory)
                - "last": only the last layer
                - [0, -1]: list of layer indices (negative indexing supported)
                - {0, 15}: set of layer indices

        After calling this method, attention weights will be captured during forward pass.
        Use `get_attention_weights()` to retrieve them and `clear_attention_weights()` to reset.

        Example usage during inference:
        ```python
        model = BrainBeacon(...)
        model.load_state_dict(torch.load("checkpoint.pt")["model_state_dict"])
        model.eval()

        # Enable attention capture for last layer only (recommended for memory efficiency)
        model.enable_attention_hooks(target_layers="last")

        with torch.no_grad():
            # Forward pass
            predictions = model(
                x_gene_id, x_connect_id, x_rna_type,
                attention_mask, esm_embedding, neighbor_gene_distribution
            )

            # Get attention weights from all layers
            attention_weights = model.get_attention_weights()
            # Returns: [(0, tensor), (1, tensor), ..., (nlayers-1, tensor)]
            # Each tensor shape: (batch_size, nheads, seq_len, seq_len)

            # Example: get attention from last layer, averaged across heads
            last_layer_attn = attention_weights[-1][1]  # (batch, nheads, seq, seq)
            avg_attn = last_layer_attn.mean(dim=1)      # (batch, seq, seq)

            # Example: visualize attention for first sample, first head
            import matplotlib.pyplot as plt
            plt.imshow(attention_weights[0][1][0, 0].cpu().numpy())
            plt.title("Layer 0, Head 0 Attention")
            plt.colorbar()
            plt.show()

        # Clean up
        model.clear_attention_weights()
        model.disable_attention_hooks()
        ```
        """
        self.disable_attention_hooks()  # Clear any existing hooks first
        self._attention_weights = []
        self._in_hook = False  # Flag to prevent recursive hook calls

        # Determine which layers to hook
        nlayers = len(self.encoder.layers)
        if target_layers is None:
            layers_to_hook = set(range(nlayers))
        elif target_layers == "last":
            layers_to_hook = {nlayers - 1}
        elif isinstance(target_layers, (list, tuple)):
            # Handle negative indices
            layers_to_hook = {(l if l >= 0 else nlayers + l) for l in target_layers}
        elif isinstance(target_layers, set):
            layers_to_hook = {(l if l >= 0 else nlayers + l) for l in target_layers}
        else:
            raise ValueError(f"Invalid target_layers: {target_layers}")

        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # Prevent recursive calls when we re-run attention inside the hook
                if self._in_hook:
                    return output

                self._in_hook = True
                try:
                    # Re-run attention with need_weights=True to get attention scores
                    query = args[0] if len(args) > 0 else kwargs.get('query')
                    key = args[1] if len(args) > 1 else kwargs.get('key', query)
                    value = args[2] if len(args) > 2 else kwargs.get('value', query)
                    key_padding_mask = kwargs.get('key_padding_mask', None)
                    attn_mask = kwargs.get('attn_mask', None)

                    with torch.no_grad():
                        _, attn_weights = module(
                            query, key, value,
                            key_padding_mask=key_padding_mask,
                            attn_mask=attn_mask,
                            need_weights=True,
                            average_attn_weights=False  # Return per-head weights: (batch, nheads, seq, seq)
                        )
                    self._attention_weights.append((layer_idx, attn_weights.detach()))
                finally:
                    self._in_hook = False
                return output
            return hook_fn

        # Only register hooks for target layers
        for idx, layer in enumerate(self.encoder.layers):
            if idx in layers_to_hook:
                hook = layer.self_attn.register_forward_hook(make_hook(idx), with_kwargs=True)
                self._attention_hooks.append(hook)

    def disable_attention_hooks(self):
        """Remove all registered attention hooks."""
        for hook in self._attention_hooks:
            hook.remove()
        self._attention_hooks = []

    def get_attention_weights(self):
        """Get captured attention weights from the last forward pass.

        Returns:
            list of tuples: [(layer_idx, attention_weights), ...]
                attention_weights shape: (batch, nheads, seq_len, seq_len)
        """
        return self._attention_weights

    def clear_attention_weights(self):
        """Clear stored attention weights to free memory."""
        self._attention_weights = []

    def encode(self, x_gene_id, x_connect_id, x_rna_type, attention_mask, esm_embedding, neighbor_gene_distribution):
        token_embedding = self.embedding(x_gene_id, x_connect_id, x_rna_type)
        if self.use_esm_emb:
            token_embedding += self.esm_embedding_projection(esm_embedding)
        if self.neighbor_enhance:
            neighbor_embedding = self.neighbor_projection(neighbor_gene_distribution)
            token_embedding += neighbor_embedding
        if self.use_pos_emb:
            pos = self.pos.to(token_embedding.device)
            pos_embedding = self.positional_embedding(pos)  # batch x (n_tokens) x dim_model
            embeddings = self.dropout(token_embedding + pos_embedding)
        else:
            embeddings = self.dropout(token_embedding)
        # Zero out density token embedding if disabled
        if not self.use_density_emb:
            embeddings[:, self.density_token_idx, :] = 0
        return self.encoder(embeddings, src_key_padding_mask=attention_mask)

    def forward(self, x_gene_id, x_connect_id, x_rna_type, attention_mask, esm_embedding, neighbor_gene_distribution):
        transformer_output = self.encode(
            x_gene_id,
            x_connect_id,
            x_rna_type,
            attention_mask,
            esm_embedding,
            neighbor_gene_distribution,
        )
        prediction = self.classifier_head(transformer_output)
        return prediction


def train_one_epoch(model, dataloader, optimizer, criterion, device, rank, writer, esm_embedding_map, global_step,
                    logger, logdir, epoch, lr_scheduler=None, max_steps=None, accumulation_steps=1):
    model.train()
    total_loss = 0.0
    accum_loss = 0.0
    scaler = GradScaler()
    optimizer.zero_grad()
    for micro_step, (masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution) in \
            enumerate(tqdm(dataloader)):
        masked_indices = masked_indices[0]
        mask = mask[0]
        real_indices = real_indices[0]
        attention_mask = attention_mask[0]
        connect_comp = connect_comp[0]
        rna_type = rna_type[0]
        neighbor_gene_distribution = neighbor_gene_distribution[0].long()

        real_indices_view = real_indices.view(-1).long()
        esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
        esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])

        masked_indices, attention_mask, connect_comp, rna_type, esm_embedding, real_indices, \
            neighbor_gene_distribution = masked_indices.to(device), attention_mask.to(device), \
            connect_comp.to(device), rna_type.to(device), esm_embedding.to(device), real_indices.to(device), \
            neighbor_gene_distribution.to(device)

        # Skip DDP gradient sync on non-update steps for efficiency
        is_update_step = (micro_step + 1) % accumulation_steps == 0
        context = model.no_sync if (not is_update_step and hasattr(model, 'no_sync')) else nullcontext
        with context():
            with autocast():
                mlm_predictions = model(
                    masked_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution
                )
                # real_indices = torch.where(mask == 1, real_indices, torch.tensor(-100, dtype=torch.long)).type(
                #     torch.int64)
                # real_indices = real_indices.to(device)
                mlm_predictions = mlm_predictions * (1 - attention_mask.unsqueeze(-1).float())
                mask_pos = mask == 0
                loss = criterion(mlm_predictions[mask_pos].reshape(-1, mlm_predictions.shape[-1]),
                                 real_indices[mask_pos].reshape(-1))
            scaler.scale(loss / accumulation_steps).backward()

        accum_loss += loss.item()

        if is_update_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += accum_loss / accumulation_steps
            global_step += 1
            if rank == 0:
                print(f"loss {accum_loss / accumulation_steps:.4f}")

            accum_loss = 0.0

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
                    checkpoint_path = os.path.join(logdir, f"epoch_{epoch}_step_{global_step}.pt")
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
            if max_steps is not None and global_step >= max_steps:
                if rank == 0:
                    if logger:
                        logger.info(f"Reached max_steps={max_steps} at global_step={global_step}, stopping epoch early.")
                break

    # Handle remaining accumulated gradients at end of epoch
    if (micro_step + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_loss += accum_loss / accumulation_steps
        global_step += 1

    return total_loss / max(len(dataloader) // accumulation_steps, 1), global_step


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
