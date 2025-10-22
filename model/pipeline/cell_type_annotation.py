import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import joblib
from torch.utils.data import Dataset
from typing import List, Union
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch import nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from model.brain_beacon import BrainBeacon

class MLPHead(nn.Module):
    def __init__(self, dim_model, num_class, hidden=512, p_drop=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_model, hidden),
            nn.BatchNorm1d(hidden),      
            nn.GELU(),                
            nn.Dropout(p_drop),

            nn.Linear(hidden, hidden//2),
            nn.BatchNorm1d(hidden//2),
            nn.GELU(),
            nn.Dropout(p_drop),

            nn.Linear(hidden//2, num_class)
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.head(x)


def masked_mean_pooling(transformer_output, mask):
    mask = mask.unsqueeze(-1)
    masked_output = transformer_output * mask
    valid_length = mask.sum(dim=1, keepdim=False)
    valid_length = torch.clamp(valid_length, min=1)
    mean_pooled = masked_output.sum(dim=1, keepdim=False) / valid_length  # (b, d)
    return mean_pooled

def masked_weighted_pooling_fixL(transformer_output, mask, rank_weight_mode="softmax", weight_decay=0.998, temperature=300.0):
    """
    Args:
        transformer_output: (B, L, D)
        mask: (B, L)
        rank_weight_mode: "none", "exp", "linear", or "softmax"
        weight_decay: only used if rank_weight_mode == "exp"
        temperature: only used if rank_weight_mode == "softmax"
    """
    mask = mask.unsqueeze(-1).float()  # (B, L, 1)
    B, L, D = transformer_output.shape

    if rank_weight_mode == "exp":
        rank_weights = torch.tensor([weight_decay ** i for i in range(L)], device=transformer_output.device)
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    elif rank_weight_mode == "linear":
        rank_weights = 1.0 - torch.arange(L, device=transformer_output.device).float() / L
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    elif rank_weight_mode == "softmax":
        rank_scores = -torch.arange(L, device=transformer_output.device).float()
        rank_weights = torch.softmax(rank_scores / temperature, dim=0) * L  # scaled softmax
        rank_weights = rank_weights.unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
    else:
        rank_weights = None

    if rank_weights is not None:
        weighted_mask = mask * rank_weights  # (B, L, 1)
        masked_output = transformer_output * weighted_mask
        valid_length = weighted_mask.sum(dim=1).clamp(min=1e-6)  # (B, 1)
    else:
        masked_output = transformer_output * mask
        valid_length = mask.sum(dim=1).clamp(min=1.0)  # (B, 1)

    mean_pooled = masked_output.sum(dim=1) / valid_length  # (B, D)
    return mean_pooled

def masked_weighted_pooling(
    transformer_output,
    mask,
    expr_weights=None,
    weight_mode="expr",
    weight_decay=0.998,
    temperature=300.0,
):
    """
    Generalized pooling with rank-based or expression-based weighting.

    Args:
        transformer_output: (B, L, D)
        mask: (B, L)
        exp: (B, L), required for expression-based modes
        weight_mode: one of ["none", "linear", "expdecay", "softmax", "expression"]
    Returns:
        (B, D)
    """
    mask = mask.float().unsqueeze(-1)  # (B, L, 1)
    B, L, D = transformer_output.shape

    # === Expression-only ===
    if weight_mode == "expression":
        if expr_weights is None:
            raise ValueError("expression must be provided when weight_mode='expression'")
        weights = expr_weights.unsqueeze(-1) * mask
        weighted_output = transformer_output * weights
        weight_sum = weights.sum(dim=1).clamp(min=1e-6)
        return weighted_output.sum(dim=1) / weight_sum

    # === Plain average ===
    if weight_mode == "none":
        weighted_output = transformer_output * mask
        weight_sum = mask.sum(dim=1).clamp(min=1e-6)
        return weighted_output.sum(dim=1) / weight_sum

    # === Rank-based ===
    valid_lengths = mask.squeeze(-1).sum(dim=1).long()
    rank_weights_list = []
    for i in range(B):
        l_i = valid_lengths[i].item()
        if weight_mode == "expdecay":
            weights = torch.tensor([weight_decay ** r for r in range(l_i)], device=transformer_output.device)
        elif weight_mode == "linear":
            weights = 1.0 - torch.arange(l_i, device=transformer_output.device).float() / l_i
        elif weight_mode == "softmax":
            scores = -torch.arange(l_i, device=transformer_output.device).float()
            weights = torch.softmax(scores / temperature, dim=0) * l_i
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")
        padded = torch.zeros(L, device=transformer_output.device)
        padded[:l_i] = weights
        rank_weights_list.append(padded)

    rank_weights = torch.stack(rank_weights_list).unsqueeze(-1)  # (B, L, 1)
    weighted_mask = mask * rank_weights
    masked_output = transformer_output * weighted_mask
    weight_sum = weighted_mask.sum(dim=1).clamp(min=1e-6)
    return masked_output.sum(dim=1) / weight_sum



class BrainBeaconCellType(nn.Module):
    def __init__(self, model_config, num_class):
        super().__init__()
        self.model_config = model_config
        self.num_class = num_class
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
        self.classification_head = nn.Sequential(
            nn.Linear(self.model_config["dim_model"], 512),   
            nn.LeakyReLU(),           
            nn.Dropout(p=0.1),  
            nn.Linear(512, 256),   
            nn.LeakyReLU(),           
            nn.Dropout(p=0.1),  
            nn.Linear(256, self.num_class)
        )
        # self.classification_head = self.classification_head = nn.Linear(self.model_config["dim_model"], self.num_class)
        # self.classification_head = MLPHead(
        #     dim_model=self.model_config["dim_model"],
        #     num_class=self.num_class,
        #     hidden=512,
        #     p_drop=0.3
        # )
        

    def forward(self, real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask, exp, return_embedding_only=False, pooling_mode="other"):
        token_embedding = self.pretrain_model.embedding(real_indices, connect_comp, rna_type)
        token_embedding += self.pretrain_model.esm_embedding_projection(esm_embedding)
        if self.model_config['neighbor_enhance']:
            neighbor_embedding = self.pretrain_model.neighbor_projection(neighbor_gene_distribution)
            token_embedding += neighbor_embedding
        pos = self.pretrain_model.pos.to(token_embedding.device)
        pos_embedding = self.pretrain_model.positional_embedding(pos)  # batch x (n_tokens) x dim_model
        embeddings = self.pretrain_model.dropout(token_embedding + pos_embedding)
        output = self.pretrain_model.encoder(embeddings, src_key_padding_mask=attention_mask)

        if pooling_mode == "mean":
            transformer_output = masked_mean_pooling(output[:, 2:, :], sequence_mask[:, 2:])
        else:
            pool_skip_tokens = self.model_config.get("pool_skip_tokens", 2)
            weight_mode = self.model_config.get("weight_mode", "none")

            if weight_mode == "expression":
                cd_weight = self.model_config.get("cd_weight", 0.05)
                aux = torch.zeros((exp.shape[0], 2), device=exp.device)  # species + platform
                cd = torch.full((exp.shape[0], 1), cd_weight, device=exp.device)  # cell_density
                gene_expr = exp[:, 3:]  # actual gene tokens
                gene_expr = gene_expr / gene_expr.sum(dim=1, keepdim=True).clamp(min=1e-6)
                exp = torch.cat([aux, cd, gene_expr], dim=1)  # shape (B, L)
                expr_weights = exp[:, pool_skip_tokens:]
            else:
                expr_weights = None
            transformer_output = masked_weighted_pooling(
                output[:, pool_skip_tokens:, :],
                sequence_mask[:, pool_skip_tokens:],
                expr_weights=expr_weights,
                weight_mode=weight_mode,
                weight_decay=self.model_config.get("weight_decay", 0.998),
                temperature=self.model_config.get("temperature", 300)
            )
        if return_embedding_only:
            return transformer_output  
        else:
            return self.classification_head(transformer_output)

class FinetuneJoblibDataset(Dataset):
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
            cell_raw_index_files,
            cell_labels_files,
            exp_files
    ):
        self.masked_indices_files = masked_indices_files
        self.mask_files = mask_files
        self.real_indices_files = real_indices_files
        self.attention_mask_files = attention_mask_files
        self.connect_comp_files = connect_comp_files
        self.rna_type_files = rna_type_files
        self.neighbor_gene_distribution_files = neighbor_gene_distribution_files
        self.file_prefix_list = file_prefix_list
        self.cell_raw_index_files = cell_raw_index_files
        self.cell_labels_files = cell_labels_files
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
            neighbor_gene_distribution_file = self.neighbor_gene_distribution_files[file_idx]
            cell_raw_index_file = self.cell_raw_index_files[file_idx]
            cell_labels_files = self.cell_labels_files[file_idx]
            exp_file = self.exp_files[file_idx]
            real_indices = joblib.load(real_indices_file)[sample_idx]
            attention_mask = joblib.load(attention_mask_file)[sample_idx]
            connect_comp = joblib.load(connect_comp_file)[sample_idx]
            rna_type = joblib.load(rna_type_file)[sample_idx]
            neighbor_gene_distribution = joblib.load(neighbor_gene_distribution_file)[sample_idx]
            exp = joblib.load(exp_file)[sample_idx]
            try:
                cell_raw_idx = joblib.load(cell_raw_index_file)[sample_idx]
                if isinstance(cell_raw_idx, np.ndarray):
                    cell_raw_idx = cell_raw_idx.tolist() if cell_raw_idx.ndim == 1 else [str(x) for x in cell_raw_idx]
                elif isinstance(cell_raw_idx, (list, tuple)):
                    cell_raw_idx = [str(x) for x in cell_raw_idx]
                else:
                    cell_raw_idx = [str(cell_raw_idx)]
            except Exception as e:
                print(f"Error processing cell_raw_idx: {e}")
                cell_raw_idx = ["unknown"]
            

            try:
                cell_labels = joblib.load(cell_labels_files)[sample_idx]

                if isinstance(cell_labels, (list, np.ndarray)):

                    try:
                        if isinstance(cell_labels, np.ndarray) and cell_labels.dtype == np.dtype('O'):

                            cell_labels = np.array([int(x) if isinstance(x, (int, float, np.number)) 
                                                    else 0 for x in cell_labels], dtype=np.int64)
                        else:
                            cell_labels = np.array(cell_labels, dtype=np.int64)
                    except (ValueError, TypeError):
                        print(f"Warning: cannot convert cell_labels to int64, using zeros. Type: {type(cell_labels)}")
                        if hasattr(cell_labels, '__len__'):
                            cell_labels = np.zeros(len(cell_labels), dtype=np.int64)
                        else:
                            cell_labels = np.array([0], dtype=np.int64)
                else:

                    try:
                        cell_labels = np.array([int(cell_labels)], dtype=np.int64)
                    except (ValueError, TypeError):
                        print(f"Warning: cannot convert single cell_label to int, using 0. Value: {cell_labels}")
                        cell_labels = np.array([0], dtype=np.int64)
            except Exception as e:
                print(f"Error loading cell_labels: {e}")

                cell_labels = np.array([0], dtype=np.int64)
            
            if real_indices is None or attention_mask is None or connect_comp is None or rna_type is None:
                print(self.file_prefix_list[idx])
                print(real_indices, attention_mask, connect_comp, rna_type)
                
            return (
                torch.as_tensor(real_indices[:, :1000], dtype=torch.long),
                torch.as_tensor(attention_mask[:, :1000], dtype=torch.bool),
                torch.as_tensor(connect_comp[:, :1000], dtype=torch.long),
                torch.as_tensor(rna_type[:, :1000], dtype=torch.long),
                torch.as_tensor(neighbor_gene_distribution[:, :1000], dtype=torch.float),
                cell_raw_idx,
                torch.as_tensor(cell_labels, dtype=torch.long),
                torch.as_tensor(exp[:, :1000], dtype=torch.float)
            )
        except Exception as e:
            print(f"Error in FinetuneJoblibDataset.__getitem__: {e}")
            print(f"Index: {idx}, file: {self.file_prefix_list[file_idx] if file_idx < len(self.file_prefix_list) else 'index_out_of_range'}, sample: {sample_idx}")
            
            if idx + 1 >= self.total_length:
                empty_tensor = torch.zeros((1, 1000), dtype=torch.long)
                empty_bool_tensor = torch.zeros((1, 1000), dtype=torch.bool)
                return (
                    empty_tensor, 
                    empty_bool_tensor,
                    empty_tensor, 
                    empty_tensor, 
                    empty_tensor, 
                    ["unknown"],
                    torch.zeros(1, dtype=torch.long)
                )
            else:
                return self.__getitem__(idx + 1)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        num_classes = pred.size(1)
        one_hot = F.one_hot(target, num_classes).float()
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / num_classes
        log_prob = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prob).sum(dim=1).mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape (num_classes,) or None

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        ce = F.nll_loss(logp, targets, reduction="none", weight=self.alpha)
        loss = ((1 - p.gather(1, targets.unsqueeze(1)).squeeze()) ** self.gamma) * ce
        return loss.mean()

class CellTypeAnnotationPipeline():
    def __init__(self, pretrain_ckpt: str, model_config: dict, device: Union[str, torch.device] = 'cpu', num_class=7):
        """
        Initialize the pipeline with model_raw and device settings.
        """
        self.device = device
        self.model_config = model_config
        self.model = None
        self.pretrain_ckpt: str = pretrain_ckpt
        self.num_class = num_class
        self.initialize_model()
        self.fitted = False 

    def initialize_model(self):
        """
        Initialize the model_raw and compute its size.
        """
        self.model = BrainBeaconCellType(self.model_config, self.num_class).to(self.device)
        
        if self.pretrain_ckpt:
            try:
                ckpt = torch.load(self.pretrain_ckpt, map_location=self.device)
                if "best_model" in self.pretrain_ckpt.lower():

                    self.model.load_state_dict(ckpt['model_state_dict'])
                else:

                    self.model.pretrain_model.load_state_dict(ckpt['model_state_dict'])
                print(f"Loaded pretrain_model checkpoint: {self.pretrain_ckpt}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                raise

        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Model size: {:.3f}MB'.format(size_all_mb))
        
    def load_dataset_anno(self, data_paths: List[str]):
        """
        Load the dataset from the given paths.
        """
        masked_indices_files_list = []
        mask_files_list = []
        real_indices_files_list = []
        attention_mask_files_list = []
        connect_comp_files_list = []
        rna_type_files_list = []
        neighbor_gene_distribution_files_list = []
        cell_raw_index_list = []
        cell_labels_list = []
        file_prefix_list = []
        exp_files_list = []
        self.data_paths = data_paths

        for prefix in os.listdir(data_paths):
            if prefix.endswith((".parquet", ".pkl", "pth")):
                continue
            file_prefix_list.append(os.path.join(data_paths, prefix))
            for file in os.listdir(os.path.join(data_paths, prefix)):
                file_path = os.path.join(data_paths, prefix, file)
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
                elif 'neighbor_gene_distribution_' in file:
                    neighbor_gene_distribution_files_list.append(file_path)
                elif 'mask_' in file:
                    mask_files_list.append(file_path)
                elif "cell_raw_index" in file:
                    cell_raw_index_list.append(file_path)
                elif "cell_labels" in file:
                    cell_labels_list.append(file_path)
                elif "exp" in file:
                    exp_files_list.append(file_path)
        # print("masked_indices: ", masked_indices_files_list)

        dataset = FinetuneJoblibDataset(
            masked_indices_files_list,
            mask_files_list,
            real_indices_files_list,
            attention_mask_files_list,
            connect_comp_files_list,
            rna_type_files_list,
            neighbor_gene_distribution_files_list,
            file_prefix_list,
            cell_raw_index_list,
            cell_labels_list,
            exp_files_list
        )
        return dataset
    
    def train(self, model, dataloader, optimizer, criterion, device, esm_embedding_map):
        model.train()
        total_loss = 0.0
        total_real_loss = 0.0
        scaler = GradScaler()        
        for real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution, _, cell_labels, exp in dataloader:
            real_indices = real_indices[0]
            attention_mask = attention_mask[0]
            connect_comp = connect_comp[0]
            rna_type = rna_type[0]
            neighbor_gene_distribution = neighbor_gene_distribution[0].long()
            real_indices_view = real_indices.view(-1).long()
            cell_labels = cell_labels[0].to(device)
            exp = exp[0]
            # Ensure correct device for indexing
            real_indices_view = real_indices_view.to(self.esm_embedding_map.device)
            esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
            esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])
            
            sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices))
            attention_mask, connect_comp, rna_type, neighbor_gene_distribution, esm_embedding, real_indices, sequence_mask, exp = \
                attention_mask.to(device), connect_comp.to(device), rna_type.to(device), \
                    neighbor_gene_distribution.to(device), esm_embedding.to(device), real_indices.to(device), sequence_mask.to(device), exp.to(device)
            
            with autocast():
                class_output = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask, exp, pooling_mode="other")   
                classification_loss = criterion(class_output, cell_labels)
                loss = classification_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

            with torch.no_grad():
                real_loss = criterion(class_output, cell_labels)
                total_real_loss += real_loss.item()
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, criterion, esm_embedding_map):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        
        with torch.no_grad():
            for real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution, _, cell_labels, exp in dataloader:
                real_indices = real_indices[0]
                attention_mask = attention_mask[0]
                connect_comp = connect_comp[0]
                rna_type = rna_type[0]
                neighbor_gene_distribution = neighbor_gene_distribution[0].long()
                real_indices_view = real_indices.view(-1).long()
                cell_labels = cell_labels[0].to(self.device)
                exp = exp[0]
                # Ensure correct device for indexing
                real_indices_view = real_indices_view.to(self.esm_embedding_map.device)
                esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1])
                
                sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices))
                attention_mask, connect_comp, rna_type, neighbor_gene_distribution, esm_embedding, real_indices, sequence_mask, exp = \
                    attention_mask.to(self.device), connect_comp.to(self.device), rna_type.to(self.device), \
                        neighbor_gene_distribution.to(self.device), esm_embedding.to(self.device), real_indices.to(self.device), sequence_mask.to(self.device), exp.to(self.device)

                class_output = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask, exp)
                classification_loss = criterion(class_output, cell_labels)
                total_loss += classification_loss.item()
                
                preds = torch.argmax(class_output, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(cell_labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, acc, f1
    
    def fit(self, data_paths, config_train: dict, test_data_paths=None, patience=10):
        """
        Finetune model_raw use traindataset.
        """

        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        self.esm_embedding_map = esm_embedding_map
        dataset = self.load_dataset_anno(data_paths)
        # testset = self.load_dataset_anno(test_data_paths)
        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=config_train["batch_size"], shuffle=True, num_workers=4,
                                prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4,
                                prefetch_factor=2)
        # test_loader = DataLoader(testset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4,
                                # prefetch_factor=2)
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = FocalLoss(gamma=2.0)
        # criterion = LabelSmoothingLoss(smoothing=0.4)
        best_model_path = os.path.join(os.path.dirname(data_paths), "best_model.pth")
        pretrain_model_path = os.path.join(os.path.dirname(data_paths), "pretrain_ckpt_new.pth")

        best_val_acc = 0.0
        best_val_f1 = 0

        # pretrain parameters
        for name, param in self.model.pretrain_model.named_parameters():
            if name.startswith('embedding'):
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4,
            weight_decay=1e-8
        )
        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=5)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(3, config_train["max_epoch"] - 10))

        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
        epochs_without_improvement = 0
        for epoch in range(0, config_train["max_epoch"]):
            epoch_loss = self.train(self.model, train_loader, optimizer, criterion, self.device, esm_embedding_map)
            logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss}, learning rate: {optimizer.param_groups[0]['lr']}")
            # val_report = self.validate(val_loader, criterion, esm_embedding_map)
            val_loss, val_acc, val_f1 = self.validate(val_loader, criterion, esm_embedding_map)
            logging.info(f"Validation Loss: {val_loss}, Accuracy: {val_acc}, F1 Score: {val_f1}")

            scheduler.step()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_macro_val_f1 = val_f1
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, best_model_path)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.pretrain_model.state_dict()
                }, pretrain_model_path)

                logging.info(
                    f"Best model_raw saved at {best_model_path} and pretrain model_raw saved at {pretrain_model_path} (epoch {epoch + 1})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience and epoch >= 20:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
        try:
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Rolled back to best model_raw parameters from checkpoint: {best_model_path}")
        except Exception as e:
            logging.error(f"Error rolling back to best model_raw: {e}")
        self.fitted = True
        return self
    
    
    
    def fit_raw(self, data_paths, config_train: dict, test_data_paths=None, patience=5):
        """
        Finetune model_raw use traindataset.
        """
        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        self.esm_embedding_map = esm_embedding_map
        dataset = self.load_dataset_anno(data_paths)

        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=config_train["batch_size"], shuffle=True, num_workers=4, prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        criterion = LabelSmoothingLoss(smoothing=0.4)
        best_model_path = os.path.join(os.path.dirname(self.data_paths), "best_model.pth")
        pretrain_model_path = os.path.join(os.path.dirname(self.data_paths), "pretrain_ckpt_new.pth")
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0

        logging.info("Stage 1: Training classification head only (pretrain model_raw frozen)")
        
        for param in self.model.pretrain_model.parameters():
            param.requires_grad = False
        
        for name, param in self.model.classification_head.named_parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-5)

        for epoch in range(10):
            epoch_loss = self.train(self.model, train_loader, optimizer, criterion, self.device, esm_embedding_map)
            logging.info(f"Stage 1 - Epoch {epoch + 1}, Loss: {epoch_loss}")
            val_loss, val_acc, val_f1 = self.validate(val_loader, criterion, esm_embedding_map)
            logging.info(f"Stage 1 - Validation Loss: {val_loss}, Accuracy: {val_acc}, F1 Score: {val_f1}")

            if val_acc > best_val_acc and val_f1 > best_val_f1:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, best_model_path)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.pretrain_model.state_dict()
                }, pretrain_model_path)
                
                logging.info(f"Stage 1 - Best model_raw saved at {best_model_path} and pretrain model_raw saved at {pretrain_model_path} (epoch {epoch + 1})")
        

        logging.info("Stage 2: Fine-tuning entire model_raw with small learning rate")

        for name, param in self.model.pretrain_model.named_parameters():
            if "embedding" in name or "positional" in name or "projection" in name:
                param.requires_grad = False
            else:

                param.requires_grad = True

        encoder_layers = list(self.model.pretrain_model.encoder.layers)
        num_layers = len(encoder_layers)
        optimizer = torch.optim.AdamW([

            {'params': [p for i in range(min(2, num_layers)) 
                      for n, p in self.model.pretrain_model.encoder.layers[i].named_parameters() 
                      if p.requires_grad], 'lr': 2e-6},

            {'params': [p for i in range(2, min(6, num_layers)) 
                      for n, p in self.model.pretrain_model.encoder.layers[i].named_parameters() 
                      if p.requires_grad], 'lr': 5e-6},

            {'params': [p for i in range(6, num_layers) 
                      for n, p in self.model.pretrain_model.encoder.layers[i].named_parameters() 
                      if p.requires_grad], 'lr': 1e-5},

            {'params': self.model.classification_head.parameters(), 'lr': 1e-4}
        ])

        epochs_without_improvement = 0
        for epoch in range(10, config_train["max_epoch"]):
            epoch_loss = self.train(self.model, train_loader, optimizer, criterion, self.device, esm_embedding_map)
            logging.info(f"Stage 2 - Epoch {epoch + 1}, Loss: {epoch_loss}")
            val_loss, val_acc, val_f1 = self.validate(val_loader, criterion, esm_embedding_map)
            logging.info(f"Stage 2 - Validation Loss: {val_loss}, Accuracy: {val_acc}, F1 Score: {val_f1}")

            if val_acc > best_val_acc and val_f1 > best_val_f1:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, best_model_path)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.pretrain_model.state_dict()
                }, pretrain_model_path)
                
                logging.info(f"Stage 2 - Best model_raw saved at {best_model_path} and pretrain model_raw saved at {pretrain_model_path} (epoch {epoch + 1})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience and epoch >= 30:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        try:
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"Rolled back to best model_raw parameters from checkpoint: {best_model_path}")
        except Exception as e:
            logging.error(f"Error rolling back to best model_raw: {e}")
        
        self.fitted = True
        return self
        
    def _infer(self, dataloader):
        """
        Run inference on new data using the pretrained model_raw.
        """
        # Ensure self.esm_embedding_map is loaded and on the correct device
        if not hasattr(self, 'esm_embedding_map') or self.esm_embedding_map is None:
            raise ValueError("ESM embedding map is not loaded. Please call fit() or load the embedding map first.")
            
        self.model.eval()
        indexed_pred = []
        class_outputs = []  # Changed from class_output to class_outputs and initialized as list
        with torch.no_grad():
            for real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution, cell_raw_idx, _, exp in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
                real_indices = real_indices[0].to(self.device)
                attention_mask = attention_mask[0].to(self.device)
                connect_comp = connect_comp[0].to(self.device)
                rna_type = rna_type[0].to(self.device)
                neighbor_gene_distribution = neighbor_gene_distribution[0].long().to(self.device)
                real_indices_view = real_indices.view(-1).long()
                exp = exp[0]
                # Get esm_embedding and ensure it's on the correct device
                real_indices_view = real_indices_view.to(self.esm_embedding_map.device)
                esm_embedding = torch.index_select(self.esm_embedding_map, dim=0, index=real_indices_view)
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1]).to(self.device)
                
                sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices))
                # Step 2: Move data to device
                real_indices, attention_mask, connect_comp, rna_type, esm_embedding, sequence_mask, neighbor_gene_distribution, exp = (
                    real_indices.to(self.device), attention_mask.to(self.device), connect_comp.to(self.device),
                    rna_type.to(self.device), esm_embedding.to(self.device), sequence_mask.to(self.device), neighbor_gene_distribution.to(self.device), exp.to(self.device)
                )
                # Step 3: Forward pass
                class_predict = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask, exp, pooling_mode="other")
                class_pred = torch.argmax(class_predict, dim=-1)
                assert len(cell_raw_idx) == class_pred.shape[0], "Batch size mismatch"
                # Collect indexed embeddings
                indexed_pred.extend(zip(cell_raw_idx, class_pred))
                class_outputs.extend(zip(cell_raw_idx, class_predict.detach().cpu()))  # Changed to class_outputs

        return indexed_pred, class_outputs  # Return class_outputs instead of class_output
    
    def predict(self, data_paths: List[str], config_train: dict):
        """
        Main method to run the entire training pipeline.
        """
        # Ensure self.esm_embedding_map is loaded and on the correct device
        if not hasattr(self, 'esm_embedding_map') or self.esm_embedding_map is None:
            print(f"Loading ESM embedding map from {config_train['esm_embedding_path']}")
            self.esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location=self.device)
            
        dataset = self.load_dataset_anno(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        indexed_pred, class_output = self._infer(data_loader)
        return indexed_pred, class_output
    
    def extract_embedding(self, data_paths: List[str], config_train: dict):
        # Ensure self.esm_embedding_map is loaded and on the correct device
        if not hasattr(self, 'esm_embedding_map') or self.esm_embedding_map is None:
            print(f"Loading ESM embedding map from {config_train['esm_embedding_path']}")
            self.esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location=self.device)
        
        dataset = self.load_dataset_anno(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        
        self.model.eval()
        indexed_embedding = []
        with torch.no_grad():
            for real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution, cell_raw_idx, _, exp in tqdm(data_loader):
                real_indices = real_indices[0].to(self.device)
                attention_mask = attention_mask[0].to(self.device)
                connect_comp = connect_comp[0].to(self.device)
                rna_type = rna_type[0].to(self.device)
                neighbor_gene_distribution = neighbor_gene_distribution[0].long().to(self.device)
                real_indices_view = real_indices.view(-1).long()
                exp = exp[0].to(self.device)
                # Get esm_embedding and ensure it's on the correct device
                real_indices_view = real_indices_view.to(self.esm_embedding_map.device)
                esm_embedding = torch.index_select(self.esm_embedding_map, dim=0, index=real_indices_view)
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1]).to(self.device)
                
                sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices)).to(self.device)

                embedding = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding, neighbor_gene_distribution, sequence_mask, exp, return_embedding_only=True)
                
                assert len(cell_raw_idx) == embedding.shape[0]
                indexed_embedding.extend(zip(cell_raw_idx, embedding.detach().cpu()))
        
        return indexed_embedding  # list of (cell_id, embedding) tuples
    
    def coral_loss(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _cov(x):
            mean = x.mean(dim=0, keepdim=True)
            xc = x - mean
            return (xc.T @ xc) / (x.size(0) - 1)
        cs, ct = _cov(source), _cov(target)
        return ((cs - ct) ** 2).mean()

    def train_with_coral(self, data_paths, test_data_paths, config_train: dict, lambda_coral=1.0, patience=5):
        self.model.train()
        device = self.device
        scaler = GradScaler()

        self.esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        esm_embedding_map = self.esm_embedding_map

        dataset = self.load_dataset_anno(data_paths)
        test_dataset = self.load_dataset_anno(test_data_paths)

        train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.1, random_state=42)
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=config_train["batch_size"], shuffle=True, num_workers=4, prefetch_factor=2)
        val_loader = DataLoader(val_subset, batch_size=config_train["batch_size"], shuffle=False, num_workers=4, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=config_train["batch_size"], shuffle=True, num_workers=4, prefetch_factor=2)

        criterion = FocalLoss(gamma=2.0)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config_train["lr"])

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_val_f1 = 0.0
        epochs_without_improvement = 0

        best_model_path = os.path.join(os.path.dirname(self.data_paths), "best_model_with_coral.pth")
        pretrain_model_path = os.path.join(os.path.dirname(self.data_paths), "pretrain_ckpt_with_coral.pth")

        for epoch in range(config_train["max_epoch"]):
            self.model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config_train['max_epoch']}", ncols=100)
            test_iter = iter(test_loader)
            total_loss = 0.0

            for batch_train in progress_bar:
                try:
                    batch_test = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    batch_test = next(test_iter)

                real_indices, attention_mask, connect_comp, rna_type, neighbor_gene_distribution, _, cell_labels, exp = batch_train
                real_indices = real_indices[0].to(device)
                attention_mask = attention_mask[0].to(device)
                connect_comp = connect_comp[0].to(device)
                rna_type = rna_type[0].to(device)
                neighbor_gene_distribution = neighbor_gene_distribution[0].long().to(device)
                cell_labels = cell_labels[0].to(device)
                exp = exp[0].to(device)
                real_indices_view = real_indices.view(-1).long().to(esm_embedding_map.device)
                esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
                esm_embedding = esm_embedding.view(real_indices.shape[0], real_indices.shape[1], -1).to(device)
                sequence_mask = torch.where(real_indices == 1, torch.zeros_like(real_indices), torch.ones_like(real_indices)).to(device)

                # with autocast():
                z_sn = self.model(real_indices, connect_comp, rna_type, attention_mask, esm_embedding,
                                neighbor_gene_distribution, sequence_mask, exp, return_embedding_only=True)
                logits = self.model.classification_head(z_sn)
                loss_focal = criterion(logits, cell_labels)

                real_indices_t, attention_mask_t, connect_comp_t, rna_type_t, neighbor_gene_distribution_t, _, _, exp_t = batch_test
                real_indices_t = real_indices_t[0].to(device)
                attention_mask_t = attention_mask_t[0].to(device)
                connect_comp_t = connect_comp_t[0].to(device)
                rna_type_t = rna_type_t[0].to(device)
                neighbor_gene_distribution_t = neighbor_gene_distribution_t[0].long().to(device)
                exp_t = exp_t[0].to(device)
                
                real_indices_view_t = real_indices_t.view(-1).long().to(esm_embedding_map.device)
                esm_embedding_t = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view_t)
                esm_embedding_t = esm_embedding_t.view(real_indices_t.shape[0], real_indices_t.shape[1], -1).to(device)
                sequence_mask_t = torch.where(real_indices_t == 1, torch.zeros_like(real_indices_t), torch.ones_like(real_indices_t)).to(device)

                with torch.no_grad():
                    z_st = self.model(real_indices_t, connect_comp_t, rna_type_t, attention_mask_t, esm_embedding_t,
                                    neighbor_gene_distribution_t, sequence_mask_t, exp_t, return_embedding_only=True)

                loss_coral = self.coral_loss(z_sn, z_st)
                loss = loss_focal + lambda_coral * loss_coral

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                progress_bar.set_postfix({
                    "focal": f"{loss_focal.item():.4f}",
                    "coral": f"{loss_coral.item():.4f}",
                    "total": f"{loss.item():.4f}"
                })

            val_loss, val_acc, val_f1 = self.validate(val_loader, criterion, esm_embedding_map)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            if val_acc > best_val_acc and val_f1 > best_val_f1:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_val_f1 = val_f1
                epochs_without_improvement = 0

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, best_model_path)

                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.pretrain_model.state_dict()
                }, pretrain_model_path)

                print(f"✅ [Epoch {epoch+1}] Best model_raw saved to {best_model_path}")
            else:
                epochs_without_improvement += 1
                print(f"⚠️ [Epoch {epoch+1}] No improvement. Patience counter: {epochs_without_improvement}/{patience}")

            if epochs_without_improvement >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break

        self.fitted = True
        return self