import os
import torch
from torch.utils.data import DataLoader
from typing import Union, List
import random
import numpy as np
from tqdm import tqdm
from model_raw.brain_beacon_refactored import BrainBeacon
from model_raw.brain_beacon_refactored import PretrainJoblibDataset
from model_raw.brain_beacon_refactored import train_one_epoch

class CellEmbeddingPipeline:
    def __init__(self, pretrain_ckpt: str, model_config: dict, device: Union[str, torch.device] = 'cpu'):
        """
        Initialize the pipeline with model_raw and device settings.
        """
        self.device = device
        self.model_config = model_config
        self.model = None
        self.initialize_model()

    def initialize_model(self):
        """
        Initialize the model_raw and compute its size.
        """
        self.model = BrainBeacon(
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
        ).to(self.device)

        # 计算模型大小
        param_size = sum(param.nelement() * param.element_size() for param in self.model.parameters())
        buffer_size = sum(buffer.nelement() * buffer.element_size() for buffer in self.model.buffers())
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('Model size: {:.3f}MB'.format(size_all_mb))

    def load_checkpoint(self, config_train: dict):
        """
        Load a checkpoint if specified in the configuration.
        """
        if config_train['pretrain_ckpt']:
            print(f"Loading checkpoint from {config_train['pretrain_ckpt']}")
            ckpt = torch.load(config_train['pretrain_ckpt'])
            self.model.load_state_dict(ckpt['model_state_dict'])
            return ckpt
        return None

    def load_dataset(self, data_paths: List[str]):
        """
        Load the dataset from the given paths.
        """
        masked_indices_files_list = []
        mask_files_list = []
        real_indices_files_list = []
        attention_mask_files_list = []
        connect_comp_files_list = []
        rna_type_files_list = []
        cell_ids_files_list = []
        file_prefix_list = []
        self.data_paths = data_paths

        for prefix in os.listdir(data_paths):
            if prefix.endswith(".parquet"):
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
                elif 'cell_ids_' in file:
                    cell_ids_files_list.append(file_path)
                elif 'mask_' in file:
                    mask_files_list.append(file_path)

        dataset = PretrainJoblibDataset(
            masked_indices_files_list,
            mask_files_list,
            real_indices_files_list,
            attention_mask_files_list,
            connect_comp_files_list,
            rna_type_files_list,
            cell_ids_files_list,
            file_prefix_list
        )
        return dataset

    def infer(self, dataloader, config_train: dict):
        """
        Run inference on new data using the pretrained model_raw.
        """
        # Load checkpoint
        ckpt = self.load_checkpoint(config_train)
        if not ckpt:
            raise ValueError("Checkpoint file is missing. Please provide a valid checkpoint.")
        
        # Switch to evaluation mode
        self.model.eval()

        # Load ESM embedding map
        esm_embedding_map = torch.load(config_train["esm_embedding_path"], map_location='cpu')
        # Inference loop
        results = []
        batch_index = 0
        output_dir = os.path.join(self.data_paths, "tmp_result")
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            for masked_indices, mask, real_indices, attention_mask, connect_comp, rna_type, cell_ids in tqdm(dataloader, desc="Processing batches", total=len(dataloader)):
                # Step 1: Flatten and map indices to ESM embedding
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
                # Step 2: Move data to device
                masked_indices, attention_mask, connect_comp, rna_type, cell_ids, esm_embedding = (
                    masked_indices.to(self.device), attention_mask.to(self.device), connect_comp.to(self.device),
                    rna_type.to(self.device), cell_ids.to(self.device), esm_embedding.to(self.device)
                )
                # Step 3: Forward pass
                predictions = self.model(masked_indices, connect_comp, rna_type, cell_ids, attention_mask, esm_embedding)
                # Step 4: Collect predictions
                # results.append(predictions.cpu().numpy())
                # Step 4: Save predictions to disk
                
                np.save(os.path.join(output_dir, f"predictions_batch_{batch_index}.npy"), predictions.cpu().numpy())
                batch_index += 1

            print(f"Results saved to {output_dir}")

        return results

    def run(self, data_paths: List[str], config_train: dict):
        """
        Main method to run the entire training pipeline.
        """
        dataset = self.load_dataset(data_paths)
        data_loader = DataLoader(dataset, batch_size=config_train["batch_size"], shuffle=True)
        
        pred = self.infer(data_loader, config_train)
        return pred