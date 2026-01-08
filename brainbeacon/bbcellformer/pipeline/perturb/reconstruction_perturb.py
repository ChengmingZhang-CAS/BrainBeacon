"""
Perturbation-specific reconstruction pipeline for BrainBeacon CellFormer.

This module provides a modified version of the reconstruction pipeline
that supports perturbation prediction tasks.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import scanpy as sc
import anndata as ad
from typing import Dict, List, Optional, Tuple
import os
import warnings

from .data_perturb import TranscriptomicDatasetPerturb
from .cellformer_perturb import OmicsFormerPerturb
from ...utils.data import data_setup


class PerturbationReconstructionPipeline:
    """
    Pipeline for perturbation prediction using modified CellFormer.
    """
    
    def __init__(self, 
                 gene_list: List[str],
                 enc_mod: str = 'transformer',
                 enc_hid: int = 1024,
                 enc_layers: int = 6,
                 post_latent_dim: int = 1024,
                 dec_mod: str = 'mlp',
                 dec_hid: int = 1024,
                 dec_layers: int = 2,
                 out_dim: int = None,
                 batch_num: int = 0,
                 dataset_num: int = 0,
                 platform_num: int = 0,
                 mask_type: str = 'input',
                 model_dropout: float = 0.1,
                 activation: str = 'gelu',
                 norm: str = 'layernorm',
                 enc_head: int = 8,
                 mask_node_rate: float = 0.5,
                 mask_feature_rate: float = 0.25,
                 drop_node_rate: float = 0.0,
                 max_batch_size: int = 5000,
                 pe_type: str = None,
                 cat_pe: bool = True,
                 gene_emb: Optional[torch.Tensor] = None,
                 latent_mod: str = 'vae',
                 use_perturbation: bool = True,
                 num_perturb_conditions: Optional[int] = None,
                 device: str = 'cuda',
                 # 新增：基因嵌入参数
                 gene_embeddings: Optional[np.ndarray] = None,
                 symbol_to_emb_idx: Optional[dict] = None,
                 condition_to_id: Optional[dict] = None,
                 case_insensitive_mapping: Optional[dict] = None,
                 # 新增：预训练模型路径
                 cellformer_pretrain_path: Optional[str] = None):
        
        self.device = device
        self.max_batch_size = max_batch_size
        self.use_perturbation = use_perturbation
        self.num_perturb_conditions = num_perturb_conditions
        
        # 保存基因嵌入信息
        self.gene_embeddings = gene_embeddings
        self.symbol_to_emb_idx = symbol_to_emb_idx
        self.condition_to_id = condition_to_id
        self.case_insensitive_mapping = case_insensitive_mapping
        
        # Set output dimension to gene list length if not specified
        if out_dim is None:
            out_dim = len(gene_list)
            
        # Initialize the perturbation-aware model_raw
        self.model = OmicsFormerPerturb(
            gene_list=gene_list,
            enc_mod=enc_mod,
            enc_hid=enc_hid,
            enc_layers=enc_layers,
            post_latent_dim=post_latent_dim,
            dec_mod=dec_mod,
            dec_hid=dec_hid,
            dec_layers=dec_layers,
            out_dim=out_dim,
            batch_num=batch_num,
            dataset_num=dataset_num,
            platform_num=platform_num,
            mask_type=mask_type,
            model_dropout=model_dropout,
            activation=activation,
            norm=norm,
            enc_head=enc_head,
            mask_node_rate=mask_node_rate,
            mask_feature_rate=mask_feature_rate,
            drop_node_rate=drop_node_rate,
            max_batch_size=max_batch_size,
            pe_type=pe_type,
            cat_pe=cat_pe,
            gene_emb=gene_emb,
            latent_mod=latent_mod,
            use_perturbation=use_perturbation,
            num_perturb_conditions=num_perturb_conditions,
            # 新增：传递基因嵌入信息
            gene_embeddings=gene_embeddings,
            symbol_to_emb_idx=symbol_to_emb_idx,
            condition_to_id=condition_to_id,
            case_insensitive_mapping=case_insensitive_mapping
        ).to(device)
        
        print(f"Initialized perturbation-aware CellFormer with {len(gene_list)} genes")
        if use_perturbation:
            print(f"Perturbation support enabled for {num_perturb_conditions} conditions")
            if gene_embeddings is not None:
                print(f"Using gene embeddings for condition initialization")
        
        # Load pretrained model_raw if path is provided
        if cellformer_pretrain_path is not None and os.path.exists(cellformer_pretrain_path):
            try:
                ckpt = torch.load(cellformer_pretrain_path, map_location=device)
                if 'model_state_dict' in ckpt:
                    state_dict = ckpt['model_state_dict']
                else:
                    state_dict = ckpt
                
                # Filter out incompatible layers (decoder layers that have size mismatch)
                filtered_state_dict = {}
                model_state_dict = self.model.state_dict()
                
                for key, value in state_dict.items():
                    if key in model_state_dict:
                        if model_state_dict[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Skipping {key}: shape mismatch {value.shape} vs {model_state_dict[key].shape}")
                    else:
                        print(f"Skipping {key}: not found in model_raw")
                
                # Load compatible layers
                self.model.load_state_dict(filtered_state_dict, strict=False)
                print(f"Loaded {len(filtered_state_dict)} compatible layers from {cellformer_pretrain_path}")
            except Exception as e:
                print(f"Warning: Failed to load pretrained model_raw from {cellformer_pretrain_path}: {e}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    def common_preprocess(self, 
                         adata: ad.AnnData, 
                         split_idx: int, 
                         covariate_fields: List[str] =None,
                         use_perturbation: bool = False,
                         perturb_flag: Optional[np.ndarray] = None,
                         perturb_gene_id: Optional[np.ndarray] = None,
                         **kwargs) -> ad.AnnData:
        """
        Preprocess data for perturbation prediction.
        """
        # Create perturbation-aware dataset
        dataset = TranscriptomicDatasetPerturb(
            adata=adata,
            split_field='split' if 'split' in adata.obs else None,
            covariate_fields=covariate_fields,
            label_fields=[],
            use_perturbation=use_perturbation,
            perturb_flag=perturb_flag,
            perturb_gene_id=perturb_gene_id
        )
        
        return dataset
    
    def fit(self, 
            adata: ad.AnnData,
            covariate_fields: List[str],
            use_perturbation: bool = False,
            perturb_flag: Optional[np.ndarray] = None,
            perturb_gene_id: Optional[np.ndarray] = None,
            epochs: int = 100,
            **kwargs) -> ad.AnnData:
        """
        Fit the perturbation-aware model_raw.
        """
        print("Starting perturbation-aware training...")
        
        # Preprocess data
        dataset = self.common_preprocess(
            adata, 0, covariate_fields, 
            use_perturbation=use_perturbation,
            perturb_flag=perturb_flag,
            perturb_gene_id=perturb_gene_id
        )
        
        # Debug: print dataset information
        print(f"Dataset length: {len(dataset)}")
        if len(dataset) > 0:
            sample_batch = dataset[0]
            print(f"Sample batch keys: {list(sample_batch.keys())}")
            if 'x_seq' in sample_batch:
                print(f"Sample batch x_seq shape: {sample_batch['x_seq'].shape}")
            if 'bb_emb' in sample_batch:
                print(f"Sample batch bb_emb shape: {sample_batch['bb_emb'].shape}")
        
        # Create data loader with custom collate function for sparse tensors
        def custom_collate(batch):
            """Custom collate function to handle sparse tensors."""
            # For now, just process one batch at a time to avoid memory issues
            return batch[0] if len(batch) > 0 else None
        
        # Use batch_size=1 to avoid memory issues
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
        
        # Training loop with multiple epochs
        self.model.train()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch_data in enumerate(dataloader):
                # 安全检查：确保批次数据有效
                if batch_data is None:
                    print(f"WARNING: Skipping empty batch {batch_idx}")
                    continue
                    
                # 检查批次大小
                if 'x_seq' in batch_data and batch_data['x_seq'].shape[0] == 0:
                    print(f"WARNING: Skipping batch {batch_idx} with zero samples")
                    continue
                
                # Move data to device
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.device)
                
                # Forward pass
                try:
                    output = self.model(batch_data)
                    
                    # Handle model_raw output (tuple of out_dict, loss)
                    if isinstance(output, tuple):
                        out_dict, loss = output
                        
                        # Backward pass
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        
                        epoch_loss += loss.item()
                        batch_count += 1
                        if epoch == 0:  # Only print detailed info for first epoch
                            print(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
                            if isinstance(out_dict, dict):
                                print(f"Batch {batch_idx}: Output keys: {list(out_dict.keys())}")
                                if 'recon' in out_dict:
                                    print(f"Batch {batch_idx}: Reconstruction shape: {out_dict['recon'].shape}")
                    else:
                        print(f"Batch {batch_idx}: Unexpected output type: {type(output)}")
                        
                except Exception as e:
                    print(f"ERROR in batch {batch_idx}: {e}")
                    print(f"Batch data keys: {list(batch_data.keys())}")
                    if 'x_seq' in batch_data:
                        print(f"x_seq shape: {batch_data['x_seq'].shape}")
                    if 'gene_mask' in batch_data:
                        print(f"gene_mask shape: {batch_data['gene_mask'].shape}")
                    continue
            
            # Print epoch summary
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
            else:
                print(f"Epoch {epoch + 1} completed. No valid batches processed.")
        
        print("Training completed!")
        
        # Save the trained model_raw
        if hasattr(self, 'output_dir') and hasattr(self, 'output_prefix'):
            model_path = os.path.join(self.output_dir, f"{self.output_prefix}_cellformer.pt")
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
        
        return adata
    
    def load_model(self, model_path: str):
        """Load a trained model_raw from checkpoint."""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found")
    
    def predict(self, 
               adata: ad.AnnData,
               covariate_fields: List[str],
               use_perturbation: bool = False,
               perturb_flag: Optional[np.ndarray] = None,
               perturb_gene_id: Optional[np.ndarray] = None,
               model_path: Optional[str] = None,
               **kwargs) -> ad.AnnData:
        """
        Make predictions using the perturbation-aware model_raw.
        """
        print("Starting perturbation-aware prediction...")
        
        # Load trained model_raw if path is provided
        if model_path is not None:
            self.load_model(model_path)
        
        # Preprocess data
        dataset = self.common_preprocess(
            adata, 0, covariate_fields,
            use_perturbation=use_perturbation,
            perturb_flag=perturb_flag,
            perturb_gene_id=perturb_gene_id
        )
        
        # Create data loader with custom collate function for sparse tensors
        def custom_collate(batch):
            """Custom collate function to handle sparse tensors."""
            # For now, just process one batch at a time to avoid memory issues
            return batch[0] if len(batch) > 0 else None
        
        # Use batch_size=1 to avoid memory issues
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
        
        # Prediction loop
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Move data to device
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        batch_data[key] = value.to(self.device)
                
                # Forward pass
                output = self.model(batch_data)
                
                # Debug: check model_raw input and intermediate states
                if batch_idx == 0:  # Only for first batch
                    print(f"Debug - Batch data keys: {list(batch_data.keys())}")
                    if 'x_seq' in batch_data:
                        print(f"Debug - x_seq shape: {batch_data['x_seq'].shape}")
                        if hasattr(batch_data['x_seq'], 'to_dense'):
                            x_dense = batch_data['x_seq'].to_dense()
                            print(f"Debug - x_seq stats: mean={x_dense.mean():.6f}, std={x_dense.std():.6f}")
                        else:
                            print(f"Debug - x_seq type: {type(batch_data['x_seq'])}")
                    if 'bb_emb' in batch_data:
                        print(f"Debug - bb_emb shape: {batch_data['bb_emb'].shape}")
                        print(f"Debug - bb_emb stats: mean={batch_data['bb_emb'].mean():.6f}, std={batch_data['bb_emb'].std():.6f}")
                    if 'perturb_flag' in batch_data:
                        print(f"Debug - perturb_flag shape: {batch_data['perturb_flag'].shape}")
                        print(f"Debug - perturb_flag unique: {torch.unique(batch_data['perturb_flag'])}")
                    if 'perturb_gene_id' in batch_data:
                        print(f"Debug - perturb_gene_id shape: {batch_data['perturb_gene_id'].shape}")
                        print(f"Debug - perturb_gene_id unique: {torch.unique(batch_data['perturb_gene_id'])}")
                
                # Handle model_raw output (tuple of out_dict, loss)
                if isinstance(output, tuple):
                    out_dict, loss = output
                    pred = out_dict.get('recon', out_dict.get('pred', None))
                    
                    # Debug: check output structure
                    if batch_idx == 0:
                        print(f"Debug - Output keys: {list(out_dict.keys())}")
                        if 'recon' in out_dict:
                            print(f"Debug - recon shape: {out_dict['recon'].shape}")
                            print(f"Debug - recon stats: mean={out_dict['recon'].mean():.6f}, std={out_dict['recon'].std():.6f}")
                        if 'latent' in out_dict:
                            print(f"Debug - latent shape: {out_dict['latent'].shape}")
                            print(f"Debug - latent stats: mean={out_dict['latent'].mean():.6f}, std={out_dict['latent'].std():.6f}")
                elif isinstance(output, dict):
                    pred = output.get('recon', output.get('pred', None))
                else:
                    pred = output
                
                if pred is not None:
                    print(f"Batch {batch_idx} prediction stats:")
                    print(f"  Shape: {pred.shape}")
                    print(f"  Mean: {pred.mean().item():.6f}")
                    print(f"  Std: {pred.std().item():.6f}")
                    print(f"  Min: {pred.min().item():.6f}")
                    print(f"  Max: {pred.max().item():.6f}")
                    print(f"  Non-zero ratio: {(pred != 0).float().mean().item():.6f}")
                    predictions.append(pred.cpu().numpy())
                else:
                    print(f"Batch {batch_idx}: No prediction found in output")
                
                print(f"Processed batch {batch_idx}")
        
        # Combine predictions
        if predictions:
            combined_pred = np.concatenate(predictions, axis=0)
            print(f"Prediction shape: {combined_pred.shape}")
            
            # Add predictions to adata
            adata.obsm['X_pred'] = combined_pred
        
        print("Prediction completed!")
        return adata
