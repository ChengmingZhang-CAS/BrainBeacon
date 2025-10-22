import os
import time
import torch
import joblib
import shutil
import torch.nn as nn
import random
import anndata as ad
import scanpy as sc
import pandas as pd
import torch
import pymn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, adjusted_rand_score
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Union, List

from config.config_train_cdniche import config_train
from model.brain_beacon import BrainBeacon
from model.utils import tokenization_h5ad, process_parquet, set_seed
from model.bbcellformer.pipeline.reconstruction import ReconstructPipeline
from model.bbcellformer.pipeline.cell_type_annotation import CellTypeAnnotationPipeline
from model.pipeline.cell_embedding import run_tokenization, run_bb_inference

from config.config_cdniche import GENE_DICT_PATH
from config.config_train_cdniche import config_train
from model.utils import get_gene_mean_path

def train_encoder_on_adata(
    adata,
    bb_embedding_path,            # path to .npz embedding file
    bb_pretrain_path,             # path to BB encoder backbone weights
    cellformer_version,          # prefix like 'cellformer', used to find .yaml/.pt
    cellformer_directory,        # path to folder with pretrained CellFormer model_raw files
    device,
    cellformer_pretrain_path=None,  # Not used here, but required by the pipeline
    use_batch=True,
    use_spatial=True,
    do_fit=True,
    fit_epochs=500,  # can be set in the pipeline
    slice_sample=False,  # NEW
    enc_mod="flowformer",
    save_model_path=None,  # optional: save .pt model_raw weights
):
    # Load AnnData file
    data = adata.copy()
    data.obs_names_make_unique()
    # set train/valid split
    np.random.seed(42)
    data.obs['split'] = 'train'
    if 'slice' not in data.obs.columns:
        data.obs['slice'] = data.obs['batch'] if 'batch' in data.obs.columns else 'default_slice'
    for batch_id in data.obs['slice'].unique():
        idx = data.obs['slice'] == batch_id
        cell_idx = np.where(idx)[0]
        n_valid = max(1, int(len(cell_idx) * 0.1))  # Ensure at least one cell is selected for validation
        valid_cells = np.random.choice(cell_idx, n_valid, replace=False)
        data.obs.iloc[valid_cells, data.obs.columns.get_loc('split')] = 'valid'

    # load brainbeacon embeddings
    data.obsm['bb_emb'] = np.load(bb_embedding_path)['embeddings']

    # Add batch info if enabled
    if use_batch and 'batch' not in data.obs.columns:
        data.obs['batch'] = data.obs['slice']

    if use_spatial and 'spatial' in data.obsm:
        all_coords = []

        for batch_id in data.obs['batch'].unique():
            idx = data.obs['batch'] == batch_id
            spatial = data.obsm['spatial'][idx]
            spatial = np.asarray(spatial)  # assure spatial is a NumPy array
            spatial_min = spatial.min(axis=0)
            spatial_max = spatial.max(axis=0)
            normalized = (spatial - spatial_min) / (spatial_max - spatial_min + 1e-8)

            data.obs.loc[idx, 'x_FOV_px'] = normalized[:, 0]
            data.obs.loc[idx, 'y_FOV_px'] = normalized[:, 1]

    # Initialize CellPLM embedding pipeline
    overwrite_config = {
        "name": f"bb_{enc_mod}",
        "enc_mod": enc_mod,
        'objective': 'imputation',
        'mask_node_rate': 0.95,
        'mask_feature_rate': 0.25,
        'max_batch_size': 2000,
        'mask_type': 'hidden',
        # 'mask_type': 'input',
    }
    # clear GPU memory before re-initializing the pipeline
    torch.cuda.empty_cache()
    pipeline = ReconstructPipeline(
        pretrain_prefix=cellformer_version,
        overwrite_config=overwrite_config,
        pretrain_directory=cellformer_directory,
        bb_pretrain_path=bb_pretrain_path,
        cellformer_pretrain_path=cellformer_pretrain_path,  # Not used here
        use_pretrain=True)
    if do_fit:
        # Only sample one slice if requested
        if slice_sample:
            # np.random.seed(42)
            rng = np.random.RandomState(None)  # randomState with local randomness
            chosen_slice = rng.choice(data.obs['slice'].unique())
            fit_data = data[data.obs['slice'] == chosen_slice].copy()
            print(f"Training only on slice: {chosen_slice} ({fit_data.n_obs} cells)")
            MAX_CELLS = 10000
            if fit_data.n_obs > MAX_CELLS:
                print(f"[Warning] Too many cells in slice ({fit_data.n_obs}), subsampling to {MAX_CELLS}")
                sampled_indices = np.random.choice(fit_data.n_obs, MAX_CELLS, replace=False)
                fit_data = fit_data[sampled_indices].copy()
                print("fit data shape:", fit_data.shape)

        else:
            fit_data = data.copy()
        pipeline.fit(
            fit_data,  # AnnData object
            train_config={'epochs': fit_epochs},
            split_field='split',
            train_split='train',
            valid_split='valid',
            device=device
        )

    if save_model_path is not None:
        torch.save(pipeline.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")
    del pipeline.model
    torch.cuda.empty_cache()
    return data


def train_encoder_on_multi_adata(
        dataset_info_list: List[dict],
        bb_ckpt_path: str,
        initial_ckpt_path: str,
        output_dir: str,
        config_train: dict,
        output_prefix: str = "brainbeacon",
        num_global_epochs: int = 100,
        per_dataset_epochs: int = 50,
        shuffle_each_epoch: bool = True,
        slice_sample: bool = True,
        cd_weight: float = 0.02,
        n_hvg: int = 1000,
        batch_size: int = 64,
        save_all_epochs: bool = False,
        enc_mod: str = "flowformer",
        device=None
) -> str:
    """
    Train CellFormer encoder on multiple datasets using BB embeddings as input.
    """
    os.makedirs(output_dir, exist_ok=True)
    current_ckpt_path = initial_ckpt_path
    save_model_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_model_dir, exist_ok=True)

    for global_epoch in range(num_global_epochs):
        print(f"\n========== Global Epoch {global_epoch + 1} ==========")
        datasets_this_epoch = dataset_info_list.copy()
        if shuffle_each_epoch:
            random.seed(global_epoch + 42)
            random.shuffle(datasets_this_epoch)

        for ds in datasets_this_epoch:
            data_name = ds["data_name"]
            data_dir = ds["data_dir"]
            adata_name = ds["adata_name"]
            specie = ds["specie"]
            assay = ds["assay"]

            print(f"\n--- Training on {data_name} ---")

            adata_path = os.path.join(data_dir, adata_name)
            adata = sc.read_h5ad(adata_path)
            adata.obs["platform"] = assay

            gene_dict_path = GENE_DICT_PATH
            BASE_DIR = "/raid/zhangchengming/BrainBeacon-master"
            gene_mean_path = get_gene_mean_path(BASE_DIR, assay, use_metacell=True)

            output_dir_epoch = os.path.join(output_dir, data_name)
            os.makedirs(output_dir_epoch, exist_ok=True)

            # --- Save checkpoint only when needed ---
            if save_all_epochs or global_epoch == num_global_epochs - 1:
                save_model_path = os.path.join(save_model_dir, f"{enc_mod}_epoch{global_epoch + 1:02d}.pt")
            else:
                save_model_path = os.path.join(save_model_dir, "tmp_last.pt")  # temporary overwrite

            # Tokenization
            bb_token_dir = os.path.join(output_dir_epoch, f"{output_prefix}_bb_token_dir")
            # bb_token_dir = os.path.join(output_dir_epoch, f"bb_token")
            # config_train["batch_size"] = batch_size  # Set batch size for tokenization
            if assay == "snrna":
                config_train["batch_size"] = min(batch_size, 16)
            else:
                config_train["batch_size"] = batch_size

            parquet_path = run_tokenization(
                adata_path=adata_path,
                bb_token_dir=bb_token_dir,
                gene_dict_path=gene_dict_path,
                mean_path=gene_mean_path,
                specie=specie,
                assay=assay,
                use_hvg=True,
                n_hvg=n_hvg,
                force_tokenize=False
            )

            # BB Inference
            bb_embedding_path = os.path.join(output_dir_epoch, f"{output_prefix}_bb_embeddings.npz")
            if os.path.exists(bb_embedding_path):
                print(f"Found existing BB embeddings at: {bb_embedding_path}")
            else:
                run_bb_inference(
                    adata=adata,
                    parquet_path=parquet_path,
                    config_train=config_train,
                    pretrain_ckpt=bb_ckpt_path,
                    device=device,
                    save_path=bb_embedding_path
                )

            # Train CellFormer encoder
            adata = train_encoder_on_adata(
                adata=adata,
                bb_embedding_path=bb_embedding_path,
                bb_pretrain_path=bb_ckpt_path,
                cellformer_version="cellformer",
                cellformer_directory=os.path.dirname(initial_ckpt_path),
                device=device,
                cellformer_pretrain_path=current_ckpt_path,
                use_batch=True,
                use_spatial=True,
                do_fit=True,
                fit_epochs=per_dataset_epochs,
                slice_sample=slice_sample,
                enc_mod=enc_mod,
                save_model_path=save_model_path
            )

            current_ckpt_path = save_model_path
            print(f"Finished training on {data_name}, model_raw saved to: {save_model_path}")

            # Clean up
            import gc
            del adata
            torch.cuda.empty_cache()
            gc.collect()

    return current_ckpt_path

def prepare_adata(
    adata_list: List[dict],
    output_dir: str,
    output_prefix: str,
    use_batch=True,
    use_spatial=True
):
    processed_data_dict = {}
    for item in adata_list:
        data_dir = item["data_dir"]
        adata_path = os.path.join(data_dir, item["adata_name"])
        adata = sc.read_h5ad(adata_path)

        # Load AnnData file
        data = adata.copy()
        data.obs_names_make_unique()
        # set train/valid split
        np.random.seed(42)
        data.obs['split'] = 'train'
        if 'slice' not in data.obs.columns:
            data.obs['slice'] = data.obs['batch'] if 'batch' in data.obs.columns else 'default_slice'
        for batch_id in data.obs['slice'].unique():
            idx = data.obs['slice'] == batch_id
            cell_idx = np.where(idx)[0]
            n_valid = max(1, int(len(cell_idx) * 0.1))  # Ensure at least one cell is selected for validation
            valid_cells = np.random.choice(cell_idx, n_valid, replace=False)
            data.obs.iloc[valid_cells, data.obs.columns.get_loc('split')] = 'valid'

        # load brainbeacon embeddings

        # Construct the path to the BrainBeacon embedding file
        bb_embedding_path = os.path.join(output_dir, item["data_name"], f"{output_prefix}_bb_embeddings.npz")
        data.obsm['bb_emb'] = np.load(bb_embedding_path)['embeddings']

        # Add batch info if enabled
        if use_batch and 'batch' not in data.obs.columns:
            data.obs['batch'] = data.obs['slice']

        if use_spatial and 'spatial' in data.obsm:
            all_coords = []

            for batch_id in data.obs['batch'].unique():
                idx = data.obs['batch'] == batch_id
                spatial = data.obsm['spatial'][idx]
                spatial = np.asarray(spatial)  # assure spatial is a NumPy array
                spatial_min = spatial.min(axis=0)
                spatial_max = spatial.max(axis=0)
                normalized = (spatial - spatial_min) / (spatial_max - spatial_min + 1e-8)

                data.obs.loc[idx, 'x_FOV_px'] = normalized[:, 0]
                data.obs.loc[idx, 'y_FOV_px'] = normalized[:, 1]
        processed_data_dict[item["data_name"]] = data

    return processed_data_dict


def run_bbcellformer_annotation(
    source_adata_list: List[dict],
    target_adata_list: List[dict],
    output_dir: str,  # The directory where the .npz file is saved
    output_prefix: str,  # The prefix used to generate the .npz filename
    bb_pretrain_path,  # path to BB encoder backbone weights
    cellformer_pretrain_path,  # Not used here, but required by the pipeline
    label_key,
    device,
    use_batch=True,
    use_spatial=True,
    do_fit=True,
    fit_epochs=500,  # can be set in the pipeline
    slice_sample=False,  # NEW
    enc_mod="flowformer",
    output_attentions=False,  # whether to return attention weights
    save_embedding_path=None,  # Optional now
    save_model_path=None,  # optional: save .pt model_raw weights
):
    source_data_dict = prepare_adata(
        adata_list=source_adata_list,
        output_dir=output_dir,
        output_prefix=output_prefix,
        use_batch=use_batch,
        use_spatial=use_spatial
    )
    data = ad.concat(
        list(source_data_dict.values()),
        join='outer',
        merge='same',
        label='source_id',
        index_unique=None
    )
    data.X = np.nan_to_num(data.X, nan=0.0)
    assert "bb_emb" in data.obsm
    print("Concatenated training adata shape:", data.shape)

    # Initialize CellPLM embedding pipeline
    overwrite_config = {
        "name": f"bb_{enc_mod}",
        "enc_mod": enc_mod,
        'objective': 'imputation',
        'mask_node_rate': 0.95,
        'mask_feature_rate': 0.25,
        'max_batch_size': 2000,
        'mask_type': 'hidden',
        # 'mask_type': 'input',
    }
    overwrite_config = {
        "name": f"bb_{enc_mod}",
        "enc_mod": enc_mod,
        'drop_node_rate': 0.3,
        'dec_layers': 1,
        'model_dropout': 0.5,
        'mask_node_rate': 0.75,
        'mask_feature_rate': 0.25,
        'dec_mod': 'mlp',
        # 'latent_mod': 'ae',
        'head_type': 'annotation',
        'out_dim': data.obs[label_key].nunique(),
        'max_batch_size': 5000,
        'mask_type': 'hidden',
        # 'mask_type': 'input',
    }
    pipeline = CellTypeAnnotationPipeline(
        pretrain_prefix="flowformer",  # Use flowformer as the encoder
        overwrite_config=overwrite_config,
        pretrain_directory= os.path.dirname(cellformer_pretrain_path),
        bb_pretrain_path=bb_pretrain_path,
        cellformer_pretrain_path=cellformer_pretrain_path,
        use_pretrain=True
    )
    if do_fit:
        # Only sample one slice if requested
        if slice_sample:
            # np.random.seed(42)
            if 'slice' not in data.obs.columns:
                data.obs['slice'] = data.obs['batch']
            rng = np.random.RandomState(None)  # randomState with local randomness
            chosen_slice = rng.choice(data.obs['slice'].unique())
            fit_data = data[data.obs['slice'] == chosen_slice].copy()
            print(f"Training only on slice: {chosen_slice} ({fit_data.n_obs} cells)")
            MAX_CELLS = 20000
            if fit_data.n_obs > MAX_CELLS:
                print(f"[Warning] Too many cells in slice ({fit_data.n_obs}), subsampling to {MAX_CELLS}")
                sampled_indices = np.random.choice(fit_data.n_obs, MAX_CELLS, replace=False)
                fit_data = fit_data[sampled_indices].copy()
                print("fit data shape:", fit_data.shape)

        else:
            fit_data = data.copy()
        pipeline.fit(
            fit_data,  # AnnData object
            train_config={'epochs': fit_epochs},
            split_field='split',
            train_split='train',
            valid_split='valid',
            label_fields=[label_key],
            device=device,
        )
    inference_config = {
        'es': 200,
        'lr': 1e-3,
        'wd': 1e-7,
        'scheduler': 'plat',
        'epochs': 2000,
        'max_eval_batch_size': 10000,
        'hvg': 3000,
        'patience': 25,
        'workers': 0,
    }
    target_data_dict = prepare_adata(
        adata_list=target_adata_list,
        output_dir=output_dir,
        output_prefix=output_prefix,
        use_batch=use_batch,
        use_spatial=use_spatial
    )

    results_dict = {}
    for data_name, target_data in target_data_dict.items():
        result = pipeline.predict(
            target_data,
            inference_config=inference_config,
            device=device
        )
        pred = result['pred']
        latent = result['latent']
        logits = result['logits']

        target_data.obsm['X_emb'] = latent.cpu().numpy()
        target_data.obsm['X_logits'] = logits.cpu().numpy()
        target_data.obs['pred_id'] = pred.cpu().numpy().astype(int)

        label_encoder = pipeline.label_encoders[label_key]
        pred_labels_str = label_encoder.inverse_transform(pred.cpu().numpy())
        target_data.obs['pred_label'] = pred_labels_str

        results_dict[data_name] = target_data
        # Optionally save embeddings to .npz
        if save_embedding_path is not None:
            np.savez_compressed(save_embedding_path, embeddings=data.obsm['X_emb'])
            print(f"Embeddings saved to {save_embedding_path}")

    # Optionally save model_raw weights
    if save_model_path is not None:
        torch.save(pipeline.model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")

    return results_dict


def run_label_transfer_pipeline(
    encoder_adata_list: List[dict],
    source_adata_list: List[dict],
    target_adata_list: List[dict],
    bb_ckpt_path: str,
    cellplm_ckpt_path: str,
    output_dir: str,
    output_prefix: str,
    config_update: dict = None,
    n_hvg: int = 1000,
    cd_weight: float = 0.02,
    use_hvg: bool = True,
    use_batch: bool = True,
    use_spatial: bool = True,
    weight_mode: str = "expression",
    force_tokenize: bool = True,
    do_fit: bool = True,
    fit_epochs: int = 500,
    shuffle_each_epoch=True,
    slice_sample=False,
    enc_mod: str = "flowformer",
    save_model: bool = True,
    save_model_path: str = None,
    do_train_encoder: bool = True,
    num_global_epochs: int = 100,
    per_dataset_epochs: int = 50,
    label_key='cell_label',
    device=None
):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(42)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config_train is None:
        raise ValueError("`config_train` must be provided.")
    config_train.update({
        "weight_mode": weight_mode,
        "cd_weight": cd_weight,
        "masking_p": 0,
        "batch_size": 64,
        "expr_mode": None,
        "use_gene_id_emb": True,
        "use_homo_emb": True,
        "use_rna_type_emb": True,
        "use_esm_emb": True,
    })
    if config_update:
        config_train.update(config_update)

    # ====== Step 1: Train encoder (if enabled) ======
    if do_train_encoder:
        encoder_ckpt_path = train_encoder_on_multi_adata(
            dataset_info_list=encoder_adata_list,
            bb_ckpt_path=bb_ckpt_path,
            initial_ckpt_path=cellplm_ckpt_path,
            output_dir=output_dir,
            config_train=config_train,
            num_global_epochs=num_global_epochs,
            per_dataset_epochs=per_dataset_epochs,
            shuffle_each_epoch=shuffle_each_epoch,
            slice_sample=slice_sample,
            cd_weight=cd_weight,
            n_hvg=n_hvg,
            enc_mod=enc_mod,
            device=device
        )
    else:
        encoder_ckpt_path = cellplm_ckpt_path  # Use existing encoder

    print(f"Using encoder checkpoint: {encoder_ckpt_path}")

    # ====== Step 2: Run annotation and prediction ======
    target_adata = run_bbcellformer_annotation(
        source_adata_list=source_adata_list,
        target_adata_list=target_adata_list,
        output_dir=output_dir,
        output_prefix=output_prefix,
        bb_pretrain_path=bb_ckpt_path,
        cellformer_pretrain_path=encoder_ckpt_path,
        label_key=label_key,
        device=device,
        use_batch=use_batch,
        use_spatial=use_spatial,
        do_fit=do_fit,
        fit_epochs=fit_epochs,
        slice_sample=slice_sample,
        enc_mod=enc_mod,
        save_model_path=save_model_path,
    )

    return target_adata

def dev_sum(df: pd.DataFrame):
    """
    Normalize a DataFrame by columns, dividing each value by the sum of its column.
    """
    v = df.values.copy()  # Use a copy to avoid modifying the original data
    col_sums = np.sum(v, axis=0)
    # Prevent division by zero
    v[:, col_sums > 0] /= col_sums[col_sums > 0]
    return pd.DataFrame(v, index=df.index, columns=df.columns)

def run_prediction_pipeline(
        adata: sc.AnnData,
        pretrained_model: pd.DataFrame,
        marker_gene_dict: dict,
        output_folder: str,
        true_label_col: str = 'SubClass',
        study_col: str = 'slice',
        layer_col: str = 'layer',
        pred_col_name: str = 'subclass_pre'
):
    """
    Run the full MetaNeighbor-based evaluation and visualization pipeline.

    Args:
        adata (sc.AnnData): Input AnnData object with gene expression and metadata.
        pretrained_model (pd.DataFrame): Pretrained MetaNeighbor reference model_raw.
        marker_gene_dict (dict): Marker gene dictionary for dotplot visualization.
        output_folder (str): Directory for saving outputs.
        true_label_col (str): Column name of true labels in adata.obs.
        study_col (str): Column name of study/sample in adata.obs.
        layer_col (str): Column name for spatial layer information in adata.obs.
        pred_col_name (str): Column name for predicted labels to be stored in adata.obs.
    """
    # --- 0. Preparation ---
    print(f"--- Pipeline started. Output will be saved to: {output_folder} ---")
    os.makedirs(output_folder, exist_ok=True)
    adata = adata.copy()

    # Ensure correct datatypes
    adata.obs[true_label_col] = adata.obs[true_label_col].astype("category")
    adata.obs[study_col] = adata.obs[study_col].astype("category")
    if layer_col in adata.obs:
        adata.obs[layer_col] = adata.obs[layer_col].astype("category")
    if "genenames" in adata.var.columns:
        adata.var_names = adata.var["genenames"].astype(str)
        adata.var_names_make_unique()
    elif "gene_symbol" in adata.var.columns:
        adata.var_names = adata.var["gene_symbol"].astype(str)
        adata.var_names_make_unique()

    # Remove duplicated gene symbols
    adata = adata[:, ~adata.var_names.duplicated()].copy()

    # --- 1. Run MetaNeighborUS ---
    print("--- 1. Running MetaNeighborUS to get predictions ---")
    pymn.MetaNeighborUS(
        adata,
        study_col=study_col,
        ct_col=pred_col_name,  # use predicted labels as input for AUROC calculation
        trained_model=pretrained_model,
        one_vs_best=True
    )

    auroc_results = adata.uns['MetaNeighborUS_1v1']
    print(f"Predictions stored in 'adata.obs[{pred_col_name}]'.")

    # AUROC heatmap
    pymn.plotMetaNeighborUS_pretrained(
        adata, cmap="coolwarm", mn_key='MetaNeighborUS_1v1',
        figsize=(10, 10), show=False
    )
    plt.savefig(os.path.join(output_folder, '0_MetaNeighborUS_AUROC_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # --- 2. Evaluation & Visualization ---
    print("\n--- 2. Starting evaluation and visualization ---")

    # a. Marker gene dotplot
    print("Saving marker gene dotplot...")
    # check if all marker genes exist in adata.var_names
    all_genes = {g for genes in marker_gene_dict.values() for g in genes}
    missing = all_genes - set(adata.var_names)

    if missing:
        print(f"[WARN] {len(missing)} marker genes not found in adata, filtering...")
        marker_gene_dict = {
            ct: [g for g in genes if g in adata.var_names]
            for ct, genes in marker_gene_dict.items()
        }
        marker_gene_dict = {ct: genes for ct, genes in marker_gene_dict.items() if genes}

    sc.pl.dotplot(adata, marker_gene_dict, groupby=pred_col_name, use_raw=False, show=False)
    plt.savefig(os.path.join(output_folder, '1a_dotplot_predicted_labels.png'), bbox_inches='tight')
    plt.close()
    sc.pl.dotplot(adata, marker_gene_dict, groupby=true_label_col, use_raw=False, show=False)
    plt.savefig(os.path.join(output_folder, '1b_dotplot_true_labels.png'), bbox_inches='tight')
    plt.close()

    # b. Distribution across spatial layers
    if layer_col in adata.obs:
        print("Analyzing distribution in layers...")
        layer_dist = pd.crosstab(adata.obs[layer_col], adata.obs[pred_col_name])
        layer_dist_norm_row = layer_dist.div(layer_dist.sum(axis=1), axis=0)
        dfwide = dev_sum(layer_dist_norm_row)
        plt.figure(figsize=(12, 4))
        sns.heatmap(dfwide, annot=False, cmap="viridis", linewidths=0.1)
        plt.title('Distribution of Predicted Cell Types across Layers')
        plt.savefig(os.path.join(output_folder, '2_layer_distribution_heatmap.png'),
                    bbox_inches='tight')
        plt.close()

    # c. Cell type proportions
    print("Comparing cell type proportions...")
    true_props = adata.obs[true_label_col].value_counts(normalize=True).sort_index()
    pred_props = adata.obs[pred_col_name].value_counts(normalize=True).sort_index()
    props_df = pd.DataFrame({'True': true_props, 'Predicted': pred_props})
    props_df.plot(kind='bar', figsize=(12, 7), position=0.5, width=0.4)
    plt.title('Proportion of Cell Types: True vs. Predicted')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(output_folder, '3_proportion_comparison.png'), bbox_inches='tight')
    plt.close()

    # d. Confusion matrix
    print("Generating confusion matrices...")
    cm_recall = pd.crosstab(
        adata.obs[true_label_col], adata.obs[pred_col_name], normalize='index'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_recall, annot=True, fmt='.2f', cmap='viridis')

    plt.title('Confusion Matrix (Normalized by True Label -> Recall)')
    plt.savefig(os.path.join(output_folder, '4a_confusion_matrix_recall.png'), bbox_inches='tight')
    plt.close()

    # e. Spatial plots
    print("Generating spatial plots...")

    # --- Unified palette (align plot_spatial_comparison) ---
    predefined_palette = {
        "L2": "#1f77b4", "RELN": "#4292c6", "VIP": "#6baed6", "VIP_RELN": "#9ecae1",
        "L2/3": "#2ca02c", "L2/3/4": "#4caf50", "L3/4/5": "#66bb6a", "SST": "#81c784", "LAMP5": "#a5d6a7",
        "L3/4": "#9467bd", "L4": "purple", "PVALB": "#b39ddb", "PV_CHC": "#c0a5e0",
        "L4/5": "#ff7f0e", "L4/5/6": "#ffa726", "L5/6": "#ffcc80",
        "ASC": "#e31a1c", "VLMC": "#ef5350",
        "L6": "#d4ac0d", "OLG": "#ffd54f",
        "MG": "#7f7f7f", "OPC": "#a0a0a0", "EC": "#f46d43",
        "unassigned": "#d0d0d0",  # 浅灰
    }

    def make_palette(categories, predefined):
        import scanpy as sc
        base_colors = sc.pl.palettes.default_102
        palette = {}
        for cat in categories:
            if cat in predefined:
                palette[cat] = predefined[cat]
        unused_colors = [c for c in base_colors if c not in palette.values()]
        i = 0
        for cat in categories:
            if cat not in palette:
                palette[cat] = unused_colors[i % len(unused_colors)]
                i += 1
        return palette

    # unified palette for both true and predicted labels
    all_categories = sorted(set(adata.obs[true_label_col].dropna().unique()) |
                            set(adata.obs[pred_col_name].dropna().unique()))
    palette_map = make_palette(all_categories, predefined_palette)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc.pl.spatial(
        adata,
        color=true_label_col,
        spot_size=100,
        palette=[palette_map[c] for c in adata.obs[true_label_col].cat.categories],
        ax=axes[0],
        show=False
    )
    axes[0].set_title(f'True Labels ({true_label_col})')

    sc.pl.spatial(
        adata,
        color=pred_col_name,
        spot_size=100,
        palette=[palette_map[c] for c in adata.obs[pred_col_name].cat.categories],
        ax=axes[1],
        show=False
    )
    axes[1].set_title(f'Predicted Labels ({pred_col_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '5_spatial_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # f. Classification metrics
    print("Calculating classification metrics...")
    report = classification_report(adata.obs[true_label_col], adata.obs[pred_col_name], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_folder, '6a_classification_report.csv'))

    ari_score = adjusted_rand_score(adata.obs[true_label_col], adata.obs[pred_col_name])
    with open(os.path.join(output_folder, '6b_ari_score.txt'), 'w') as f:
        f.write(f"Adjusted Rand Index (ARI): {ari_score:.4f}\n")

    print(f"\n--- Pipeline finished successfully. All outputs are in {output_folder} ---")
    return adata
