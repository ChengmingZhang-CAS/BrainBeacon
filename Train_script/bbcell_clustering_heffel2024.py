"""
This script processes the Steroseq Mouse Ma2024Aging dataset for cell clustering using the BrainBeacon model_raw.
"""
import os
import torch

from brainbeacon.tokenizer import tokenization_h5ad, process_parquet, set_seed
from brainbeacon.pipeline.cell_embedding import CellEmbeddingPipeline, run_bbcellformer_pipeline
from brainbeacon.bbcellformer.pipeline.reconstruction import ReconstructPipeline
from brainbeacon.configs.config import resolve_path
from brainbeacon.configs.config_train import config_train
import os
import yaml



# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Define base paths and dataset info
BASE_DIR = "/raid/zhangchengming/BrainBeacon-master"
ablation_config_path = os.path.join(BASE_DIR, "brainbeacon", "configs", "ablation_configs", "train_xsmall.yaml")

with open(ablation_config_path, "r", encoding="utf-8") as f:
    yaml_cfg = yaml.safe_load(f) or {}

config_train.update(yaml_cfg)

def main():
    # ========== 1. Dataset and Basic Setup ==========
    dataset_name = "heffel2024"
    specie = 'human'
    assay = 'merfish'

    # ========== 2. Input Paths ==========
    input_data_dir = os.path.join(BASE_DIR, "data", "MERFISH_Human_Heffel2024Temporally3D", "processed")
    adata_path = os.path.join(input_data_dir, "Heffel2024Temporally3D.h5ad")
    gene_dict_path = resolve_path("GENE_DICT_PATH")
    prior_path = resolve_path("PRIOR_DIR")

    # ========== 3. Pretrained Checkpoints ==========
    pretrain_dir = resolve_path("PRETRAIN_DIR")
    # bb_ckpt_name = "epoch_6_hv.pt"
    bb_ckpt_name = "epoch_0_step_800000_0.33B.pt"
    bb_ckpt_name = "epoch_1.pt"

    bb_ckpt_path = os.path.join(pretrain_dir, "ABL1", bb_ckpt_name)
    # cellformer_ckpt_name  = "cellformer.ckpt"
    # cellformer_ckpt_name = "cellformer_epoch10_bb_epoch_0_step_800000_0.33B.pt"
    # cellplm_ckpt_path = os.path.join(pretrain_dir, cellformer_ckpt_name)
    cellplm_ckpt_path = os.path.join(
        BASE_DIR,
        "downstream_tasks", "train_cellformer", "epoch_0_step_800000_0.33B",
        "cellformer_epoch100.pt"
    )
    # ========== 4. Output Naming ==========
    cd_weight = 0.02
    n_hvg = 1000
    # do_fit = False
    do_fit = True
    fit_epochs = 1
    # method_name = f"bbcellformer_{bb_ckpt_name.replace('.pt', '').replace('.ckpt', '')}_hvg{n_hvg}_cd{cd_weight}"
    method_name = f"bbcell_cf100_{bb_ckpt_name.replace('.pt', '').replace('.ckpt', '')}_hvg{n_hvg}_cd{cd_weight}"
    if do_fit:
        method_name += f"_fit{fit_epochs}"
    output_prefix = f"{dataset_name}_{method_name}"
    output_dir = os.path.join(BASE_DIR, "downstream_tasks", "cell_clustering", "outputs", dataset_name, method_name)

    # ========== 5. Run Pipeline ==========
    adata = run_bbcellformer_pipeline(
        adata_path=adata_path,
        specie=specie,
        assay=assay,
        gene_dict_path=gene_dict_path,
        bb_ckpt_path=bb_ckpt_path,
        cellplm_ckpt_path=cellplm_ckpt_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        config_train=config_train,
        n_hvg=n_hvg,
        cd_weight=cd_weight,
        use_hvg=True,
        use_batch=True,
        use_spatial=True,
        weight_mode="expression",
        force_tokenize=False,
        do_fit=do_fit,
        fit_epochs=fit_epochs,
        device=device
    )

    print("Reconstruction complete. Embeddings and model_raw saved.")
    print("adata:", adata)


if __name__ == "__main__":
    main()
