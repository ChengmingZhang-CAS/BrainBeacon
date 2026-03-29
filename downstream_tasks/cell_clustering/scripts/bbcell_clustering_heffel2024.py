"""
This script runs BrainBeacon for cell clustering benchmark.
"""

import os
import torch

from brainbeacon.pipeline.cell_embedding import run_bbcellformer_pipeline
from brainbeacon.configs.config import resolve_path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
if device.type == "cuda":
    print(f"using gpu: {torch.cuda.get_device_name(torch.cuda.current_device())}")

base_dir = "/raid/zhangchengming/BrainBeacon-master"


def main():
    dataset_name = "heffel2024"
    specie = 'human'
    assay = 'merfish'
    method_name = "bb_abl1_mean"

    input_data_dir = os.path.join(base_dir, "data", "MERFISH_Human_Heffel2024Temporally3D", "processed")
    adata_path = os.path.join(input_data_dir, "Heffel2024Temporally3D.h5ad")
    gene_dict_path = resolve_path("GENE_DICT_PATH")

    pretrain_dir = os.path.join(resolve_path("PRETRAIN_DIR"), "ABL1")
    stage1_ckpt_path = os.path.join(pretrain_dir, "stage1_step_60000.pt")
    stage2_ckpt_path = os.path.join(pretrain_dir, "stage2_epoch_200.pt")
    stage2_ckpt_path = os.path.join(pretrain_dir, "ABL_1_mean_latest.pt")

    config_override = {
        "nlayers": 4,
        "nheads": 4,
        "dim_feedforward": 256,
        "use_esm_embedding": True,
        "gene_id": True,
        "neighbor_enhance": True,
    }

    n_hvg = 1000
    cd_weight = 0.02
    # do_fit = True
    do_fit = False
    fit_epochs = 10
    use_hvg = True
    use_batch = True
    use_spatial = True
    weight_mode = "expression"
    force_tokenize = False

    output_dir = os.path.join(base_dir, "downstream_tasks", "cell_clustering", "outputs")
    output_dir = os.path.join(output_dir, dataset_name, method_name)
    output_prefix = f"{dataset_name}_{method_name}"
    os.makedirs(output_dir, exist_ok=True)

    adata = run_bbcellformer_pipeline(
        adata_path=adata_path,
        specie=specie,
        assay=assay,
        gene_dict_path=gene_dict_path,
        stage1_ckpt_path=stage1_ckpt_path,
        stage2_ckpt_path=stage2_ckpt_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        config_override=config_override,
        n_hvg=n_hvg,
        cd_weight=cd_weight,
        use_hvg=use_hvg,
        use_batch=use_batch,
        use_spatial=use_spatial,
        weight_mode=weight_mode,
        force_tokenize=force_tokenize,
        do_fit=do_fit,
        fit_epochs=fit_epochs,
        device=device,
    )

    print("run finished.")
    print("adata:", adata)


if __name__ == "__main__":
    main()