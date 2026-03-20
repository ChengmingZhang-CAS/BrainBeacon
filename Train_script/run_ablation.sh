#!/bin/bash
# =============================================================================
# BrainBeacon Ablation Study Script
# 4 nodes x 8 GPUs = 32 GPUs, model config: train_xsmall.yaml
# Usage: bash run_ablation.sh <EXP_ID>
#   e.g. bash run_ablation.sh ABL1
#        bash run_ablation.sh ALL   # run all experiments sequentially
# =============================================================================
set -e

# ---- Environment setup ----
apt-get update -qq && apt install -y -qq libgl1-mesa-glx > /dev/null 2>&1

cd /cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/jiaoyifeng/code/brainbeacon_new
export MASTER_PORT=50200
echo "WORLD_SIZE=${WORLD_SIZE} RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"

# ---- Constants ----
PYTHON=/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/share_dataplatform/code/env_seg/bin/python
CONFIG=brainbeacon/configs/ablation_configs/train_xsmall.yaml
MAX_STEPS=60000
NPROC=8

run_experiment() {
    local exp_id=$1
    shift
    echo "=============================================="
    echo "  Running experiment: ${exp_id}"
    echo "  Extra args: $@"
    echo "=============================================="

    ${PYTHON} -m torch.distributed.launch \
        --nproc_per_node=${NPROC} \
        --nnodes=${WORLD_SIZE} \
        --node_rank=${RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        train_ddp_sampler.py \
        --nproc_per_node=${NPROC} \
        --world_size=${WORLD_SIZE} \
        --node_rank=${RANK} \
        --nnode=${WORLD_SIZE} \
        --config ${CONFIG} \
        --max_steps ${MAX_STEPS} \
        --exp_name ${exp_id} \
        "$@"
}

EXP=${1:-ABL1}

case ${EXP} in

    # ==================================================================
    # Component ablation (ABL1-ABL7): low data = 30M samples
    # ==================================================================

    ABL1)
        # Base model: all components enabled
        run_experiment ABL1 \
            --max_total_samples 30000000
        ;;

    ABL2)
        # w/o gene_id embedding
        run_experiment ABL2 \
            --use_gene_id_emb 0 \
            --max_total_samples 30000000
        ;;

    ABL3)
        # w/o homologous connected component embedding
        run_experiment ABL3 \
            --use_homo_emb 0 \
            --max_total_samples 30000000
        ;;

    ABL4)
        # w/o ESM protein embedding + RNA type embedding
        run_experiment ABL4 \
            --use_esm_emb 0 \
            --use_rna_type_emb 0 \
            --max_total_samples 30000000
        ;;

    ABL5)
        # w/o neighbor gene distribution (deviation)
        run_experiment ABL5 \
            --neighbor_enhance 0 \
            --max_total_samples 30000000
        ;;

    ABL6)
        # w/o density token embedding
        run_experiment ABL6 \
            --use_density_emb 0 \
            --max_total_samples 30000000
        ;;

    ABL7)
        # w/o positional embedding
        run_experiment ABL7 \
            --use_pos_emb 0 \
            --max_total_samples 30000000
        ;;

    # ==================================================================
    # Data volume ablation (ABL13-ABL14): gene_id disabled per table
    # ==================================================================

    ABL13)
        # Median data volume: 60M samples
        run_experiment ABL13 \
            --use_gene_id_emb 0 \
            --max_total_samples 60000000
        ;;

    ABL14)
        # Large data volume: 90M samples
        run_experiment ABL14 \
            --use_gene_id_emb 0 \
            --max_total_samples 90000000
        ;;

    # ==================================================================
    # Run all supported experiments sequentially
    # ==================================================================

    ALL)
        for exp in ABL1 ABL2 ABL3 ABL4 ABL5 ABL6 ABL7 ABL13 ABL14; do
            bash "$0" ${exp}
        done
        ;;

    *)
        echo "Unknown experiment: ${EXP}"
        echo "Supported: ABL1 ABL2 ABL3 ABL4 ABL5 ABL6 ABL7 ABL13 ABL14 ALL"
        echo ""
        echo "Skipped (need code support):"
        echo "  ABL8  - w/o weight_sampling"
        echo "  ABL9  - leave-out species"
        echo "  ABL10 - leave-out platform"
        echo "  ABL11 - median model size"
        echo "  ABL12 - large model size"
        echo "  ABL15 - token length"
        echo "  ABL16 - stage1 only"
        echo "  ABL17 - stage2 only"
        echo "  ABL18 - token length"
        exit 1
        ;;
esac
