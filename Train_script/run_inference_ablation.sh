#!/bin/bash
# =============================================================================
# BrainBeacon inference ablation runner for inference_main.py
#
# Usage:
#   bash run_inference_ablation.sh ABL1
#   bash run_inference_ablation.sh ALL
#
# Default paths:
#   BASE_CONFIG=brainbeacon/configs/bb_inference_batch_example.yaml
#   GENE_DICT_PATH=prior_knowledge/gene_dict.h5ad
#   CHECKPOINT_ROOT=/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/jiaoyifeng/code/brainbeacon_new
#   OUTPUT_ROOT=/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/subsample_traindata_20per
#
# Optional environment variables:
#   PYTHON_BIN=python
#   ABL1_CKPT=/path/to/ABL1_checkpoint.pt
#   ABL2_CKPT=/path/to/ABL2_checkpoint.pt
#   ...
#   TOKEN_ROOT=/path/to/token_cache_root
#   NPZ_ROOT=/path/to/optional_npz_root
#   DEVICE=cuda:0
#   OBSM_KEY_PREFIX=bb_emb
#   FORCE_TOKENIZE=0
# =============================================================================
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

PYTHON_BIN=${PYTHON_BIN:-python}
INFER_SCRIPT=${INFER_SCRIPT:-inference_main.py}
BASE_CONFIG=${BASE_CONFIG:-brainbeacon/configs/bb_inference_batch_example.yaml}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/jiaoyifeng/code/brainbeacon_new}
GENE_DICT_PATH=${GENE_DICT_PATH:-prior_knowledge/gene_dict.h5ad}
OUTPUT_ROOT=${OUTPUT_ROOT:-/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/BrainST_impu/subsample_traindata_20per}
TOKEN_ROOT=${TOKEN_ROOT:-${OUTPUT_ROOT}/token_cache}
NPZ_ROOT=${NPZ_ROOT:-}
DEVICE=${DEVICE:-cuda:0}
OBSM_KEY_PREFIX=${OBSM_KEY_PREFIX:-bb_emb}
FORCE_TOKENIZE=${FORCE_TOKENIZE:-0}
EXP=${1:-ABL1}

usage() {
    cat <<'EOF'
Usage:
  bash run_inference_ablation.sh <EXP_ID>

Supported:
  ABL1 ABL2 ABL3 ABL4 ABL5 ABL6 ABL7 ABL13 ABL14 ALL

Examples:
  bash run_inference_ablation.sh ABL1

  ABL1_CKPT=/path/to/ABL1_epoch_100.pt \
  ABL2_CKPT=/path/to/ABL2_epoch_100.pt \
  bash run_inference_ablation.sh ALL

Notes:
  1. This script mirrors run_ablation.sh for inference.
  2. Token cache is shared across experiments by default; output h5ad files
     are still written under separate ${EXP_ID} directories.
  3. ABL13 and ABL14 differ from ABL2 mainly by checkpoint, because the
     data-volume ablation happens at training time, not inference time.
EOF
}

require_file() {
    local path=$1
    local label=$2
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] ${label} not found: ${path}" >&2
        exit 1
    fi
}

resolve_ckpt() {
    local exp_id=$1
    local env_var="${exp_id}_CKPT"
    local explicit_ckpt="${!env_var:-}"

    if [[ -n "${explicit_ckpt}" ]]; then
        echo "${explicit_ckpt}"
        return
    fi

    if [[ -z "${CHECKPOINT_ROOT}" ]]; then
        echo "[ERROR] Missing checkpoint for ${exp_id}. Set ${env_var} or CHECKPOINT_ROOT." >&2
        exit 1
    fi

    local candidates=()
    while IFS= read -r path; do
        candidates+=("${path}")
    done < <(find "${CHECKPOINT_ROOT}" -type f -name '*.pt' \
        \( -path "*/${exp_id}_*/*" -o -path "*/${exp_id}/*" -o -name "${exp_id}.pt" \) | sort -V)

    if [[ ${#candidates[@]} -eq 0 ]]; then
        echo "[ERROR] No checkpoint found for ${exp_id} under ${CHECKPOINT_ROOT}" >&2
        exit 1
    fi

    echo "${candidates[${#candidates[@]}-1]}"
}

run_experiment() {
    local exp_id=$1
    local ckpt_path
    ckpt_path=$(resolve_ckpt "${exp_id}")

    require_file "${INFER_SCRIPT}" "Inference script"
    require_file "${BASE_CONFIG}" "Base config"
    require_file "${GENE_DICT_PATH}" "Gene dictionary"
    require_file "${ckpt_path}" "Checkpoint"

    local exp_output_root="${OUTPUT_ROOT}/${exp_id}"
    local shared_token_root="${TOKEN_ROOT}"
    local exp_obsm_key="${OBSM_KEY_PREFIX}_${exp_id}"
    mkdir -p "${exp_output_root}" "${shared_token_root}"

    local -a cmd=(
        "${PYTHON_BIN}" "${INFER_SCRIPT}"
        --config "${BASE_CONFIG}"
        --pretrain-ckpt "${ckpt_path}"
        --gene-dict-path "${GENE_DICT_PATH}"
        --device "${DEVICE}"
        --obsm-key "${exp_obsm_key}"
        --force-tokenize "${FORCE_TOKENIZE}"
        # Model parameters aligned with brainbeacon/configs/ablation_configs/train_xsmall.yaml
        --set "dim_model=256"
        --set "nheads=4"
        --set "dim_feedforward=256"
        --set "nlayers=4"
        --set "dropout=0.1"
        --set "n_tokens=92076"
        --set "n_connect_comp=46714"
        --set "n_aux=20"
        --set "n_rna_type=33"
        --set "num_neighbors=4"
        --set "context_length=1000"
        --set "ems_embedding_dim=5120"
        --set "use_esm_embedding=true"
        --set "use_esm_emb=true"
        --set "token_data_path_template=${shared_token_root}/{index}_{adata_stem}_bb_token_dir"
        --set "output_h5ad_template=${exp_output_root}/{index}_{adata_stem}_with_bb_emb.h5ad"
    )

    if [[ -n "${NPZ_ROOT}" ]]; then
        mkdir -p "${NPZ_ROOT}/${exp_id}"
        cmd+=(--set "npz_save_path_template=${NPZ_ROOT}/${exp_id}/{index}_{adata_stem}_bb_embeddings.npz")
    fi

    case "${exp_id}" in
        ABL1)
            ;;
        ABL2)
            cmd+=(--use-gene-id-emb 0)
            ;;
        ABL3)
            cmd+=(--use-homo-emb 0)
            ;;
        ABL4)
            cmd+=(--use-esm-embedding 0 --use-rna-type-emb 0)
            ;;
        ABL5)
            cmd+=(--neighbor-enhance 0)
            ;;
        ABL6)
            cmd+=(--use-density-emb 0)
            ;;
        ABL7)
            cmd+=(--use-pos-emb 0)
            ;;
        ABL13|ABL14)
            cmd+=(--use-gene-id-emb 0)
            ;;
        *)
            echo "[ERROR] Unknown experiment: ${exp_id}" >&2
            usage
            exit 1
            ;;
    esac

    echo "=============================================="
    echo "Running inference experiment: ${exp_id}"
    echo "Checkpoint: ${ckpt_path}"
    echo "Output root: ${exp_output_root}"
    echo "Shared token root: ${shared_token_root}"
    echo "obsm key: ${exp_obsm_key}"
    echo "force_tokenize: ${FORCE_TOKENIZE}"
    echo "=============================================="
    "${cmd[@]}"
}

case "${EXP}" in
    ABL1|ABL2|ABL3|ABL4|ABL5|ABL6|ABL7|ABL13|ABL14)
        run_experiment "${EXP}"
        ;;
    ALL)
        for exp in ABL1 ABL2 ABL3 ABL4 ABL5 ABL6 ABL7 ABL13 ABL14; do
            run_experiment "${exp}"
        done
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[ERROR] Unknown experiment: ${EXP}" >&2
        usage
        exit 1
        ;;
esac
