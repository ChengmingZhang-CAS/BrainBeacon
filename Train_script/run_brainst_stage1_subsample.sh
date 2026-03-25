#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${CONFIG:-${SCRIPT_DIR}/brainst_stage1_subsample.yaml}
INFER_SCRIPT=${INFER_SCRIPT:-${SCRIPT_DIR}/inference_main.py}
DEVICE=${DEVICE:-cuda:0}
FORCE_TOKENIZE=${FORCE_TOKENIZE:-0}

PRETRAIN_CKPT=${PRETRAIN_CKPT:-}
if [[ $# -gt 0 && -z "${PRETRAIN_CKPT}" ]]; then
    PRETRAIN_CKPT=$1
    shift
fi

if [[ -z "${PRETRAIN_CKPT}" ]]; then
    cat <<'EOF' >&2
Usage:
  bash Train_script/run_brainst_stage1_subsample.sh /path/to/brainbeacon_checkpoint.pt

Environment overrides:
  PRETRAIN_CKPT=/path/to/brainbeacon_checkpoint.pt
  DEVICE=cuda:0
  FORCE_TOKENIZE=0
  OUTPUT_ROOT=/cpfs01/.../AfterStage1
  TOKEN_ROOT=/cpfs01/.../_bb_token_cache
  PATH_ROOT=/cpfs01/.../subsample_traindata_20per
  GENE_DICT_PATH=prior_knowledge/gene_dict.h5ad

Run this on a machine where /cpfs01/projects-HDD/... is mounted.
EOF
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config not found: ${CONFIG}" >&2
    exit 1
fi

if [[ ! -f "${INFER_SCRIPT}" ]]; then
    echo "[ERROR] Inference script not found: ${INFER_SCRIPT}" >&2
    exit 1
fi

cmd=(
    "${PYTHON_BIN}" "${INFER_SCRIPT}"
    --config "${CONFIG}"
    --pretrain-ckpt "${PRETRAIN_CKPT}"
    --device "${DEVICE}"
    --force-tokenize "${FORCE_TOKENIZE}"
)

if [[ -n "${GENE_DICT_PATH:-}" ]]; then
    cmd+=(--gene-dict-path "${GENE_DICT_PATH}")
fi

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    cmd+=(--set "output_h5ad_template=${OUTPUT_ROOT}/{relative_path}")
fi

if [[ -n "${TOKEN_ROOT:-}" ]]; then
    cmd+=(--set "token_data_path_template=${TOKEN_ROOT}/{relative_dir}/{adata_stem}_bb_token_dir")
fi

if [[ -n "${PATH_ROOT:-}" ]]; then
    cmd+=(--set "path_root=${PATH_ROOT}")
fi

if [[ -n "${OBSM_KEY:-}" ]]; then
    cmd+=(--obsm-key "${OBSM_KEY}")
fi

cmd+=("$@")

echo "Config: ${CONFIG}"
echo "Checkpoint: ${PRETRAIN_CKPT}"
echo "Device: ${DEVICE}"
echo "Force tokenize: ${FORCE_TOKENIZE}"
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
    echo "Override output root: ${OUTPUT_ROOT}"
fi
if [[ -n "${TOKEN_ROOT:-}" ]]; then
    echo "Override token root: ${TOKEN_ROOT}"
fi

"${cmd[@]}"
