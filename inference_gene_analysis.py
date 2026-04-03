#!/usr/bin/env python3
"""
inference_gene_analysis.py
==========================

BrainBeacon Stage 1 gene-level analysis script.
从预训练的 BrainBeacon 编码器中提取 **基因级别** 的注意力矩阵和隐藏表征,
用于下游验证分析（共表达关系、跨物种同源性等）。

功能概述 / Overview
-------------------
本脚本提供两个核心接口, 均以 adata (AnnData, n cells x p genes) 作为输入:

  接口1 — Attention Matrix  (n, p, p)
      提取 Transformer 编码器的 gene-gene 注意力权重矩阵。
      用途: 验证模型是否捕获了已知的基因共表达关系 (gene A attend to gene B)。
      输出已按 adata.var_names 的原始基因顺序反索引, 不在 token 序列中的基因补 0。

  接口2 — Gene Embedding    (n, p, d)
      提取 Transformer 编码器最后一层每个基因位置的隐藏表征 (d = dim_model = 256)。
      用途: 验证跨物种 gene embedding 是否保留了同源性
            (同源基因表征相似, 非同源基因表征不相似)。
      同样反索引到原始基因顺序, 缺失基因补 0。

内存管理 / Memory Management
-----------------------------
  --group-by <obs_column>
      按 adata.obs 中的某列 (如 cell_type) 对结果取均值。
      这将输出维度从 (n_cells, p, p) 降为 (n_groups, p, p),
      大幅减少内存 (例如 50k cells -> 20 groups)。
  --max-cells N
      仅处理前 N 个细胞 (用于调试或内存受限场景)。
  --dtype float16
      以半精度存储输出 (默认), 节省 50% 内存。

Token 序列结构 / Token Sequence Layout
---------------------------------------
BrainBeacon tokenization 将每个 cell 编码为长度 1000 的 token 序列:

  Position 0:  Species token   (物种标识, token_id < 20)
  Position 1:  Assay token     (平台标识, token_id < 20)
  Position 2:  Density token   (细胞密度分箱, token_id < 20)
  Position 3+: Gene tokens     (基因 token, token_id >= 20, 按表达量降序排列)
  Padding:     token_id == 1   (填充位)

基因反索引逻辑:
  token_id >= AUX_TOKEN_OFFSET(20) 的位置是基因 token。
  gene_dict_index = token_id - 20
  gene_name = gene_dict.var.index[gene_dict_index]
  adata_col = adata.var_names 中该 gene_name 的位置

Usage / 使用方式
----------------
  # 提取 attention + embedding, 按 cell type 分组 (推荐, 省内存)
  python inference_gene_analysis.py \\
    --adata-path /path/to/data.h5ad \\
    --pretrain-ckpt /path/to/checkpoint.pt \\
    --gene-dict-path /path/to/gene_dict.h5ad \\
    --token-data-path /path/to/existing/tokens \\
    --mode both \\
    --group-by cell_type \\
    --output-dir /path/to/output

  # 只提取 attention, 逐 cell 输出, 限制 1000 个 cell
  python inference_gene_analysis.py \\
    --adata-path /path/to/data.h5ad \\
    --pretrain-ckpt /path/to/checkpoint.pt \\
    --mode attention \\
    --max-cells 1000

Output Files / 输出文件
-----------------------
  {prefix}_attention.npz :
      attention  (n, p, p)   gene-gene attention matrix (float16)
      gene_names (p,)        adata 原始基因名
      labels     (n,)        cell name 或 group label
      valid_mask (n, p)      每个基因是否有有效值 (1=有效, 0=缺失)
      cell_counts (n,)       [仅 group_by] 每组包含的 cell 数量

  {prefix}_embedding.npz :
      embedding  (n, p, d)   gene-level hidden states (float16)
      gene_names (p,)        adata 原始基因名
      labels     (n,)        cell name 或 group label
      valid_mask (n, p)      每个基因是否有有效值
      cell_counts (n,)       [仅 group_by] 每组包含的 cell 数量
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tqdm import tqdm

# ============================================================================
# 常量 / Constants
# ============================================================================

# BrainBeacon tokenization 将基因 token ID 偏移 20:
#   token_id 0-19  : 辅助 token (species, assay, density, padding, mask, CLS, ...)
#   token_id 20+   : 基因 token (gene_dict 中第 i 个基因 -> token_id = i + 20)
AUX_TOKEN_OFFSET = 20


# ============================================================================
# 配置加载工具 / Config Loading Helpers
# (从 inference_main.py 简化移植)
# ============================================================================

def parse_scalar(value: str) -> Any:
    """将字符串解析为 Python 标量值 (bool / int / float / None / str)。

    用于解析 CLI --set KEY=VALUE 中的 VALUE 部分。
    """
    text = value.strip()
    lowered = text.lower()
    # 布尔值
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    # None
    if lowered in {"none", "null"}:
        return None
    # 整数
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    # 浮点数 (含科学计数法)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)", text):
        return float(text)
    # 默认: 字符串
    return text


def parse_key_value_override(text: str) -> tuple[str, Any]:
    """解析 'KEY=VALUE' 格式的配置覆盖字符串。"""
    if "=" not in text:
        raise ValueError(f"Invalid override {text!r}. Expected KEY=VALUE.")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {text!r}. Empty key.")
    return key, parse_scalar(value)


def load_config(config_path: str | None, overrides: list[str]) -> dict[str, Any]:
    """加载模型配置。

    优先级 (从低到高):
      1. brainbeacon/configs/config_train.py 中的默认值
      2. YAML 配置文件 (--config)
      3. CLI 覆盖 (--set KEY=VALUE)
    """
    from brainbeacon.configs.config_train import config_train as default_config_train

    # 深拷贝默认配置, 避免修改全局变量
    config = copy.deepcopy(default_config_train)

    # 合并 YAML 配置
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}
        if not isinstance(yaml_config, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
        config.update(yaml_config)

    # 应用 CLI 覆盖
    for override in overrides:
        key, value = parse_key_value_override(override)
        config[key] = value

    return config


# ============================================================================
# 基因映射工具 / Gene Mapping Utilities
# ============================================================================

def build_token_to_gene_map(gene_dict_path: str) -> dict[int, str]:
    """从 gene dictionary h5ad 构建 token_id -> gene_name 的反向映射。

    BrainBeacon 的 tokenization 过程:
      1. 加载 gene_dict.h5ad, 其 .var.index 包含所有已知基因名
         (Ensembl ID 或 gene symbol)
      2. 将 adata 中的基因与 gene_dict 对齐
      3. 对每个匹配的基因: token_id = gene_dict 中的位置索引 + AUX_TOKEN_OFFSET (20)

    因此反向映射为: token_id -> gene_dict.var.index[token_id - 20]

    Parameters
    ----------
    gene_dict_path : str
        基因字典 h5ad 文件路径 (如 gene_dict.h5ad 或 model_h5ad_1211.h5ad)。

    Returns
    -------
    dict[int, str]
        {token_id: gene_name} 映射, 覆盖所有 gene_dict 中的基因。
    """
    import scanpy as sc

    gene_dict = sc.read_h5ad(gene_dict_path)
    token_to_gene: dict[int, str] = {}
    for i, gene_name in enumerate(gene_dict.var.index):
        token_to_gene[i + AUX_TOKEN_OFFSET] = gene_name
    print(f"Gene dictionary loaded: {len(token_to_gene)} genes from {gene_dict_path}")
    return token_to_gene


def build_adata_gene_index(adata) -> dict[str, int]:
    """构建 gene_name -> adata.var 列位置 的映射。

    用于将 gene_dict 中的基因名映射回 adata 的原始基因顺序。

    Returns
    -------
    dict[str, int]
        {gene_name: column_index} 映射。
    """
    return {name: i for i, name in enumerate(adata.var_names)}


def _match_gene_name(gene_name: str, adata_gene_idx: dict[str, int]) -> int | None:
    """将 gene_dict 中的基因名匹配到 adata.var 的列位置。

    匹配策略:
      1. 精确匹配 (区分大小写)
      2. 回退: 大小写不敏感匹配

    Parameters
    ----------
    gene_name : str
        来自 gene_dict 的基因名。
    adata_gene_idx : dict[str, int]
        adata 的 {gene_name: col_index} 映射。

    Returns
    -------
    int or None
        匹配到的 adata 列索引, 未匹配返回 None。
    """
    # 精确匹配
    if gene_name in adata_gene_idx:
        return adata_gene_idx[gene_name]
    # 大小写不敏感回退
    gene_lower = gene_name.lower()
    for name, idx in adata_gene_idx.items():
        if name.lower() == gene_lower:
            return idx
    return None


# ============================================================================
# 核心推理函数 / Core Inference
# ============================================================================

def run_gene_level_inference(
    adata,
    token_data_path: str,
    config_train: dict,
    pretrain_ckpt: str,
    gene_dict_path: str,
    device: torch.device,
    mode: str = "both",
    group_by: str | None = None,
    attention_layers: str = "last",
    max_cells: int | None = None,
    out_dtype: str = "float16",
) -> dict[str, Any]:
    """运行 BrainBeacon Stage 1 推理, 提取基因级别的注意力矩阵和/或隐藏表征。

    整体流程:
      1. 构建反向映射: token_id -> gene_dict gene_name -> adata column index
      2. 加载预训练模型 (BrainBeaconCellCluster -> BrainBeacon.encode())
      3. 加载 tokenized 数据集 (joblib 格式)
      4. 逐 batch 推理:
         a. 前向传播获取 encoder hidden states (batch, seq_len, dim_model)
         b. [可选] 通过 attention hooks 捕获注意力权重 (batch, seq_len, seq_len)
         c. 对每个 cell, 使用 real_indices 将 token 位置映射回 adata 基因位置
         d. 填充到 (n, p, p) 或 (n, p, d) 输出矩阵
      5. [如果 group_by] 对每组取均值

    Parameters
    ----------
    adata : AnnData
        输入 anndata, 包含 p 个基因。用于确定输出的基因顺序和分组信息。
    token_data_path : str
        tokenized 数据目录路径 (包含 tokens-XXXX/ 子目录)。
    config_train : dict
        模型配置字典 (来自 config_train.py 或 YAML)。
    pretrain_ckpt : str
        预训练 BrainBeacon checkpoint 文件路径。
    gene_dict_path : str
        基因字典 h5ad 文件路径, 用于构建 token_id -> gene_name 映射。
    device : torch.device
        推理设备 (cuda / cpu)。
    mode : str
        提取模式:
        - "attention": 仅提取注意力矩阵
        - "embedding": 仅提取基因表征
        - "both":      同时提取 (默认)
    group_by : str or None
        如果指定, 按 adata.obs 中该列的值对结果取组均值。
        推荐用于大数据集以控制内存 (如 "cell_type")。
    attention_layers : str
        注意力提取范围:
        - "last": 仅最后一层 (默认, 最节省内存)
        - "all":  所有层 (取均值, 内存占用 x nlayers)
    max_cells : int or None
        限制处理的最大 cell 数量 (用于调试)。
    out_dtype : str
        输出数组精度: "float16" (默认) 或 "float32"。

    Returns
    -------
    dict[str, Any]
        结果字典, 包含以下 key:
        - "gene_names" : np.ndarray (p,)      — adata 原始基因名列表
        - "attention"  : np.ndarray (n, p, p)  — gene-gene 注意力矩阵 [仅 mode 含 attention]
        - "embedding"  : np.ndarray (n, p, d)  — gene-level 隐藏表征 [仅 mode 含 embedding]
        - "valid_mask" : np.ndarray (n, p)     — 标记每个基因是否有有效值 (1=有效, 0=缺失/padding)
        - "labels"     : np.ndarray (n,)       — 行标签 (cell name 或 group label)
        - "cell_counts": np.ndarray (n,)       — 每组 cell 数量 [仅 group_by 模式]

    内存估算 (以 MERFISH 50k cells x 298 genes 为例):
        per-cell attention:  50000 x 298 x 298 x 2 bytes = ~8.9 GB (float16)
        group-by (20 types): 20 x 298 x 298 x 2 bytes = ~3.5 MB (float16)
    """
    from brainbeacon.pipeline.cell_embedding import (
        CellEmbeddingPipeline,
        normalize_brainbeacon_model_config,
    )
    from torch.utils.data import DataLoader

    # 确定需要提取哪些输出
    want_attention = mode in ("attention", "both")
    want_embedding = mode in ("embedding", "both")
    np_dtype = np.float16 if out_dtype == "float16" else np.float32

    # ------------------------------------------------------------------
    # Step 1: 构建反向映射 (token_id -> adata column index)
    # ------------------------------------------------------------------
    # token_to_gene: {token_id: gene_name} — 从 gene_dict 构建
    token_to_gene = build_token_to_gene_map(gene_dict_path)
    # adata_gene_idx: {gene_name: col_index} — 从 adata.var_names 构建
    adata_gene_idx = build_adata_gene_index(adata)
    p = len(adata.var_names)  # adata 中的基因总数

    # 预计算 token_id -> adata 列索引 的直接映射 (跳过中间的 gene_name 查找)
    # 这样在推理循环中每个 cell 只需做 dict.get() 即可
    token_to_adata_col: dict[int, int] = {}
    n_mapped = 0
    for token_id, gene_name in token_to_gene.items():
        col = _match_gene_name(gene_name, adata_gene_idx)
        if col is not None:
            token_to_adata_col[token_id] = col
            n_mapped += 1
    print(f"Gene mapping: {n_mapped}/{len(token_to_gene)} dict genes mapped to adata "
          f"({p} adata genes)")
    if n_mapped == 0:
        raise RuntimeError(
            f"No genes from gene_dict could be mapped to adata.var_names. "
            f"Check naming convention: gene_dict uses names like "
            f"'{next(iter(token_to_gene.values()), '?')}', "
            f"adata uses names like '{adata.var_names[0] if p > 0 else '?'}'. "
            f"Possible Ensembl ID vs gene symbol mismatch."
        )
    mapping_ratio = n_mapped / max(len(token_to_gene), 1)
    if mapping_ratio < 0.01:
        import warnings
        warnings.warn(
            f"Only {n_mapped}/{len(token_to_gene)} ({mapping_ratio:.1%}) genes mapped. "
            f"Output will be mostly zeros. Check gene naming convention.",
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Step 2: 初始化模型
    # ------------------------------------------------------------------
    # normalize_brainbeacon_model_config 处理 legacy config key 兼容性
    # (如 use_esm_embedding <-> use_esm_emb, gene_id <-> use_gene_id_emb)
    config_train = normalize_brainbeacon_model_config(config_train)
    # gene-level 分析使用 batch_size=1, 避免跨 cell 混淆
    config_train["batch_size"] = 1

    # CellEmbeddingPipeline 内部创建 BrainBeaconCellCluster 模型并加载 checkpoint
    # BrainBeaconCellCluster.forward() 调用 BrainBeacon.encode():
    #   输入: (x_gene_id, x_connect_id, x_rna_type, attention_mask, esm_embedding,
    #          neighbor_gene_distribution)
    #   输出: (batch, seq_len, dim_model) — Transformer 编码器的隐藏状态
    pipeline = CellEmbeddingPipeline(
        pretrain_ckpt=pretrain_ckpt, model_config=config_train, device=device
    )
    model = pipeline.model
    model.eval()
    dim_model = config_train["dim_model"]  # 隐藏层维度 (默认 256)

    # ------------------------------------------------------------------
    # Step 3: 加载 tokenized 数据集
    # ------------------------------------------------------------------
    # ZeroshotJoblibDataset 从 tokens-XXXX/ 目录加载 joblib 文件
    # 每个样本返回:
    #   real_indices:              (1, seq_len) 真实基因 token ID (未 mask)
    #   attention_mask:            (1, seq_len) padding mask (True=padding)
    #   connect_comp:              (1, seq_len) 同源连通分量 ID
    #   rna_type:                  (1, seq_len) RNA 类型 ID
    #   cell_raw_idx:              (1,) 原始 cell 名称/索引
    #   neighbor_gene_distribution:(1, seq_len) 邻域基因分布偏差分箱 (0-5)
    #   exp:                       (1, seq_len) 基因表达量
    dataset = pipeline.load_dataset(token_data_path)
    n_total = len(dataset)
    if max_cells is not None:
        n_total = min(n_total, max_cells)
    print(f"Total cells to process: {n_total}")

    # num_workers=0: gene-level 分析 batch_size=1, I/O 不是瓶颈
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # Step 4: 加载 ESM2 蛋白质嵌入
    # ------------------------------------------------------------------
    # ESM2 embedding: (n_tokens, 5120) — 每个 gene token 对应一个 5120 维蛋白质特征
    # 模型内部通过 esm_embedding_projection 将 5120 -> dim_model
    esm_embedding_map = torch.load(
        config_train["esm_embedding_path"], map_location="cpu"
    )

    # ------------------------------------------------------------------
    # Step 5: 准备输出累加器
    # ------------------------------------------------------------------
    if group_by is not None:
        # group_by 模式: 按 adata.obs[group_by] 分组, 输出维度 = n_groups
        if group_by not in adata.obs.columns:
            raise ValueError(f"Column '{group_by}' not found in adata.obs")
        groups = adata.obs[group_by].astype(str).values
        unique_groups = sorted(set(groups))
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        n_out = len(unique_groups)
        labels = np.array(unique_groups)
        # 构建 cell_name -> group_idx 映射, 在推理循环中用于确定结果行索引
        cell_to_group: dict[str, int] = {}
        for cell_name, group_name in zip(adata.obs_names, groups):
            cell_to_group[cell_name] = group_to_idx[group_name]
        cell_counts = np.zeros(n_out, dtype=np.int64)
        print(f"Grouping by '{group_by}': {n_out} groups")
    else:
        # per-cell 模式: 每个 cell 一行, 输出维度 = n_cells
        n_out = n_total
        labels = None  # 推理过程中逐步收集 cell 名称
        cell_counts = None

    # 使用 float32 作为累加精度 (避免 float16 累加精度丢失), 最后再转换
    acc_dtype = np.float32
    # attention 累加器: (n_out, p, p), 存储 gene-gene attention 权重的和
    attn_acc = np.zeros((n_out, p, p), dtype=acc_dtype) if want_attention else None
    # embedding 累加器: (n_out, p, dim_model), 存储 gene hidden state 的和
    emb_acc = np.zeros((n_out, p, dim_model), dtype=acc_dtype) if want_embedding else None
    # valid 累加器: (n_out, p), 统计每个基因位置被多少个 cell 覆盖
    # (用于 group_by 模式下的均值计算)
    valid_acc = np.zeros((n_out, p), dtype=acc_dtype)
    cell_names_list: list[str] = []
    n_skipped_no_group = 0  # group_by 模式下因 cell name 不匹配被跳过的 cell 数
    first_skipped_cell_name: str | None = None  # 记录第一个被跳过的 cell name, 用于错误信息

    # 打印内存占用估算
    mem_mb = 0
    if attn_acc is not None:
        mem_mb += attn_acc.nbytes / 1024 / 1024
    if emb_acc is not None:
        mem_mb += emb_acc.nbytes / 1024 / 1024
    print(f"Accumulator memory: {mem_mb:.1f} MB")

    # ------------------------------------------------------------------
    # Step 6: 注册 attention hooks (如果需要提取注意力)
    # ------------------------------------------------------------------
    # BrainBeacon.enable_attention_hooks() 在 Transformer encoder 的
    # self-attention 层上注册 forward hook, 捕获 attention weight:
    #   权重形状: (batch, n_heads, seq_len, seq_len)
    # "last" 仅在最后一层注册 (最省内存), "all" 在所有层注册
    if want_attention:
        hook_target = None if attention_layers == "all" else attention_layers
        model.pretrain_model.enable_attention_hooks(target_layers=hook_target)

    # ------------------------------------------------------------------
    # Step 7: 推理循环
    # ------------------------------------------------------------------
    t_start = time.time()
    n_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, total=n_total, desc="Gene analysis")):
            if n_processed >= n_total:
                break

            # --- 解包 batch ---
            # DataLoader 返回的每个 tensor 外层多一个 batch 维度 [0] 是因为
            # ZeroshotJoblibDataset.__getitem__ 返回的 tensor 已含 batch 维度
            real_indices, attention_mask, connect_comp, rna_type, cell_raw_idx, \
                neighbor_gene_distribution, exp = batch

            real_indices = real_indices[0]               # (B, seq_len) 基因 token ID
            attention_mask = attention_mask[0]            # (B, seq_len) padding mask
            connect_comp = connect_comp[0]                # (B, seq_len) 同源连通分量
            rna_type = rna_type[0]                        # (B, seq_len) RNA 类型
            neighbor_gene_distribution = neighbor_gene_distribution[0].long()  # (B, seq_len) 邻域偏差
            # cell_raw_idx 解包: Dataset 返回 list[str] (长度 B),
            # DataLoader collate 后变为 list[tuple[str]] 即 [('name1',), ('name2',), ...]
            # 需要提取每个元素中的实际字符串
            cell_raw_idx = [
                item[0] if isinstance(item, (list, tuple)) else str(item)
                for item in cell_raw_idx
            ]

            # --- ESM embedding 查表 ---
            # 将每个 token_id 映射到对应的 5120 维 ESM2 蛋白质嵌入
            real_indices_view = real_indices.view(-1).long()
            esm_embedding = torch.index_select(esm_embedding_map, dim=0, index=real_indices_view)
            esm_embedding = esm_embedding.view(
                real_indices.shape[0], real_indices.shape[1], esm_embedding.shape[-1]
            )

            # --- 移至推理设备 ---
            real_indices_d = real_indices.to(device)
            attention_mask_d = attention_mask.to(device)
            connect_comp_d = connect_comp.to(device)
            rna_type_d = rna_type.to(device)
            esm_embedding_d = esm_embedding.to(device)
            neighbor_gene_distribution_d = neighbor_gene_distribution.to(device)

            # --- 前向传播 ---
            # model = BrainBeaconCellCluster, 其 forward() 调用:
            #   self.pretrain_model.encode(x_gene_id, x_connect_id, x_rna_type,
            #                              attention_mask, esm_embedding,
            #                              neighbor_gene_distribution)
            # 返回 Transformer encoder 的隐藏状态 (非 pooled, 非 classifier head)
            # 形状: (B, seq_len, dim_model)
            hidden_states = model(
                real_indices_d, connect_comp_d, rna_type_d,
                attention_mask_d, esm_embedding_d, neighbor_gene_distribution_d,
                None  # sequence_mask 参数被 forward() 忽略 (del sequence_mask)
            )
            hidden_states = hidden_states.detach().cpu().numpy()  # (B, seq_len, dim_model)

            # --- 提取 attention 权重 ---
            if want_attention:
                # get_attention_weights() 返回 hook 捕获的注意力权重列表:
                #   [(layer_idx, weights), ...]
                #   weights 形状: (B, n_heads, seq_len, seq_len)
                attn_weights_raw = model.pretrain_model.get_attention_weights()

                # 对每一层: 在 head 维度取均值 -> (B, seq_len, seq_len)
                attn_per_layer = []
                for layer_idx, w in attn_weights_raw:
                    attn_per_layer.append(w.mean(dim=1).cpu().numpy())

                # 跨层取均值 -> (B, seq_len, seq_len)
                # 这给出了一个综合的 gene-gene attention 视图
                batch_attention = np.mean(attn_per_layer, axis=0)

                # 清除已捕获的权重, 释放内存
                model.pretrain_model.clear_attention_weights()
            else:
                batch_attention = None

            real_indices_np = real_indices.numpy()  # (B, seq_len)

            # --- 逐 cell 处理: 反索引到 adata 基因顺序 ---
            B = real_indices_np.shape[0]
            for b in range(B):
                if n_processed >= n_total:
                    break

                cell_name = str(cell_raw_idx[b])

                # 确定结果矩阵中的行索引
                if group_by is not None:
                    # group_by 模式: 查找 cell 所属的组
                    if cell_name not in cell_to_group:
                        # cell 在 tokenized 数据中但名称与 adata.obs_names 不匹配, 跳过
                        if first_skipped_cell_name is None:
                            first_skipped_cell_name = cell_name
                        n_skipped_no_group += 1
                        n_processed += 1
                        continue
                    out_idx = cell_to_group[cell_name]
                    cell_counts[out_idx] += 1
                else:
                    # per-cell 模式: 按处理顺序分配行索引
                    out_idx = n_processed
                    cell_names_list.append(cell_name)

                # ----------------------------------------------------------
                # 核心映射: token 序列位置 -> adata 基因列位置
                # ----------------------------------------------------------
                # real_indices[b] 是该 cell 的 token 序列:
                #   [species_id, assay_id, density_id, gene1_token, gene2_token, ..., PAD, PAD, ...]
                # 我们只关心 token_id >= 20 的位置 (基因 token)
                cell_tokens = real_indices_np[b]  # (seq_len,)
                token_positions = []  # 在 token 序列中的位置 (用于索引 hidden_states/attention)
                adata_cols = []       # 对应的 adata.var 列位置 (用于填充输出矩阵)

                for pos in range(len(cell_tokens)):
                    token_id = int(cell_tokens[pos])
                    if token_id < AUX_TOKEN_OFFSET:
                        # 辅助 token (species/assay/density) 或 padding, 跳过
                        continue
                    col = token_to_adata_col.get(token_id)
                    if col is not None:
                        token_positions.append(pos)
                        adata_cols.append(col)

                if not token_positions:
                    # 该 cell 没有任何基因映射成功 (罕见情况)
                    n_processed += 1
                    continue

                # 转为 numpy 数组以支持向量化索引
                token_positions = np.array(token_positions)  # (k,) k = 有效基因数
                adata_cols = np.array(adata_cols)              # (k,)

                # ----------------------------------------------------------
                # 填充 gene embedding 矩阵
                # ----------------------------------------------------------
                if want_embedding:
                    cell_hidden = hidden_states[b]                  # (seq_len, dim_model)
                    gene_embs = cell_hidden[token_positions]        # (k, dim_model)
                    if group_by is not None:
                        # group 模式: 累加, 之后除以有效 cell 数
                        emb_acc[out_idx][adata_cols] += gene_embs
                    else:
                        # per-cell 模式: 直接赋值
                        emb_acc[out_idx][adata_cols] = gene_embs

                # ----------------------------------------------------------
                # 填充 attention 矩阵
                # ----------------------------------------------------------
                if want_attention:
                    cell_attn = batch_attention[b]  # (seq_len, seq_len)
                    # 用 np.ix_ 提取基因位置之间的 attention 子矩阵
                    # attn_sub[i, j] = cell_attn[token_positions[i], token_positions[j]]
                    # 即: gene_i attend to gene_j 的权重
                    attn_sub = cell_attn[np.ix_(token_positions, token_positions)]  # (k, k)
                    if group_by is not None:
                        # group 模式: 累加
                        attn_acc[out_idx][np.ix_(adata_cols, adata_cols)] += attn_sub
                    else:
                        # per-cell 模式: 直接赋值
                        attn_acc[out_idx][np.ix_(adata_cols, adata_cols)] = attn_sub

                # ----------------------------------------------------------
                # 更新 valid mask (记录哪些基因有有效值)
                # ----------------------------------------------------------
                if group_by is not None:
                    # group 模式: 累计每个基因位置被覆盖的 cell 次数
                    valid_acc[out_idx][adata_cols] += 1.0
                else:
                    # per-cell 模式: 二值标记
                    valid_acc[out_idx][adata_cols] = 1.0

                n_processed += 1

    # ------------------------------------------------------------------
    # Step 7.5: group_by 模式下的 cell name 匹配检查
    # ------------------------------------------------------------------
    if group_by is not None and n_processed > 0:
        skip_ratio = n_skipped_no_group / n_processed
        if n_skipped_no_group == n_processed:
            raise RuntimeError(
                f"All {n_processed} cells were skipped because their names in the "
                f"tokenized dataset do not match adata.obs_names. "
                f"Token cell name example: {first_skipped_cell_name!r}; "
                f"adata.obs_names example: {adata.obs_names[0]!r}. "
                f"Result would be all zeros."
            )
        if skip_ratio > 0.5:
            import warnings
            warnings.warn(
                f"{n_skipped_no_group}/{n_processed} ({skip_ratio:.0%}) cells skipped "
                f"due to cell name mismatch between tokenized data and adata.obs_names. "
                f"Many groups may have zero values.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Step 8: 清理资源
    # ------------------------------------------------------------------
    if want_attention:
        model.pretrain_model.disable_attention_hooks()
    del pipeline, esm_embedding_map
    torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"Inference completed: {n_processed} cells in {t_elapsed:.1f}s")

    # ------------------------------------------------------------------
    # Step 9: group_by 模式下的均值归一化
    # ------------------------------------------------------------------
    if group_by is not None:
        for g in range(n_out):
            count = max(cell_counts[g], 1)  # 防止除以零
            if want_embedding:
                # gene embedding 均值: 按每个基因位置的实际覆盖 cell 数除
                # (因为不同 cell 可能 tokenize 了不同的基因子集)
                for j in range(p):
                    if valid_acc[g, j] > 0:
                        emb_acc[g, j] /= valid_acc[g, j]
            if want_attention:
                # attention 均值: 按组内 cell 总数除
                # (保持 attention matrix 行和归一化)
                attn_acc[g] /= count
            # valid_acc 归一化为 [0, 1] 比例 (该基因在组内的覆盖率)
            valid_acc[g] = np.where(valid_acc[g] > 0, valid_acc[g] / count, 0)

    # ------------------------------------------------------------------
    # Step 10: 组装返回结果
    # ------------------------------------------------------------------
    result: dict[str, Any] = {
        "gene_names": np.array(list(adata.var_names)),  # (p,) 原始基因名
        "valid_mask": valid_acc.astype(np_dtype),        # (n, p) 有效标记
    }
    if want_attention:
        result["attention"] = attn_acc.astype(np_dtype)  # (n, p, p) 注意力矩阵
    if want_embedding:
        result["embedding"] = emb_acc.astype(np_dtype)   # (n, p, d) 基因表征
    if group_by is not None:
        result["labels"] = labels              # (n,) group 名称
        result["cell_counts"] = cell_counts    # (n,) 每组 cell 数
    else:
        result["labels"] = np.array(cell_names_list)  # (n,) cell 名称

    return result


# ============================================================================
# 结果保存与摘要打印 / Save & Print Utilities
# ============================================================================

def save_results(result: dict[str, Any], output_dir: str, prefix: str = "gene_analysis"):
    """将分析结果保存为压缩 npz 文件。

    输出文件:
      {prefix}_attention.npz — attention matrix + metadata
      {prefix}_embedding.npz — gene embeddings + metadata

    npz 文件可用 np.load() 加载:
      data = np.load("gene_analysis_attention.npz")
      attn = data["attention"]       # (n, p, p)
      genes = data["gene_names"]     # (p,)
      labels = data["labels"]        # (n,)
    """
    os.makedirs(output_dir, exist_ok=True)

    if "attention" in result:
        path = os.path.join(output_dir, f"{prefix}_attention.npz")
        np.savez_compressed(
            path,
            attention=result["attention"],
            gene_names=result["gene_names"],
            labels=result["labels"],
            valid_mask=result["valid_mask"],
            **({"cell_counts": result["cell_counts"]} if "cell_counts" in result else {}),
        )
        shape = result["attention"].shape
        size_mb = result["attention"].nbytes / 1024 / 1024
        print(f"Attention saved: {path}  shape={shape}  size={size_mb:.1f}MB")

    if "embedding" in result:
        path = os.path.join(output_dir, f"{prefix}_embedding.npz")
        np.savez_compressed(
            path,
            embedding=result["embedding"],
            gene_names=result["gene_names"],
            labels=result["labels"],
            valid_mask=result["valid_mask"],
            **({"cell_counts": result["cell_counts"]} if "cell_counts" in result else {}),
        )
        shape = result["embedding"].shape
        size_mb = result["embedding"].nbytes / 1024 / 1024
        print(f"Embedding saved: {path}  shape={shape}  size={size_mb:.1f}MB")


def print_summary(result: dict[str, Any]):
    """打印分析结果的摘要统计信息。"""
    print("\n" + "=" * 60)
    print("GENE ANALYSIS SUMMARY")
    print("=" * 60)
    p = len(result["gene_names"])
    n = len(result["labels"])
    print(f"  Genes (p):          {p}")
    print(f"  Output rows (n):    {n}")
    if "cell_counts" in result:
        counts = result["cell_counts"]
        print(f"  Group sizes:        min={counts.min()}, "
              f"max={counts.max()}, "
              f"total={counts.sum()}")
        empty_groups = int((counts == 0).sum())
        if empty_groups > 0:
            import warnings
            empty_labels = result["labels"][counts == 0]
            warnings.warn(
                f"{empty_groups} group(s) have 0 cells and will be all zeros: "
                f"{list(empty_labels[:5])}"
                f"{'...' if empty_groups > 5 else ''}",
                stacklevel=2,
            )
    valid = result["valid_mask"]
    avg_valid = valid.sum(axis=1).mean()
    print(f"  Avg valid genes:    {avg_valid:.1f} / {p}")
    if "attention" in result:
        attn = result["attention"]
        print(f"  Attention shape:    {attn.shape}  dtype={attn.dtype}")
        print(f"  Attention memory:   {attn.nbytes / 1024 / 1024:.1f} MB")
    if "embedding" in result:
        emb = result["embedding"]
        print(f"  Embedding shape:    {emb.shape}  dtype={emb.dtype}")
        print(f"  Embedding memory:   {emb.nbytes / 1024 / 1024:.1f} MB")
    print("=" * 60)


# ============================================================================
# 物种/平台自动推断 / Species & Assay Inference
# (从文件路径推断, 作为用户未显式指定时的回退)
# ============================================================================

def infer_specie_from_path(adata_path: str) -> str | None:
    """从文件路径推断物种 (human / mouse / macaque / marmoset)。"""
    text = str(adata_path).lower()
    if "marmoset" in text:
        return "marmoset"
    if any(t in text for t in ("macaque", "macaca")):
        return "macaque"
    if "human" in text:
        return "human"
    if "mouse" in text:
        return "mouse"
    return None


def infer_assay_from_path(adata_path: str) -> str | None:
    """从文件路径推断测序平台 (merfish / xenium / starmap / ...)。"""
    text = str(adata_path).lower()
    if "merfish" in text:
        return "merfish"
    if "xenium" in text:
        return "xenium"
    if "starmap" in text:
        return "starmap"
    if "slideseq" in text:
        return "slideseqv2"
    if "stereo" in text:
        return "stereo"
    if "snrna" in text or "scrna" in text:
        return "snrna"
    return None


# ============================================================================
# 命令行接口 / CLI
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Extract gene-level attention matrices and embeddings from BrainBeacon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 按 cell type 分组, 提取 attention + embedding (推荐)
  python inference_gene_analysis.py \\
    --adata-path data.h5ad --pretrain-ckpt ckpt.pt \\
    --gene-dict-path gene_dict.h5ad \\
    --token-data-path tokens/ \\
    --mode both --group-by cell_type

  # 逐 cell 输出 attention, 限制 1000 cell
  python inference_gene_analysis.py \\
    --adata-path data.h5ad --pretrain-ckpt ckpt.pt \\
    --mode attention --max-cells 1000
""",
    )

    # --- 输入路径 ---
    g_input = parser.add_argument_group("Input")
    g_input.add_argument("--config", type=str, default=None,
                         help="YAML config path (覆盖默认 config_train).")
    g_input.add_argument("--adata-path", type=str, required=True,
                         help="Input h5ad path (AnnData with p genes).")
    g_input.add_argument("--pretrain-ckpt", type=str, default=None,
                         help="BrainBeacon pretrained checkpoint (.pt).")
    g_input.add_argument("--gene-dict-path", type=str, default=None,
                         help="Gene dictionary h5ad (用于 token -> gene name 映射).")
    g_input.add_argument("--token-data-path", type=str, default=None,
                         help="Pre-tokenized data directory. 如未设置则自动 tokenize.")

    # --- Tokenization 参数 ---
    g_token = parser.add_argument_group("Tokenization")
    g_token.add_argument("--tokenize-specie", type=str, default=None,
                         help="物种 (human/mouse/macaque/marmoset). 默认从路径推断.")
    g_token.add_argument("--tokenize-assay", type=str, default=None,
                         help="平台 (merfish/xenium/starmap/...). 默认从路径推断.")
    g_token.add_argument("--use-hvg", type=str, default="true",
                         help="是否使用高变基因 (默认 true).")
    g_token.add_argument("--n-hvg", type=int, default=1000,
                         help="高变基因数量 (默认 1000).")
    g_token.add_argument("--force-tokenize", type=str, default="false",
                         help="强制重新 tokenize (即使已有 token 文件).")

    # --- 分析模式 ---
    g_mode = parser.add_argument_group("Analysis Mode")
    g_mode.add_argument("--mode", type=str, default="both",
                        choices=["attention", "embedding", "both"],
                        help="提取内容: attention / embedding / both (默认 both).")
    g_mode.add_argument("--attention-layers", type=str, default="last",
                        choices=["last", "all"],
                        help="Transformer attention 层: last (仅最后层) / all (所有层均值).")

    # --- 内存管理 ---
    g_mem = parser.add_argument_group("Memory Management")
    g_mem.add_argument("--group-by", type=str, default=None,
                       help="按 adata.obs 列分组取均值 (如 cell_type). 大幅减少内存.")
    g_mem.add_argument("--max-cells", type=int, default=None,
                       help="限制处理 cell 数 (调试用).")
    g_mem.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "float32"],
                       help="输出精度 (默认 float16, 节省 50%% 内存).")

    # --- 输出 ---
    g_out = parser.add_argument_group("Output")
    g_out.add_argument("--output-dir", type=str, default=None,
                       help="输出目录. 默认与 adata 同目录.")
    g_out.add_argument("--output-prefix", type=str, default="gene_analysis",
                       help="输出文件名前缀 (默认 gene_analysis).")

    # --- 设备 & 杂项 ---
    g_misc = parser.add_argument_group("Device & Misc")
    g_misc.add_argument("--device", type=str, default=None,
                        help="Torch device (cpu / cuda / cuda:0). 默认自动检测.")
    g_misc.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="覆盖 config 中的任意 key. 可多次使用.")
    g_misc.add_argument("--print-config", action="store_true",
                        help="打印解析后的完整 config 后再运行.")

    return parser


def main() -> None:
    """主入口: 解析参数 -> 加载数据 -> tokenize -> 推理 -> 保存结果。"""
    parser = build_parser()
    args = parser.parse_args()

    # 延迟 import 以加速 --help 响应
    from anndata import read_h5ad
    from brainbeacon.pipeline.cell_embedding import (
        normalize_brainbeacon_model_config,
        run_tokenization,
    )
    from brainbeacon.configs.config import resolve_path

    # ------------------------------------------------------------------
    # 1. 加载配置
    # ------------------------------------------------------------------
    config = load_config(args.config, args.set)
    config["masking_p"] = 0  # inference 模式: 不做 masking
    config = normalize_brainbeacon_model_config(config)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=True, allow_unicode=True))

    # ------------------------------------------------------------------
    # 2. 解析关键路径
    # ------------------------------------------------------------------
    pretrain_ckpt = args.pretrain_ckpt or config.get("pretrain_ckpt")
    gene_dict_path = (args.gene_dict_path
                      or config.get("gene_dict_path")
                      or resolve_path("GENE_DICT_PATH"))
    if not pretrain_ckpt:
        raise ValueError("Missing --pretrain-ckpt or config pretrain_ckpt")
    if not gene_dict_path:
        raise ValueError("Missing --gene-dict-path or config gene_dict_path")

    # ------------------------------------------------------------------
    # 3. 设置推理设备
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 4. 加载 AnnData
    # ------------------------------------------------------------------
    print(f"Loading adata: {args.adata_path}")
    adata = read_h5ad(args.adata_path)
    print(f"  Shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    if args.group_by:
        n_groups = adata.obs[args.group_by].nunique()
        print(f"  Groups ({args.group_by}): {n_groups}")

    # ------------------------------------------------------------------
    # 5. Tokenization (如果需要)
    # ------------------------------------------------------------------
    if args.token_data_path and os.path.isdir(args.token_data_path):
        # 已有 tokenized 数据, 直接使用
        token_data_path = args.token_data_path
        print(f"Using existing tokens: {token_data_path}")
    else:
        # 需要 tokenize: 推断物种和平台
        specie = args.tokenize_specie or infer_specie_from_path(args.adata_path)
        assay = args.tokenize_assay or infer_assay_from_path(args.adata_path)
        if not specie:
            raise ValueError("Cannot infer specie from path. Set --tokenize-specie.")
        if not assay:
            raise ValueError("Cannot infer assay from path. Set --tokenize-assay.")

        # 默认 token 输出目录: 与 adata 同目录, 名称加后缀
        token_data_path = args.token_data_path or str(
            Path(args.adata_path).with_name(
                Path(args.adata_path).stem + "_gene_analysis_tokens"
            )
        )
        use_hvg = args.use_hvg.lower() in ("true", "1", "yes")
        force_tokenize = args.force_tokenize.lower() in ("true", "1", "yes")
        print(f"Tokenizing: specie={specie}, assay={assay}")
        token_data_path = run_tokenization(
            adata_path=args.adata_path,
            bb_token_dir=token_data_path,
            gene_dict_path=gene_dict_path,
            specie=specie,
            assay=assay,
            use_hvg=use_hvg,
            n_hvg=args.n_hvg,
            force_tokenize=force_tokenize,
        )

    # ------------------------------------------------------------------
    # 6. 运行 gene-level 推理
    # ------------------------------------------------------------------
    result = run_gene_level_inference(
        adata=adata,
        token_data_path=token_data_path,
        config_train=config,
        pretrain_ckpt=pretrain_ckpt,
        gene_dict_path=gene_dict_path,
        device=device,
        mode=args.mode,
        group_by=args.group_by,
        attention_layers=args.attention_layers,
        max_cells=args.max_cells,
        out_dtype=args.dtype,
    )

    # ------------------------------------------------------------------
    # 7. 输出结果
    # ------------------------------------------------------------------
    print_summary(result)

    output_dir = args.output_dir or str(Path(args.adata_path).parent)
    save_results(result, output_dir, prefix=args.output_prefix)

    print("\nDone.")


if __name__ == "__main__":
    main()
