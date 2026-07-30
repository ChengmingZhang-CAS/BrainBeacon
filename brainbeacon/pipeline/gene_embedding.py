"""Hook-based gene-level attention and embedding extraction for BrainBeacon."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import logging
from numbers import Integral, Real
import os
from pathlib import Path
import platform
import re
import stat
import sys
from typing import Any, Mapping
import uuid
import warnings

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader

from brainbeacon.pipeline.cell_embedding import (
    CellEmbeddingPipeline,
    normalize_brainbeacon_model_config,
    run_tokenization,
)
from brainbeacon.configs.config import MAX_LENGTH as TOKEN_SEQUENCE_LIMIT

_TOKENIZER_FUNCTION = run_tokenization

FORMAT_VERSION = 2
VALID_OUTPUT_AXES = {"compact", "full"}
VALID_ATTENTION_AVERAGES = {"population", "coobserved", "both"}
VALID_HVG_SOURCES = {"raw", "x_log"}
VALID_HVG_FLAVORS = {"seurat_v3", "seurat", "cell_ranger"}
REQUIRED_CONFIG_KEYS = {
    "dim_model",
    "nheads",
    "nlayers",
    "n_aux",
    "strict_load",
    "use_dev_abs",
}

AUX_TOKEN_OFFSET = 20
VALID_MODES = {"attention", "embedding", "both"}
VALID_ATTENTION_LAYER_KEYWORDS = {"last", "all"}
VALID_DTYPES = {"float16", "float32"}
REQUIRED_TOKEN_FILE_PREFIXES = (
    "real_indices_",
    "attention_mask_",
    "connect_comp_",
    "rna_type_",
    "cell_raw_index_",
    "neighbor_gene_distribution_",
    "exp_",
)
ZEROSHOT_LOADER_FAMILY_TOKENS = {
    prefix: ("cell_raw_index" if prefix == "cell_raw_index_" else prefix)
    for prefix in REQUIRED_TOKEN_FILE_PREFIXES
}
TOKEN_MANIFEST_NAME = "token_manifest.json"
TOKENIZATION_LOCK_NAME = ".tokenization.lock"
TOKEN_SCHEMA_NAME = "brainbeacon_gene_tokens"
TOKEN_SCHEMA_VERSION = 2
RESULT_SCHEMA_NAME = "brainbeacon_gene_analysis_result"
RESULT_SCHEMA_VERSION = 2
RESULT_COMMON_NPZ_KEYS = (
    "gene_names",
    "original_gene_indices",
    "gene_token_ids",
    "labels",
    "cell_counts",
    "gene_counts",
    "gene_coverage",
    "valid_mask",
)

logger = logging.getLogger(__name__)


def _parse_attention_layer_items(
    value: str,
) -> tuple[str, tuple[int, ...] | None]:
    if not isinstance(value, str):
        raise TypeError("attention_layers must be a string")
    requested = value.strip()
    if not requested:
        raise ValueError("attention_layers must not be empty")
    if requested in VALID_ATTENTION_LAYER_KEYWORDS:
        return requested, None
    items = [item.strip() for item in requested.split(",")]
    if any(not item for item in items):
        raise ValueError("attention_layers contains an empty item")
    if any(re.fullmatch(r"[+-]?\d+", item) is None for item in items):
        raise ValueError(
            "attention_layers must be 'last', 'all', or comma-separated integers"
        )
    return requested, tuple(int(item) for item in items)


def resolve_attention_layers(
    value: str,
    *,
    nlayers: int,
) -> dict[str, object]:
    if not isinstance(nlayers, Integral) or isinstance(
        nlayers,
        (bool, np.bool_),
    ):
        raise TypeError("nlayers must be a positive integer")
    nlayers = int(nlayers)
    if nlayers <= 0:
        raise ValueError("nlayers must be a positive integer")
    requested, raw_indices = _parse_attention_layer_items(value)
    if requested == "last":
        indices = (nlayers - 1,)
        hook_target: str | tuple[int, ...] | None = "last"
    elif requested == "all":
        indices = tuple(range(nlayers))
        hook_target = None
    else:
        assert raw_indices is not None
        resolved: list[int] = []
        for index in raw_indices:
            if index < -nlayers or index >= nlayers:
                raise ValueError(
                    f"attention layer index {index} is outside "
                    f"[-{nlayers}, {nlayers - 1}]"
                )
            canonical = index if index >= 0 else nlayers + index
            if canonical in resolved:
                raise ValueError(
                    "attention_layers contains duplicate effective layer "
                    f"{canonical}"
                )
            resolved.append(canonical)
        indices = tuple(resolved)
        hook_target = indices
    return {
        "requested": requested,
        "indices": indices,
        "hook_target": hook_target,
        "aggregation": "mean",
    }


def _validate_options(
    *,
    mode: str,
    attention_layers: str,
    out_dtype: str,
    max_cells: int | None,
    hvg_source: str = "raw",
    hvg_flavor: str = "seurat_v3",
    output_axis: str = "compact",
    attention_average: str = "both",
    memory_limit_gib: float | None = None,
    allow_unverified_legacy_tokens: bool = False,
    use_hvg: bool = True,
) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    _parse_attention_layer_items(attention_layers)
    if out_dtype not in VALID_DTYPES:
        raise ValueError(
            f"out_dtype must be one of {sorted(VALID_DTYPES)}, got {out_dtype!r}"
        )
    if max_cells is not None:
        if not isinstance(max_cells, Integral) or isinstance(
            max_cells, (bool, np.bool_)
        ):
            raise TypeError("max_cells must be a positive integer or None")
        if int(max_cells) <= 0:
            raise ValueError("max_cells must be a positive integer or None")
    if output_axis not in VALID_OUTPUT_AXES:
        raise ValueError(
            f"output_axis must be one of {sorted(VALID_OUTPUT_AXES)}, "
            f"got {output_axis!r}"
        )
    if attention_average not in VALID_ATTENTION_AVERAGES:
        raise ValueError(
            "attention_average must be one of "
            f"{sorted(VALID_ATTENTION_AVERAGES)}, got {attention_average!r}"
        )
    if hvg_source not in VALID_HVG_SOURCES:
        raise ValueError(
            f"hvg_source must be one of {sorted(VALID_HVG_SOURCES)}, "
            f"got {hvg_source!r}"
        )
    if hvg_flavor not in VALID_HVG_FLAVORS:
        raise ValueError(
            f"hvg_flavor must be one of {sorted(VALID_HVG_FLAVORS)}, "
            f"got {hvg_flavor!r}"
        )
    if hvg_flavor in {"seurat", "cell_ranger"} and hvg_source != "x_log":
        raise ValueError(
            f"hvg_flavor={hvg_flavor!r} requires hvg_source='x_log'"
        )
    if not isinstance(allow_unverified_legacy_tokens, (bool, np.bool_)):
        raise TypeError("allow_unverified_legacy_tokens must be boolean")
    if not isinstance(use_hvg, (bool, np.bool_)):
        raise TypeError("use_hvg must be boolean")
    if memory_limit_gib is not None:
        if not isinstance(memory_limit_gib, Real) or isinstance(
            memory_limit_gib, (bool, np.bool_)
        ):
            raise TypeError("memory_limit_gib must be a positive finite number or None")
        if not np.isfinite(memory_limit_gib) or float(memory_limit_gib) <= 0:
            raise ValueError("memory_limit_gib must be a positive finite number or None")


def _require_config(config: Mapping[str, Any]) -> dict[str, Any]:
    missing = sorted(REQUIRED_CONFIG_KEYS.difference(config))
    if missing:
        raise KeyError("Missing required config keys: " + ", ".join(missing))

    local = deepcopy(dict(config))
    for key in ("strict_load", "use_dev_abs"):
        if not isinstance(local[key], (bool, np.bool_)):
            raise TypeError(f"config[{key!r}] must be boolean")
    if not isinstance(local["n_aux"], Integral) or isinstance(
        local["n_aux"], (bool, np.bool_)
    ):
        raise TypeError("config['n_aux'] must be a positive integer")
    if int(local["n_aux"]) <= 0:
        raise ValueError("config['n_aux'] must be a positive integer")
    return local


def select_smoke_adata(
    adata_path: str | Path,
    *,
    group_by: str,
    group_value: str,
    n_cells: int,
    n_genes: int,
    gene_filter_source: str = "x",
) -> ad.AnnData:
    """Materialize a deterministic, bounded subset from a large H5AD."""
    source_path = Path(adata_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {source_path}")
    if n_cells < 3:
        raise ValueError("n_cells must be at least 3 for tokenizer gene filtering")
    if n_genes <= 0:
        raise ValueError("n_genes must be positive")
    if gene_filter_source not in {"x", "raw"}:
        raise ValueError(
            "gene_filter_source must be either 'x' or 'raw'"
        )

    source = ad.read_h5ad(source_path, backed="r")
    try:
        if group_by not in source.obs.columns:
            raise ValueError(f"Column {group_by!r} not found in adata.obs")
        group_values = source.obs[group_by]
        matching = np.flatnonzero(
            group_values.notna().to_numpy()
            & (group_values.astype(str).to_numpy() == group_value)
        )
        if matching.size < n_cells:
            raise ValueError(
                f"Only {matching.size} cells have {group_by}={group_value!r}; "
                f"requested {n_cells}"
            )
        selected = source[matching[:n_cells], :].to_memory()
    finally:
        source.file.close()

    detection_matrix = selected.X
    if gene_filter_source == "raw":
        if selected.raw is None:
            raise ValueError("gene_filter_source='raw' requires adata.raw")
        if (
            not selected.var_names.is_unique
            or not selected.raw.var_names.is_unique
        ):
            raise ValueError(
                "Raw smoke gene filtering requires unique current and raw var_names"
            )
        raw_positions = selected.raw.var_names.get_indexer(selected.var_names)
        if np.any(raw_positions < 0):
            missing = (
                selected.var_names[raw_positions < 0].astype(str).tolist()
            )
            raise ValueError(
                "adata.raw is missing current genes: " + ", ".join(missing)
            )
        detection_matrix = selected.raw.X[:, raw_positions]

    detected_cells = np.asarray((detection_matrix != 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(detected_cells >= 3)
    if eligible.size < n_genes:
        raise ValueError(
            f"Only {eligible.size} genes are detected in at least three selected "
            f"cells; requested {n_genes}"
        )

    if "highly_variable_rank" in selected.var.columns:
        ranks = np.asarray(selected.var["highly_variable_rank"], dtype=float)
        ranks = np.where(np.isfinite(ranks), ranks, np.inf)
        eligible = eligible[np.argsort(ranks[eligible], kind="stable")]

    selected = selected[:, eligible[:n_genes]].copy()
    if selected.obs[group_by].astype(str).ne(group_value).any():
        raise RuntimeError("Selected AnnData contains an unexpected group label")
    return selected


def load_analysis_adata(
    adata_path: str | Path,
    *,
    smoke: bool,
    group_by: str,
    group_value: str,
    n_cells: int,
    n_genes: int,
    gene_filter_source: str = "x",
) -> ad.AnnData:
    """Load either a deterministic smoke subset or the complete AnnData."""
    if smoke:
        return select_smoke_adata(
            adata_path,
            group_by=group_by,
            group_value=group_value,
            n_cells=n_cells,
            n_genes=n_genes,
            gene_filter_source=gene_filter_source,
        )

    source_path = Path(adata_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"AnnData file not found: {source_path}")
    return ad.read_h5ad(source_path)


def _validate_unique_gene_names(names, label: str) -> list[str]:
    string_names = [str(name) for name in names]
    if len(set(string_names)) != len(string_names):
        raise ValueError(f"{label} gene names must be unique")
    folded = [name.casefold() for name in string_names]
    if len(set(folded)) != len(folded):
        raise ValueError(f"{label} gene names contain a case-fold collision")
    return string_names


def build_token_to_gene_map(
    gene_dict_path: str | Path,
    n_aux: int,
) -> dict[int, str]:
    import scanpy as sc

    if not isinstance(n_aux, Integral) or isinstance(n_aux, (bool, np.bool_)):
        raise TypeError("n_aux must be a positive integer")
    n_aux = int(n_aux)
    if n_aux <= 0:
        raise ValueError("n_aux must be a positive integer")
    gene_dict = sc.read_h5ad(gene_dict_path)
    if "gene_id" not in gene_dict.var.columns:
        raise KeyError("gene dictionary requires var['gene_id']")
    names = _validate_unique_gene_names(
        gene_dict.var_names,
        "gene dictionary",
    )
    gene_ids: list[int] = []
    seen_gene_ids: set[int] = set()
    max_gene_id = int(np.iinfo(np.int64).max) - n_aux
    for raw_gene_id in gene_dict.var["gene_id"].array:
        if isinstance(raw_gene_id, (bool, np.bool_)) or not isinstance(
            raw_gene_id, Integral
        ):
            raise ValueError("gene_id values must be finite integers")
        gene_id = int(raw_gene_id)
        if gene_id < 0:
            raise ValueError("gene_id values must be non-negative")
        if gene_id > max_gene_id:
            raise ValueError("gene_id plus n_aux overflows int64 token IDs")
        if gene_id in seen_gene_ids:
            raise ValueError("gene_id values must be unique")
        seen_gene_ids.add(gene_id)
        gene_ids.append(gene_id)
    if len(gene_ids) != len(names):
        raise RuntimeError("gene_id and gene-name lengths do not match")
    token_ids_python = [gene_id + n_aux for gene_id in gene_ids]
    if any(token_id < n_aux for token_id in token_ids_python):
        raise ValueError("gene token IDs must not overlap the auxiliary-token range")
    if len(set(token_ids_python)) != len(token_ids_python):
        raise ValueError("gene_id values must be unique")
    token_ids = np.asarray(token_ids_python, dtype=np.int64)
    return {
        int(token_id): gene_name
        for token_id, gene_name in zip(token_ids, names)
    }


class _GeneNameIndex(dict[str, int]):
    def __init__(self, names: list[str]):
        super().__init__((name, index) for index, name in enumerate(names))
        self.folded = {
            name.casefold(): index
            for index, name in enumerate(names)
        }


def build_adata_gene_index(adata) -> _GeneNameIndex:
    names = _validate_unique_gene_names(adata.var_names, "AnnData")
    return _GeneNameIndex(names)


def _match_gene_name(
    gene_name: str,
    adata_gene_idx: _GeneNameIndex,
) -> int | None:
    if gene_name in adata_gene_idx:
        return adata_gene_idx[gene_name]
    return adata_gene_idx.folded.get(gene_name.casefold())


def _has_token_joblib_bundles(token_data_path: str) -> bool:
    try:
        _inspect_required_token_jobs(Path(token_data_path))
    except (FileNotFoundError, NotADirectoryError, RuntimeError):
        return False
    return True


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be a positive integer")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _as_unicode(values) -> np.ndarray:
    array = np.asarray(values)
    converted = np.asarray(
        [str(value) for value in array.reshape(-1)],
        dtype=str,
    )
    return converted.reshape(array.shape)


def _hash_index(index) -> str:
    digest = hashlib.sha256()
    digest.update(b"brainbeacon-index-v1\0")
    digest.update(str(len(index)).encode("ascii"))
    digest.update(b"\0")
    for value in index:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _update_digest_array(
    digest,
    array,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> None:
    values = np.asarray(array)
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(values.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    flat = values.reshape(-1) if values.flags.c_contiguous else None
    if flat is not None:
        items = max(1, int(chunk_bytes) // max(values.dtype.itemsize, 1))
        for start in range(0, flat.size, items):
            chunk = np.ascontiguousarray(flat[start : start + items])
            digest.update(memoryview(chunk).cast("B"))
        return
    contiguous = np.ascontiguousarray(values)
    digest.update(memoryview(contiguous).cast("B"))


def _hash_matrix(
    matrix,
    *,
    column_indices=None,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    """Hash dense, CSR, or CSC matrices in bounded row chunks."""
    from scipy import sparse

    if not isinstance(chunk_bytes, Integral) or isinstance(chunk_bytes, bool):
        raise TypeError("chunk_bytes must be a positive integer")
    if int(chunk_bytes) <= 0:
        raise ValueError("chunk_bytes must be a positive integer")
    if len(matrix.shape) != 2:
        raise ValueError("matrix hashing requires a two-dimensional matrix")
    if sparse.issparse(matrix) and getattr(matrix, "format", None) not in {
        "csr",
        "csc",
    }:
        raise TypeError(
            "matrix hashing supports dense, CSR, and CSC inputs; "
            f"unsupported sparse format: {getattr(matrix, 'format', 'unknown')}"
        )

    selected = None
    if column_indices is not None:
        selected = np.asarray(column_indices, dtype=np.int64)
        if selected.ndim != 1:
            raise ValueError("column_indices must be one-dimensional")
        if selected.size and (
            int(selected.min()) < 0 or int(selected.max()) >= matrix.shape[1]
        ):
            raise IndexError("column_indices are outside the matrix")
    output_n_vars = matrix.shape[1] if selected is None else int(selected.size)

    digest = hashlib.sha256()
    digest.update(b"brainbeacon-matrix-v1\0")
    digest.update(
        json.dumps([int(matrix.shape[0]), output_n_vars], separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    dtype_value = getattr(matrix, "dtype", None)
    if dtype_value is None:
        dtype_value = np.asarray(matrix[: min(1, matrix.shape[0])]).dtype
    dtype = np.dtype(dtype_value)
    digest.update(dtype.str.encode("ascii"))
    digest.update(b"\0")

    bytes_per_row = max(1, output_n_vars * max(dtype.itemsize, 1))
    rows_per_chunk = max(1, int(chunk_bytes) // bytes_per_row)
    if sparse.issparse(matrix):
        digest.update(b"sparse-logical-v1\0")
        record_dtype = np.dtype(
            [("row", "<i8"), ("column", "<i8"), ("value", dtype.str)],
            align=False,
        )
        for start in range(0, matrix.shape[0], rows_per_chunk):
            stop = min(matrix.shape[0], start + rows_per_chunk)
            block = matrix[start:stop]
            if selected is not None:
                block = block[:, selected]
            coo = block.tocoo(copy=False)
            if coo.nnz:
                order = np.lexsort((coo.col, coo.row))
                rows = np.asarray(coo.row[order], dtype=np.int64) + start
                cols = np.asarray(coo.col[order], dtype=np.int64)
                data = np.asarray(coo.data[order])
            else:
                rows = np.empty(0, dtype=np.int64)
                cols = np.empty(0, dtype=np.int64)
                data = np.empty(0, dtype=dtype)
            records = np.empty(rows.size, dtype=record_dtype)
            records["row"] = rows
            records["column"] = cols
            records["value"] = data
            digest.update(memoryview(records).cast("B"))
    else:
        digest.update(b"dense-v1\0")
        for start in range(0, matrix.shape[0], rows_per_chunk):
            stop = min(matrix.shape[0], start + rows_per_chunk)
            block = matrix[start:stop]
            if selected is not None:
                block = block[:, selected]
            contiguous = np.ascontiguousarray(block)
            digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _hash_obs(obs) -> str:
    import pandas as pd

    # Deliberately hash every obs column. This conservative provenance policy
    # may invalidate reuse for metadata-only changes, but cannot omit a real
    # tokenizer dependency as tokenizer behavior evolves.
    digest = hashlib.sha256()
    digest.update(b"brainbeacon-obs-v1\0")
    digest.update(_hash_index(obs.index).encode("ascii"))
    for column in obs.columns:
        name = str(column).encode("utf-8")
        digest.update(len(name).to_bytes(8, "little", signed=False))
        digest.update(name)
        series = obs[column]
        digest.update(str(series.dtype).encode("utf-8"))
        hashes = pd.util.hash_pandas_object(
            series,
            index=False,
            categorize=True,
        ).to_numpy(dtype=np.uint64, copy=False)
        _update_digest_array(digest, hashes)
    return digest.hexdigest()


def _hash_spatial(adata) -> str | None:
    if "spatial" not in adata.obsm:
        return None
    return _hash_matrix(adata.obsm["spatial"])


def _source_sha256(value) -> str:
    source_path = inspect.getsourcefile(value)
    if source_path and Path(source_path).is_file():
        return _sha256_file(source_path)
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        source = repr(value).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _canonical_mapping_sha256(namespace: str, values: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(values),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(namespace.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _build_adata_content_sha256(
    *,
    shape,
    x_sha256: str,
    obs_names_sha256: str,
    var_names_sha256: str,
    obs_sha256: str,
    spatial_sha256: str | None,
) -> str:
    return _canonical_mapping_sha256(
        "brainbeacon-adata-content-v2",
        {
            "shape": [int(shape[0]), int(shape[1])],
            "x_sha256": x_sha256,
            "obs_names_sha256": obs_names_sha256,
            "var_names_sha256": var_names_sha256,
            "obs_sha256": obs_sha256,
            "spatial_sha256": spatial_sha256,
        },
    )


def _build_hvg_source_content_sha256(
    *,
    source: str,
    shape,
    matrix_sha256: str,
    obs_names_sha256: str,
    var_names_sha256: str,
) -> str:
    return _canonical_mapping_sha256(
        "brainbeacon-hvg-source-v1",
        {
            "source": source,
            "shape": [int(shape[0]), int(shape[1])],
            "matrix_sha256": matrix_sha256,
            "obs_names_sha256": obs_names_sha256,
            "var_names_sha256": var_names_sha256,
        },
    )


def _build_unused_hvg_source_fingerprint() -> dict[str, Any]:
    reason = "use_hvg_false"
    return {
        "used": False,
        "source": "unused",
        "reason": reason,
        "content_sha256": _canonical_mapping_sha256(
            "brainbeacon-hvg-source-unused-v1",
            {"used": False, "source": "unused", "reason": reason},
        ),
    }


def _build_tokenizer_dependency_sha256(
    *,
    tokenizer_source_sha256: str,
    wrapper_source_sha256: str,
    config_source_sha256: str,
    stage1_config_source_sha256: str,
    max_length: int,
    aux_token_length: int,
    tokenizer_config_n_aux: int,
    tokenizer_config_n_tokens: int,
) -> str:
    return _canonical_mapping_sha256(
        "brainbeacon-tokenizer-dependency-v2",
        {
            "tokenizer_source_sha256": tokenizer_source_sha256,
            "wrapper_source_sha256": wrapper_source_sha256,
            "config_source_sha256": config_source_sha256,
            "stage1_config_source_sha256": stage1_config_source_sha256,
            "max_length": int(max_length),
            "aux_token_length": int(aux_token_length),
            "tokenizer_config_n_aux": int(tokenizer_config_n_aux),
            "tokenizer_config_n_tokens": int(tokenizer_config_n_tokens),
        },
    )


def _tokenizer_metadata() -> dict[str, Any]:
    import importlib
    import brainbeacon.configs.config as tokenizer_config
    import brainbeacon.tokenizer as tokenizer_module

    stage1_config_module = importlib.import_module(
        "brainbeacon.configs.stage1_config"
    )
    tokenizer_source_sha256 = _source_sha256(tokenizer_module)
    wrapper_source_sha256 = _source_sha256(_TOKENIZER_FUNCTION)
    config_source_sha256 = _source_sha256(tokenizer_config)
    stage1_config_source_sha256 = _source_sha256(stage1_config_module)
    max_length = int(tokenizer_module.MAX_LENGTH)
    aux_token_length = int(tokenizer_module.AUX_TOKEN)
    runtime_config = tokenizer_module.config_train
    for key in ("n_aux", "n_tokens"):
        value = runtime_config.get(key) if isinstance(runtime_config, Mapping) else None
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(
                f"brainbeacon.tokenizer.config_train[{key!r}] must be a positive integer"
            )
    tokenizer_config_n_aux = int(runtime_config["n_aux"])
    tokenizer_config_n_tokens = int(runtime_config["n_tokens"])
    if max_length <= 0 or aux_token_length <= 0:
        raise ValueError("Tokenizer MAX_LENGTH and AUX_TOKEN must be positive")

    dependency_sha256 = _build_tokenizer_dependency_sha256(
        tokenizer_source_sha256=tokenizer_source_sha256,
        wrapper_source_sha256=wrapper_source_sha256,
        config_source_sha256=config_source_sha256,
        stage1_config_source_sha256=stage1_config_source_sha256,
        max_length=max_length,
        aux_token_length=aux_token_length,
        tokenizer_config_n_aux=tokenizer_config_n_aux,
        tokenizer_config_n_tokens=tokenizer_config_n_tokens,
    )
    try:
        package_version = importlib.metadata.version("brainbeacon")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    return {
        "tokenizer_module": tokenizer_module.__name__,
        "tokenizer_qualname": "tokenization_h5ad",
        "tokenizer_source_sha256": tokenizer_source_sha256,
        "tokenizer_wrapper_module": getattr(
            _TOKENIZER_FUNCTION,
            "__module__",
            "unknown",
        ),
        "tokenizer_wrapper_qualname": getattr(
            _TOKENIZER_FUNCTION,
            "__qualname__",
            "run_tokenization",
        ),
        "tokenizer_wrapper_source_sha256": wrapper_source_sha256,
        "tokenizer_config_source_sha256": config_source_sha256,
        "stage1_config_source_sha256": stage1_config_source_sha256,
        "tokenizer_dependency_sha256": dependency_sha256,
        "tokenizer_package_version": package_version,
        "tokenizer_code_version": f"source-sha256:{dependency_sha256}",
        "max_length": max_length,
        "aux_token_length": aux_token_length,
        "tokenizer_config_n_aux": tokenizer_config_n_aux,
        "tokenizer_config_n_tokens": tokenizer_config_n_tokens,
    }


def _adata_fingerprint(adata) -> dict[str, Any]:
    x_sha256 = _hash_matrix(adata.X)
    obs_names_sha256 = _hash_index(adata.obs_names)
    var_names_sha256 = _hash_index(adata.var_names)
    obs_sha256 = _hash_obs(adata.obs)
    spatial_sha256 = _hash_spatial(adata)
    shape = [int(adata.n_obs), int(adata.n_vars)]
    return {
        "shape": shape,
        "adata_content_sha256": _build_adata_content_sha256(
            shape=shape,
            x_sha256=x_sha256,
            obs_names_sha256=obs_names_sha256,
            var_names_sha256=var_names_sha256,
            obs_sha256=obs_sha256,
            spatial_sha256=spatial_sha256,
        ),
        "x_sha256": x_sha256,
        "obs_names_sha256": obs_names_sha256,
        "var_names_sha256": var_names_sha256,
        "obs_sha256": obs_sha256,
        "spatial_sha256": spatial_sha256,
    }


def _hvg_source_fingerprint(adata, hvg_source: str) -> dict[str, Any]:
    if hvg_source == "raw":
        if adata.raw is None:
            raise ValueError("hvg_source='raw' requires adata.raw")
        matrix = adata.raw.X
        obs_names = adata.raw.obs_names
        var_names = adata.raw.var_names
    elif hvg_source == "x_log":
        matrix = adata.X
        obs_names = adata.obs_names
        var_names = adata.var_names
    else:
        raise ValueError(f"Unsupported hvg_source: {hvg_source!r}")
    shape = [int(matrix.shape[0]), int(matrix.shape[1])]
    matrix_sha256 = _hash_matrix(matrix)
    obs_names_sha256 = _hash_index(obs_names)
    var_names_sha256 = _hash_index(var_names)
    return {
        "used": True,
        "source": hvg_source,
        "shape": shape,
        "matrix_sha256": matrix_sha256,
        "obs_names_sha256": obs_names_sha256,
        "var_names_sha256": var_names_sha256,
        "content_sha256": _build_hvg_source_content_sha256(
            source=hvg_source,
            shape=shape,
            matrix_sha256=matrix_sha256,
            obs_names_sha256=obs_names_sha256,
            var_names_sha256=var_names_sha256,
        ),
    }


def _build_token_identity(
    *,
    adata,
    original_adata=None,
    gene_dict_path: str,
    species: str,
    assay: str,
    use_hvg: bool,
    n_hvg: int,
    hvg_source: str,
    hvg_flavor: str,
    use_dev_abs: bool,
    n_aux: int,
    max_cells: int | None = None,
    original_fingerprint: Mapping[str, Any] | None = None,
    token_work_matches_original: bool = False,
) -> dict[str, Any]:
    original = adata if original_adata is None else original_adata
    tokenizer_metadata = _tokenizer_metadata()
    if not (
        int(n_aux)
        == tokenizer_metadata["aux_token_length"]
        == tokenizer_metadata["tokenizer_config_n_aux"]
    ):
        raise ValueError(
            "config n_aux must match tokenizer AUX_TOKEN and "
            "brainbeacon.tokenizer.config_train['n_aux']: "
            f"{n_aux}, {tokenizer_metadata['aux_token_length']}, "
            f"{tokenizer_metadata['tokenizer_config_n_aux']}"
        )
    original_fingerprint_value = (
        _adata_fingerprint(original)
        if original_fingerprint is None
        else deepcopy(dict(original_fingerprint))
    )
    token_work_fingerprint = (
        original_fingerprint_value
        if token_work_matches_original or original is adata
        else _adata_fingerprint(adata)
    )
    hvg_source_fingerprint = (
        _hvg_source_fingerprint(original, hvg_source)
        if use_hvg
        else _build_unused_hvg_source_fingerprint()
    )
    selected_gene_names = [str(name) for name in adata.var_names]
    return {
        "original_input": original_fingerprint_value,
        "hvg_source": hvg_source_fingerprint,
        "token_work": token_work_fingerprint,
        "gene_dictionary": {
            "sha256": _sha256_file(gene_dict_path),
        },
        "selection": {
            "use_hvg": bool(use_hvg),
            "n_hvg": int(n_hvg),
            "hvg_source": hvg_source,
            "hvg_flavor": hvg_flavor,
            "max_cells": None if max_cells is None else int(max_cells),
            "operation_order": "hvg_then_max_cells",
            "selected_gene_names": selected_gene_names,
            "selected_gene_axis_sha256": _hash_index(adata.var_names),
        },
        "tokenization": {
            "species": str(species),
            "assay": str(assay),
            "use_dev_abs": bool(use_dev_abs),
            "n_aux": int(n_aux),
            "configured_n_aux": int(n_aux),
            "tokenizer_use_hvg": False,
            **tokenizer_metadata,
        },
    }


def _inspect_required_token_jobs(root: Path) -> dict[str, Any]:
    if not root.exists():
        raise FileNotFoundError(f"Token directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Token path is not a directory: {root}")
    token_dirs = [
        item
        for item in sorted(root.iterdir())
        if item.name.startswith("tokens-")
    ]
    if not token_dirs:
        raise RuntimeError(f"No tokens-* directories found under {root}")

    required_paths: list[str] = []
    family_paths = {prefix: [] for prefix in REQUIRED_TOKEN_FILE_PREFIXES}
    bundle_count = 0
    bundle_paths: list[dict[str, Path]] = []
    for token_dir in token_dirs:
        if token_dir.is_symlink():
            raise RuntimeError(f"Token directory symlink is forbidden: {token_dir}")
        if not token_dir.is_dir():
            raise RuntimeError(f"Token entry is not a directory: {token_dir}")
        entries = list(token_dir.iterdir())
        for entry in entries:
            if entry.is_symlink():
                raise RuntimeError(f"Token artifact symlink is forbidden: {entry}")
            if not entry.is_file():
                raise RuntimeError(f"Nested token entry is not allowed: {entry}")
        names = {entry.name for entry in entries}
        for name in names:
            loader_matches = [
                prefix
                for prefix, token in ZEROSHOT_LOADER_FAMILY_TOKENS.items()
                if token in name
            ]
            if not loader_matches:
                continue
            canonical_matches = [
                prefix
                for prefix in REQUIRED_TOKEN_FILE_PREFIXES
                if name.startswith(prefix)
                and name.endswith(".job")
                and len(name) > len(prefix) + len(".job")
            ]
            if len(loader_matches) != 1 or canonical_matches != loader_matches:
                raise RuntimeError(
                    "Token filename is ambiguous or unsafe for the Zeroshot "
                    f"loader substring rules: {token_dir / name}"
                )
        suffix_sets = []
        for prefix in REQUIRED_TOKEN_FILE_PREFIXES:
            suffixes = {
                name[len(prefix):]
                for name in names
                if name.startswith(prefix) and name.endswith(".job")
            }
            if not suffixes:
                raise RuntimeError(
                    f"Token directory {token_dir} is missing required family {prefix}"
                )
            suffix_sets.append(suffixes)
        if any(suffixes != suffix_sets[0] for suffixes in suffix_sets[1:]):
            raise RuntimeError(
                "Required token job families are incomplete or contain an "
                f"unlisted required job in {token_dir}"
            )
        suffixes = sorted(suffix_sets[0])
        bundle_count += len(suffixes)
        for suffix in suffixes:
            paired_paths = {}
            for prefix in REQUIRED_TOKEN_FILE_PREFIXES:
                path = token_dir / f"{prefix}{suffix}"
                relative = path.relative_to(root).as_posix()
                required_paths.append(relative)
                family_paths[prefix].append(relative)
                paired_paths[prefix] = path
            bundle_paths.append(paired_paths)
    return {
        "required_paths": sorted(required_paths),
        "families": {
            prefix.rstrip("_"): sorted(paths)
            for prefix, paths in family_paths.items()
        },
        "bundle_count": int(bundle_count),
        "bundle_paths": bundle_paths,
        "token_dirs": token_dirs,
    }


def _token_payload_batch_shape(
    batch: Any,
    family: str,
    path: Path,
) -> tuple[int, ...]:
    if family == "cell_raw_index_":
        if not isinstance(batch, (np.ndarray, list, tuple)):
            raise RuntimeError(
                f"Token payload batch must be an array or sequence: {path}"
            )
        if isinstance(batch, np.ndarray) and batch.ndim != 1:
            raise RuntimeError(
                "cell_raw_index token payload batches must be one-dimensional: "
                f"{path}"
            )
        if any(not np.isscalar(value) for value in batch):
            raise RuntimeError(
                "cell_raw_index token payload batches must contain scalar IDs: "
                f"{path}"
            )
        cell_count = len(batch)
        shape = (int(cell_count),)
    else:
        shape = getattr(batch, "shape", None)
        if shape is None:
            try:
                batch = np.asarray(batch)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"Token payload batch is not a rectangular 2D array: {path}"
                ) from error
            shape = batch.shape
        if len(shape) != 2:
            raise RuntimeError(
                f"Token payload batch must be a two-dimensional array: {path}"
            )
        dtype = getattr(batch, "dtype", None)
        try:
            has_object_dtype = dtype is not None and np.dtype(dtype).hasobject
        except TypeError:
            has_object_dtype = False
        if has_object_dtype:
            raise RuntimeError(
                f"Token payload batch must not use ragged/object dtype: {path}"
            )
        shape = tuple(int(dimension) for dimension in shape)
        if any(dimension <= 0 for dimension in shape):
            raise RuntimeError(
                f"Token payload batch has an empty or zero dimension: {path}"
            )
    if shape[0] <= 0:
        raise RuntimeError(f"Token payload batch has zero cells: {path}")
    return shape


def _token_payload_batch_cell_count(batch: Any, family: str, path: Path) -> int:
    return int(_token_payload_batch_shape(batch, family, path)[0])


def _validate_token_bundle_payloads(structure: Mapping[str, Any]) -> dict[str, int]:
    import joblib

    loader_sample_count = 0
    tokenized_cell_count = 0
    for paired_paths in structure["bundle_paths"]:
        expected_outer_length = None
        expected_batch_cell_counts = None
        expected_non_cell_shapes = None
        ordered_families = ("cell_raw_index_",) + tuple(
            prefix
            for prefix in REQUIRED_TOKEN_FILE_PREFIXES
            if prefix != "cell_raw_index_"
        )
        for family in ordered_families:
            path = paired_paths[family]
            payload = joblib.load(path)
            try:
                if not isinstance(payload, (list, tuple)):
                    raise RuntimeError(
                        f"Token payload outer object must be a list of batches: {path}"
                    )
                outer_length = len(payload)
                if outer_length <= 0:
                    raise RuntimeError(f"Token payload outer batch list is empty: {path}")
                batch_shapes = tuple(
                    _token_payload_batch_shape(batch, family, path)
                    for batch in payload
                )
                batch_cell_counts = tuple(shape[0] for shape in batch_shapes)
            finally:
                del payload
            if expected_outer_length is None:
                expected_outer_length = outer_length
                expected_batch_cell_counts = batch_cell_counts
                loader_sample_count += outer_length
                tokenized_cell_count += sum(batch_cell_counts)
            elif outer_length != expected_outer_length:
                raise RuntimeError(
                    "Token payload outer batch lengths do not match within bundle: "
                    f"{path}"
                )
            elif batch_cell_counts != expected_batch_cell_counts:
                raise RuntimeError(
                    "Token payload batch cell dimensions do not match cell_raw_index: "
                    f"{path}"
                )
            if family != "cell_raw_index_":
                if expected_non_cell_shapes is None:
                    expected_non_cell_shapes = batch_shapes
                elif batch_shapes != expected_non_cell_shapes:
                    raise RuntimeError(
                        "Token payload full 2D shapes do not match within bundle: "
                        f"{path}"
                    )
    if loader_sample_count <= 0 or tokenized_cell_count <= 0:
        raise RuntimeError("Token payload validation found no loader samples or cells")
    return {
        "loader_sample_count": int(loader_sample_count),
        "tokenized_cell_count": int(tokenized_cell_count),
    }


def _token_artifact_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    structure = _inspect_required_token_jobs(root)
    artifacts = []
    for token_dir in structure["token_dirs"]:
        for path in sorted(token_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Unsafe token artifact: {path}")
            artifacts.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": int(path.stat().st_size),
                    "sha256": _sha256_file(path),
                }
            )
    return artifacts, structure


def _environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "anndata": ad.__version__,
        "torch": torch.__version__,
    }
    for package in ("scanpy", "scipy", "brainbeacon"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _build_token_manifest(root: str | Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    artifacts, structure = _token_artifact_inventory(root)
    payload_counts = _validate_token_bundle_payloads(structure)
    return {
        "schema_name": TOKEN_SCHEMA_NAME,
        "schema_version": TOKEN_SCHEMA_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "producer": {
            "code": "gene_embedding_v2.py",
            "code_sha256": _sha256_file(__file__),
            "format_version": FORMAT_VERSION,
            "versions": _environment_versions(),
        },
        "identity": deepcopy(dict(identity)),
        "output": {
            "loader_sample_count": payload_counts["loader_sample_count"],
            "tokenized_cell_count": payload_counts["tokenized_cell_count"],
            "bundle_count": structure["bundle_count"],
            "families": structure["families"],
            "artifacts": artifacts,
        },
    }


def _require_manifest_mapping(
    mapping: Mapping[str, Any],
    key: str,
    required: set[str],
) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Token manifest {key} must be an object")
    missing = sorted(required.difference(value))
    if missing:
        raise RuntimeError(
            f"Token manifest {key} is missing fields: {', '.join(missing)}"
        )
    return value


def _require_manifest_sha256(value: Any, label: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"Token manifest {label} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(
            f"Token manifest {label} must be a SHA256 hex digest"
        ) from exc


def _validate_manifest_shape(value: Any, label: str) -> None:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(
            not isinstance(item, Integral)
            or isinstance(item, bool)
            or int(item) < 0
            for item in value
        )
    ):
        raise RuntimeError(f"Token manifest {label} must contain two nonnegative integers")


def _validate_token_manifest_header(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise RuntimeError("Token manifest root must be an object")
    if manifest.get("schema_name") != TOKEN_SCHEMA_NAME:
        raise RuntimeError("Unsupported token manifest schema name")
    if manifest.get("schema_version") != TOKEN_SCHEMA_VERSION:
        raise RuntimeError("Unsupported token manifest schema version")
    if manifest.get("state") != "complete":
        raise RuntimeError("Token manifest state must be complete")
    if not isinstance(manifest.get("created_at"), str) or not manifest["created_at"]:
        raise RuntimeError("Token manifest created_at is missing or invalid")

    producer = _require_manifest_mapping(
        manifest,
        "producer",
        {"code", "code_sha256", "format_version", "versions"},
    )
    if not isinstance(producer["code"], str) or not producer["code"]:
        raise RuntimeError("Token manifest producer code is invalid")
    _require_manifest_sha256(producer["code_sha256"], "producer code SHA256")
    if (
        not isinstance(producer["format_version"], Integral)
        or isinstance(producer["format_version"], bool)
        or int(producer["format_version"]) != FORMAT_VERSION
    ):
        raise RuntimeError("Token manifest producer format version is invalid")
    versions = producer["versions"]
    required_versions = {"python", "numpy", "anndata", "scanpy", "scipy", "brainbeacon"}
    if not isinstance(versions, Mapping) or not required_versions.issubset(versions):
        raise RuntimeError("Token manifest producer versions are incomplete")
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise RuntimeError("Token manifest producer versions must be non-empty strings")

    identity = _require_manifest_mapping(
        manifest,
        "identity",
        {
            "original_input",
            "hvg_source",
            "token_work",
            "gene_dictionary",
            "selection",
            "tokenization",
        },
    )
    for section_name in ("original_input", "token_work"):
        section = _require_manifest_mapping(
            identity,
            section_name,
            {
                "shape",
                "adata_content_sha256",
                "x_sha256",
                "obs_names_sha256",
                "var_names_sha256",
                "obs_sha256",
                "spatial_sha256",
            },
        )
        _validate_manifest_shape(section["shape"], f"identity.{section_name}.shape")
        for field in (
            "adata_content_sha256",
            "x_sha256",
            "obs_names_sha256",
            "var_names_sha256",
            "obs_sha256",
        ):
            _require_manifest_sha256(
                section[field],
                f"identity.{section_name}.{field}",
            )
        _require_manifest_sha256(
            section["spatial_sha256"],
            f"identity.{section_name}.spatial_sha256",
            allow_none=True,
        )
        expected_content_sha256 = _build_adata_content_sha256(
            shape=section["shape"],
            x_sha256=section["x_sha256"],
            obs_names_sha256=section["obs_names_sha256"],
            var_names_sha256=section["var_names_sha256"],
            obs_sha256=section["obs_sha256"],
            spatial_sha256=section["spatial_sha256"],
        )
        if section["adata_content_sha256"] != expected_content_sha256:
            raise RuntimeError(
                f"Token manifest identity {section_name} content digest is inconsistent"
            )

    hvg_source = _require_manifest_mapping(
        identity,
        "hvg_source",
        {"used", "source", "content_sha256"},
    )
    if not isinstance(hvg_source["used"], bool):
        raise RuntimeError("Token manifest identity hvg_source.used must be boolean")
    _require_manifest_sha256(
        hvg_source["content_sha256"],
        "identity.hvg_source.content_sha256",
    )
    if hvg_source["used"]:
        required_hvg_fields = {
            "shape",
            "matrix_sha256",
            "obs_names_sha256",
            "var_names_sha256",
        }
        missing_hvg_fields = sorted(required_hvg_fields.difference(hvg_source))
        if missing_hvg_fields:
            raise RuntimeError(
                "Token manifest identity hvg_source is missing fields: "
                + ", ".join(missing_hvg_fields)
            )
        if hvg_source["source"] not in VALID_HVG_SOURCES:
            raise RuntimeError("Token manifest identity hvg_source.source is invalid")
        _validate_manifest_shape(hvg_source["shape"], "identity.hvg_source.shape")
        for field in ("matrix_sha256", "obs_names_sha256", "var_names_sha256"):
            _require_manifest_sha256(
                hvg_source[field],
                f"identity.hvg_source.{field}",
            )
        expected_hvg_content_sha256 = _build_hvg_source_content_sha256(
            source=hvg_source["source"],
            shape=hvg_source["shape"],
            matrix_sha256=hvg_source["matrix_sha256"],
            obs_names_sha256=hvg_source["obs_names_sha256"],
            var_names_sha256=hvg_source["var_names_sha256"],
        )
    else:
        if hvg_source.get("source") != "unused" or hvg_source.get("reason") != "use_hvg_false":
            raise RuntimeError("Token manifest unused HVG source sentinel is invalid")
        expected_hvg_content_sha256 = _build_unused_hvg_source_fingerprint()[
            "content_sha256"
        ]
    if hvg_source["content_sha256"] != expected_hvg_content_sha256:
        raise RuntimeError("Token manifest identity HVG source content digest is inconsistent")

    gene_dictionary = _require_manifest_mapping(
        identity,
        "gene_dictionary",
        {"sha256"},
    )
    _require_manifest_sha256(gene_dictionary["sha256"], "identity gene dictionary SHA256")

    selection = _require_manifest_mapping(
        identity,
        "selection",
        {
            "use_hvg",
            "n_hvg",
            "hvg_source",
            "hvg_flavor",
            "max_cells",
            "operation_order",
            "selected_gene_names",
            "selected_gene_axis_sha256",
        },
    )
    if not isinstance(selection["use_hvg"], bool):
        raise RuntimeError("Token manifest identity selection.use_hvg must be boolean")
    if (
        not isinstance(selection["n_hvg"], Integral)
        or isinstance(selection["n_hvg"], bool)
        or int(selection["n_hvg"]) <= 0
    ):
        raise RuntimeError("Token manifest identity selection.n_hvg is invalid")
    if selection["hvg_source"] not in VALID_HVG_SOURCES:
        raise RuntimeError("Token manifest identity selection.hvg_source is invalid")
    if selection["hvg_flavor"] not in VALID_HVG_FLAVORS:
        raise RuntimeError("Token manifest identity selection.hvg_flavor is invalid")
    if selection["operation_order"] != "hvg_then_max_cells":
        raise RuntimeError("Token manifest identity selection operation order is invalid")
    max_cells = selection["max_cells"]
    if max_cells is not None and (
        not isinstance(max_cells, Integral)
        or isinstance(max_cells, bool)
        or int(max_cells) <= 0
    ):
        raise RuntimeError("Token manifest identity selection.max_cells is invalid")
    selected_names = selection["selected_gene_names"]
    if not isinstance(selected_names, list) or any(
        not isinstance(name, str) for name in selected_names
    ):
        raise RuntimeError("Token manifest identity selected gene names are invalid")
    _require_manifest_sha256(
        selection["selected_gene_axis_sha256"],
        "identity selected gene axis SHA256",
    )
    if selection["use_hvg"] != hvg_source["used"]:
        raise RuntimeError("Token manifest identity HVG usage fields are inconsistent")
    if hvg_source["used"] and selection["hvg_source"] != hvg_source["source"]:
        raise RuntimeError("Token manifest identity HVG source fields are inconsistent")
    if selection["selected_gene_axis_sha256"] != _hash_index(selected_names):
        raise RuntimeError("Token manifest identity selected gene axis SHA is inconsistent")
    token_work = identity["token_work"]
    if (
        token_work["var_names_sha256"] != selection["selected_gene_axis_sha256"]
        or int(token_work["shape"][1]) != len(selected_names)
    ):
        raise RuntimeError("Token manifest identity token-work gene axis is inconsistent")
    if (
        hvg_source["used"]
        and hvg_source["obs_names_sha256"]
        != identity["original_input"]["obs_names_sha256"]
    ):
        raise RuntimeError("Token manifest identity HVG observation axis is inconsistent")

    tokenization = _require_manifest_mapping(
        identity,
        "tokenization",
        {
            "species",
            "assay",
            "use_dev_abs",
            "n_aux",
            "configured_n_aux",
            "tokenizer_use_hvg",
            "tokenizer_module",
            "tokenizer_qualname",
            "tokenizer_source_sha256",
            "tokenizer_wrapper_module",
            "tokenizer_wrapper_qualname",
            "tokenizer_wrapper_source_sha256",
            "tokenizer_config_source_sha256",
            "stage1_config_source_sha256",
            "tokenizer_dependency_sha256",
            "tokenizer_package_version",
            "tokenizer_code_version",
            "max_length",
            "aux_token_length",
            "tokenizer_config_n_aux",
            "tokenizer_config_n_tokens",
        },
    )
    for field in (
        "species",
        "assay",
        "tokenizer_module",
        "tokenizer_qualname",
        "tokenizer_wrapper_module",
        "tokenizer_wrapper_qualname",
        "tokenizer_package_version",
        "tokenizer_code_version",
    ):
        if not isinstance(tokenization[field], str) or not tokenization[field]:
            raise RuntimeError(f"Token manifest identity tokenization.{field} is invalid")
    for field in ("use_dev_abs", "tokenizer_use_hvg"):
        if not isinstance(tokenization[field], bool):
            raise RuntimeError(f"Token manifest identity tokenization.{field} must be boolean")
    if tokenization["tokenizer_use_hvg"] is not False:
        raise RuntimeError("Token manifest tokenizer_use_hvg must be false")
    for field in (
        "n_aux",
        "configured_n_aux",
        "max_length",
        "aux_token_length",
        "tokenizer_config_n_aux",
        "tokenizer_config_n_tokens",
    ):
        if (
            not isinstance(tokenization[field], Integral)
            or isinstance(tokenization[field], bool)
            or int(tokenization[field]) <= 0
        ):
            raise RuntimeError(f"Token manifest identity tokenization.{field} is invalid")
    if not (
        int(tokenization["n_aux"])
        == int(tokenization["configured_n_aux"])
        == int(tokenization["aux_token_length"])
        == int(tokenization["tokenizer_config_n_aux"])
    ):
        raise RuntimeError("Token manifest identity auxiliary-token config is inconsistent")
    for field in (
        "tokenizer_source_sha256",
        "tokenizer_wrapper_source_sha256",
        "tokenizer_config_source_sha256",
        "stage1_config_source_sha256",
        "tokenizer_dependency_sha256",
    ):
        _require_manifest_sha256(
            tokenization[field],
            f"identity tokenization {field}",
        )
    expected_dependency_sha256 = _build_tokenizer_dependency_sha256(
        tokenizer_source_sha256=tokenization["tokenizer_source_sha256"],
        wrapper_source_sha256=tokenization["tokenizer_wrapper_source_sha256"],
        config_source_sha256=tokenization["tokenizer_config_source_sha256"],
        stage1_config_source_sha256=tokenization["stage1_config_source_sha256"],
        max_length=tokenization["max_length"],
        aux_token_length=tokenization["aux_token_length"],
        tokenizer_config_n_aux=tokenization["tokenizer_config_n_aux"],
        tokenizer_config_n_tokens=tokenization["tokenizer_config_n_tokens"],
    )
    if tokenization["tokenizer_dependency_sha256"] != expected_dependency_sha256:
        raise RuntimeError("Token manifest identity tokenizer dependency digest is inconsistent")
    expected_code_version = "source-sha256:" + expected_dependency_sha256
    if tokenization["tokenizer_code_version"] != expected_code_version:
        raise RuntimeError("Token manifest identity tokenizer code version is inconsistent")

    output = _require_manifest_mapping(
        manifest,
        "output",
        {
            "loader_sample_count",
            "tokenized_cell_count",
            "bundle_count",
            "families",
            "artifacts",
        },
    )
    for field, minimum in (
        ("loader_sample_count", 1),
        ("tokenized_cell_count", 1),
        ("bundle_count", 1),
    ):
        value = output[field]
        if (
            not isinstance(value, Integral)
            or isinstance(value, bool)
            or int(value) < minimum
        ):
            raise RuntimeError(f"Token manifest output {field} is invalid")
    if not isinstance(output["families"], Mapping):
        raise RuntimeError("Token manifest output families must be an object")
    required_families = {prefix.rstrip("_") for prefix in REQUIRED_TOKEN_FILE_PREFIXES}
    if set(output["families"]) != required_families or any(
        not isinstance(paths, list) or any(not isinstance(path, str) for path in paths)
        for paths in output["families"].values()
    ):
        raise RuntimeError("Token manifest output families are incomplete or invalid")
    if not isinstance(output["artifacts"], list) or not output["artifacts"]:
        raise RuntimeError("Token manifest output artifact inventory is invalid")
    for artifact in output["artifacts"]:
        if not isinstance(artifact, Mapping) or not {"path", "size", "sha256"}.issubset(artifact):
            raise RuntimeError("Token manifest output artifact record is invalid")
        if not isinstance(artifact["path"], str) or not artifact["path"]:
            raise RuntimeError("Token manifest output artifact path is invalid")
        if (
            not isinstance(artifact["size"], Integral)
            or isinstance(artifact["size"], bool)
            or int(artifact["size"]) < 0
        ):
            raise RuntimeError("Token manifest output artifact size is invalid")
        _require_manifest_sha256(artifact["sha256"], "output artifact SHA256")


def _load_token_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / TOKEN_MANIFEST_NAME
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Token manifest is missing or unsafe: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read token manifest: {path}") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("Token manifest root must be a JSON object")
    _validate_token_manifest_header(manifest)
    return manifest


def _token_manifest_matches(root: str | Path, expected_identity: Mapping[str, Any]) -> bool:
    manifest = _load_token_manifest(root)
    return manifest["identity"] == dict(expected_identity)


def _validate_relative_artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("Token artifact path must be a non-empty relative path")
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise RuntimeError(f"Unsafe token artifact path escape: {relative!r}")
    target = root.joinpath(*candidate_relative.parts)
    current = root
    for part in candidate_relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Token artifact symlink is forbidden: {relative}")
    try:
        target.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Token artifact is missing or escapes root: {relative}") from exc
    if not target.is_file():
        raise RuntimeError(f"Token artifact is not a regular file: {relative}")
    return target


def _validate_token_artifacts(
    root: str | Path,
    manifest: Mapping[str, Any] | None = None,
) -> None:
    root = Path(root)
    current_manifest = _load_token_manifest(root) if manifest is None else dict(manifest)
    _validate_token_manifest_header(current_manifest)
    output = current_manifest.get("output")
    if not isinstance(output, Mapping) or not isinstance(output.get("artifacts"), list):
        raise RuntimeError("Token manifest artifact inventory is missing")

    listed_paths = []
    for artifact in output["artifacts"]:
        if not isinstance(artifact, Mapping):
            raise RuntimeError("Token artifact record must be an object")
        relative = artifact.get("path")
        listed_paths.append(relative)
        target = _validate_relative_artifact_path(root, relative)
        expected_size = artifact.get("size")
        if not isinstance(expected_size, Integral) or isinstance(expected_size, bool):
            raise RuntimeError(f"Invalid artifact size for {relative}")
        if target.stat().st_size != int(expected_size):
            raise RuntimeError(f"Token artifact size mismatch: {relative}")
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or _sha256_file(target) != expected_hash:
            raise RuntimeError(f"Token artifact SHA256 mismatch: {relative}")
    if len(listed_paths) != len(set(listed_paths)):
        raise RuntimeError("Token manifest contains duplicate artifact paths")

    structure = _inspect_required_token_jobs(root)
    actual_paths = sorted(
        path.relative_to(root).as_posix()
        for token_dir in structure["token_dirs"]
        for path in token_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if sorted(listed_paths) != actual_paths:
        raise RuntimeError(
            "Token artifact inventory has a missing or unlisted file under tokens-*"
        )
    listed_required = sorted(
        relative
        for relative in listed_paths
        if isinstance(relative, str)
        and any(Path(relative).name.startswith(prefix) for prefix in REQUIRED_TOKEN_FILE_PREFIXES)
        and relative.endswith(".job")
    )
    if listed_required != structure["required_paths"]:
        raise RuntimeError("Token directory has a missing or unlisted required job artifact")
    if output.get("bundle_count") != structure["bundle_count"]:
        raise RuntimeError("Token bundle count does not match the manifest")
    if output.get("families") != structure["families"]:
        raise RuntimeError("Token artifact families do not match the manifest")

    payload_counts = _validate_token_bundle_payloads(structure)
    if output.get("loader_sample_count") != payload_counts["loader_sample_count"]:
        raise RuntimeError("Loader sample count does not match the manifest")
    if output.get("tokenized_cell_count") != payload_counts["tokenized_cell_count"]:
        raise RuntimeError("Tokenized cell count does not match the manifest")


def _write_token_manifest(root: str | Path, manifest: Mapping[str, Any]) -> Path:
    root = Path(root)
    _validate_token_manifest_header(manifest)
    _validate_token_artifacts(root, manifest)
    payload = (
        json.dumps(
            manifest,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target = root / TOKEN_MANIFEST_NAME
    temporary = root / f".{TOKEN_MANIFEST_NAME}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite token manifest: {target}") from exc
    else:
        temporary.unlink()
        try:
            directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    return target


def _acquire_persistent_tokenization_lock(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / TOKENIZATION_LOCK_NAME
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Persistent tokenization lock already exists at {lock_path}; "
            "use a new empty token_data_path"
        ) from exc
    try:
        payload = (
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "state": "claimed",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path


def _matrix_is_count_like(matrix, chunk_size: int = 1_000_000) -> bool:
    from scipy import sparse

    if not isinstance(chunk_size, Integral) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    if sparse.issparse(matrix):
        values = matrix.data if hasattr(matrix, "data") else matrix.values()
    elif isinstance(matrix, np.ndarray):
        values = matrix
    else:
        values = np.asarray(matrix)

    if isinstance(values, np.ndarray) and values.dtype == object:
        return all(
            _matrix_is_count_like(item, chunk_size=chunk_size)
            for item in values.flat
        )
    if not isinstance(values, np.ndarray):
        return all(
            _matrix_is_count_like(item, chunk_size=chunk_size)
            for item in values
        )

    if values.ndim == 0:
        chunks = (values,)
    else:
        trailing_size = int(np.prod(values.shape[1:], dtype=np.int64))
        rows_per_chunk = max(1, int(chunk_size) // max(trailing_size, 1))
        chunks = (
            values[start : start + rows_per_chunk]
            for start in range(0, values.shape[0], rows_per_chunk)
        )
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            return False
        if (chunk < 0).any():
            return False
        if not np.allclose(chunk, np.rint(chunk), atol=1e-6, rtol=0.0):
            return False
    return True


def _select_hvg_adata(
    adata,
    *,
    n_hvg: int,
    hvg_source: str,
    hvg_flavor: str,
):
    import scanpy as sc

    if not isinstance(n_hvg, Integral) or isinstance(n_hvg, (bool, np.bool_)):
        raise TypeError(f"n_hvg must be an integer between 1 and {adata.n_vars}")
    if not 1 <= int(n_hvg) <= adata.n_vars:
        raise ValueError(f"n_hvg must be between 1 and {adata.n_vars}")
    if hvg_source not in VALID_HVG_SOURCES:
        raise ValueError(
            f"hvg_source must be one of {sorted(VALID_HVG_SOURCES)}, "
            f"got {hvg_source!r}"
        )
    if hvg_flavor not in VALID_HVG_FLAVORS:
        raise ValueError(
            f"hvg_flavor must be one of {sorted(VALID_HVG_FLAVORS)}, "
            f"got {hvg_flavor!r}"
        )
    if hvg_flavor in {"seurat", "cell_ranger"} and hvg_source != "x_log":
        raise ValueError(
            f"hvg_flavor={hvg_flavor!r} requires hvg_source='x_log'"
        )

    if hvg_source == "raw":
        if adata.raw is None:
            raise ValueError("hvg_source='raw' requires adata.raw")
        if not adata.var_names.is_unique or not adata.raw.var_names.is_unique:
            raise ValueError("AnnData and raw gene names must be unique")
        raw_positions = adata.raw.var_names.get_indexer(adata.var_names)
        if (raw_positions < 0).any():
            missing = [
                str(name)
                for name, position in zip(adata.var_names, raw_positions)
                if position < 0
            ]
            raise ValueError(
                "adata.raw is missing current genes: " + ", ".join(missing)
            )
        hvg_adata = adata.raw[:, adata.var_names].to_adata()
    else:
        hvg_adata = adata.copy()

    if hvg_flavor == "seurat_v3" and not _matrix_is_count_like(hvg_adata.X):
        raise ValueError(
            "hvg_flavor='seurat_v3' requires a finite, nonnegative, "
            "count-like matrix"
        )

    sc.pp.highly_variable_genes(
        hvg_adata,
        flavor=hvg_flavor,
        n_top_genes=int(n_hvg),
        subset=False,
        inplace=True,
        check_values=True,
    )
    if "highly_variable" not in hvg_adata.var.columns:
        raise RuntimeError("Scanpy did not report highly_variable genes")
    selected_mask = np.asarray(hvg_adata.var["highly_variable"], dtype=bool)
    selected_hvg_names = [
        str(name) for name in hvg_adata.var_names[selected_mask]
    ]
    if len(selected_hvg_names) != int(n_hvg):
        raise RuntimeError(
            f"Scanpy selected {len(selected_hvg_names)} HVGs; requested {n_hvg}"
        )

    selected = adata[:, selected_hvg_names].copy()
    report = {
        "requested_hvg_count": int(n_hvg),
        "selected_hvg_names": selected_hvg_names,
    }
    return selected, report


def _prepare_token_data(
    *,
    adata,
    original_adata=None,
    gene_dict_path: str,
    token_data_path: str,
    species: str,
    assay: str,
    use_hvg: bool,
    n_hvg: int,
    hvg_source: str,
    hvg_flavor: str,
    use_dev_abs: bool,
    n_aux: int,
    allow_unverified_legacy_tokens: bool,
    max_cells: int | None = None,
    original_fingerprint: Mapping[str, Any] | None = None,
    token_work_matches_original: bool = False,
) -> str:
    root = Path(token_data_path)
    if root.is_symlink():
        raise RuntimeError(f"token_data_path symlinks are forbidden: {root}")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"token_data_path is not a directory: {root}")
    expected_identity = _build_token_identity(
        adata=adata,
        original_adata=adata if original_adata is None else original_adata,
        gene_dict_path=gene_dict_path,
        species=species,
        assay=assay,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        hvg_source=hvg_source,
        hvg_flavor=hvg_flavor,
        use_dev_abs=use_dev_abs,
        n_aux=n_aux,
        max_cells=max_cells,
        original_fingerprint=original_fingerprint,
        token_work_matches_original=token_work_matches_original,
    )
    manifest_path = root / TOKEN_MANIFEST_NAME
    if manifest_path.exists() or manifest_path.is_symlink():
        if not _token_manifest_matches(root, expected_identity):
            raise RuntimeError(
                "Token provenance mismatch; use a new empty token_data_path"
            )
        _validate_token_artifacts(root)
        return str(root)
    lock_path = root / TOKENIZATION_LOCK_NAME
    if lock_path.exists() or lock_path.is_symlink():
        raise RuntimeError(
            f"Persistent tokenization lock exists under {root}; "
            "a prior or concurrent attempt owns this path. "
            "Use a new empty token_data_path; existing files were preserved."
        )
    if _has_token_joblib_bundles(str(root)):
        if allow_unverified_legacy_tokens:
            structure = _inspect_required_token_jobs(root)
            _validate_token_bundle_payloads(structure)
            warnings.warn(
                "Reusing unverified legacy token bundle without a manifest",
                RuntimeWarning,
                stacklevel=2,
            )
            return str(root)
        raise RuntimeError(
            "Token manifest is missing; provenance cannot be verified. "
            "Use a new empty token_data_path or explicitly set "
            "allow_unverified_legacy_tokens=True"
        )
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(
            f"Token directory is non-empty but incomplete: {root}. "
            "Use a new empty directory; existing files were preserved."
        )
    lock_path = _acquire_persistent_tokenization_lock(root)
    post_lock_entries = [
        path
        for path in root.iterdir()
        if path.name != lock_path.name
    ]
    if post_lock_entries:
        raise RuntimeError(
            f"Token directory changed while acquiring {lock_path}; "
            "use a new empty token_data_path. Existing files and lock were preserved."
        )
    run_tokenization(
        adata=adata,
        gene_dict_path=gene_dict_path,
        bb_token_dir=str(root),
        species=species,
        assay=assay,
        use_hvg=False,
        n_hvg=n_hvg,
        use_dev_abs=use_dev_abs,
        force_tokenize=False,
    )
    if not _has_token_joblib_bundles(str(root)):
        raise RuntimeError(
            f"Tokenization did not produce a complete token bundle under {root}; "
            "no complete manifest was published"
        )
    manifest = _build_token_manifest(root, expected_identity)
    _write_token_manifest(root, manifest)
    return str(root)


def _collect_gene_axis(
    *,
    adata,
    token_data_path: str | Path,
    gene_dict_path: str | Path,
    n_aux: int,
    output_axis: str,
    max_cells: int | None = None,
) -> dict[str, Any]:
    """Pre-pass real token IDs and build one stable output gene axis."""
    import joblib

    if output_axis not in VALID_OUTPUT_AXES:
        raise ValueError(
            f"output_axis must be one of {sorted(VALID_OUTPUT_AXES)}, "
            f"got {output_axis!r}"
        )
    if max_cells is not None:
        if not isinstance(max_cells, Integral) or isinstance(
            max_cells, (bool, np.bool_)
        ):
            raise TypeError("max_cells must be a positive integer or None")
        if int(max_cells) <= 0:
            raise ValueError("max_cells must be a positive integer or None")
        max_cells = int(max_cells)

    token_to_gene = build_token_to_gene_map(gene_dict_path, n_aux=n_aux)
    adata_gene_idx = build_adata_gene_index(adata)
    adata_names = [str(name) for name in adata.var_names]

    token_to_original_col: dict[int, int] = {}
    original_col_to_token: dict[int, int] = {}
    for token_id, gene_name in token_to_gene.items():
        original_col = _match_gene_name(gene_name, adata_gene_idx)
        if original_col is None:
            continue
        if original_col in original_col_to_token:
            prior = original_col_to_token[original_col]
            raise ValueError(
                "Gene dictionary does not round-trip uniquely to AnnData: "
                f"tokens {prior} and {token_id} map to {adata_names[original_col]!r}"
            )
        token_to_original_col[token_id] = original_col
        original_col_to_token[original_col] = token_id

    structure = _inspect_required_token_jobs(Path(token_data_path))
    observed_original_cols: set[int] = set()
    tokenized_cell_count = 0
    raw_sequence_length = 0
    effective_sequence_length = 0
    max_inner_batch_size = 0
    stop = False
    for paired_paths in structure["bundle_paths"]:
        payload = joblib.load(paired_paths["real_indices_"])
        try:
            if not isinstance(payload, (list, tuple)) or not payload:
                raise RuntimeError(
                    "real_indices token payload must be a non-empty list of batches: "
                    f"{paired_paths['real_indices_']}"
                )
            for batch in payload:
                token_batch = np.asarray(batch)
                if token_batch.ndim != 2 or token_batch.shape[0] <= 0 or token_batch.shape[1] <= 0:
                    raise RuntimeError(
                        "real_indices token batches must be non-empty 2D arrays: "
                        f"{paired_paths['real_indices_']}"
                    )
                if not np.issubdtype(token_batch.dtype, np.integer):
                    if not np.isfinite(token_batch).all() or not np.equal(
                        token_batch, np.floor(token_batch)
                    ).all():
                        raise RuntimeError("real_indices token IDs must be finite integers")
                    token_batch = token_batch.astype(np.int64)
                raw_sequence_length = max(
                    raw_sequence_length,
                    int(token_batch.shape[1]),
                )
                effective_sequence_length = max(
                    effective_sequence_length,
                    min(int(token_batch.shape[1]), int(TOKEN_SEQUENCE_LIMIT)),
                )
                max_inner_batch_size = max(
                    max_inner_batch_size,
                    int(token_batch.shape[0]),
                )
                remaining = (
                    token_batch.shape[0]
                    if max_cells is None
                    else max_cells - tokenized_cell_count
                )
                if remaining <= 0:
                    stop = True
                    break
                used = token_batch[:remaining, : int(TOKEN_SEQUENCE_LIMIT)]
                tokenized_cell_count += int(used.shape[0])
                for token_id in np.unique(used):
                    token_id = int(token_id)
                    if token_id < n_aux:
                        continue
                    if token_id not in token_to_gene:
                        raise RuntimeError(f"Unknown gene token ID {token_id}")
                    original_col = token_to_original_col.get(token_id)
                    if original_col is None:
                        raise RuntimeError(
                            f"Gene token ID {token_id} ({token_to_gene[token_id]!r}) "
                            "does not round-trip to adata.var_names"
                        )
                    observed_original_cols.add(original_col)
                if max_cells is not None and tokenized_cell_count >= max_cells:
                    stop = True
                    break
        finally:
            del payload
        if stop:
            break

    if tokenized_cell_count <= 0 or effective_sequence_length <= 0:
        raise RuntimeError("Token pre-pass found no tokenized cells")
    observed_indices = np.asarray(sorted(observed_original_cols), dtype=np.int64)
    if observed_indices.size == 0:
        raise RuntimeError("Token pre-pass found no observed AnnData genes")
    if output_axis == "compact":
        original_indices = observed_indices.copy()
        axis_mode = "global_union"
    else:
        original_indices = np.arange(adata.n_vars, dtype=np.int64)
        axis_mode = "full_original"

    gene_token_ids = np.asarray(
        [original_col_to_token.get(int(index), -1) for index in original_indices],
        dtype=np.int64,
    )
    gene_has_token = gene_token_ids >= int(n_aux)
    gene_names = np.asarray(
        [adata_names[int(index)] for index in original_indices],
        dtype=str,
    )
    token_to_output_col = {
        int(token_id): output_col
        for output_col, token_id in enumerate(gene_token_ids)
        if int(token_id) >= int(n_aux)
    }
    observed_mask = np.asarray(
        [index in observed_original_cols for index in range(adata.n_vars)],
        dtype=bool,
    )
    tokenized_gene_names = [
        adata_names[index]
        for index in range(adata.n_vars)
        if observed_mask[index]
    ]
    dropped_gene_names = [
        adata_names[index]
        for index in range(adata.n_vars)
        if not observed_mask[index]
    ]
    dropped_gene_reasons = [
        (
            "not_in_gene_dictionary"
            if index not in original_col_to_token
            else "not_observed_in_token_bundles"
        )
        for index in range(adata.n_vars)
        if not observed_mask[index]
    ]
    return {
        "axis_mode": axis_mode,
        "gene_names": gene_names,
        "original_gene_indices": original_indices,
        "observed_original_gene_indices": observed_indices,
        "gene_token_ids": gene_token_ids,
        "gene_has_token": np.asarray(gene_has_token, dtype=bool),
        "token_to_output_col": token_to_output_col,
        "tokenized_gene_names": tokenized_gene_names,
        "dropped_gene_names": dropped_gene_names,
        "dropped_gene_reasons": dropped_gene_reasons,
        "original_n_vars": int(adata.n_vars),
        "original_gene_axis_hash": _hash_index(adata.var_names),
        "tokenized_cell_count": int(tokenized_cell_count),
        "sequence_length": int(effective_sequence_length),
        "raw_sequence_length": int(raw_sequence_length),
        "effective_sequence_length": int(effective_sequence_length),
        "max_inner_batch_size": int(max_inner_batch_size),
    }


def _count_dtype_for_max_cells(max_cells: int) -> np.dtype:
    """Choose the smallest supported exact counter dtype for a cell bound."""
    if not isinstance(max_cells, Integral) or isinstance(max_cells, (bool, np.bool_)):
        raise TypeError("max_cells must be a non-negative integer")
    max_cells = int(max_cells)
    if max_cells < 0:
        raise ValueError("max_cells must be a non-negative integer")
    if max_cells <= int(np.iinfo(np.uint32).max):
        return np.dtype(np.uint32)
    if max_cells <= int(np.iinfo(np.uint64).max):
        return np.dtype(np.uint64)
    raise OverflowError("cell count exceeds the supported uint64 range")


def _estimate_aggregation_bytes(
    *,
    output_rows: int,
    gene_count: int,
    dim_model: int,
    want_attention: bool,
    want_embedding: bool,
    attention_average: str,
    batch_size: int,
    nheads: int,
    sequence_length: int,
    nlayers: int,
    attention_layers: str,
    out_dtype: str,
    count_dtype: str | np.dtype | type = np.uint32,
) -> dict[str, int]:
    """Return a conservative byte estimate before model or accumulator setup."""
    if attention_average not in VALID_ATTENTION_AVERAGES:
        raise ValueError(f"Unsupported attention_average: {attention_average!r}")
    _parse_attention_layer_items(attention_layers)
    if out_dtype not in VALID_DTYPES:
        raise ValueError(f"Unsupported out_dtype: {out_dtype!r}")
    dimensions = {
        "output_rows": output_rows,
        "gene_count": gene_count,
        "dim_model": dim_model,
        "batch_size": batch_size,
        "nheads": nheads,
        "sequence_length": sequence_length,
        "nlayers": nlayers,
    }
    for label, value in dimensions.items():
        if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{label} must be a positive integer")
        if int(value) <= 0:
            raise ValueError(f"{label} must be a positive integer")
    output_rows = int(output_rows)
    gene_count = int(gene_count)
    dim_model = int(dim_model)
    out_itemsize = np.dtype(out_dtype).itemsize
    resolved_count_dtype = np.dtype(count_dtype)
    if resolved_count_dtype not in (np.dtype(np.uint32), np.dtype(np.uint64)):
        raise ValueError("count_dtype must be uint32 or uint64")
    count_itemsize = resolved_count_dtype.itemsize
    pair_elements = output_rows * gene_count * gene_count
    gene_elements = output_rows * gene_count
    embedding_elements = gene_elements * dim_model
    attention_outputs = 2 if attention_average == "both" else 1
    layer_selection = resolve_attention_layers(
        attention_layers,
        nlayers=int(nlayers),
    )
    hook_layers = len(layer_selection["indices"])
    float32_finalize_elements = [gene_elements]
    if want_attention:
        float32_finalize_elements.append(pair_elements)
    if want_embedding:
        float32_finalize_elements.append(embedding_elements)
    float32_finalize_workspace_bytes = (
        max(float32_finalize_elements) * np.dtype(np.float32).itemsize
        if out_dtype == "float16"
        else 0
    )
    computes_coobserved = (
        want_attention
        and attention_average in {"coobserved", "both"}
    )
    mask_itemsize = np.dtype(np.bool_).itemsize
    pair_divide_mask_bytes = (
        pair_elements * mask_itemsize
        if computes_coobserved
        else 0
    )
    other_divide_mask_workspace_bytes = (
        0
        if computes_coobserved
        else max(
            output_rows,
            gene_elements if want_embedding else 0,
        )
        * mask_itemsize
    )
    aggregation_components = {
        "attention_sum_bytes": pair_elements * 4 if want_attention else 0,
        "pair_count_bytes": (
            pair_elements * count_itemsize if want_attention else 0
        ),
        "attention_output_bytes": (
            pair_elements * out_itemsize * attention_outputs
            if want_attention else 0
        ),
        "pair_coverage_bytes": pair_elements * out_itemsize if want_attention else 0,
        "embedding_sum_bytes": embedding_elements * 4 if want_embedding else 0,
        "gene_count_bytes": gene_elements * count_itemsize,
        "cell_count_bytes": output_rows * count_itemsize,
        "valid_coverage_bytes": gene_elements * out_itemsize,
        "embedding_output_bytes": (
            embedding_elements * out_itemsize if want_embedding else 0
        ),
        "float32_finalize_workspace_bytes": float32_finalize_workspace_bytes,
        "pair_divide_mask_bytes": pair_divide_mask_bytes,
        "other_divide_mask_workspace_bytes": (
            other_divide_mask_workspace_bytes
        ),
    }
    aggregation_host_bytes = int(sum(aggregation_components.values()))
    reduced_batch_bytes = (
        int(batch_size) * int(sequence_length) ** 2 * 4
        if want_attention else 0
    )
    hook_raw_weights_device_bytes = (
        reduced_batch_bytes * int(nheads) * hook_layers
        if want_attention else 0
    )
    hook_mean_temporary_device_bytes = reduced_batch_bytes
    hook_per_layer_host_bytes = reduced_batch_bytes * hook_layers
    hook_numpy_stack_host_bytes = reduced_batch_bytes * hook_layers
    hook_reduced_batch_host_bytes = reduced_batch_bytes
    host_total_bytes = (
        aggregation_host_bytes
        + hook_per_layer_host_bytes
        + hook_numpy_stack_host_bytes
        + hook_reduced_batch_host_bytes
    )
    device_total_bytes = (
        hook_raw_weights_device_bytes
        + hook_mean_temporary_device_bytes
    )
    components = {
        **aggregation_components,
        "aggregation_host_bytes": aggregation_host_bytes,
        "hook_raw_weights_bytes": hook_raw_weights_device_bytes,
        "hook_raw_weights_device_bytes": hook_raw_weights_device_bytes,
        "hook_mean_temporary_device_bytes": hook_mean_temporary_device_bytes,
        "hook_per_layer_host_bytes": hook_per_layer_host_bytes,
        "hook_numpy_stack_host_bytes": hook_numpy_stack_host_bytes,
        "hook_reduced_batch_host_bytes": hook_reduced_batch_host_bytes,
        "host_total_bytes": int(host_total_bytes),
        "device_total_bytes": int(device_total_bytes),
        "total_bytes": int(host_total_bytes + device_total_bytes),
    }
    return components


def _parse_slurm_memory_bytes(value: str) -> int:
    import re

    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?)B?\s*", value, re.I)
    if match is None:
        raise ValueError(f"Unsupported Slurm memory value: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2).upper()
    power = {"K": 1, "M": 2, "G": 3, "T": 4}.get(suffix, 2)
    result = int(number * 1024**power)
    if result <= 0:
        raise ValueError(f"Slurm memory value must be positive: {value!r}")
    return result


def _slurm_memory_remaining_bytes(rss_bytes: int) -> int | None:
    allocations: list[int] = []
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        try:
            allocations.append(_parse_slurm_memory_bytes(per_node))
        except ValueError:
            pass
    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpu_count = os.environ.get("SLURM_CPUS_ON_NODE")
    if per_cpu and cpu_count:
        try:
            allocations.append(
                _parse_slurm_memory_bytes(per_cpu) * int(cpu_count)
            )
        except ValueError:
            pass
    if not allocations:
        return None
    return max(0, min(allocations) - int(rss_bytes))


def _read_cgroup_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not value or value == "max":
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _cgroup_memory_remaining_bytes() -> tuple[int, str] | None:
    v2_roots = [Path("/sys/fs/cgroup")]
    v1_roots = [Path("/sys/fs/cgroup/memory")]
    try:
        entries = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError, OSError):
        entries = []
    for entry in entries:
        fields = entry.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, relative = fields[1], fields[2].lstrip("/")
        if controllers == "":
            v2_roots.insert(0, Path("/sys/fs/cgroup") / relative)
        elif "memory" in controllers.split(","):
            v1_roots.insert(0, Path("/sys/fs/cgroup/memory") / relative)

    remaining: list[tuple[int, str]] = []
    for root in dict.fromkeys(v2_roots):
        limit = _read_cgroup_integer(root / "memory.max")
        current = _read_cgroup_integer(root / "memory.current")
        if limit is not None and current is not None:
            remaining.append((max(0, limit - current), "cgroup_v2_remaining"))
    for root in dict.fromkeys(v1_roots):
        limit = _read_cgroup_integer(root / "memory.limit_in_bytes")
        current = _read_cgroup_integer(root / "memory.usage_in_bytes")
        if limit is not None and current is not None:
            remaining.append((max(0, limit - current), "cgroup_v1_remaining"))
    return min(remaining) if remaining else None


def _host_memory_candidates() -> dict[str, int]:
    try:
        import psutil

        available_bytes = int(psutil.virtual_memory().available)
        rss_bytes = int(psutil.Process().memory_info().rss)
    except (ImportError, AttributeError, OSError):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        available_bytes = page_size * available_pages
        rss_bytes = 0
    candidates = {"psutil_available": available_bytes}
    cgroup = _cgroup_memory_remaining_bytes()
    if cgroup is not None:
        remaining, source = cgroup
        candidates[source] = int(remaining)
    slurm_remaining = _slurm_memory_remaining_bytes(rss_bytes)
    if slurm_remaining is not None:
        candidates["slurm_remaining"] = int(slurm_remaining)
    return candidates


def _check_memory_budget(
    estimate: Mapping[str, int],
    *,
    memory_limit_gib: float | None,
    shape: tuple[int, ...],
    mode: str,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    explicit_limit_bytes = None
    if memory_limit_gib is not None:
        if not isinstance(memory_limit_gib, Real) or isinstance(
            memory_limit_gib, (bool, np.bool_)
        ):
            raise TypeError("memory_limit_gib must be a positive finite number or None")
        if not np.isfinite(memory_limit_gib) or float(memory_limit_gib) <= 0:
            raise ValueError("memory_limit_gib must be a positive finite number or None")
        explicit_limit_bytes = int(float(memory_limit_gib) * 1024**3)

    candidates = {
        source: int(value)
        for source, value in _host_memory_candidates().items()
        if int(value) >= 0
    }
    if not candidates:
        raise RuntimeError("No positive host-memory budget candidate is available")
    limiting_host_source, host_available_bytes = min(
        candidates.items(),
        key=lambda item: item[1],
    )
    host_limit_bytes = int(host_available_bytes * 0.70)
    host_limit_source = f"70% {limiting_host_source}"
    if (
        explicit_limit_bytes is not None
        and explicit_limit_bytes < host_limit_bytes
    ):
        host_limit_bytes = explicit_limit_bytes
        host_limit_source = "explicit memory_limit_gib upper bound"

    resolved_device = torch.device(device)
    host_estimated_bytes = int(estimate.get("host_total_bytes", estimate["total_bytes"]))
    device_estimated_bytes = int(estimate.get("device_total_bytes", 0))
    if resolved_device.type == "cuda":
        cuda_free_bytes, _cuda_total_bytes = torch.cuda.mem_get_info(resolved_device)
        device_limit_bytes = int(int(cuda_free_bytes) * 0.70)
        device_limit_source = "70% cuda_free"
        if (
            explicit_limit_bytes is not None
            and explicit_limit_bytes < device_limit_bytes
        ):
            device_limit_bytes = explicit_limit_bytes
            device_limit_source = "explicit memory_limit_gib upper bound"
    else:
        host_estimated_bytes += device_estimated_bytes
        device_estimated_bytes = 0
        device_limit_bytes = 0
        device_limit_source = "merged_into_host"

    if (
        host_estimated_bytes > host_limit_bytes
        or (
            resolved_device.type == "cuda"
            and device_estimated_bytes > device_limit_bytes
        )
    ):
        raise MemoryError(
            "Gene aggregation memory preflight failed: "
            f"host estimated={host_estimated_bytes / 1024**3:.3f} GiB, "
            f"host limit={host_limit_bytes / 1024**3:.3f} GiB, "
            f"host source={host_limit_source}; "
            f"device estimated={device_estimated_bytes / 1024**3:.3f} GiB, "
            f"device limit={device_limit_bytes / 1024**3:.3f} GiB, "
            f"device source={device_limit_source}; "
            f"shape={tuple(shape)}, mode={mode}"
        )
    return {
        **dict(estimate),
        "host_estimated_bytes": int(host_estimated_bytes),
        "host_limit_bytes": int(host_limit_bytes),
        "host_limit_source": host_limit_source,
        "device_estimated_bytes": int(device_estimated_bytes),
        "device_limit_bytes": int(device_limit_bytes),
        "device_limit_source": device_limit_source,
        "limit_bytes": int(host_limit_bytes),
        "limit_source": host_limit_source,
    }


def _validate_finite(name: str, value) -> None:
    array = np.asarray(value)
    if not array.size:
        return
    minimum = np.min(array)
    maximum = np.max(array)
    if not bool(np.isfinite(minimum)) or not bool(np.isfinite(maximum)):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _validate_float16_range(name: str, value) -> None:
    array = np.asarray(value)
    if not array.size:
        return
    minimum = np.min(array)
    maximum = np.max(array)
    if not bool(np.isfinite(minimum)) or not bool(np.isfinite(maximum)):
        raise FloatingPointError(f"{name} contains NaN or Inf")
    limit = float(np.finfo(np.float16).max)
    if minimum < -limit or maximum > limit:
        max_abs = max(abs(float(minimum)), abs(float(maximum)))
        raise OverflowError(
            f"{name} exceeds float16 range: "
            f"max_abs={max_abs:.8g}, limit={limit:.8g}"
        )


def _finalize_float_output(
    name: str,
    value,
    out_dtype: str,
) -> np.ndarray:
    if out_dtype not in VALID_DTYPES:
        raise ValueError(
            f"out_dtype must be one of {sorted(VALID_DTYPES)}, got {out_dtype!r}"
        )
    array = np.asarray(value)
    if out_dtype == "float16":
        _validate_float16_range(name, array)
        target_dtype = np.float16
    else:
        _validate_finite(name, array)
        target_dtype = np.float32
    with np.errstate(over="ignore", invalid="ignore"):
        converted = array.astype(target_dtype, copy=False)
    _validate_finite(
        f"{name} after {out_dtype} conversion",
        converted,
    )
    return converted


def _average_attention(
    captured: list[tuple[int, torch.Tensor]],
) -> np.ndarray:
    if not captured:
        raise RuntimeError("Attention hooks captured no weights")
    per_layer = []
    for layer_index, weights in captured:
        detached = weights.detach()
        if detached.numel():
            minimum = torch.amin(detached)
            maximum = torch.amax(detached)
            if (
                not bool(torch.isfinite(minimum).item())
                or not bool(torch.isfinite(maximum).item())
            ):
                raise FloatingPointError(
                    f"hook attention layer {layer_index} contains NaN or Inf"
                )
        reduced = detached.mean(dim=1).cpu().numpy()
        _validate_finite(
            f"mean hook attention layer {layer_index}",
            reduced,
        )
        per_layer.append(reduced)
    averaged = np.mean(per_layer, axis=0)
    _validate_finite("averaged hook attention", averaged)
    return averaged


def _unpack_cell_names(values) -> list[str]:
    return [
        str(item[0] if isinstance(item, (list, tuple)) else item)
        for item in values
    ]


def _count_tokenized_cells(dataset) -> int:
    try:
        import joblib

        total = 0
        for path in getattr(dataset, "cell_raw_index_files", []):
            for item in joblib.load(path):
                if isinstance(item, np.ndarray):
                    total += int(item.size if item.ndim == 0 else len(item))
                elif isinstance(item, (list, tuple)):
                    total += len(item)
                else:
                    total += 1
        return total or len(dataset)
    except Exception as exc:
        warnings.warn(
            f"Could not count cells in token bundles; using dataset length: {exc}",
            stacklevel=2,
        )
        return len(dataset)


def run_gene_level_inference(
    *,
    adata,
    token_data_path: str,
    config: Mapping[str, Any],
    pretrain_ckpt: str,
    gene_dict_path: str,
    esm_embedding_path: str,
    device: torch.device,
    mode: str,
    group_by: str | None,
    attention_layers: str,
    max_cells: int | None,
    out_dtype: str,
    output_axis: str,
    attention_average: str,
    memory_limit_gib: float | None,
) -> dict[str, Any]:
    """Extract per-cell or grouped gene attention and hidden representations."""
    config_local = normalize_brainbeacon_model_config(deepcopy(dict(config)))
    config_local["batch_size"] = 1
    config_local["gene_dict_path"] = gene_dict_path
    config_local["pretrain_ckpt"] = pretrain_ckpt
    config_local["esm_embedding_path"] = esm_embedding_path

    want_attention = mode in {"attention", "both"}
    want_embedding = mode in {"embedding", "both"}

    group_values = None
    if group_by is not None:
        if group_by not in adata.obs.columns:
            raise ValueError(f"Column {group_by!r} not found in adata.obs")
        group_series = adata.obs[group_by]
        if group_series.isna().any():
            raise ValueError(
                f"{group_by!r} contains real missing values; "
                "label or remove them explicitly"
            )
        group_values = np.asarray(
            [str(value) for value in group_series.to_numpy()],
            dtype=str,
        )

    axis = _collect_gene_axis(
        adata=adata,
        token_data_path=token_data_path,
        gene_dict_path=gene_dict_path,
        n_aux=int(config_local["n_aux"]),
        output_axis=output_axis,
        max_cells=max_cells,
    )
    token_to_col = axis["token_to_output_col"]
    n_total = int(axis["tokenized_cell_count"])
    count_dtype = _count_dtype_for_max_cells(n_total)
    gene_count = len(axis["gene_names"])
    dim_model = int(config_local["dim_model"])
    if group_by is None:
        output_rows = n_total
        labels = None
        cell_to_group = None
    else:
        labels = np.asarray(sorted(set(group_values)), dtype=str)
        group_to_index = {label: index for index, label in enumerate(labels)}
        cell_to_group = {
            str(cell): group_to_index[group]
            for cell, group in zip(adata.obs_names, group_values)
        }
        output_rows = len(labels)

    memory_estimate = _estimate_aggregation_bytes(
        output_rows=output_rows,
        gene_count=gene_count,
        dim_model=dim_model,
        want_attention=want_attention,
        want_embedding=want_embedding,
        attention_average=attention_average,
        batch_size=int(axis["max_inner_batch_size"]),
        nheads=int(config_local["nheads"]),
        sequence_length=int(axis["effective_sequence_length"]),
        nlayers=int(config_local["nlayers"]),
        attention_layers=attention_layers,
        out_dtype=out_dtype,
        count_dtype=count_dtype,
    )
    memory_report = _check_memory_budget(
        memory_estimate,
        memory_limit_gib=memory_limit_gib,
        shape=(output_rows, gene_count),
        mode=f"{mode}/{output_axis}/{attention_layers}/{attention_average}",
        device=device,
    )

    pipeline = CellEmbeddingPipeline(
        pretrain_ckpt=pretrain_ckpt,
        model_config=config_local,
        device=device,
    )
    model = pipeline.model
    model.eval()
    dataset = pipeline.load_dataset(token_data_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    esm_map = torch.load(esm_embedding_path, map_location="cpu")
    esm_map = pipeline.apply_homo_mean_to_esm_map(esm_map)
    cell_counts = np.zeros(output_rows, dtype=count_dtype)

    attention_sum = (
        np.zeros((output_rows, gene_count, gene_count), dtype=np.float32)
        if want_attention
        else None
    )
    pair_counts = (
        np.zeros((output_rows, gene_count, gene_count), dtype=count_dtype)
        if want_attention
        else None
    )
    embedding_sum = (
        np.zeros((output_rows, gene_count, dim_model), dtype=np.float32)
        if want_embedding
        else None
    )
    gene_counts = np.zeros((output_rows, gene_count), dtype=count_dtype)
    cell_labels: list[str] = []
    processed = 0
    skipped = 0
    layer_selection = resolve_attention_layers(
        attention_layers,
        nlayers=int(config_local["nlayers"]),
    )
    hook_target = layer_selection["hook_target"]

    try:
        if want_attention:
            model.pretrain_model.enable_attention_hooks(
                target_layers=hook_target
            )
        with torch.no_grad():
            for batch in loader:
                if processed >= n_total:
                    break
                (
                    real_indices,
                    attention_mask,
                    connect_comp,
                    rna_type,
                    cell_raw_idx,
                    neighbor_distribution,
                    _exp,
                ) = batch
                real_indices = real_indices[0]
                attention_mask = attention_mask[0]
                connect_comp = connect_comp[0]
                rna_type = rna_type[0]
                neighbor_distribution = neighbor_distribution[0].long()
                cell_names = _unpack_cell_names(cell_raw_idx)

                flat_indices = real_indices.reshape(-1).long()
                esm = torch.index_select(esm_map, 0, flat_indices).reshape(
                    real_indices.shape[0],
                    real_indices.shape[1],
                    esm_map.shape[-1],
                )
                hidden = model(
                    real_indices.to(device),
                    connect_comp.to(device),
                    rna_type.to(device),
                    attention_mask.to(device),
                    esm.to(device),
                    neighbor_distribution.to(device),
                    None,
                ).detach().cpu().numpy()
                _validate_finite("hidden output", hidden)

                batch_attention = None
                if want_attention:
                    try:
                        batch_attention = _average_attention(
                            model.pretrain_model.get_attention_weights()
                        )
                    finally:
                        model.pretrain_model.clear_attention_weights()

                token_array = real_indices.numpy()
                for batch_index in range(token_array.shape[0]):
                    if processed >= n_total:
                        break
                    cell_name = cell_names[batch_index]
                    if group_by is None:
                        output_index = processed
                        cell_labels.append(cell_name)
                    elif cell_name not in cell_to_group:
                        skipped += 1
                        processed += 1
                        continue
                    else:
                        output_index = cell_to_group[cell_name]
                    cell_counts[output_index] += 1

                    positions = []
                    columns = []
                    for position, token_id in enumerate(token_array[batch_index]):
                        token_id = int(token_id)
                        if token_id < int(config_local["n_aux"]):
                            continue
                        column = token_to_col.get(token_id)
                        if column is not None:
                            positions.append(position)
                            columns.append(column)

                    if positions:
                        positions_array = np.asarray(positions, dtype=np.intp)
                        columns_array = np.asarray(columns, dtype=np.intp)
                        if np.unique(columns_array).size != columns_array.size:
                            raise RuntimeError(
                                f"Tokenized cell {cell_name!r} contains duplicate "
                                "gene output columns"
                            )
                        if want_embedding:
                            values = hidden[batch_index, positions_array]
                            embedding_sum[output_index, columns_array] += values
                        if want_attention:
                            values = batch_attention[batch_index][
                                np.ix_(positions_array, positions_array)
                            ]
                            attention_sum[output_index][
                                np.ix_(columns_array, columns_array)
                            ] += values
                            pair_counts[output_index][
                                np.ix_(columns_array, columns_array)
                            ] += 1
                        gene_counts[output_index, columns_array] += 1
                    processed += 1
    finally:
        if want_attention:
            try:
                model.pretrain_model.clear_attention_weights()
            finally:
                model.pretrain_model.disable_attention_hooks()

    if group_by is not None:
        if processed and skipped == processed:
            raise RuntimeError("All tokenized cells were absent from adata.obs_names")
        if processed and skipped / processed > 0.5:
            warnings.warn(
                "More than 50% of tokenized cells were skipped",
                stacklevel=2,
            )

    gene_coverage = np.zeros(gene_counts.shape, dtype=np.float32)
    np.divide(
        gene_counts,
        cell_counts[:, None],
        out=gene_coverage,
        where=cell_counts[:, None] > 0,
    )
    gene_coverage = _finalize_float_output(
        "gene_coverage",
        gene_coverage,
        out_dtype,
    )

    result: dict[str, Any] = {
        "gene_names": np.asarray(axis["gene_names"], dtype=str),
        "labels": (
            labels
            if group_by is not None
            else np.asarray(cell_labels, dtype=str)
        ),
        "gene_counts": gene_counts,
        "gene_coverage": gene_coverage,
        "valid_mask": gene_coverage,
        "cell_counts": cell_counts,
        "skipped_cell_count": int(skipped),
        "axis_mode": axis["axis_mode"],
        "original_gene_indices": axis["original_gene_indices"],
        "observed_original_gene_indices": axis["observed_original_gene_indices"],
        "gene_token_ids": axis["gene_token_ids"],
        "gene_has_token": axis["gene_has_token"],
        "original_n_vars": axis["original_n_vars"],
        "original_gene_axis_hash": axis["original_gene_axis_hash"],
        "tokenized_gene_names": axis["tokenized_gene_names"],
        "dropped_gene_names": axis["dropped_gene_names"],
        "dropped_gene_reasons": axis["dropped_gene_reasons"],
        "tokenized_cell_count": axis["tokenized_cell_count"],
        "sequence_length": axis["sequence_length"],
        "raw_sequence_length": axis["raw_sequence_length"],
        "effective_sequence_length": axis["effective_sequence_length"],
        "max_inner_batch_size": axis["max_inner_batch_size"],
        "memory_estimate": memory_report,
    }
    if want_attention:
        pair_coverage = np.zeros(pair_counts.shape, dtype=np.float32)
        np.divide(
            pair_counts,
            cell_counts[:, None, None],
            out=pair_coverage,
            where=cell_counts[:, None, None] > 0,
        )
        pair_coverage = _finalize_float_output(
            "pair_coverage",
            pair_coverage,
            out_dtype,
        )
        result["pair_counts"] = pair_counts
        result["pair_coverage"] = pair_coverage
        if attention_average in {"population", "both"}:
            attention_population = np.zeros(attention_sum.shape, dtype=np.float32)
            np.divide(
                attention_sum,
                cell_counts[:, None, None],
                out=attention_population,
                where=cell_counts[:, None, None] > 0,
            )
            attention_population = _finalize_float_output(
                "attention_population",
                attention_population,
                out_dtype,
            )
            result["attention_population"] = attention_population
        if attention_average in {"coobserved", "both"}:
            attention_coobserved = np.zeros(attention_sum.shape, dtype=np.float32)
            np.divide(
                attention_sum,
                pair_counts,
                out=attention_coobserved,
                where=pair_counts > 0,
            )
            attention_coobserved = _finalize_float_output(
                "attention_coobserved",
                attention_coobserved,
                out_dtype,
            )
            result["attention_coobserved"] = attention_coobserved
        if output_axis == "full" and attention_average == "population":
            result["attention"] = result["attention_population"]
    if want_embedding:
        embedding = np.zeros(embedding_sum.shape, dtype=np.float32)
        np.divide(
            embedding_sum,
            gene_counts[:, :, None],
            out=embedding,
            where=gene_counts[:, :, None] > 0,
        )
        embedding = _finalize_float_output(
            "embedding",
            embedding,
            out_dtype,
        )
        result["embedding"] = embedding
    return result


def _require_file(path: str, label: str) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return str(resolved)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Result provenance contains a non-finite float")
        return value
    raise TypeError(
        "Result provenance contains a non-JSON-safe value: "
        f"{type(value).__name__}"
    )


def _require_sha256_value(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Result provenance {label} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(
            f"Result provenance {label} must be a SHA256 hex digest"
        ) from exc


def _validate_saved_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("result provenance config must be a mapping")
    try:
        required = _require_config(value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"result provenance config is invalid: {exc}") from exc
    normalized = normalize_brainbeacon_model_config(required)
    config = _json_safe(deepcopy(dict(value)))
    if config != _json_safe(normalized):
        raise ValueError(
            "result provenance config must contain the complete normalized aliases"
        )
    batch_size = config.get("batch_size")
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size != 1
    ):
        raise ValueError("result provenance config batch_size must equal 1")
    for left, right in (
        ("use_esm_emb", "use_esm_embedding"),
        ("use_gene_id_emb", "gene_id"),
        ("use_cell_density", "use_density_emb"),
        ("use_gene_deviation", "neighbor_enhance"),
    ):
        if left in config and right in config and bool(config[left]) != bool(config[right]):
            raise ValueError(
                f"result provenance config aliases disagree: {left}, {right}"
            )
    return config


def _validate_saved_run(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("result provenance run must be a mapping")
    required = {
        "species",
        "assay",
        "mode",
        "group_by",
        "attention_layers",
        "use_hvg",
        "n_hvg",
        "max_cells",
        "out_dtype",
        "hvg_source",
        "hvg_flavor",
        "output_axis",
        "attention_average",
        "allow_unverified_legacy_tokens",
    }
    if set(value) != required:
        missing = sorted(required.difference(value))
        extra = sorted(set(value).difference(required))
        raise ValueError(
            "result provenance run fields are incomplete or unexpected: "
            f"missing={missing}, extra={extra}"
        )
    run = _json_safe(deepcopy(dict(value)))
    if not isinstance(run["species"], str) or not run["species"]:
        raise ValueError("result provenance run species must be non-empty")
    if not isinstance(run["assay"], str) or not run["assay"]:
        raise ValueError("result provenance run assay must be non-empty")
    if run["group_by"] is not None and not isinstance(run["group_by"], str):
        raise ValueError("result provenance run group_by must be a string or None")
    for key in ("use_hvg", "allow_unverified_legacy_tokens"):
        if not isinstance(run[key], bool):
            raise ValueError(f"result provenance run {key} must be boolean")
    if (
        not isinstance(run["n_hvg"], int)
        or isinstance(run["n_hvg"], bool)
        or run["n_hvg"] <= 0
    ):
        raise ValueError("result provenance run n_hvg must be a positive integer")
    _validate_options(
        mode=run["mode"],
        attention_layers=run["attention_layers"],
        out_dtype=run["out_dtype"],
        max_cells=run["max_cells"],
        hvg_source=run["hvg_source"],
        hvg_flavor=run["hvg_flavor"],
        output_axis=run["output_axis"],
        attention_average=run["attention_average"],
        allow_unverified_legacy_tokens=run[
            "allow_unverified_legacy_tokens"
        ],
        use_hvg=run["use_hvg"],
    )
    return run


def _validate_file_fingerprint_record(
    value: Any,
    label: str,
) -> dict[str, Any]:
    required = {"path", "size", "sha256"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(
            f"result provenance {label} fingerprint fields are invalid"
        )
    record = dict(value)
    if not isinstance(record["path"], str) or not Path(record["path"]).is_absolute():
        raise ValueError(
            f"result provenance {label} fingerprint path must be absolute"
        )
    if (
        not isinstance(record["size"], int)
        or isinstance(record["size"], bool)
        or record["size"] < 0
    ):
        raise ValueError(
            f"result provenance {label} fingerprint size must be nonnegative"
        )
    _require_sha256_value(record["sha256"], f"{label}.sha256")
    return record


def _validate_input_fingerprint(value: Any) -> dict[str, Any]:
    required = {
        "shape",
        "adata_content_sha256",
        "x_sha256",
        "obs_names_sha256",
        "var_names_sha256",
        "obs_sha256",
        "spatial_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("result provenance input fingerprint fields are invalid")
    fingerprint = dict(value)
    shape = fingerprint["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 0
            for size in shape
        )
    ):
        raise ValueError("result provenance input fingerprint shape is invalid")
    for field in (
        "adata_content_sha256",
        "x_sha256",
        "obs_names_sha256",
        "var_names_sha256",
        "obs_sha256",
    ):
        _require_sha256_value(fingerprint[field], f"input.{field}")
    spatial_sha256 = fingerprint["spatial_sha256"]
    if spatial_sha256 is not None:
        _require_sha256_value(spatial_sha256, "input.spatial_sha256")
    expected_content = _build_adata_content_sha256(
        shape=shape,
        x_sha256=fingerprint["x_sha256"],
        obs_names_sha256=fingerprint["obs_names_sha256"],
        var_names_sha256=fingerprint["var_names_sha256"],
        obs_sha256=fingerprint["obs_sha256"],
        spatial_sha256=spatial_sha256,
    )
    if fingerprint["adata_content_sha256"] != expected_content:
        raise ValueError(
            "result provenance input fingerprint content digest is inconsistent"
        )
    return fingerprint


def _validate_token_fingerprint(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("result provenance token_data fingerprint must be a mapping")
    token_data = dict(value)
    if not isinstance(token_data.get("verified"), bool):
        raise ValueError("result provenance token_data.verified must be boolean")
    required = (
        {
            "path",
            "verified",
            "manifest_sha256",
            "identity_sha256",
            "artifact_inventory_sha256",
        }
        if token_data["verified"]
        else {
            "path",
            "verified",
            "legacy_opt_in",
            "artifact_inventory_sha256",
        }
    )
    if set(token_data) != required:
        raise ValueError("result provenance token_data fields are invalid")
    if (
        not isinstance(token_data["path"], str)
        or not Path(token_data["path"]).is_absolute()
    ):
        raise ValueError("result provenance token_data path must be absolute")
    if token_data["verified"]:
        for field in (
            "manifest_sha256",
            "identity_sha256",
            "artifact_inventory_sha256",
        ):
            _require_sha256_value(token_data[field], f"token_data.{field}")
    else:
        if token_data["legacy_opt_in"] is not True:
            raise ValueError(
                "result provenance legacy token_data must record legacy_opt_in=true"
            )
        _require_sha256_value(
            token_data["artifact_inventory_sha256"],
            "token_data.artifact_inventory_sha256",
        )
    return token_data


def _validate_result_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("result provenance must be a mapping")
    required = {
        "config",
        "run",
        "input",
        "token_data",
        "checkpoint",
        "gene_dictionary",
        "esm_embedding",
        "code",
    }
    missing = sorted(required.difference(value))
    extra = sorted(set(value).difference(required))
    if missing or extra:
        raise ValueError(
            "result provenance fields are incomplete or unexpected: "
            f"missing={missing}, extra={extra}"
        )
    provenance = _json_safe(deepcopy(dict(value)))
    for section in required:
        if not isinstance(provenance[section], Mapping):
            raise ValueError(f"result provenance {section} must be a mapping")
    provenance["config"] = _validate_saved_config(provenance["config"])
    provenance["run"] = _validate_saved_run(provenance["run"])
    provenance["input"] = _validate_input_fingerprint(provenance["input"])
    provenance["token_data"] = _validate_token_fingerprint(
        provenance["token_data"]
    )
    if (
        not provenance["token_data"]["verified"]
        and not provenance["run"]["allow_unverified_legacy_tokens"]
    ):
        raise ValueError(
            "result provenance uses legacy unverified token_data without "
            "run allow_unverified_legacy_tokens permission"
        )
    for section in ("checkpoint", "gene_dictionary", "esm_embedding", "code"):
        provenance[section] = _validate_file_fingerprint_record(
            provenance[section],
            section,
        )
    return provenance


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    digest = hashlib.sha256()
    before = None
    after = None
    try:
        before_stat = os.fstat(descriptor)
        if not stat.S_ISREG(before_stat.st_mode):
            raise FileNotFoundError(
                f"Fingerprint source is not a regular file: {resolved}"
            )
        before = (
            int(before_stat.st_dev),
            int(before_stat.st_ino),
            int(before_stat.st_size),
            int(before_stat.st_mtime_ns),
            int(before_stat.st_ctime_ns),
        )
        while True:
            chunk = os.read(descriptor, 8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_stat = os.fstat(descriptor)
        after = (
            int(after_stat.st_dev),
            int(after_stat.st_ino),
            int(after_stat.st_size),
            int(after_stat.st_mtime_ns),
            int(after_stat.st_ctime_ns),
        )
    finally:
        os.close(descriptor)
    try:
        path_stat = os.stat(resolved, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"Fingerprint source path changed during stable snapshot: {resolved}"
        ) from exc
    path_snapshot = (
        int(path_stat.st_dev),
        int(path_stat.st_ino),
        int(path_stat.st_size),
        int(path_stat.st_mtime_ns),
        int(path_stat.st_ctime_ns),
    )
    if before != after or after != path_snapshot:
        raise RuntimeError(
            f"Fingerprint source changed during stable snapshot: {resolved}"
        )
    return {
        "path": str(resolved),
        "size": int(after[2]),
        "sha256": digest.hexdigest(),
    }


def _stable_manifest_token_inventory_sha256(
    root: Path,
    manifest: Mapping[str, Any],
) -> str:
    records = []
    for artifact in manifest["output"]["artifacts"]:
        relative = artifact["path"]
        target = _validate_relative_artifact_path(root, relative)
        fingerprint = _file_fingerprint(target)
        if (
            fingerprint["size"] != artifact["size"]
            or fingerprint["sha256"] != artifact["sha256"]
        ):
            raise RuntimeError(
                f"Token artifact changed during provenance snapshot: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size": fingerprint["size"],
                "sha256": fingerprint["sha256"],
            }
        )
    return _canonical_mapping_sha256(
        "brainbeacon-result-token-artifacts-v2",
        {
            "artifacts": records,
            "loader_sample_count": manifest["output"]["loader_sample_count"],
            "tokenized_cell_count": manifest["output"]["tokenized_cell_count"],
            "bundle_count": manifest["output"]["bundle_count"],
            "families": manifest["output"]["families"],
        },
    )


def _stable_legacy_token_inventory_sha256(root: Path) -> str:
    structure = _inspect_required_token_jobs(root)
    _validate_token_bundle_payloads(structure)
    records = []
    for token_dir in structure["token_dirs"]:
        for path in sorted(token_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Unsafe legacy token artifact: {path}")
            fingerprint = _file_fingerprint(path)
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "size": fingerprint["size"],
                    "sha256": fingerprint["sha256"],
                }
            )
    return _canonical_mapping_sha256(
        "brainbeacon-result-legacy-token-artifacts-v2",
        {
            "artifacts": records,
            "bundle_count": structure["bundle_count"],
            "families": structure["families"],
        },
    )


def _build_result_provenance(
    *,
    adata,
    config: Mapping[str, Any],
    token_data_path: str | Path,
    pretrain_ckpt: str | Path,
    gene_dict_path: str | Path,
    esm_embedding_path: str | Path,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    token_root = Path(token_data_path).resolve(strict=True)
    token_manifest_path = token_root / TOKEN_MANIFEST_NAME
    gene_dictionary = _file_fingerprint(gene_dict_path)
    if token_manifest_path.is_file() and not token_manifest_path.is_symlink():
        manifest_before = _file_fingerprint(token_manifest_path)
        token_manifest = _load_token_manifest(token_root)
        _validate_token_artifacts(token_root, token_manifest)
        artifact_inventory_sha256 = _stable_manifest_token_inventory_sha256(
            token_root,
            token_manifest,
        )
        manifest_after = _file_fingerprint(token_manifest_path)
        if manifest_before != manifest_after:
            raise RuntimeError(
                "Token manifest changed during provenance snapshot"
            )
        manifest_gene_hash = token_manifest["identity"]["gene_dictionary"][
            "sha256"
        ]
        if manifest_gene_hash != gene_dictionary["sha256"]:
            raise RuntimeError(
                "Token manifest gene-dictionary fingerprint changed before save"
            )
        input_fingerprint = _adata_fingerprint(adata)
        if input_fingerprint != token_manifest["identity"]["original_input"]:
            raise RuntimeError(
                "Input AnnData fingerprint changed or disagrees with token provenance"
            )
        token_data = {
            "path": str(token_root),
            "verified": True,
            "manifest_sha256": manifest_after["sha256"],
            "identity_sha256": _canonical_mapping_sha256(
                "brainbeacon-result-token-identity-v1",
                token_manifest["identity"],
            ),
            "artifact_inventory_sha256": artifact_inventory_sha256,
        }
    else:
        input_fingerprint = _adata_fingerprint(adata)
        token_data = {
            "path": str(token_root),
            "verified": False,
            "legacy_opt_in": True,
            "artifact_inventory_sha256": _stable_legacy_token_inventory_sha256(
                token_root
            ),
        }
    provenance = {
        "config": _json_safe(deepcopy(dict(config))),
        "run": _json_safe(deepcopy(dict(run))),
        "input": input_fingerprint,
        "token_data": token_data,
        "checkpoint": _file_fingerprint(pretrain_ckpt),
        "gene_dictionary": gene_dictionary,
        "esm_embedding": _file_fingerprint(esm_embedding_path),
        "code": _file_fingerprint(__file__),
    }
    return _validate_result_provenance(provenance)


def _validate_prefix(prefix: str) -> str:
    if (
        not isinstance(prefix, str)
        or not prefix
        or prefix in {".", ".."}
        or Path(prefix).name != prefix
        or "\x00" in prefix
    ):
        raise ValueError("prefix must be a non-empty filename component")
    return prefix


def _validate_common_npz_arrays(
    payload: Mapping[str, np.ndarray],
) -> tuple[int, int, np.dtype]:
    required = {
        "gene_names",
        "original_gene_indices",
        "gene_token_ids",
        "labels",
        "cell_counts",
        "gene_counts",
        "gene_coverage",
        "valid_mask",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("Result is missing output arrays: " + ", ".join(missing))
    if any(np.asarray(value).dtype.kind == "O" for value in payload.values()):
        raise ValueError("NPZ payload contains object dtype")
    gene_names = np.asarray(payload["gene_names"])
    labels = np.asarray(payload["labels"])
    if gene_names.ndim != 1 or gene_names.dtype.kind != "U":
        raise ValueError("gene_names must be a one-dimensional Unicode array")
    if labels.ndim != 1 or labels.dtype.kind != "U":
        raise ValueError("labels must be a one-dimensional Unicode array")
    rows = int(labels.size)
    genes = int(gene_names.size)
    expected_shapes = {
        "original_gene_indices": (genes,),
        "gene_token_ids": (genes,),
        "cell_counts": (rows,),
        "gene_counts": (rows, genes),
        "gene_coverage": (rows, genes),
        "valid_mask": (rows, genes),
    }
    for key, shape in expected_shapes.items():
        if np.asarray(payload[key]).shape != shape:
            raise ValueError(
                f"{key} shape must be {shape}, got {np.asarray(payload[key]).shape}"
            )
    for key in ("original_gene_indices", "gene_token_ids"):
        if np.asarray(payload[key]).dtype.kind not in {"i", "u"}:
            raise ValueError(f"{key} must have an integer dtype")
    count_dtype = np.asarray(payload["cell_counts"]).dtype
    if count_dtype not in {np.dtype(np.uint32), np.dtype(np.uint64)}:
        raise ValueError("cell_counts must have uint32 or uint64 dtype")
    if np.asarray(payload["gene_counts"]).dtype != count_dtype:
        raise ValueError(
            "gene_counts must have the same uint32 or uint64 dtype as cell_counts"
        )
    for key in ("gene_coverage", "valid_mask"):
        if np.asarray(payload[key]).dtype.kind not in {"b", "f"}:
            raise ValueError(f"{key} must have a boolean or floating dtype")
        _validate_finite(key, payload[key])
    if not np.array_equal(payload["gene_coverage"], payload["valid_mask"]):
        raise ValueError("gene_coverage and valid_mask must contain identical values")
    return rows, genes, count_dtype


def _common_axis_sha256(payload: Mapping[str, np.ndarray]) -> str:
    return _canonical_mapping_sha256(
        "brainbeacon-result-common-axis-v1",
        {
            "gene_names": np.asarray(payload["gene_names"]).tolist(),
            "original_gene_indices": np.asarray(
                payload["original_gene_indices"]
            ).tolist(),
            "gene_token_ids": np.asarray(payload["gene_token_ids"]).tolist(),
            "labels": np.asarray(payload["labels"]).tolist(),
        },
    )


def _validate_npz_payload(
    payload: Mapping[str, np.ndarray],
    kind: str,
) -> str:
    rows, genes, count_dtype = _validate_common_npz_arrays(payload)
    common_keys = {
        "gene_names",
        "original_gene_indices",
        "gene_token_ids",
        "labels",
        "cell_counts",
        "gene_counts",
        "gene_coverage",
        "valid_mask",
    }
    if kind == "attention":
        attention_keys = {
            key
            for key in ("attention_population", "attention_coobserved")
            if key in payload
        }
        if not attention_keys:
            raise ValueError("Attention NPZ requires at least one attention average")
        expected = common_keys | attention_keys | {"pair_counts", "pair_coverage"}
        if set(payload) != expected:
            raise ValueError("Attention NPZ keys do not match the v2 schema")
        for key in attention_keys | {"pair_coverage"}:
            array = np.asarray(payload[key])
            if array.shape != (rows, genes, genes) or array.dtype.kind != "f":
                raise ValueError(f"{key} has an invalid shape or dtype")
            _validate_finite(key, array)
        pair_counts = np.asarray(payload["pair_counts"])
        if (
            pair_counts.shape != (rows, genes, genes)
            or pair_counts.dtype != count_dtype
        ):
            raise ValueError("pair_counts has an invalid shape or dtype")
    elif kind == "embedding":
        if set(payload) != common_keys | {"embedding"}:
            raise ValueError("Embedding NPZ keys do not match the v2 schema")
        embedding = np.asarray(payload["embedding"])
        if (
            embedding.ndim != 3
            or embedding.shape[:2] != (rows, genes)
            or embedding.dtype.kind != "f"
        ):
            raise ValueError("embedding has an invalid shape or dtype")
        _validate_finite("embedding", embedding)
    else:
        raise ValueError(f"Unknown result artifact kind: {kind}")
    return _common_axis_sha256(payload)


def _build_result_payloads(
    result: Mapping[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    for key in (
        "gene_names",
        "labels",
        "original_gene_indices",
        "gene_token_ids",
        "cell_counts",
        "gene_counts",
        "gene_coverage",
        "valid_mask",
    ):
        if key not in result:
            raise ValueError(f"Result is missing required output key: {key}")
    common = {
        "gene_names": _as_unicode(result["gene_names"]),
        "original_gene_indices": np.asarray(result["original_gene_indices"]),
        "gene_token_ids": np.asarray(result["gene_token_ids"]),
        "labels": _as_unicode(result["labels"]),
        "cell_counts": np.asarray(result["cell_counts"]),
        "gene_counts": np.asarray(result["gene_counts"]),
        "gene_coverage": np.asarray(result["gene_coverage"]),
        "valid_mask": np.asarray(result["valid_mask"]),
    }
    _validate_common_npz_arrays(common)
    payloads: dict[str, dict[str, np.ndarray]] = {}
    population = result.get("attention_population")
    legacy_attention = result.get("attention")
    if population is None and legacy_attention is not None:
        population = legacy_attention
    elif population is not None and legacy_attention is not None and not np.array_equal(
        population,
        legacy_attention,
    ):
        raise ValueError("attention alias disagrees with attention_population")
    coobserved = result.get("attention_coobserved")
    if population is not None or coobserved is not None:
        if "pair_counts" not in result or "pair_coverage" not in result:
            raise ValueError("Attention output requires pair_counts and pair_coverage")
        attention_payload = dict(common)
        if population is not None:
            attention_payload["attention_population"] = np.asarray(population)
        if coobserved is not None:
            attention_payload["attention_coobserved"] = np.asarray(coobserved)
        attention_payload["pair_counts"] = np.asarray(result["pair_counts"])
        attention_payload["pair_coverage"] = np.asarray(result["pair_coverage"])
        _validate_npz_payload(attention_payload, "attention")
        payloads["attention"] = attention_payload
    if "embedding" in result:
        embedding_payload = {**common, "embedding": np.asarray(result["embedding"])}
        _validate_npz_payload(embedding_payload, "embedding")
        payloads["embedding"] = embedding_payload
    if not payloads:
        raise ValueError("Result contains neither attention nor embedding output")
    axis_sha256 = _common_axis_sha256(common)
    axis_mode = result.get("axis_mode")
    if not isinstance(axis_mode, str) or not axis_mode:
        raise ValueError("Result axis_mode must be a non-empty string")
    original_axis_sha256 = result.get("original_gene_axis_hash")
    _require_sha256_value(original_axis_sha256, "original_gene_axis_hash")
    axis = {
        "mode": axis_mode,
        "sha256": axis_sha256,
        "gene_sha256": _hash_index(common["gene_names"]),
        "label_sha256": _hash_index(common["labels"]),
        "original_gene_axis_sha256": original_axis_sha256,
        "gene_count": int(common["gene_names"].size),
        "label_count": int(common["labels"].size),
    }
    return payloads, axis


def _validate_result_context(
    provenance: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, np.ndarray]],
    axis: Mapping[str, Any],
) -> None:
    run = provenance["run"]
    expected_kinds = {
        "attention": {"attention"},
        "embedding": {"embedding"},
        "both": {"attention", "embedding"},
    }[run["mode"]]
    if set(payloads) != expected_kinds:
        raise ValueError(
            "result provenance run mode disagrees with attention/embedding payloads"
        )
    if "attention" in payloads:
        actual_attention = {
            key
            for key in ("attention_population", "attention_coobserved")
            if key in payloads["attention"]
        }
        expected_attention = {
            "population": {"attention_population"},
            "coobserved": {"attention_coobserved"},
            "both": {"attention_population", "attention_coobserved"},
        }[run["attention_average"]]
        if actual_attention != expected_attention:
            raise ValueError(
                "result provenance attention_average disagrees with attention keys"
            )
    expected_axis_mode = {
        "compact": "global_union",
        "full": "full_original",
    }[run["output_axis"]]
    if axis["mode"] != expected_axis_mode:
        raise ValueError(
            "result provenance output_axis disagrees with result axis mode"
        )
    expected_float_dtype = np.dtype(run["out_dtype"])
    for payload in payloads.values():
        for key, value in payload.items():
            dtype = np.asarray(value).dtype
            if dtype.kind == "f" and dtype != expected_float_dtype:
                raise ValueError(
                    "result provenance out_dtype disagrees with floating output "
                    f"{key}: expected {expected_float_dtype}, got {dtype}"
                )


def _write_npz_temp(path: str | Path, payload: Mapping[str, np.ndarray]) -> None:
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())


def _array_content_sha256(
    value,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise ValueError("Object arrays cannot be hashed as safe NPZ content")
    digest = hashlib.sha256()
    digest.update(b"brainbeacon-result-array-v1\0")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    if array.dtype.kind in {"U", "S"}:
        for value in array.flat:
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little", signed=False))
            digest.update(encoded)
        return digest.hexdigest()
    buffer_elements = max(1, int(chunk_bytes) // max(array.dtype.itemsize, 1))
    iterator = np.nditer(
        array,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=buffer_elements,
    )
    for chunk in iterator:
        contiguous = np.ascontiguousarray(chunk)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _array_schema(value) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": [int(size) for size in array.shape],
        "dtype": array.dtype.str,
        "sha256": _array_content_sha256(array),
    }


def _npz_payload_schema(
    payload: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    return {
        key: _array_schema(payload[key])
        for key in sorted(payload)
    }


def _common_content_sha256(
    key_schema: Mapping[str, Mapping[str, Any]],
) -> str:
    missing = sorted(set(RESULT_COMMON_NPZ_KEYS).difference(key_schema))
    if missing:
        raise ValueError(
            "NPZ schema is missing common arrays: " + ", ".join(missing)
        )
    return _canonical_mapping_sha256(
        "brainbeacon-result-common-content-v1",
        {
            key: dict(key_schema[key])
            for key in RESULT_COMMON_NPZ_KEYS
        },
    )


def _validate_npz_temp(
    path: str | Path,
    *,
    kind: str,
    expected_schema: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    actual_schema: dict[str, dict[str, Any]] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_schema):
                raise ValueError(f"Temporary {kind} NPZ keys do not match schema")
            for key in sorted(archive.files):
                # NumPy materializes one compressed member here. Validation
                # therefore adds at most one member-sized reload peak, not a
                # second in-memory copy of the complete NPZ result set.
                array = archive[key]
                try:
                    schema = _array_schema(array)
                finally:
                    del array
                expected = dict(expected_schema[key])
                if schema != expected:
                    raise ValueError(
                        f"Temporary {kind} NPZ {key} dtype, shape, or content hash "
                        "does not match the pre-write array"
                    )
                actual_schema[key] = schema
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Temporary {kind} NPZ validation failed: {path}") from exc
    return {
        "key_schema": actual_schema,
        "common_content_sha256": _common_content_sha256(actual_schema),
    }


def _publish_no_clobber(source: str | Path, target: str | Path) -> None:
    source = Path(source)
    target = Path(target)
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing output: {target}") from exc
    except OSError as exc:
        raise RuntimeError(
            "Same-filesystem hard-link publication is required and failed for "
            f"{target}: {exc}"
        ) from exc


def _acquire_result_lock(root: Path, prefix: str) -> Path:
    lock_path = root / f"{prefix}.lock"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"An exclusive result writer lock already exists: {lock_path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as handle:
        payload = (
            json.dumps(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "prefix": prefix,
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return lock_path


def _aggregation_definitions(payloads: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, str]:
    definitions = {
        "gene_coverage": "gene_counts / successfully processed cell_counts",
        "valid_mask": "compatibility alias with values identical to gene_coverage",
    }
    attention = payloads.get("attention")
    if attention is not None:
        definitions["pair_counts"] = (
            "number of successfully processed cells where query and key genes co-occur"
        )
        definitions["pair_coverage"] = (
            "pair_counts / successfully processed cell_counts"
        )
        if "attention_population" in attention:
            definitions["attention_population"] = (
                "attention_sum / successfully processed cell_counts; absent pairs are zero"
            )
        if "attention_coobserved" in attention:
            definitions["attention_coobserved"] = (
                "attention_sum / pair_counts where query and key genes co-occur"
            )
    if "embedding" in payloads:
        definitions["embedding"] = (
            "embedding_sum / gene_counts for cells where each gene is observed"
        )
    return definitions


def _write_json_temp(path: str | Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_result_manifest_mapping(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_name",
        "schema_version",
        "format_version",
        "state",
        "created_at",
        "prefix",
        "axis",
        "aggregation",
        "config",
        "provenance",
        "versions",
        "artifacts",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required:
        raise ValueError("Result manifest fields do not match the v2 schema")
    if manifest["schema_name"] != RESULT_SCHEMA_NAME:
        raise ValueError("Result manifest schema_name is invalid")
    if manifest["schema_version"] != RESULT_SCHEMA_VERSION:
        raise ValueError("Result manifest schema_version is invalid")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Result manifest format_version is invalid")
    if manifest["state"] != "complete":
        raise ValueError("Result manifest state must be complete")
    created_at = manifest["created_at"]
    if not isinstance(created_at, str):
        raise ValueError("Result manifest created_at is invalid")
    parsed_time = datetime.fromisoformat(created_at)
    if parsed_time.utcoffset() != timezone.utc.utcoffset(parsed_time):
        raise ValueError("Result manifest created_at must use UTC")
    prefix = _validate_prefix(manifest["prefix"])

    provenance = _validate_result_provenance(manifest["provenance"])
    if manifest["config"] != provenance["config"]:
        raise ValueError("Result manifest config disagrees with provenance")
    versions = manifest["versions"]
    required_versions = {
        "python",
        "numpy",
        "anndata",
        "scanpy",
        "scipy",
        "brainbeacon",
        "torch",
    }
    if (
        not isinstance(versions, Mapping)
        or not required_versions.issubset(versions)
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise ValueError("Result manifest versions are incomplete")
    aggregation = manifest["aggregation"]
    if (
        not isinstance(aggregation, Mapping)
        or not aggregation
        or any(not isinstance(value, str) or not value for value in aggregation.values())
    ):
        raise ValueError("Result manifest aggregation definitions are invalid")

    axis = manifest["axis"]
    axis_fields = {
        "mode",
        "sha256",
        "gene_sha256",
        "label_sha256",
        "original_gene_axis_sha256",
        "gene_count",
        "label_count",
    }
    if not isinstance(axis, Mapping) or set(axis) != axis_fields:
        raise ValueError("Result manifest axis fields are invalid")
    for field in (
        "sha256",
        "gene_sha256",
        "label_sha256",
        "original_gene_axis_sha256",
    ):
        _require_sha256_value(axis[field], f"manifest.axis.{field}")
    for field in ("gene_count", "label_count"):
        if (
            not isinstance(axis[field], int)
            or isinstance(axis[field], bool)
            or axis[field] < 0
        ):
            raise ValueError(f"Result manifest axis {field} is invalid")
    expected_axis_mode = {
        "compact": "global_union",
        "full": "full_original",
    }[provenance["run"]["output_axis"]]
    if axis["mode"] != expected_axis_mode:
        raise ValueError("Result manifest axis mode disagrees with provenance")

    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Result manifest artifacts are missing")
    artifact_fields = {
        "kind",
        "path",
        "keys",
        "key_schema",
        "size",
        "sha256",
        "axis_sha256",
        "common_content_sha256",
    }
    kinds = []
    common_hashes = []
    expected_float_dtype = np.dtype(provenance["run"]["out_dtype"])
    label_count = int(axis["label_count"])
    gene_count = int(axis["gene_count"])
    dim_model = provenance["config"].get("dim_model")
    if (
        not isinstance(dim_model, Integral)
        or isinstance(dim_model, (bool, np.bool_))
        or int(dim_model) <= 0
    ):
        raise ValueError("Result manifest config dim_model must be positive")
    dim_model = int(dim_model)
    shared_count_dtype: np.dtype | None = None
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != artifact_fields:
            raise ValueError("Result manifest artifact fields are invalid")
        kind = artifact["kind"]
        if kind not in {"attention", "embedding"}:
            raise ValueError("Result manifest artifact kind is invalid")
        kinds.append(kind)
        if artifact["path"] != f"{prefix}_{kind}.npz":
            raise ValueError("Result manifest artifact path is invalid")
        if (
            not isinstance(artifact["size"], int)
            or isinstance(artifact["size"], bool)
            or artifact["size"] < 0
        ):
            raise ValueError("Result manifest artifact size is invalid")
        _require_sha256_value(artifact["sha256"], "manifest artifact SHA256")
        _require_sha256_value(
            artifact["common_content_sha256"],
            "manifest artifact common content SHA256",
        )
        if artifact["axis_sha256"] != axis["sha256"]:
            raise ValueError("Result manifest artifact axis hash is inconsistent")
        key_schema = artifact["key_schema"]
        if not isinstance(key_schema, Mapping) or not key_schema:
            raise ValueError("Result manifest key schema is invalid")
        if artifact["keys"] != sorted(key_schema):
            raise ValueError("Result manifest artifact keys are inconsistent")
        attention_keys = {
            "population": {"attention_population"},
            "coobserved": {"attention_coobserved"},
            "both": {"attention_population", "attention_coobserved"},
        }[provenance["run"]["attention_average"]]
        common_keys = set(RESULT_COMMON_NPZ_KEYS)
        expected_keys = (
            common_keys | attention_keys | {"pair_counts", "pair_coverage"}
            if kind == "attention"
            else common_keys | {"embedding"}
        )
        if set(key_schema) != expected_keys:
            raise ValueError(
                f"Result manifest {kind} key set does not match the exact schema"
            )
        for key, schema in key_schema.items():
            if not isinstance(key, str) or not isinstance(schema, Mapping):
                raise ValueError("Result manifest key schema entry is invalid")
            if set(schema) != {"shape", "dtype", "sha256"}:
                raise ValueError("Result manifest key schema fields are invalid")
            shape = schema["shape"]
            if (
                not isinstance(shape, list)
                or any(
                    not isinstance(size, int) or isinstance(size, bool) or size < 0
                    for size in shape
                )
            ):
                raise ValueError("Result manifest key shape is invalid")
            dtype = np.dtype(schema["dtype"])
            if dtype.kind == "O":
                raise ValueError("Result manifest object dtype is forbidden")
            if dtype.kind == "f" and dtype != expected_float_dtype:
                raise ValueError("Result manifest floating dtype is inconsistent")
            _require_sha256_value(schema["sha256"], "manifest key SHA256")

        expected_common_shapes = {
            "gene_names": [gene_count],
            "original_gene_indices": [gene_count],
            "gene_token_ids": [gene_count],
            "labels": [label_count],
            "cell_counts": [label_count],
            "gene_counts": [label_count, gene_count],
            "gene_coverage": [label_count, gene_count],
            "valid_mask": [label_count, gene_count],
        }
        for key, shape in expected_common_shapes.items():
            if key_schema[key]["shape"] != shape:
                raise ValueError(
                    f"Result manifest common shape is invalid for {key}"
                )
        if np.dtype(key_schema["gene_names"]["dtype"]).kind != "U":
            raise ValueError("Result manifest gene_names dtype must be Unicode")
        if np.dtype(key_schema["labels"]["dtype"]).kind != "U":
            raise ValueError("Result manifest labels dtype must be Unicode")
        for key in ("original_gene_indices", "gene_token_ids"):
            if np.dtype(key_schema[key]["dtype"]).kind not in {"i", "u"}:
                raise ValueError(f"Result manifest {key} dtype must be integer")
        count_dtype = np.dtype(key_schema["cell_counts"]["dtype"])
        if count_dtype not in {np.dtype(np.uint32), np.dtype(np.uint64)}:
            raise ValueError("Result manifest count dtype must be uint32 or uint64")
        if np.dtype(key_schema["gene_counts"]["dtype"]) != count_dtype:
            raise ValueError("Result manifest cell/gene count dtypes disagree")
        if shared_count_dtype is None:
            shared_count_dtype = count_dtype
        elif shared_count_dtype != count_dtype:
            raise ValueError("Result manifest common count dtypes disagree")
        for key in ("gene_coverage", "valid_mask"):
            if np.dtype(key_schema[key]["dtype"]) != expected_float_dtype:
                raise ValueError(f"Result manifest {key} dtype is invalid")
        if kind == "attention":
            matrix_shape = [label_count, gene_count, gene_count]
            for key in attention_keys | {"pair_counts", "pair_coverage"}:
                if key_schema[key]["shape"] != matrix_shape:
                    raise ValueError(
                        f"Result manifest attention shape is invalid for {key}"
                    )
            if np.dtype(key_schema["pair_counts"]["dtype"]) != count_dtype:
                raise ValueError("Result manifest pair/common count dtypes disagree")
            for key in attention_keys | {"pair_coverage"}:
                if np.dtype(key_schema[key]["dtype"]) != expected_float_dtype:
                    raise ValueError(
                        f"Result manifest attention dtype is invalid for {key}"
                    )
        else:
            if key_schema["embedding"]["shape"] != [
                label_count,
                gene_count,
                dim_model,
            ]:
                raise ValueError("Result manifest embedding shape is invalid")
            if np.dtype(key_schema["embedding"]["dtype"]) != expected_float_dtype:
                raise ValueError("Result manifest embedding dtype is invalid")
        actual_common = _common_content_sha256(key_schema)
        if actual_common != artifact["common_content_sha256"]:
            raise ValueError("Result manifest common content hash is inconsistent")
        common_hashes.append(actual_common)
    if len(kinds) != len(set(kinds)) or len(set(common_hashes)) != 1:
        raise ValueError("Result manifest artifact kinds or common arrays disagree")
    expected_kinds = {
        "attention": {"attention"},
        "embedding": {"embedding"},
        "both": {"attention", "embedding"},
    }[provenance["run"]["mode"]]
    if set(kinds) != expected_kinds:
        raise ValueError("Result manifest mode disagrees with artifact kinds")
    attention = next(
        (artifact for artifact in artifacts if artifact["kind"] == "attention"),
        None,
    )
    if attention is not None:
        actual_attention = {
            key
            for key in ("attention_population", "attention_coobserved")
            if key in attention["key_schema"]
        }
        expected_attention = {
            "population": {"attention_population"},
            "coobserved": {"attention_coobserved"},
            "both": {"attention_population", "attention_coobserved"},
        }[provenance["run"]["attention_average"]]
        if actual_attention != expected_attention:
            raise ValueError("Result manifest attention average is inconsistent")


def _validate_result_manifest_temp(
    path: str | Path,
    expected: Mapping[str, Any],
) -> None:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_result_manifest_mapping(manifest)
        if manifest != expected:
            raise ValueError("Result manifest differs from the expected transaction")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Temporary result manifest validation failed: {path}") from exc


def _fsync_directory(root: str | Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(root), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_results(
    result: dict[str, Any],
    output_dir: str,
    prefix: str = "gene_analysis",
) -> None:
    prefix = _validate_prefix(prefix)
    provenance = _validate_result_provenance(result.get("provenance"))
    payloads, axis = _build_result_payloads(result)
    _validate_result_context(provenance, payloads, axis)
    expected_schemas = {
        kind: _npz_payload_schema(payload)
        for kind, payload in payloads.items()
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {root}")
    lock_path = _acquire_result_lock(root, prefix)
    _fsync_directory(root)
    targets = {
        kind: root / f"{prefix}_{kind}.npz"
        for kind in payloads
    }
    manifest_target = root / f"{prefix}_manifest.json"
    canonical_targets = [
        root / f"{prefix}_attention.npz",
        root / f"{prefix}_embedding.npz",
        manifest_target,
    ]
    existing = [
        path for path in canonical_targets if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs: "
            + ", ".join(str(path) for path in existing)
        )

    transaction_id = uuid.uuid4().hex
    temporary_npz = {
        kind: root / f".{prefix}.{transaction_id}.{kind}.npz"
        for kind in payloads
    }
    temporary_manifest = root / f".{prefix}.{transaction_id}.manifest.json"
    validated: dict[str, dict[str, Any]] = {}
    for kind, payload in payloads.items():
        _write_npz_temp(temporary_npz[kind], payload)
        validated[kind] = _validate_npz_temp(
            temporary_npz[kind],
            kind=kind,
            expected_schema=expected_schemas[kind],
        )
    common_hashes = {
        item["common_content_sha256"]
        for item in validated.values()
    }
    if len(common_hashes) != 1:
        raise RuntimeError("Temporary NPZ common arrays disagree across artifacts")

    artifacts = []
    for kind in payloads:
        temporary = temporary_npz[kind]
        artifacts.append(
            {
                "kind": kind,
                "path": targets[kind].name,
                "keys": sorted(validated[kind]["key_schema"]),
                "key_schema": validated[kind]["key_schema"],
                "size": int(temporary.stat().st_size),
                "sha256": _sha256_file(temporary),
                "axis_sha256": axis["sha256"],
                "common_content_sha256": validated[kind][
                    "common_content_sha256"
                ],
            }
        )

    manifest = {
        "schema_name": RESULT_SCHEMA_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "format_version": FORMAT_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prefix": prefix,
        "axis": axis,
        "aggregation": _aggregation_definitions(payloads),
        "config": provenance["config"],
        "provenance": provenance,
        "versions": _environment_versions(),
        "artifacts": artifacts,
    }
    _write_json_temp(temporary_manifest, manifest)
    _validate_result_manifest_temp(temporary_manifest, manifest)

    for kind in payloads:
        _publish_no_clobber(temporary_npz[kind], targets[kind])
    _fsync_directory(root)
    _publish_no_clobber(temporary_manifest, manifest_target)
    _fsync_directory(root)

    for path in [*temporary_npz.values(), temporary_manifest]:
        path.unlink()
    lock_path.unlink()
    _fsync_directory(root)


def run_gene_analysis(
    *,
    adata,
    config: Mapping[str, Any],
    gene_dict_path: str,
    pretrain_ckpt: str,
    esm_embedding_path: str,
    token_data_path: str,
    species: str,
    assay: str,
    mode: str = "both",
    group_by: str | None = None,
    attention_layers: str = "last",
    use_hvg: bool = True,
    n_hvg: int = 1000,
    max_cells: int | None = None,
    device: str | torch.device | None = None,
    out_dtype: str = "float16",
    hvg_source: str = "raw",
    hvg_flavor: str = "seurat_v3",
    output_axis: str = "compact",
    attention_average: str = "both",
    memory_limit_gib: float | None = None,
    allow_unverified_legacy_tokens: bool = False,
    output_dir: str | None = None,
    output_prefix: str = "gene_analysis",
) -> dict[str, Any]:
    """Run safe token preparation and GitHub-style hook inference.

    The input AnnData and config mapping are never modified in place. Results
    are returned in memory; NPZ files are written only when output_dir is set.
    """
    _validate_options(
        mode=mode,
        attention_layers=attention_layers,
        out_dtype=out_dtype,
        max_cells=max_cells,
        hvg_source=hvg_source,
        hvg_flavor=hvg_flavor,
        output_axis=output_axis,
        attention_average=attention_average,
        allow_unverified_legacy_tokens=allow_unverified_legacy_tokens,
        use_hvg=use_hvg,
        memory_limit_gib=memory_limit_gib,
    )
    config_local = _require_config(config)
    resolve_attention_layers(
        attention_layers,
        nlayers=int(config_local["nlayers"]),
    )
    gene_dict_path = _require_file(gene_dict_path, "gene dictionary")
    pretrain_ckpt = _require_file(pretrain_ckpt, "checkpoint")
    esm_embedding_path = _require_file(esm_embedding_path, "ESM embedding")
    if not species or not assay:
        raise ValueError("species and assay must be non-empty")
    if group_by is not None and group_by not in adata.obs.columns:
        raise ValueError(f"Column {group_by!r} not found in adata.obs")
    if group_by is not None and adata.obs[group_by].isna().any():
        raise ValueError(
            f"{group_by!r} contains real missing values; "
            "label or remove them explicitly"
        )

    logger.info(
        "Effective tokenization config: use_dev_abs=%s n_aux=%s "
        "hvg_source=%s hvg_flavor=%s species=%s assay=%s",
        config_local["use_dev_abs"],
        config_local["n_aux"],
        hvg_source,
        hvg_flavor,
        species,
        assay,
    )

    hvg_report = None
    original_fingerprint = None
    if use_hvg:
        adata_work, hvg_report = _select_hvg_adata(
            adata,
            n_hvg=n_hvg,
            hvg_source=hvg_source,
            hvg_flavor=hvg_flavor,
        )
    else:
        original_fingerprint = _adata_fingerprint(adata)
        adata_work = adata.copy()
    if max_cells is not None and max_cells < adata_work.n_obs:
        adata_work = adata_work[:max_cells].copy()
    inference_adata = ad.AnnData(
        X=None,
        obs=adata_work.obs.copy(deep=True),
        var=adata_work.var.copy(deep=True),
        shape=adata_work.shape,
    )
    token_work_matches_original = (
        not use_hvg
        and (max_cells is None or int(max_cells) >= int(adata.n_obs))
    )

    config_local["gene_dict_path"] = gene_dict_path
    config_local["pretrain_ckpt"] = pretrain_ckpt
    config_local["esm_embedding_path"] = esm_embedding_path

    token_path = _prepare_token_data(
        adata=adata_work,
        original_adata=adata,
        gene_dict_path=gene_dict_path,
        token_data_path=token_data_path,
        species=species,
        assay=assay,
        use_hvg=use_hvg,
        n_hvg=n_hvg,
        hvg_source=hvg_source,
        hvg_flavor=hvg_flavor,
        use_dev_abs=config_local["use_dev_abs"],
        n_aux=config_local["n_aux"],
        allow_unverified_legacy_tokens=allow_unverified_legacy_tokens,
        max_cells=max_cells,
        original_fingerprint=original_fingerprint,
        token_work_matches_original=token_work_matches_original,
    )
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    provenance_before = None
    effective_config = None
    run_context = None
    if output_dir is not None:
        effective_config = normalize_brainbeacon_model_config(
            deepcopy(config_local)
        )
        effective_config["batch_size"] = 1
        run_context = {
            "species": species,
            "assay": assay,
            "mode": mode,
            "group_by": group_by,
            "attention_layers": attention_layers,
            "use_hvg": bool(use_hvg),
            "n_hvg": int(n_hvg),
            "max_cells": None if max_cells is None else int(max_cells),
            "out_dtype": out_dtype,
            "hvg_source": hvg_source,
            "hvg_flavor": hvg_flavor,
            "output_axis": output_axis,
            "attention_average": attention_average,
            "allow_unverified_legacy_tokens": bool(
                allow_unverified_legacy_tokens
            ),
        }
        provenance_before = _build_result_provenance(
            adata=adata,
            config=effective_config,
            token_data_path=token_path,
            pretrain_ckpt=pretrain_ckpt,
            gene_dict_path=gene_dict_path,
            esm_embedding_path=esm_embedding_path,
            run=run_context,
        )
    result = dict(run_gene_level_inference(
        adata=inference_adata,
        token_data_path=token_path,
        config=config_local,
        pretrain_ckpt=pretrain_ckpt,
        gene_dict_path=gene_dict_path,
        esm_embedding_path=esm_embedding_path,
        device=resolved_device,
        mode=mode,
        group_by=group_by,
        attention_layers=attention_layers,
        max_cells=max_cells,
        out_dtype=out_dtype,
        output_axis=output_axis,
        attention_average=attention_average,
        memory_limit_gib=(
            None if memory_limit_gib is None else float(memory_limit_gib)
        ),
    ))
    if hvg_report is not None:
        result.update(hvg_report)
    if output_dir is not None:
        provenance_after = _build_result_provenance(
            adata=adata,
            config=effective_config,
            token_data_path=token_path,
            pretrain_ckpt=pretrain_ckpt,
            gene_dict_path=gene_dict_path,
            esm_embedding_path=esm_embedding_path,
            run=run_context,
        )
        if provenance_before != provenance_after:
            raise RuntimeError(
                "Result provenance changed during model inference; "
                "dependencies, input, tokens, or code drifted"
            )
        result["provenance"] = provenance_after
        save_results(result, output_dir, output_prefix)
    return result
