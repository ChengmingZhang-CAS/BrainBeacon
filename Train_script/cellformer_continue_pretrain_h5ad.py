"""
Continued CellFormer pretraining on multiple h5ad files that already contain
BrainBeacon embeddings in ``adata.obsm["bb_emb"]``.

This script intentionally skips:
1. BrainBeacon tokenization
2. BrainBeacon inference
3. Full-dataset prediction after each fit

Instead, it trains CellFormer directly from h5ad files and stores checkpoints
after each dataset so training can continue across many files.

Default batching mode is ``whole_h5ad``:
- one h5ad -> one training batch/unit
- no internal chunking
- validation is disabled until a later slice/FOV pipeline is enabled
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from brainbeacon.bbcellformer.pipeline.reconstruction import ReconstructPipeline
from brainbeacon.bbcellformer.utils.data import TranscriptomicDataset, XDict
from brainbeacon.configs.config import DEFAULT_PATHS
from brainbeacon.configs.config_train import config_train as GLOBAL_TRAIN_CONFIG
from brainbeacon.tokenizer import set_seed


DEFAULT_DATA_ROOT = (
    "/cpfs01/projects-HDD/cfff-c7cd658afc74_HDD/public/"
    "BrainST_impu/subsample_traindata_20per/AfterStage1"
)
DEFAULT_BB_CKPT = os.path.join(DEFAULT_PATHS["PRETRAIN_DIR"], "epoch_0_step_800000_0.33B.pt")


class Logger:
    def __init__(self, filename: str) -> None:
        self.terminal = sys.__stdout__
        self.log = open(filename, "w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue pretraining CellFormer with bb_emb-equipped h5ad files.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT, help="Directory containing h5ad files.")
    parser.add_argument("--output-root", type=str, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument(
        "--bb-ckpt-path",
        type=str,
        default=DEFAULT_BB_CKPT,
        help="BrainBeacon checkpoint used to initialize CellFormer gene embeddings.",
    )
    parser.add_argument(
        "--initial-ckpt-path",
        type=str,
        default=None,
        help="Initial CellFormer checkpoint. If omitted, the default pretrained CellFormer under PRETRAIN_DIR is used.",
    )
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Initialize CellFormer from config without loading a CellFormer checkpoint for the first dataset.",
    )
    parser.add_argument("--cellformer-version", type=str, default="cellformer", help="Checkpoint prefix for CellFormer.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument("--num-global-epochs", type=int, default=20, help="How many passes over all h5ad files.")
    parser.add_argument("--per-dataset-epochs", type=int, default=1, help="Epochs to train on each h5ad per pass.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling h5ad order between global epochs.")
    parser.add_argument("--default-assay", type=str, default="stereo", help="Fallback assay when platform is missing.")
    parser.add_argument("--default-specie", type=str, default=None, help="Fallback specie when organism is missing.")
    parser.add_argument("--slice-key", type=str, default="slice", help="obs key that marks biological slices.")
    parser.add_argument(
        "--catalog-cache-path",
        type=str,
        default=None,
        help="Optional reusable catalog cache JSON. Point multiple ablation runs at the same file to skip repeated h5ad scanning.",
    )
    parser.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Ignore any existing catalog cache and rebuild it from the current data root.",
    )
    parser.add_argument(
        "--prepare-catalog-only",
        action="store_true",
        help="Build or refresh the catalog cache, write meta files, then exit before training.",
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="balanced_slice",
        choices=["balanced_slice", "all_h5ad"],
        help="How to sample data each global epoch. `balanced_slice` keeps platform/specie slice-unit counts balanced.",
    )
    parser.add_argument(
        "--slices-per-group",
        type=int,
        default=None,
        help="Optional fixed number of slice units sampled per platform/specie group each global epoch. "
             "Large slices are pre-split by `max_cells_per_unit`. Defaults to the minimum available unit count across groups.",
    )
    parser.add_argument(
        "--batch-mode",
        type=str,
        default="whole_h5ad",
        choices=["whole_h5ad", "slice", "chunk"],
        help="Training unit inside each h5ad. Default keeps the whole h5ad intact.",
    )
    parser.add_argument(
        "--train-unit-key",
        type=str,
        default="train_unit",
        help="obs key used as internal training unit. Created automatically if missing.",
    )
    parser.add_argument(
        "--max-cells-per-unit",
        type=int,
        default=200000,
        help="Maximum cells per spatial train_unit. Slices are only split when they exceed this limit.",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.1, help="Validation ratio at train_unit level.")
    parser.add_argument("--enc-mod", type=str, default="flowformer", help="Encoder module name.")
    parser.add_argument(
        "--pe-type",
        type=str,
        default="fourier",
        choices=["fourier", "sin", "learnable", "naive", "none"],
        help="Spatial positional encoding type. Default is `fourier`.",
    )
    parser.add_argument("--mask-type", type=str, default="hidden", choices=["hidden", "input"])
    parser.add_argument("--mask-node-rate", type=float, default=0.75)
    parser.add_argument("--mask-feature-rate", type=float, default=0.25)
    parser.add_argument(
        "--drop-node-rate",
        type=float,
        default=0.0,
        help="Set to 0 to avoid the old training path silently subsetting cells.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=200000,
        help="Forwarded into CellFormer config. Keep aligned with max-cells-per-unit when drop_node_rate=0.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--scheduler", type=str, default="plat", choices=["plat", "none"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--bb-emb-dim",
        type=int,
        default=1024,
        help="Expected bb_emb width. Also used to update the global CellFormer bb branch input dim.",
    )
    parser.add_argument(
        "--pretrain-dir",
        type=str,
        default=None,
        help="Optional override for PRETRAIN_DIR used by the vendored CellFormer loader.",
    )
    parser.add_argument(
        "--gene-dict-path",
        type=str,
        default=None,
        help="Optional override for GENE_DICT_PATH used by the vendored CellFormer loader.",
    )
    parser.add_argument(
        "--esm-embed-path",
        type=str,
        default=None,
        help="Optional override for ESM_EMBED_PATH used by the vendored CellFormer loader.",
    )
    return parser.parse_args()


def discover_h5ad_files(data_root: str) -> list[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    files = sorted(root.rglob("*.h5ad"))
    if not files:
        raise FileNotFoundError(f"No .h5ad files found under: {data_root}")
    return files


def infer_assay_from_path(path: Path, default_assay: str) -> str:
    name = path.as_posix().lower()
    assay_tokens = {
        "merfish": "merfish",
        "xenium": "xenium",
        "starmap": "starmap",
        "slideseqv2": "slideseqv2",
        "slideseq": "slideseqv2",
        "stereo": "stereo",
        "stereoseq": "stereo",
        "snrna": "snrna",
    }
    for token, assay in assay_tokens.items():
        if token in name:
            return assay
    return default_assay


def canonicalize_assay(value: str | None, default_assay: str) -> str:
    if value is None:
        return default_assay
    text = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "merfish": "merfish",
        "merscope": "merfish",
        "xenium": "xenium",
        "starmap": "starmap",
        "slideseq": "slideseqv2",
        "slideseqv2": "slideseqv2",
        "stereo": "stereo",
        "stereoseq": "stereo",
        "snrna": "snrna",
        "scrna": "snrna",
    }
    return mapping.get(text, str(value).strip().lower() or default_assay)


def infer_specie_from_path(path: Path, default_specie: str | None = None) -> str:
    text = path.as_posix().lower()
    if "marmoset" in text:
        return "marmoset"
    if any(token in text for token in ("macaque", "macaqe", "macque", "macaca")):
        return "macaque"
    if "human" in text:
        return "human"
    if "mouse" in text:
        return "mouse"
    if default_specie:
        return default_specie
    raise ValueError(f"Cannot infer specie from path: {path}")


def canonicalize_specie(value: str | None, default_specie: str | None = None) -> str:
    if value is None:
        if default_specie is None:
            raise ValueError("specie is missing and no default was provided.")
        return default_specie
    text = str(value).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    mapping = {
        "human": "human",
        "homosapiens": "human",
        "hsapiens": "human",
        "mouse": "mouse",
        "musmusculus": "mouse",
        "macaque": "macaque",
        "macaqe": "macaque",
        "macque": "macaque",
        "macaca": "macaque",
        "marmoset": "marmoset",
    }
    return mapping.get(text, str(value).strip().lower() or (default_specie or "unknown"))


def infer_unique_obs_value(adata, keys: list[str]) -> str | None:
    for key in keys:
        if key not in adata.obs.columns:
            continue
        values = [
            str(value).strip()
            for value in adata.obs[key].dropna().unique().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        if len(values) == 1:
            return values[0]
    return None


def infer_uns_value(adata, keys: list[str]) -> str | None:
    for key in keys:
        value = adata.uns.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def sanitize_x_matrix(adata) -> None:
    if sp.issparse(adata.X):
        adata.X = adata.X.tocsr(copy=True)
        if np.isnan(adata.X.data).any():
            adata.X.data = np.nan_to_num(adata.X.data, nan=0.0)
    else:
        adata.X = np.nan_to_num(np.asarray(adata.X), nan=0.0).astype(np.float32, copy=False)


def ensure_platform(adata, h5ad_path: Path, default_assay: str) -> None:
    if "platform" in adata.obs.columns and adata.obs["platform"].nunique() == 1:
        adata.obs["platform"] = canonicalize_assay(str(adata.obs["platform"].iloc[0]), default_assay)
        return
    inferred = canonicalize_assay(infer_assay_from_path(h5ad_path, default_assay=default_assay), default_assay)
    adata.obs["platform"] = inferred


def ensure_specie(adata, h5ad_path: Path, default_specie: str | None) -> None:
    if "specie" in adata.obs.columns and adata.obs["specie"].nunique() == 1:
        adata.obs["specie"] = canonicalize_specie(str(adata.obs["specie"].iloc[0]), default_specie)
        return
    if "species" in adata.obs.columns and adata.obs["species"].nunique() == 1:
        adata.obs["specie"] = canonicalize_specie(str(adata.obs["species"].iloc[0]), default_specie)
        return

    inferred = (
        infer_unique_obs_value(adata, ["specie", "species", "organism"])
        or infer_uns_value(adata, ["specie", "species", "organism"])
        or infer_specie_from_path(h5ad_path, default_specie=default_specie)
    )
    adata.obs["specie"] = canonicalize_specie(str(inferred), default_specie)


def ensure_slice_column(adata, slice_key: str, fallback_name: str) -> None:
    if slice_key in adata.obs.columns:
        adata.obs[slice_key] = adata.obs[slice_key].astype(str)
        return
    if "batch" in adata.obs.columns:
        adata.obs[slice_key] = adata.obs["batch"].astype(str)
    else:
        adata.obs[slice_key] = fallback_name


def get_spatial_coords(adata) -> np.ndarray | None:
    if "spatial" not in adata.obsm:
        return None
    coords = np.asarray(adata.obsm["spatial"])
    if coords.ndim != 2 or coords.shape[1] == 0:
        return None
    if np.isnan(coords).any():
        coords = np.nan_to_num(coords, nan=0.0)
    return coords


def sort_indices_by_spatial(indices: np.ndarray, spatial_coords: np.ndarray | None) -> np.ndarray:
    if len(indices) <= 1 or spatial_coords is None:
        return indices.copy()
    dims = min(2, spatial_coords.shape[1])
    if dims == 0:
        return indices.copy()
    local_coords = spatial_coords[indices][:, :dims]
    if dims == 1:
        order = np.argsort(local_coords[:, 0], kind="mergesort")
        return indices[order]
    order = np.lexsort((local_coords[:, 1], local_coords[:, 0]))
    return indices[order]


def choose_fov_grid_shape(
    target_units: int,
    span_x: float,
    span_y: float,
) -> tuple[int, int]:
    if target_units <= 1:
        return 1, 1

    eps = 1e-8
    if span_x <= eps and span_y <= eps:
        return target_units, 1
    if span_x <= eps:
        return 1, target_units
    if span_y <= eps:
        return target_units, 1

    aspect = span_x / max(span_y, eps)
    best: tuple[int, int] | None = None
    best_score: tuple[float, float, float] | None = None
    for nx in range(1, target_units + 1):
        ny = math.ceil(target_units / nx)
        excess = float(nx * ny - target_units)
        grid_aspect = nx / ny
        aspect_error = abs(math.log(grid_aspect) - math.log(aspect))
        balance_error = abs(nx - ny)
        score = (excess, aspect_error, balance_error)
        if best_score is None or score < best_score:
            best_score = score
            best = (nx, ny)
    assert best is not None
    return best


def split_indices_by_fov_grid(
    indices: np.ndarray,
    spatial_coords: np.ndarray,
    nx: int,
    ny: int,
) -> list[np.ndarray]:
    local_coords = spatial_coords[indices][:, :2]
    x = local_coords[:, 0]
    y = local_coords[:, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    eps = 1e-8

    if nx == 1 or x_max - x_min <= eps:
        x_bin = np.zeros(len(indices), dtype=int)
    else:
        x_scaled = (x - x_min) / (x_max - x_min + eps)
        x_bin = np.minimum((x_scaled * nx).astype(int), nx - 1)

    if ny == 1 or y_max - y_min <= eps:
        y_bin = np.zeros(len(indices), dtype=int)
    else:
        y_scaled = (y - y_min) / (y_max - y_min + eps)
        y_bin = np.minimum((y_scaled * ny).astype(int), ny - 1)

    partitions: list[np.ndarray] = []
    for tile_y in range(ny):
        for tile_x in range(nx):
            tile_mask = (x_bin == tile_x) & (y_bin == tile_y)
            if not tile_mask.any():
                continue
            tile_indices = indices[tile_mask]
            partitions.append(sort_indices_by_spatial(tile_indices, spatial_coords))
    return partitions


def build_fov_recursive_partitions(
    indices: np.ndarray,
    spatial_coords: np.ndarray,
    max_cells_per_unit: int,
    *,
    depth: int = 0,
    max_depth: int = 12,
) -> list[np.ndarray]:
    if len(indices) == 0:
        return []
    if len(indices) <= max_cells_per_unit:
        return [sort_indices_by_spatial(indices, spatial_coords)]

    local_coords = spatial_coords[indices][:, :2]
    span_x = float(local_coords[:, 0].max() - local_coords[:, 0].min())
    span_y = float(local_coords[:, 1].max() - local_coords[:, 1].min())
    eps = 1e-8

    if depth >= max_depth or (span_x <= eps and span_y <= eps):
        n_units = math.ceil(len(indices) / max_cells_per_unit)
        ordered = sort_indices_by_spatial(indices, spatial_coords)
        return [chunk.copy() for chunk in np.array_split(ordered, n_units) if len(chunk) > 0]

    target_units = math.ceil(len(indices) / max_cells_per_unit)
    nx, ny = choose_fov_grid_shape(target_units=target_units, span_x=span_x, span_y=span_y)
    coarse_partitions = split_indices_by_fov_grid(indices, spatial_coords, nx=nx, ny=ny)

    if len(coarse_partitions) <= 1:
        n_units = math.ceil(len(indices) / max_cells_per_unit)
        ordered = sort_indices_by_spatial(indices, spatial_coords)
        return [chunk.copy() for chunk in np.array_split(ordered, n_units) if len(chunk) > 0]

    partitions: list[np.ndarray] = []
    for chunk_indices in coarse_partitions:
        if len(chunk_indices) <= max_cells_per_unit:
            partitions.append(chunk_indices.copy())
        else:
            partitions.extend(
                build_fov_recursive_partitions(
                    chunk_indices,
                    spatial_coords=spatial_coords,
                    max_cells_per_unit=max_cells_per_unit,
                    depth=depth + 1,
                    max_depth=max_depth,
                )
            )
    return partitions


def build_slice_unit_partitions(
    slice_indices: np.ndarray,
    spatial_coords: np.ndarray | None,
    max_cells_per_unit: int,
) -> list[np.ndarray]:
    if len(slice_indices) == 0:
        return []
    if len(slice_indices) <= max_cells_per_unit:
        return [sort_indices_by_spatial(slice_indices, spatial_coords)]
    if spatial_coords is None:
        ordered_indices = slice_indices.copy()
        n_units = math.ceil(len(ordered_indices) / max_cells_per_unit)
        return [chunk.copy() for chunk in np.array_split(ordered_indices, n_units) if len(chunk) > 0]
    return build_fov_recursive_partitions(
        slice_indices,
        spatial_coords=spatial_coords,
        max_cells_per_unit=max_cells_per_unit,
    )


def split_large_slices_into_train_units(
    adata,
    slice_key: str,
    train_unit_key: str,
    max_cells_per_unit: int,
    seed: int,
) -> None:
    if train_unit_key in adata.obs.columns:
        adata.obs[train_unit_key] = adata.obs[train_unit_key].astype(str)
        return

    train_units = np.empty(adata.n_obs, dtype=object)
    slice_values = adata.obs[slice_key].astype(str).to_numpy()
    spatial_coords = get_spatial_coords(adata)

    for slice_name in pd_unique_preserve_order(slice_values):
        indices = np.where(slice_values == slice_name)[0]
        if len(indices) == 0:
            continue
        partitions = build_slice_unit_partitions(
            indices,
            spatial_coords=spatial_coords,
            max_cells_per_unit=max_cells_per_unit,
        )
        for chunk_id, chunk_indices in enumerate(partitions):
            unit_name = f"{slice_name}__unit{chunk_id:05d}"
            train_units[chunk_indices] = unit_name

    adata.obs[train_unit_key] = train_units


def assign_train_valid_split(
    adata,
    slice_key: str,
    train_unit_key: str,
    valid_ratio: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    split = np.full(adata.n_obs, "train", dtype=object)
    slice_values = adata.obs[slice_key].astype(str).to_numpy()
    train_units = adata.obs[train_unit_key].astype(str).to_numpy()
    valid_units: set[str] = set()

    for slice_name in pd_unique_preserve_order(slice_values):
        slice_unit_names = pd_unique_preserve_order(train_units[slice_values == slice_name])
        if len(slice_unit_names) <= 1:
            continue
        n_valid = max(1, int(round(len(slice_unit_names) * valid_ratio)))
        n_valid = min(n_valid, len(slice_unit_names) - 1)
        chosen = rng.choice(slice_unit_names, size=n_valid, replace=False)
        valid_units.update(str(x) for x in np.atleast_1d(chosen))

    if not valid_units:
        all_units = pd_unique_preserve_order(train_units)
        if len(all_units) > 1:
            valid_units.add(str(rng.choice(all_units)))

    valid_mask = np.isin(train_units, list(valid_units))
    split[valid_mask] = "valid"
    adata.obs["split"] = split


def normalize_spatial_per_slice(adata, slice_key: str) -> None:
    if "spatial" not in adata.obsm:
        return

    slice_values = adata.obs[slice_key].astype(str).to_numpy()
    coords = np.asarray(adata.obsm["spatial"])
    x_norm = np.zeros(adata.n_obs, dtype=np.float32)
    y_norm = np.zeros(adata.n_obs, dtype=np.float32)

    for slice_name in pd_unique_preserve_order(slice_values):
        idx = np.where(slice_values == slice_name)[0]
        slice_coords = coords[idx]
        coord_min = slice_coords.min(axis=0)
        coord_max = slice_coords.max(axis=0)
        normalized = (slice_coords - coord_min) / (coord_max - coord_min + 1e-8)
        x_norm[idx] = normalized[:, 0].astype(np.float32, copy=False)
        y_norm[idx] = normalized[:, 1].astype(np.float32, copy=False)

    adata.obs["x_FOV_px"] = x_norm
    adata.obs["y_FOV_px"] = y_norm


def pd_unique_preserve_order(values: np.ndarray) -> np.ndarray:
    _, index = np.unique(values, return_index=True)
    return values[np.sort(index)]


def build_slice_sampling_catalog(
    h5ad_files: list[Path],
    default_assay: str,
    default_specie: str | None,
    slice_key: str,
    max_cells_per_unit: int,
) -> list[dict]:
    catalog: list[dict] = []

    for h5ad_path in h5ad_files:
        adata = sc.read_h5ad(h5ad_path, backed="r")
        try:
            obs = adata.obs.copy()
            spatial_coords = get_spatial_coords(adata)
            if slice_key in obs.columns:
                slice_values = obs[slice_key].astype(str).to_numpy()
            elif "batch" in obs.columns:
                slice_values = obs["batch"].astype(str).to_numpy()
            else:
                slice_values = np.repeat(h5ad_path.stem, adata.n_obs)

            platform = infer_unique_obs_value(adata, ["platform", "assay", "technology", "tech", "modality"])
            if not platform:
                platform = infer_uns_value(adata, ["platform", "assay", "technology", "tech", "modality"])
            if not platform:
                platform = infer_assay_from_path(h5ad_path, default_assay=default_assay)
            platform = canonicalize_assay(platform, default_assay)

            specie = infer_unique_obs_value(adata, ["specie", "species", "organism"])
            if not specie:
                specie = infer_uns_value(adata, ["specie", "species", "organism"])
            if not specie:
                specie = infer_specie_from_path(h5ad_path, default_specie=default_specie)
            specie = canonicalize_specie(specie, default_specie)

            for slice_name in pd_unique_preserve_order(slice_values):
                slice_indices = np.where(slice_values == slice_name)[0]
                source_slice_n_cells = int(len(slice_indices))
                partitions = build_slice_unit_partitions(
                    slice_indices,
                    spatial_coords=spatial_coords,
                    max_cells_per_unit=max_cells_per_unit,
                )
                for slice_unit_index, partition in enumerate(partitions):
                    catalog.append(
                        {
                            "h5ad_path": str(h5ad_path),
                            "dataset_name": h5ad_path.stem,
                            "platform": str(platform),
                            "specie": str(specie),
                            "slice_name": f"{slice_name}__unit{slice_unit_index:05d}",
                            "source_slice_name": str(slice_name),
                            "slice_unit_name": f"{slice_name}__unit{slice_unit_index:05d}",
                            "slice_unit_index": slice_unit_index,
                            "slice_unit_total": len(partitions),
                            "n_cells": int(len(partition)),
                            "source_slice_n_cells": source_slice_n_cells,
                            "group_key": (str(platform), str(specie)),
                        }
                    )
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    if not catalog:
        raise RuntimeError("Slice sampling catalog is empty.")
    return catalog


def summarize_sampling_catalog(catalog: list[dict]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in catalog:
        grouped[item["group_key"]].append(item)

    for (platform, specie), items in sorted(grouped.items()):
        summary[f"{platform}__{specie}"] = {
            "num_slice_units": len(items),
            "num_source_slices": len({(item["h5ad_path"], item["source_slice_name"]) for item in items}),
            "num_datasets": len({item["h5ad_path"] for item in items}),
            "num_cells": int(sum(item["n_cells"] for item in items)),
        }
    return summary


def build_catalog_file_records(h5ad_files: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in h5ad_files:
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return records


def normalize_catalog_entries(catalog: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for raw_item in catalog:
        item = dict(raw_item)
        platform = str(item["platform"])
        specie = str(item["specie"])
        item["h5ad_path"] = str(item["h5ad_path"])
        item["dataset_name"] = str(item["dataset_name"])
        item["platform"] = platform
        item["specie"] = specie
        item["slice_name"] = str(item["slice_name"])
        item["source_slice_name"] = str(item["source_slice_name"])
        item["slice_unit_name"] = str(item["slice_unit_name"])
        item["slice_unit_index"] = int(item["slice_unit_index"])
        item["slice_unit_total"] = int(item["slice_unit_total"])
        item["n_cells"] = int(item["n_cells"])
        item["source_slice_n_cells"] = int(item["source_slice_n_cells"])
        item["group_key"] = (platform, specie)
        normalized.append(item)
    if not normalized:
        raise RuntimeError("Catalog cache does not contain any slice units.")
    return normalized


def build_catalog_payload(
    *,
    data_root: str,
    h5ad_files: list[Path],
    default_assay: str,
    default_specie: str | None,
    slice_key: str,
    max_cells_per_unit: int,
    catalog: list[dict],
) -> dict:
    return {
        "cache_version": 1,
        "data_root": str(Path(data_root).resolve()),
        "default_assay": default_assay,
        "default_specie": default_specie,
        "slice_key": slice_key,
        "max_cells_per_unit": int(max_cells_per_unit),
        "num_h5ad_files": len(h5ad_files),
        "files": build_catalog_file_records(h5ad_files),
        "catalog": catalog,
    }


def write_catalog_payload(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_catalog_payload(
    *,
    cache_path: Path,
    data_root: str,
    h5ad_files: list[Path],
    default_assay: str,
    default_specie: str | None,
    slice_key: str,
    max_cells_per_unit: int,
) -> dict | None:
    if not cache_path.exists():
        return None

    with open(cache_path, "r") as handle:
        payload = json.load(handle)

    expected = {
        "cache_version": 1,
        "data_root": str(Path(data_root).resolve()),
        "default_assay": default_assay,
        "default_specie": default_specie,
        "slice_key": slice_key,
        "max_cells_per_unit": int(max_cells_per_unit),
        "num_h5ad_files": len(h5ad_files),
        "files": build_catalog_file_records(h5ad_files),
    }

    mismatch_reasons: list[str] = []
    for key, expected_value in expected.items():
        actual_value = payload.get(key)
        if actual_value != expected_value:
            mismatch_reasons.append(key)

    if mismatch_reasons:
        mismatch_text = ", ".join(mismatch_reasons)
        print(f"Catalog cache mismatch at {cache_path}: {mismatch_text}. Rebuilding catalog.")
        return None

    payload["catalog"] = normalize_catalog_entries(payload.get("catalog", []))
    return payload


def get_or_create_catalog_payload(
    *,
    cache_path: Path,
    data_root: str,
    h5ad_files: list[Path],
    default_assay: str,
    default_specie: str | None,
    slice_key: str,
    max_cells_per_unit: int,
    rebuild_catalog: bool,
) -> tuple[dict, str]:
    if not rebuild_catalog:
        payload = load_catalog_payload(
            cache_path=cache_path,
            data_root=data_root,
            h5ad_files=h5ad_files,
            default_assay=default_assay,
            default_specie=default_specie,
            slice_key=slice_key,
            max_cells_per_unit=max_cells_per_unit,
        )
        if payload is not None:
            print(f"Loaded slice sampling catalog from cache: {cache_path}")
            return payload, "cache"

    catalog = build_slice_sampling_catalog(
        h5ad_files=h5ad_files,
        default_assay=default_assay,
        default_specie=default_specie,
        slice_key=slice_key,
        max_cells_per_unit=max_cells_per_unit,
    )
    payload = build_catalog_payload(
        data_root=data_root,
        h5ad_files=h5ad_files,
        default_assay=default_assay,
        default_specie=default_specie,
        slice_key=slice_key,
        max_cells_per_unit=max_cells_per_unit,
        catalog=catalog,
    )
    write_catalog_payload(cache_path, payload)
    print(f"Saved slice sampling catalog to cache: {cache_path}")
    return payload, "rebuilt"


def build_epoch_training_plan(
    *,
    catalog: list[dict],
    sampling_mode: str,
    slices_per_group: int | None,
    global_epoch: int,
    seed: int,
    no_shuffle: bool,
) -> list[dict]:
    if sampling_mode == "all_h5ad":
        h5ad_paths = sorted({item["h5ad_path"] for item in catalog})
        if not no_shuffle:
            random.Random(seed + global_epoch).shuffle(h5ad_paths)
        return [{"h5ad_path": path, "selected_slice_units": None} for path in h5ad_paths]

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in catalog:
        grouped[item["group_key"]].append(item)

    if not grouped:
        raise RuntimeError("No platform/specie groups available for balanced sampling.")

    available_counts = {group_key: len(items) for group_key, items in grouped.items()}
    if slices_per_group is None:
        target_slice_units = min(available_counts.values())
    else:
        target_slice_units = int(slices_per_group)
        too_small = {group_key: count for group_key, count in available_counts.items() if count < target_slice_units}
        if too_small:
            preview = ", ".join(f"{platform}/{specie}={count}" for (platform, specie), count in sorted(too_small.items()))
            raise ValueError(
                f"`slices_per_group={target_slice_units}` is larger than some groups: {preview}."
            )

    rng = random.Random(seed + global_epoch)
    selected_entries: list[dict] = []
    for group_key, items in sorted(grouped.items()):
        candidates = list(items)
        rng.shuffle(candidates)
        selected_entries.extend(candidates[:target_slice_units])

    plan_by_file: dict[str, list[dict]] = defaultdict(list)
    file_order: list[str] = []
    seen_files: set[str] = set()
    for entry in selected_entries:
        h5ad_path = entry["h5ad_path"]
        plan_by_file[h5ad_path].append(entry)
        if h5ad_path not in seen_files:
            seen_files.add(h5ad_path)
            file_order.append(h5ad_path)

    if not no_shuffle:
        rng.shuffle(file_order)

    return [
        {
            "h5ad_path": h5ad_path,
            "selected_slice_units": sorted(
                plan_by_file[h5ad_path],
                key=lambda item: (item["source_slice_name"], item["slice_unit_index"]),
            ),
        }
        for h5ad_path in file_order
    ]


def prepare_training_adata(
    h5ad_path: Path,
    default_assay: str,
    default_specie: str | None,
    slice_key: str,
    batch_mode: str,
    train_unit_key: str,
    max_cells_per_unit: int,
    valid_ratio: float,
    seed: int,
    bb_emb_dim: int,
    selected_slice_units: list[dict] | None = None,
):
    adata = sc.read_h5ad(h5ad_path)
    adata.obs_names_make_unique()
    sanitize_x_matrix(adata)

    if "bb_emb" not in adata.obsm:
        raise KeyError(f"{h5ad_path} does not contain adata.obsm['bb_emb']")
    if adata.obsm["bb_emb"].shape[0] != adata.n_obs:
        raise ValueError(
            f"{h5ad_path} has bb_emb rows={adata.obsm['bb_emb'].shape[0]} but n_obs={adata.n_obs}"
        )
    if adata.obsm["bb_emb"].shape[1] != bb_emb_dim:
        raise ValueError(
            f"{h5ad_path} has bb_emb dim={adata.obsm['bb_emb'].shape[1]}, expected {bb_emb_dim}"
        )

    ensure_platform(adata, h5ad_path=h5ad_path, default_assay=default_assay)
    ensure_specie(adata, h5ad_path=h5ad_path, default_specie=default_specie)
    ensure_slice_column(adata, slice_key=slice_key, fallback_name=h5ad_path.stem)

    preassigned_train_units: np.ndarray | None = None
    if selected_slice_units is not None:
        slice_series = adata.obs[slice_key].astype(str).to_numpy()
        spatial_coords = get_spatial_coords(adata)
        keep_mask = np.zeros(adata.n_obs, dtype=bool)
        selected_train_units = np.full(adata.n_obs, "", dtype=object)
        selected_by_slice: dict[str, dict[str, int | set[int]]] = {}

        for item in selected_slice_units:
            source_slice_name = str(item["source_slice_name"])
            unit_total = int(item["slice_unit_total"])
            unit_index = int(item["slice_unit_index"])
            payload = selected_by_slice.setdefault(
                source_slice_name,
                {"slice_unit_total": unit_total, "slice_unit_indices": set()},
            )
            if int(payload["slice_unit_total"]) != unit_total:
                raise RuntimeError(
                    f"Inconsistent slice unit totals for {h5ad_path} slice {source_slice_name}: "
                    f"{payload['slice_unit_total']} vs {unit_total}"
                )
            payload["slice_unit_indices"].add(unit_index)

        for source_slice_name, payload in selected_by_slice.items():
            slice_indices = np.where(slice_series == source_slice_name)[0]
            if len(slice_indices) == 0:
                raise RuntimeError(f"{h5ad_path} did not contain requested slice: {source_slice_name}")

            split_indices = build_slice_unit_partitions(
                slice_indices,
                spatial_coords=spatial_coords,
                max_cells_per_unit=max_cells_per_unit,
            )
            actual_total = int(payload["slice_unit_total"])
            if len(split_indices) != actual_total:
                raise RuntimeError(
                    f"{h5ad_path} slice {source_slice_name} built {len(split_indices)} units but sampling plan requested "
                    f"{actual_total}."
                )
            for unit_index in sorted(payload["slice_unit_indices"]):
                if unit_index >= len(split_indices):
                    raise RuntimeError(
                        f"{h5ad_path} slice {source_slice_name} requested unit {unit_index} "
                        f"but only {len(split_indices)} units exist."
                    )
                chunk_indices = split_indices[unit_index]
                if len(chunk_indices) == 0:
                    continue
                keep_mask[chunk_indices] = True
                selected_train_units[chunk_indices] = f"{source_slice_name}__unit{unit_index:05d}"

        if not keep_mask.any():
            requested_units = [item["slice_unit_name"] for item in selected_slice_units]
            raise RuntimeError(f"{h5ad_path} did not contain any of the requested slice units: {requested_units}")
        adata = adata[keep_mask].copy()
        preassigned_train_units = selected_train_units[keep_mask].astype(str)

    if preassigned_train_units is not None:
        adata.obs[train_unit_key] = preassigned_train_units
        assign_train_valid_split(
            adata=adata,
            slice_key=slice_key,
            train_unit_key=train_unit_key,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    elif batch_mode == "whole_h5ad":
        adata.obs[train_unit_key] = h5ad_path.stem
        adata.obs["split"] = "train"
    elif batch_mode == "slice":
        adata.obs[train_unit_key] = adata.obs[slice_key].astype(str)
        assign_train_valid_split(
            adata=adata,
            slice_key=slice_key,
            train_unit_key=train_unit_key,
            valid_ratio=valid_ratio,
            seed=seed,
        )
    else:
        split_large_slices_into_train_units(
            adata=adata,
            slice_key=slice_key,
            train_unit_key=train_unit_key,
            max_cells_per_unit=max_cells_per_unit,
            seed=seed,
        )
        assign_train_valid_split(
            adata=adata,
            slice_key=slice_key,
            train_unit_key=train_unit_key,
            valid_ratio=valid_ratio,
            seed=seed,
        )

    adata.obs["batch"] = adata.obs[train_unit_key].astype(str)
    normalize_spatial_per_slice(adata, slice_key=slice_key)
    return adata


def evaluate_split_loss(model, dataloader: DataLoader, split_name: str, device: torch.device) -> float | None:
    losses = []
    with torch.no_grad():
        model.eval()
        for data_dict in dataloader:
            if data_dict["split"] is None:
                continue
            split_values = np.asarray(data_dict["split"])
            if split_values.size == 0 or split_values[0] != split_name:
                continue

            input_dict = data_dict.copy()
            del input_dict["gene_list"], input_dict["split"]
            for key in input_dict:
                input_dict[key] = input_dict[key].to(device)
            x_dict = XDict(input_dict)
            _, loss = model(x_dict, data_dict["gene_list"])
            losses.append(loss.item())

    if not losses:
        return None
    return float(np.mean(losses))


def load_optimizer_scheduler_state(
    checkpoint_path: str | None,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
) -> bool:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return False

    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        return False

    loaded = False
    if "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        loaded = True
    if scheduler is not None and "scheduler_state_dict" in state and state["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    return loaded


def fit_pipeline_on_adata(
    pipeline: ReconstructPipeline,
    adata,
    current_ckpt_path: str | None,
    epochs: int,
    lr: float,
    wd: float,
    scheduler_name: str,
    patience: int,
    workers: int,
    device: torch.device,
) -> tuple[dict, dict | None, dict | None, float, float | None]:
    pipeline.model.to(device)
    processed = pipeline.common_preprocess(adata, 0, covariate_fields=None, ensembl_auto_conversion=False)
    dataset = TranscriptomicDataset(processed, split_field="split")
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=workers)

    optimizer = torch.optim.AdamW(pipeline.model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=patience, factor=0.9) if scheduler_name == "plat" else None
    optimizer_loaded = load_optimizer_scheduler_state(current_ckpt_path, optimizer, scheduler)

    train_cells = int((processed.obs["split"] == "train").sum())
    valid_cells = int((processed.obs["split"] == "valid").sum())
    print(f"Filtered data shape: {processed.shape}, train cells: {train_cells}, valid cells: {valid_cells}")

    best_state = deepcopy(pipeline.model.state_dict())
    best_valid = None
    last_train = float("nan")
    last_valid = None

    for epoch_idx in range(epochs):
        pipeline.model.train()
        epoch_losses = []

        if not optimizer_loaded and epoch_idx < 5:
            warmup_lr = lr * float(epoch_idx + 1) / 5.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        for data_dict in dataloader:
            split_values = np.asarray(data_dict["split"])
            if split_values.size == 0 or split_values[0] != "train":
                continue

            input_dict = data_dict.copy()
            del input_dict["gene_list"], input_dict["split"]
            for key in input_dict:
                input_dict[key] = input_dict[key].to(device)
            x_dict = XDict(input_dict)
            _, loss = pipeline.model(x_dict, data_dict["gene_list"])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(pipeline.model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        if not epoch_losses:
            raise RuntimeError("No training batches were produced. Check train_unit splitting and split assignment.")

        last_train = float(np.mean(epoch_losses))
        last_valid = evaluate_split_loss(pipeline.model, dataloader, split_name="valid", device=device)
        metric = last_valid if last_valid is not None else last_train

        if scheduler is not None:
            scheduler.step(metric)

        if best_valid is None or metric <= best_valid:
            best_valid = metric
            best_state = deepcopy(pipeline.model.state_dict())

        if last_valid is None:
            print(f"Epoch {epoch_idx + 1}/{epochs} | train loss: {last_train:.4f}")
        else:
            print(
                f"Epoch {epoch_idx + 1}/{epochs} | train loss: {last_train:.4f} | "
                f"valid loss: {last_valid:.4f}"
            )

    pipeline.model.load_state_dict(best_state)
    pipeline.fitted = True
    return best_state, optimizer.state_dict(), scheduler.state_dict() if scheduler is not None else None, last_train, last_valid


def save_training_checkpoint(
    save_path: Path,
    model_state_dict: dict,
    optimizer_state_dict: dict | None,
    scheduler_state_dict: dict | None,
    *,
    global_epoch: int,
    dataset_index: int,
    dataset_path: str,
    dataset_name: str,
    n_obs: int,
    n_vars: int,
    n_train_units: int,
    last_train_loss: float,
    last_valid_loss: float | None,
) -> None:
    payload = {
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "global_epoch": global_epoch,
        "dataset_index": dataset_index,
        "dataset_path": dataset_path,
        "dataset_name": dataset_name,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "n_train_units": n_train_units,
        "last_train_loss": last_train_loss,
        "last_valid_loss": last_valid_loss,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, save_path)


def build_path_dict(args: argparse.Namespace) -> dict | None:
    path_dict = {}
    if args.pretrain_dir:
        path_dict["PRETRAIN_DIR"] = args.pretrain_dir
    if args.gene_dict_path:
        path_dict["GENE_DICT_PATH"] = args.gene_dict_path
    if args.esm_embed_path:
        path_dict["ESM_EMBED_PATH"] = args.esm_embed_path
    return path_dict or None


def prepare_runtime(args: argparse.Namespace) -> tuple[torch.device, Path, Path, Path]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir = output_root / "logs"
    ckpt_dir = output_root / "checkpoints"
    meta_dir = output_root / "meta"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sys.stdout = Logger(str(log_path))
    sys.stderr = sys.stdout

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, ckpt_dir, meta_dir, log_path


def main() -> None:
    args = parse_args()
    device, ckpt_dir, meta_dir, log_path = prepare_runtime(args)
    path_dict = build_path_dict(args)

    GLOBAL_TRAIN_CONFIG["dim_model"] = args.bb_emb_dim
    set_seed(args.seed, deterministic=True)

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    h5ad_files = discover_h5ad_files(args.data_root)
    catalog_cache_path = Path(args.catalog_cache_path) if args.catalog_cache_path else (meta_dir / "catalog.json")
    catalog_payload, catalog_source = get_or_create_catalog_payload(
        cache_path=catalog_cache_path,
        data_root=args.data_root,
        h5ad_files=h5ad_files,
        default_assay=args.default_assay,
        default_specie=args.default_specie,
        slice_key=args.slice_key,
        max_cells_per_unit=args.max_cells_per_unit,
        rebuild_catalog=args.rebuild_catalog,
    )
    slice_catalog = catalog_payload["catalog"]
    run_catalog_path = meta_dir / "catalog.json"
    if run_catalog_path != catalog_cache_path:
        write_catalog_payload(run_catalog_path, catalog_payload)

    sampling_summary = summarize_sampling_catalog(slice_catalog)
    effective_batch_mode = args.batch_mode
    if args.sampling_mode == "balanced_slice" and effective_batch_mode != "chunk":
        effective_batch_mode = "chunk"
        print("sampling_mode=balanced_slice uses pre-split slice units; switching batch_mode to chunk.")

    manifest = {
        "data_root": args.data_root,
        "num_h5ad_files": len(h5ad_files),
        "files": [str(path) for path in h5ad_files],
        "sampling_mode": args.sampling_mode,
        "effective_batch_mode": effective_batch_mode,
        "sampling_summary": sampling_summary,
        "catalog_cache_path": str(catalog_cache_path),
        "catalog_source": catalog_source,
        "log_path": str(log_path),
        "args": vars(args),
    }
    with open(meta_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    if args.prepare_catalog_only:
        print("Catalog preparation completed; exiting because --prepare-catalog-only was set.")
        return

    current_ckpt_path = args.initial_ckpt_path
    latest_ckpt_path = ckpt_dir / "latest.pt"
    history_path = meta_dir / "history.jsonl"

    for global_epoch in range(1, args.num_global_epochs + 1):
        print(f"\n========== Global Epoch {global_epoch}/{args.num_global_epochs} ==========")
        epoch_plan = build_epoch_training_plan(
            catalog=slice_catalog,
            sampling_mode=args.sampling_mode,
            slices_per_group=args.slices_per_group,
            global_epoch=global_epoch,
            seed=args.seed,
            no_shuffle=args.no_shuffle,
        )
        if args.sampling_mode == "balanced_slice":
            group_counter: dict[tuple[str, str], int] = defaultdict(int)
            for item in epoch_plan:
                for selected_unit in item["selected_slice_units"] or []:
                    group_counter[(selected_unit["platform"], selected_unit["specie"])] += 1
            group_text = ", ".join(
                f"{platform}/{specie}={count} units"
                for (platform, specie), count in sorted(group_counter.items())
            )
            print(f"Balanced slice-unit plan: {group_text}")
        epoch_checkpoint_meta = None

        for dataset_index, plan_item in enumerate(epoch_plan, start=1):
            h5ad_path = Path(plan_item["h5ad_path"])
            selected_slice_units = plan_item["selected_slice_units"]
            selected_slice_unit_text = (
                f", selected units: {len(selected_slice_units)}, source slices: "
                f"{len({item['source_slice_name'] for item in selected_slice_units})}"
                if selected_slice_units is not None else ""
            )
            print(f"\n--- [{dataset_index}/{len(epoch_plan)}] Training on {h5ad_path.name}{selected_slice_unit_text} ---")
            adata = prepare_training_adata(
                h5ad_path=h5ad_path,
                default_assay=args.default_assay,
                default_specie=args.default_specie,
                slice_key=args.slice_key,
                batch_mode=effective_batch_mode,
                train_unit_key=args.train_unit_key,
                max_cells_per_unit=args.max_cells_per_unit,
                valid_ratio=args.valid_ratio,
                seed=args.seed + global_epoch + dataset_index,
                bb_emb_dim=args.bb_emb_dim,
                selected_slice_units=selected_slice_units,
            )
            print(
                f"adata shape: {adata.shape}, slices: {adata.obs[args.slice_key].nunique()}, "
                f"train units: {adata.obs[args.train_unit_key].nunique()}, "
                f"platform/specie: {adata.obs['platform'].iloc[0]}/{adata.obs['specie'].iloc[0]}"
            )

            overwrite_config = {
                "name": f"bb_{args.enc_mod}",
                "enc_mod": args.enc_mod,
                "objective": "imputation",
                "pe_type": None if args.pe_type == "none" else args.pe_type,
                "use_hidden_pe": args.pe_type != "none",
                "mask_node_rate": args.mask_node_rate,
                "mask_feature_rate": args.mask_feature_rate,
                "drop_node_rate": args.drop_node_rate,
                "max_batch_size": args.max_batch_size,
                "mask_type": args.mask_type,
            }
            print(
                f"Model config override: pe_type={overwrite_config['pe_type']}, "
                f"use_hidden_pe={overwrite_config['use_hidden_pe']}, mask_type={args.mask_type}"
            )

            use_pretrained = not (args.random_init and current_ckpt_path is None)
            if use_pretrained:
                print(f"Initializing CellFormer from checkpoint: {current_ckpt_path or '[default pretrained checkpoint]'}")
            else:
                print("Initializing CellFormer from config with random weights for the first dataset.")

            pipeline = ReconstructPipeline(
                pretrain_prefix=args.cellformer_version,
                overwrite_config=overwrite_config,
                pretrain_directory=args.pretrain_dir or DEFAULT_PATHS["PRETRAIN_DIR"],
                bb_pretrain_path=args.bb_ckpt_path,
                cellformer_pretrain_path=current_ckpt_path,
                path_dict=path_dict,
                use_pretrain=use_pretrained,
            )

            model_state_dict, optimizer_state_dict, scheduler_state_dict, last_train_loss, last_valid_loss = fit_pipeline_on_adata(
                pipeline=pipeline,
                adata=adata,
                current_ckpt_path=current_ckpt_path,
                epochs=args.per_dataset_epochs,
                lr=args.lr,
                wd=args.wd,
                scheduler_name=args.scheduler,
                patience=args.patience,
                workers=args.workers,
                device=device,
            )

            n_train_units = int(adata.obs[args.train_unit_key].nunique())
            save_training_checkpoint(
                save_path=latest_ckpt_path,
                model_state_dict=model_state_dict,
                optimizer_state_dict=optimizer_state_dict,
                scheduler_state_dict=scheduler_state_dict,
                global_epoch=global_epoch,
                dataset_index=dataset_index,
                dataset_path=str(h5ad_path),
                dataset_name=h5ad_path.stem,
                n_obs=int(adata.n_obs),
                n_vars=int(adata.n_vars),
                n_train_units=n_train_units,
                last_train_loss=last_train_loss,
                last_valid_loss=last_valid_loss,
            )
            epoch_checkpoint_meta = {
                "global_epoch": global_epoch,
                "dataset_index": dataset_index,
                "dataset_path": str(h5ad_path),
                "dataset_name": h5ad_path.stem,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "n_train_units": n_train_units,
                "selected_slices": (
                    sorted({item["source_slice_name"] for item in selected_slice_units})
                    if selected_slice_units is not None else None
                ),
                "selected_slice_units": (
                    [item["slice_unit_name"] for item in selected_slice_units]
                    if selected_slice_units is not None else None
                ),
                "last_train_loss": last_train_loss,
                "last_valid_loss": last_valid_loss,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "scheduler_state_dict": scheduler_state_dict,
            }

            history_record = {
                "global_epoch": global_epoch,
                "dataset_index": dataset_index,
                "dataset_path": str(h5ad_path),
                "latest_checkpoint_path": str(latest_ckpt_path),
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "n_train_units": n_train_units,
                "selected_slices": (
                    sorted({item["source_slice_name"] for item in selected_slice_units})
                    if selected_slice_units is not None else None
                ),
                "selected_slice_units": (
                    [item["slice_unit_name"] for item in selected_slice_units]
                    if selected_slice_units is not None else None
                ),
                "last_train_loss": last_train_loss,
                "last_valid_loss": last_valid_loss,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            with open(history_path, "a") as handle:
                handle.write(json.dumps(history_record) + "\n")

            current_ckpt_path = str(latest_ckpt_path)
            print(f"Updated latest checkpoint: {latest_ckpt_path}")

            del pipeline
            del adata
            torch.cuda.empty_cache()
            gc.collect()

        if epoch_checkpoint_meta is not None:
            epoch_ckpt_path = ckpt_dir / f"epoch_{global_epoch:03d}.pt"
            save_training_checkpoint(
                save_path=epoch_ckpt_path,
                model_state_dict=epoch_checkpoint_meta["model_state_dict"],
                optimizer_state_dict=epoch_checkpoint_meta["optimizer_state_dict"],
                scheduler_state_dict=epoch_checkpoint_meta["scheduler_state_dict"],
                global_epoch=global_epoch,
                dataset_index=epoch_checkpoint_meta["dataset_index"],
                dataset_path=epoch_checkpoint_meta["dataset_path"],
                dataset_name=epoch_checkpoint_meta["dataset_name"],
                n_obs=epoch_checkpoint_meta["n_obs"],
                n_vars=epoch_checkpoint_meta["n_vars"],
                n_train_units=epoch_checkpoint_meta["n_train_units"],
                last_train_loss=epoch_checkpoint_meta["last_train_loss"],
                last_valid_loss=epoch_checkpoint_meta["last_valid_loss"],
            )
            print(f"Saved epoch checkpoint: {epoch_ckpt_path}")

    print(f"\nTraining completed. Latest checkpoint: {latest_ckpt_path}")


if __name__ == "__main__":
    main()
