#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from brainbeacon.configs.config_train import config_train as default_config_train


BOOL_TRUE = {"1", "true", "t", "yes", "y", "on"}
BOOL_FALSE = {"0", "false", "f", "no", "n", "off"}


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if value is None:
        raise ValueError("Boolean value cannot be None.")

    text = str(value).strip().lower()
    if text in BOOL_TRUE:
        return True
    if text in BOOL_FALSE:
        return False
    raise ValueError(f"Cannot parse boolean value from: {value!r}")


def parse_scalar(value: str) -> Any:
    text = value.strip()
    lowered = text.lower()

    if lowered in BOOL_TRUE:
        return True
    if lowered in BOOL_FALSE:
        return False
    if lowered in {"none", "null"}:
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+|\d+[eE][+-]?\d+|\d+\.\d*[eE][+-]?\d+)", text):
        return float(text)
    return text


def parse_key_value_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise ValueError(f"Invalid override {text!r}. Expected KEY=VALUE.")
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"Invalid override {text!r}. Empty key is not allowed.")
    return key, parse_scalar(value)


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def resolve_single_value(cli_value: Any, config: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def load_config(config_path: str | None, overrides: list[str]) -> dict[str, Any]:
    config = copy.deepcopy(default_config_train)

    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        if not isinstance(yaml_config, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
        config.update(yaml_config)

    for override in overrides:
        key, value = parse_key_value_override(override)
        config[key] = value

    return config


def build_path_context(index: int, adata_path: str) -> dict[str, Any]:
    path = Path(adata_path)
    return {
        "index": index,
        "adata_path": str(path),
        "adata_name": path.name,
        "adata_stem": path.stem,
        "adata_suffix": path.suffix,
        "adata_parent": str(path.parent),
    }


def format_template_path(template: str, context: dict[str, Any]) -> str:
    return str(template).format(**context)


def default_output_h5ad_path(adata_path: str) -> str:
    path = Path(adata_path)
    return str(path.with_name(f"{path.stem}_with_bb_emb.h5ad"))


def default_npz_save_path(adata_path: str) -> str:
    path = Path(adata_path)
    return str(path.with_name(f"{path.stem}_bb_embeddings.npz"))


def default_token_dir_path(adata_path: str) -> str:
    path = Path(adata_path)
    return str(Path.cwd() / "bb_token_dirs" / f"{path.stem}_bb_token_dir")


def discover_adata_paths(cli_adata_path: str | None, config: dict[str, Any]) -> list[str]:
    paths: list[str] = []

    if cli_adata_path:
        paths.append(cli_adata_path)

    paths.extend(str(item) for item in ensure_list(config.get("adata_path")))
    paths.extend(str(item) for item in ensure_list(config.get("adata_paths")))

    parent_dirs = [str(item) for item in ensure_list(config.get("adata_parent_dirs"))]
    if parent_dirs:
        adata_glob = str(config.get("adata_glob", "*.h5ad"))
        recursive = parse_bool(config.get("adata_recursive", True))
        for parent_dir in parent_dirs:
            parent = Path(parent_dir)
            iterator = parent.rglob(adata_glob) if recursive else parent.glob(adata_glob)
            paths.extend(str(path) for path in sorted(iterator))

    paths = dedupe_preserve_order(paths)
    if not paths:
        raise ValueError("No h5ad inputs were provided. Set `adata_path` or `adata_paths` in config, or pass `--adata-path`.")
    return paths


def resolve_path_outputs(
    *,
    item_paths: list[str],
    explicit_single: str | None,
    explicit_list: list[str] | None,
    template: str | None,
    directory: str | None,
    default_builder,
) -> list[str | None]:
    count = len(item_paths)
    contexts = [build_path_context(index=i, adata_path=path) for i, path in enumerate(item_paths)]

    if explicit_list is not None:
        values = [str(item) if item is not None else None for item in explicit_list]
        if len(values) != count:
            raise ValueError(f"Expected {count} paths, but got {len(values)}.")
        return values

    if explicit_single is not None:
        if count != 1:
            raise ValueError("A single explicit path can only be used with one input h5ad.")
        return [str(explicit_single)]

    if template is not None:
        return [format_template_path(str(template), context) for context in contexts]

    if directory is not None:
        dir_path = Path(directory)
        return [str(dir_path / Path(default_builder(path)).name) for path in item_paths]

    return [default_builder(path) for path in item_paths]


def ensure_parent_dir(path: str | None) -> None:
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def ensure_unique_paths(paths: list[str | None], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for path in paths:
        if path is None:
            continue
        if path in seen:
            duplicates.append(path)
            continue
        seen.add(path)

    if duplicates:
        duplicate_preview = ", ".join(duplicates[:3])
        raise ValueError(
            f"Resolved duplicate {label} paths: {duplicate_preview}. "
            f"Use a template with `{{index}}`, for example `{label}_template: /path/to/{{index}}_{{adata_stem}}...`."
        )


def resolve_device(device_value: str | None) -> torch.device:
    if device_value:
        return torch.device(device_value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def canonicalize_specie(value: Any) -> str | None:
    if value is None:
        return None

    token = normalize_text(value)
    mapping = {
        "human": "human",
        "homosapiens": "human",
        "hsapiens": "human",
        "mouse": "mouse",
        "musmusculus": "mouse",
        "mmusculus": "mouse",
        "marmoset": "marmoset",
        "commonmarmoset": "marmoset",
        "macaque": "macaque",
        "macaqe": "macaque",
        "macque": "macaque",
        "macaca": "macaque",
    }
    return mapping.get(token, str(value).strip().lower() or None)


def canonicalize_assay(value: Any) -> str | None:
    if value is None:
        return None

    token = normalize_text(value)
    mapping = {
        "merfish": "merfish",
        "merscope": "merfish",
        "xenium": "xenium",
        "starmap": "starmap",
        "slideseq": "slideseqv2",
        "slideseqv2": "slideseqv2",
        "stereo": "stereo",
        "stereoseq": "stereo",
        "stereoseqv1": "stereo",
        "snrna": "snrna",
        "scrna": "snrna",
        "scrnaseq": "snrna",
        "snrnaseq": "snrna",
        "visium": "visium",
    }
    return mapping.get(token, str(value).strip().lower() or None)


def infer_specie_from_path(adata_path: str) -> str | None:
    text = str(adata_path).lower()
    if "marmoset" in text:
        return "marmoset"
    if any(token in text for token in ("macaque", "macaqe", "macque", "macaca")):
        return "macaque"
    if "human" in text:
        return "human"
    if "mouse" in text:
        return "mouse"
    return None


def infer_assay_from_path(adata_path: str) -> str | None:
    text = str(adata_path).lower()
    if "private" in Path(adata_path).parts:
        return "stereo"
    if "merfish" in text:
        return "merfish"
    if "xenium" in text:
        return "xenium"
    if "starmap" in text:
        return "starmap"
    if "slideseqv2" in text or "slideseq" in text or "slide-seq" in text:
        return "slideseqv2"
    if "stereoseq" in text or "stereo" in text:
        return "stereo"
    if "snrna" in text or "scrna" in text:
        return "snrna"
    if "visium" in text:
        return "visium"
    return None


def infer_unique_obs_value(adata, keys: list[str]) -> str | None:
    for key in keys:
        if key not in adata.obs.columns:
            continue
        values = [
            str(value).strip()
            for value in adata.obs[key].dropna().unique().tolist()
            if str(value).strip() and str(value).strip().lower() != "nan"
        ]
        values = dedupe_preserve_order(values)
        if len(values) == 1:
            return values[0]
    return None


def infer_uns_value(adata, keys: list[str]) -> str | None:
    for key in keys:
        value = adata.uns.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def infer_specie_from_adata(adata) -> str | None:
    value = infer_unique_obs_value(adata, ["specie", "species", "organism"])
    if value is None:
        value = infer_uns_value(adata, ["specie", "species", "organism"])
    return canonicalize_specie(value)


def infer_assay_from_adata(adata) -> str | None:
    value = infer_unique_obs_value(adata, ["platform", "assay", "technology", "tech", "modality"])
    if value is None:
        value = infer_uns_value(adata, ["platform", "assay", "technology", "tech", "modality"])
    return canonicalize_assay(value)


def resolve_mapping_value(mapping: dict[str, Any] | None, adata_path: str) -> Any:
    if not mapping:
        return None

    path = Path(adata_path)
    candidates = [
        str(path),
        path.as_posix(),
        path.name,
        path.stem,
    ]
    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def resolve_tokenization_metadata(
    *,
    adata_path: str,
    adata,
    config: dict[str, Any],
    cli_specie: str | None,
    cli_assay: str | None,
) -> tuple[str, str]:
    specie = canonicalize_specie(
        resolve_mapping_value(config.get("tokenize_specie_by_path"), adata_path)
        or cli_specie
        or config.get("tokenize_specie")
        or infer_specie_from_adata(adata)
        or infer_specie_from_path(adata_path)
    )
    assay = canonicalize_assay(
        resolve_mapping_value(config.get("tokenize_assay_by_path"), adata_path)
        or cli_assay
        or config.get("tokenize_assay")
        or infer_assay_from_adata(adata)
        or infer_assay_from_path(adata_path)
    )

    if not specie:
        raise ValueError(
            f"Cannot determine tokenization specie for {adata_path}. "
            "Set `tokenize_specie`, `tokenize_specie_by_path`, or pass `--tokenize-specie`."
        )
    if not assay:
        raise ValueError(
            f"Cannot determine tokenization assay for {adata_path}. "
            "Set `tokenize_assay`, `tokenize_assay_by_path`, or pass `--tokenize-assay`."
        )
    return specie, assay


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BrainBeacon tokenization + inference and save embeddings into h5ad.obsm."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--adata-path", type=str, default=None, help="Single input h5ad path.")
    parser.add_argument("--pretrain-ckpt", type=str, default=None, help="BrainBeacon checkpoint path.")
    parser.add_argument("--gene-dict-path", type=str, default=None, help="BrainBeacon gene dictionary path.")
    parser.add_argument("--token-data-path", type=str, default=None, help="Single token output directory for a single input h5ad.")
    parser.add_argument("--output-h5ad", type=str, default=None, help="Single output h5ad path for a single input h5ad.")
    parser.add_argument("--save-path", type=str, default=None, help="Optional single npz output path for a single input h5ad.")
    parser.add_argument("--obsm-key", type=str, default=None, help="obsm key used to save BrainBeacon embeddings.")
    parser.add_argument("--device", type=str, default=None, help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--tokenize-specie", type=str, default=None, help="Global specie override used for tokenization.")
    parser.add_argument("--tokenize-assay", type=str, default=None, help="Global assay override used for tokenization.")
    parser.add_argument("--use-hvg", type=str, default=None, help="Whether tokenization uses HVG selection.")
    parser.add_argument("--n-hvg", type=int, default=None, help="Number of HVGs for tokenization.")
    parser.add_argument("--min-genes", type=int, default=None, help="Minimum genes per cell during tokenization.")
    parser.add_argument("--min-cells", type=int, default=None, help="Minimum cells per gene during tokenization.")
    parser.add_argument("--force-tokenize", type=str, default=None, help="Whether to force regeneration of token files.")
    parser.add_argument("--use-dev-abs", type=str, default=None, help="Pass through to run_tokenization.")
    parser.add_argument("--token-batch-size", type=int, default=None, help="Effective token batch size written into token joblib bundles.")
    parser.add_argument("--dataloader-num-workers", type=int, default=None, help="Number of DataLoader workers for inference.")
    parser.add_argument("--joblib-cache-size", type=int, default=None, help="How many token joblib bundles to cache per worker.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor when num_workers > 0.")
    parser.add_argument("--pin-memory", type=str, default=None, help="Whether to enable DataLoader pin_memory.")
    parser.add_argument("--inference-amp", type=str, default=None, help="Whether to enable AMP during inference.")
    parser.add_argument("--amp-dtype", type=str, default=None, help="AMP dtype, for example float16 or bfloat16.")
    parser.add_argument("--neighbor-enhance", type=str, default=None, help="Override BrainBeacon neighbor_enhance.")
    parser.add_argument("--use-gene-id-emb", type=str, default=None, help="Override BrainBeacon use_gene_id_emb.")
    parser.add_argument("--use-homo-emb", type=str, default=None, help="Override BrainBeacon use_homo_emb.")
    parser.add_argument("--use-rna-type-emb", type=str, default=None, help="Override BrainBeacon use_rna_type_emb.")
    parser.add_argument("--use-pos-emb", type=str, default=None, help="Override BrainBeacon use_pos_emb.")
    parser.add_argument("--use-density-emb", type=str, default=None, help="Override BrainBeacon use_density_emb.")
    parser.add_argument("--density-token-idx", type=int, default=None, help="Override BrainBeacon density_token_idx.")
    parser.add_argument("--use-esm-embedding", type=str, default=None, help="Override BrainBeacon use_esm_embedding.")
    parser.add_argument("--use-esm-emb", type=str, default=None, help="Alias of --use-esm-embedding.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config before running.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config key. Can be used multiple times.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from anndata import read_h5ad

    import numpy as np

    from brainbeacon.pipeline.cell_embedding import (
        CellEmbeddingPipeline,
        normalize_brainbeacon_model_config,
        run_tokenization,
    )

    config = load_config(args.config, args.set)

    optional_overrides = {
        "pretrain_ckpt": args.pretrain_ckpt,
        "gene_dict_path": args.gene_dict_path,
        "obsm_key": args.obsm_key,
        "min_genes": args.min_genes,
        "min_cells": args.min_cells,
        "token_batch_size": args.token_batch_size,
        "dataloader_num_workers": args.dataloader_num_workers,
        "joblib_cache_size": args.joblib_cache_size,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": parse_bool(args.pin_memory) if args.pin_memory is not None else None,
        "inference_amp": parse_bool(args.inference_amp) if args.inference_amp is not None else None,
        "amp_dtype": args.amp_dtype,
        "neighbor_enhance": parse_bool(args.neighbor_enhance) if args.neighbor_enhance is not None else None,
        "use_gene_id_emb": parse_bool(args.use_gene_id_emb) if args.use_gene_id_emb is not None else None,
        "gene_id": parse_bool(args.use_gene_id_emb) if args.use_gene_id_emb is not None else None,
        "use_homo_emb": parse_bool(args.use_homo_emb) if args.use_homo_emb is not None else None,
        "use_rna_type_emb": parse_bool(args.use_rna_type_emb) if args.use_rna_type_emb is not None else None,
        "use_pos_emb": parse_bool(args.use_pos_emb) if args.use_pos_emb is not None else None,
        "use_density_emb": parse_bool(args.use_density_emb) if args.use_density_emb is not None else None,
        "density_token_idx": args.density_token_idx,
    }

    use_esm_override = args.use_esm_embedding if args.use_esm_embedding is not None else args.use_esm_emb
    if use_esm_override is not None:
        optional_overrides["use_esm_embedding"] = parse_bool(use_esm_override)
        optional_overrides["use_esm_emb"] = parse_bool(use_esm_override)

    for key, value in optional_overrides.items():
        if value is not None:
            config[key] = value

    config["masking_p"] = 0
    config = normalize_brainbeacon_model_config(config)

    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=True, allow_unicode=True))

    pretrain_ckpt = resolve_single_value(args.pretrain_ckpt, config, ["pretrain_ckpt"])
    gene_dict_path = resolve_single_value(args.gene_dict_path, config, ["gene_dict_path"])
    if not pretrain_ckpt:
        raise ValueError("Missing BrainBeacon checkpoint. Set `pretrain_ckpt` in config or pass `--pretrain-ckpt`.")
    if not gene_dict_path:
        raise ValueError("Missing gene dictionary path. Set `gene_dict_path` in config or pass `--gene-dict-path`.")

    obsm_key = resolve_single_value(args.obsm_key, config, ["obsm_key"], default="bb_emb")
    device = resolve_device(resolve_single_value(args.device, config, ["device"]))

    use_hvg = parse_bool(resolve_single_value(args.use_hvg, config, ["use_hvg"], default=True))
    n_hvg = int(resolve_single_value(args.n_hvg, config, ["n_hvg"], default=1000))
    min_genes = int(resolve_single_value(args.min_genes, config, ["min_genes"], default=0))
    min_cells = int(resolve_single_value(args.min_cells, config, ["min_cells"], default=3))
    force_tokenize = parse_bool(resolve_single_value(args.force_tokenize, config, ["force_tokenize"], default=True))
    use_dev_abs = parse_bool(resolve_single_value(args.use_dev_abs, config, ["use_dev_abs"], default=False))
    token_batch_size = int(resolve_single_value(args.token_batch_size, config, ["token_batch_size", "batch_size"], default=16))

    adata_paths = discover_adata_paths(args.adata_path, config)

    token_data_paths = resolve_path_outputs(
        item_paths=adata_paths,
        explicit_single=resolve_single_value(args.token_data_path, config, ["token_data_path"]),
        explicit_list=ensure_list(config.get("token_data_paths")) or None,
        template=config.get("token_data_path_template"),
        directory=config.get("token_data_dir"),
        default_builder=default_token_dir_path,
    )
    output_h5ad_paths = resolve_path_outputs(
        item_paths=adata_paths,
        explicit_single=resolve_single_value(args.output_h5ad, config, ["output_h5ad"]),
        explicit_list=ensure_list(config.get("output_h5ad_paths")) or None,
        template=config.get("output_h5ad_template"),
        directory=config.get("output_h5ad_dir"),
        default_builder=default_output_h5ad_path,
    )
    ensure_unique_paths(token_data_paths, "token_data_path")
    ensure_unique_paths(output_h5ad_paths, "output_h5ad")

    npz_enabled = any(
        value is not None
        for value in (
            args.save_path,
            config.get("npz_save_path"),
            config.get("save_path"),
            config.get("npz_save_paths"),
            config.get("save_paths"),
            config.get("npz_save_path_template"),
            config.get("save_path_template"),
            config.get("npz_save_dir"),
        )
    )
    if not npz_enabled:
        npz_save_paths = [None] * len(adata_paths)
    else:
        npz_save_paths = resolve_path_outputs(
            item_paths=adata_paths,
            explicit_single=resolve_single_value(args.save_path, config, ["npz_save_path", "save_path"]),
            explicit_list=ensure_list(resolve_single_value(None, config, ["npz_save_paths", "save_paths"])) or None,
            template=resolve_single_value(None, config, ["npz_save_path_template", "save_path_template"]),
            directory=config.get("npz_save_dir"),
            default_builder=default_npz_save_path,
        )
        ensure_unique_paths(npz_save_paths, "npz_save_path")

    print(f"Resolved {len(adata_paths)} h5ad files.")
    print(f"Using device: {device}")
    print("masking_p is fixed to 0 for inference.")
    print(f"Token batch size: {token_batch_size}")
    print(f"DataLoader workers: {int(config.get('dataloader_num_workers', 4))}")

    config["batch_size"] = 1
    pipeline = CellEmbeddingPipeline(pretrain_ckpt=pretrain_ckpt, model_config=config, device=device)

    try:
        for index, (adata_path, token_data_path, output_h5ad, npz_save_path) in enumerate(
            zip(adata_paths, token_data_paths, output_h5ad_paths, npz_save_paths),
            start=1,
        ):
            if os.path.exists(output_h5ad):
                print(f"[{index}/{len(adata_paths)}] Skipping existing output: {output_h5ad}")
                continue

            print(f"[{index}/{len(adata_paths)}] Reading {adata_path}")
            adata = read_h5ad(adata_path)

            tokenize_specie, tokenize_assay = resolve_tokenization_metadata(
                adata_path=adata_path,
                adata=adata,
                config=config,
                cli_specie=args.tokenize_specie,
                cli_assay=args.tokenize_assay,
            )
            print(
                f"[{index}/{len(adata_paths)}] Tokenizing with specie={tokenize_specie}, assay={tokenize_assay} -> {token_data_path}"
            )

            token_data_path = run_tokenization(
                adata_path=adata_path,
                bb_token_dir=token_data_path,
                gene_dict_path=gene_dict_path,
                specie=tokenize_specie,
                assay=tokenize_assay,
                use_hvg=use_hvg,
                n_hvg=n_hvg,
                force_tokenize=force_tokenize,
                use_dev_abs=use_dev_abs,
            )

            if npz_save_path is not None:
                ensure_parent_dir(npz_save_path)
            ensure_parent_dir(output_h5ad)

            print(f"[{index}/{len(adata_paths)}] Running BrainBeacon inference from {token_data_path}")
            try:
                pred = pipeline.run(data_paths=token_data_path, config_train=config)

                pred_indices, pred_embeddings = zip(*[(str(idx[0]), emb.numpy()) for idx, emb in pred])
                pred_indices = np.array(pred_indices)
                pred_embeddings = np.array(pred_embeddings)
                obs_names = np.array(adata.obs_names)
                dim = pred_embeddings.shape[1]

                if np.array_equal(pred_indices, obs_names):
                    embeddings = pred_embeddings
                else:
                    # Align by cell index: map predictions to adata obs order
                    pred_map = dict(zip(pred_indices, pred_embeddings))
                    matched = sum(1 for name in obs_names if name in pred_map)
                    missing = len(obs_names) - matched
                    print(
                        f"[{index}/{len(adata_paths)}] Index alignment: {matched}/{len(obs_names)} matched, "
                        f"{missing} missing, {len(pred_indices)} predicted"
                    )
                    embeddings = np.zeros((len(obs_names), dim), dtype=pred_embeddings.dtype)
                    for j, name in enumerate(obs_names):
                        if name in pred_map:
                            embeddings[j] = pred_map[name]

                if npz_save_path is not None:
                    np.savez_compressed(npz_save_path, embeddings=embeddings)
                    print(f"[{index}/{len(adata_paths)}] Embeddings saved to {npz_save_path}")

                adata.obsm[obsm_key] = embeddings
                adata.write_h5ad(output_h5ad)
                print(f"[{index}/{len(adata_paths)}] Saved h5ad to {output_h5ad}")
            except Exception as e:
                print(f"[{index}/{len(adata_paths)}] ERROR processing {adata_path}: {e}")
                continue

            del adata, pred, pred_indices, pred_embeddings, embeddings
    finally:
        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
