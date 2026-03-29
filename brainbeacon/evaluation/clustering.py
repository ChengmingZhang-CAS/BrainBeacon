from typing import Dict, Optional, Sequence, Tuple

import os
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.decomposition import PCA

from .metrics import (
    compute_clustering_metrics,
    compute_embedding_metrics,
    compute_spatial_metrics,
)
from .plotting import save_method_plots


# =============================================================================
# Basic helpers
# =============================================================================

def _to_numpy_1d(x, name: str) -> np.ndarray:
    """Convert input to a 1D NumPy array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, but got shape {arr.shape}.")
    return arr


def _to_numpy_2d(x, name: str) -> np.ndarray:
    """Convert input to a 2D NumPy array."""
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {arr.shape}.")
    return arr


def _normalize_log1p(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Apply library-size normalization and log1p transform."""
    X = np.asarray(X, dtype=np.float32)
    libsize = X.sum(axis=1, keepdims=True)
    X = X / (libsize + 1e-8) * target_sum
    X = np.log1p(X)
    return X


def reduce_dimensions(
    X: np.ndarray,
    n_components: Optional[int] = 50,
    random_state: int = 0,
) -> np.ndarray:
    """Run PCA if feature dimension is larger than n_components."""
    X = _to_numpy_2d(X, "X")
    if n_components is None or X.shape[1] <= n_components:
        return X.astype(np.float32, copy=False)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)
    return X_pca.astype(np.float32, copy=False)


def _ensure_dir(path: Optional[str]):
    """Create directory if needed."""
    if path:
        os.makedirs(path, exist_ok=True)


# =============================================================================
# Embedding preparation
# =============================================================================

def load_embedding_from_npz(npz_path, array_key=None):
    data = np.load(npz_path, allow_pickle=True)

    if array_key is None:
        keys = list(data.keys())
        if len(keys) == 0:
            raise ValueError(f"No arrays found in npz file: {npz_path}")
        array_key = keys[0]

    if array_key not in data:
        raise KeyError(
            f"{array_key} not found in {npz_path}. "
            f"Available keys: {list(data.keys())}"
        )

    X_emb = data[array_key]
    X_emb = X_emb.toarray() if hasattr(X_emb, "toarray") else np.asarray(X_emb)

    if X_emb.ndim != 2:
        raise ValueError(f"Embedding must be 2D, but got shape {X_emb.shape}")

    return X_emb.astype(np.float32, copy=False)


def attach_embeddings_to_adata(
    adata,
    method_npz_paths,
    method_embedding_keys=None,
    array_key=None,
):
    if method_embedding_keys is None:
        method_embedding_keys = {}

    final_keys = {}

    for method_name, npz_path in method_npz_paths.items():
        emb_key = method_embedding_keys.get(method_name, f"X_{method_name}")
        X_emb = load_embedding_from_npz(npz_path, array_key=array_key)

        if X_emb.shape[0] != adata.n_obs:
            raise ValueError(
                f"Cell number mismatch for {method_name}: "
                f"{X_emb.shape[0]} rows in embedding vs {adata.n_obs} cells in adata."
            )

        adata.obsm[emb_key] = X_emb
        final_keys[method_name] = emb_key

    return final_keys


def _get_pca_embedding(
    adata: AnnData,
    n_components: int = 50,
    normalize_before_pca: bool = True,
    target_sum: float = 1e4,
    random_state: int = 0,
) -> np.ndarray:
    """Compute PCA embedding from adata.X for one slice."""
    X = _to_numpy_2d(adata.X, "adata.X")
    if normalize_before_pca:
        X = _normalize_log1p(X, target_sum=target_sum)
    X_pca = reduce_dimensions(X, n_components=n_components, random_state=random_state)
    return X_pca


def get_embedding(
    adata: AnnData,
    method_name: str,
    embedding_key: Optional[str] = None,
    pca_dim: int = 50,
    normalize_before_pca: bool = True,
    target_sum: float = 1e4,
    random_state: int = 0,
) -> Tuple[np.ndarray, str]:
    """
    Get embedding for one slice.

    Logic
    -----
    - If method_name == 'pca', compute PCA from adata.X.
    - Otherwise, read embedding from adata.obsm[embedding_key].
    """
    method_name = str(method_name)
    method_name_lower = method_name.lower()

    if method_name_lower == "pca":
        X_emb = _get_pca_embedding(
            adata=adata,
            n_components=pca_dim,
            normalize_before_pca=normalize_before_pca,
            target_sum=target_sum,
            random_state=random_state,
        )
        return X_emb, "X_pca"

    if embedding_key is None:
        raise ValueError("embedding_key must be provided for non-PCA methods.")
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm.")

    X_emb = _to_numpy_2d(adata.obsm[embedding_key], embedding_key)
    return X_emb.astype(np.float32, copy=False), embedding_key


# =============================================================================
# Leiden clustering
# =============================================================================

def _run_leiden_once(
    X: np.ndarray,
    resolution: float,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 0,
    pca_dim: Optional[int] = None,
) -> np.ndarray:
    """Run Leiden once on one embedding matrix."""
    X = _to_numpy_2d(X, "X")
    X_use = reduce_dimensions(X, n_components=pca_dim, random_state=random_state)

    adata_tmp = AnnData(X=np.zeros((X_use.shape[0], 1), dtype=np.float32))
    adata_tmp.obsm["X_emb"] = X_use

    sc.pp.neighbors(
        adata_tmp,
        use_rep="X_emb",
        n_neighbors=n_neighbors,
        metric=metric,
    )
    sc.tl.leiden(
        adata_tmp,
        resolution=resolution,
        random_state=random_state,
    )
    return adata_tmp.obs["leiden"].to_numpy().astype(str)


def _search_leiden_resolution(
    X: np.ndarray,
    target_n_clusters: int,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 0,
    pca_dim: Optional[int] = None,
    max_iterations: int = 10,
    tolerance_ratio: float = 0.1,
):
    """Search Leiden resolution to match target cluster number."""
    if target_n_clusters < 1:
        raise ValueError("target_n_clusters must be >= 1.")
    if tolerance_ratio < 0:
        raise ValueError("tolerance_ratio must be >= 0.")

    X = _to_numpy_2d(X, "X")
    X_use = reduce_dimensions(X, n_components=pca_dim, random_state=random_state)

    adata_tmp = AnnData(X=np.zeros((X_use.shape[0], 1), dtype=np.float32))
    adata_tmp.obsm["X_emb"] = X_use
    sc.pp.neighbors(
        adata_tmp,
        use_rep="X_emb",
        n_neighbors=n_neighbors,
        metric=metric,
    )

    tolerance = max(1, int(np.ceil(target_n_clusters * tolerance_ratio)))
    init_resolution = max(0.1, target_n_clusters / 10.0)

    sc.tl.leiden(
        adata_tmp,
        resolution=init_resolution,
        random_state=random_state,
        key_added="leiden",
    )
    y_pred = adata_tmp.obs["leiden"].to_numpy().astype(str)

    pred_n_clusters = len(np.unique(y_pred))
    best_resolution = init_resolution
    best_pred = y_pred
    best_diff = abs(pred_n_clusters - target_n_clusters)

    history = [{
        "resolution": float(init_resolution),
        "n_clusters": int(pred_n_clusters),
        "diff": int(best_diff),
    }]

    if best_diff <= tolerance:
        return best_resolution, best_pred, history

    if pred_n_clusters < target_n_clusters:
        low, high = init_resolution, min(10.0, init_resolution * 5.0)
    else:
        low, high = max(0.01, init_resolution / 5.0), init_resolution

    for _ in range(max_iterations):
        mid = (low + high) / 2.0

        sc.tl.leiden(
            adata_tmp,
            resolution=mid,
            random_state=random_state,
            key_added="leiden",
        )
        y_mid = adata_tmp.obs["leiden"].to_numpy().astype(str)
        n_mid = len(np.unique(y_mid))
        diff_mid = abs(n_mid - target_n_clusters)

        history.append({
            "resolution": float(mid),
            "n_clusters": int(n_mid),
            "diff": int(diff_mid),
        })

        if diff_mid < best_diff:
            best_diff = diff_mid
            best_resolution = mid
            best_pred = y_mid

        if diff_mid <= tolerance:
            break

        if n_mid < target_n_clusters:
            low = mid
        else:
            high = mid

    return best_resolution, best_pred, history


def cluster_leiden(
    X: np.ndarray,
    target_n_clusters: Optional[int] = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 0,
    pca_dim: Optional[int] = None,
    max_iterations: int = 10,
    return_info: bool = False,
):
    """Run Leiden clustering on one embedding matrix."""
    X = _to_numpy_2d(X, "X")

    if target_n_clusters is not None:
        final_resolution, y_pred, history = _search_leiden_resolution(
            X=X,
            target_n_clusters=target_n_clusters,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            pca_dim=pca_dim,
            max_iterations=max_iterations,
        )
    else:
        y_pred = _run_leiden_once(
            X=X,
            resolution=resolution,
            n_neighbors=n_neighbors,
            metric=metric,
            random_state=random_state,
            pca_dim=pca_dim,
        )
        final_resolution = resolution
        history = None

    if not return_info:
        return y_pred

    unique_labels, counts = np.unique(y_pred, return_counts=True)
    info = {
        "resolution": float(final_resolution),
        "n_clusters": int(len(unique_labels)),
        "cluster_sizes": {str(label): int(count) for label, count in zip(unique_labels, counts)},
        "history": history,
    }
    return y_pred, info


# =============================================================================
# UMAP
# =============================================================================

def _prepare_umap(
    adata: AnnData,
    embedding_key: str,
    umap_key: str = "X_umap",
    n_neighbors: int = 15,
    metric: str = "euclidean",
    random_state: int = 0,
):
    """Compute UMAP from an existing embedding and store it in adata.obsm[umap_key]."""
    if embedding_key not in adata.obsm:
        raise KeyError(f"{embedding_key} not found in adata.obsm.")

    X_emb = _to_numpy_2d(adata.obsm[embedding_key], embedding_key)

    adata_tmp = AnnData(X=np.zeros((X_emb.shape[0], 1), dtype=np.float32))
    adata_tmp.obsm["X_emb"] = X_emb

    sc.pp.neighbors(
        adata_tmp,
        use_rep="X_emb",
        n_neighbors=n_neighbors,
        metric=metric,
    )
    sc.tl.umap(adata_tmp, random_state=random_state)

    adata.obsm[umap_key] = _to_numpy_2d(adata_tmp.obsm["X_umap"], "X_umap")
    return adata


# =============================================================================
# Single evaluation
# =============================================================================

def evaluate_clustering(
    adata: AnnData,
    method_name: str,
    label_key: str,
    embedding_key: Optional[str] = None,
    spatial_key: Optional[str] = "spatial",
    cluster_key: Optional[str] = None,
    target_n_clusters: Optional[int] = None,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    graph_metric: str = "euclidean",
    random_state: int = 0,
    embedding_pca_dim: int = 50,
    leiden_pca_dim: Optional[int] = None,
    normalize_before_pca: bool = True,
    target_sum: float = 1e4,
    max_iterations: int = 10,
    return_details: bool = False,
    copy: bool = False,
    compute_umap: bool = False,
    umap_key: Optional[str] = None,
    save_plots: bool = False,
    plot_dir: Optional[str] = None,
    plot_prefix: Optional[str] = None,
    plot_ground_truth: bool = False,
    spatial_point_size: float = 8,
    umap_point_size: float = 8,
):
    """Evaluate clustering on one slice AnnData."""
    method_name = str(method_name)
    label_key = str(label_key)

    if label_key not in adata.obs:
        raise KeyError(f"{label_key} not found in adata.obs.")

    adata_out = adata.copy() if copy else adata

    X_emb, final_embedding_key = get_embedding(
        adata=adata_out,
        method_name=method_name,
        embedding_key=embedding_key,
        pca_dim=embedding_pca_dim,
        normalize_before_pca=normalize_before_pca,
        target_sum=target_sum,
        random_state=random_state,
    )

    if final_embedding_key not in adata_out.obsm:
        adata_out.obsm[final_embedding_key] = X_emb

    y_true = adata_out.obs[label_key].to_numpy()

    if target_n_clusters is None:
        target_n_clusters = int(pd.Series(y_true).nunique())

    if cluster_key is None:
        cluster_key = f"{method_name}_{label_key}_cluster"

    y_pred, cluster_info = cluster_leiden(
        X=X_emb,
        target_n_clusters=target_n_clusters,
        resolution=resolution,
        n_neighbors=n_neighbors,
        metric=graph_metric,
        random_state=random_state,
        pca_dim=leiden_pca_dim,
        max_iterations=max_iterations,
        return_info=True,
    )
    print(
        f"[Clustering] label={label_key} | "
        f"n_labels_true={int(pd.Series(y_true).nunique())} | "
        f"target_n_clusters={int(target_n_clusters)} | "
        f"n_clusters_pred={int(cluster_info['n_clusters'])} | "
        f"resolution={float(cluster_info['resolution']):.2f} | "
        f"n_search_steps={len(cluster_info['history'])}"
    )

    adata_out.obs[cluster_key] = pd.Categorical(y_pred.astype(str))

    if compute_umap:
        if umap_key is None:
            umap_key = f"X_umap_{method_name}_{label_key}"
        _prepare_umap(
            adata=adata_out,
            embedding_key=final_embedding_key,
            umap_key=umap_key,
            n_neighbors=n_neighbors,
            metric=graph_metric,
            random_state=random_state,
        )

    results = {
        "method": method_name,
        "embedding_key": final_embedding_key,
        "label_key": label_key,
        "cluster_key": cluster_key,
        "n_cells": int(adata_out.n_obs),
        "n_clusters_true": int(pd.Series(y_true).nunique()),
        "n_clusters_pred": int(cluster_info["n_clusters"]),
        "resolution": float(cluster_info["resolution"]),
        "n_neighbors": int(n_neighbors),
        "graph_metric": graph_metric,
        "random_state": int(random_state),
    }

    results.update(compute_clustering_metrics(y_true=y_true, y_pred=y_pred))
    results.update(compute_embedding_metrics(X=X_emb, labels=y_pred))

    if spatial_key is not None and spatial_key in adata_out.obsm:
        spatial = _to_numpy_2d(adata_out.obsm[spatial_key], spatial_key)
        results.update(compute_spatial_metrics(y_pred=y_pred, spatial=spatial))
    else:
        results["neighbor_agreement"] = np.nan
        results["label_entropy"] = np.nan

    if return_details:
        results["cluster_sizes"] = cluster_info["cluster_sizes"]
        results["search_history"] = cluster_info["search_history"]

    if save_plots:
        if plot_dir is None:
            raise ValueError("plot_dir must be provided when save_plots=True.")

        save_method_plots(
            adata=adata_out,
            label_key=label_key,
            cluster_key=cluster_key,
            plot_dir=plot_dir,
            plot_prefix=plot_prefix if plot_prefix is not None else f"{method_name}_{label_key}",
            spatial_key=spatial_key,
            umap_key=umap_key if umap_key is not None else f"X_umap_{method_name}_{label_key}",
            method_name=method_name,
            ari=results.get("ari", np.nan),
            plot_ground_truth=plot_ground_truth,
            spatial_point_size=spatial_point_size,
            umap_point_size=umap_point_size,
        )

    return adata_out, results


# =============================================================================
# Benchmark loop
# =============================================================================

def _infer_slice_key(
    adata: AnnData,
    slice_key: Optional[str] = None,
    candidate_slice_keys: Optional[Sequence[str]] = None,
) -> str:
    """Infer slice key from adata.obs."""
    if slice_key is not None:
        if slice_key not in adata.obs.columns:
            raise KeyError(f"{slice_key} not found in adata.obs.")
        return slice_key

    if candidate_slice_keys is None:
        candidate_slice_keys = [
            "slice_id", "slice", "section_id", "sample", "sample_id", "batch", "batch_id"
        ]

    for key in candidate_slice_keys:
        if key in adata.obs.columns:
            return key

    raise ValueError("Failed to infer slice_key. Please provide slice_key explicitly.")


def _normalize_selection(values: Optional[Sequence[str]]) -> Optional[set]:
    """Normalize optional selection list to a string set."""
    if values is None:
        return None
    return {str(v) for v in values}


def run_clustering_benchmark(
    adata: AnnData,
    label_keys: Sequence[str],
    methods: Sequence[str],
    method_embedding_keys: Optional[Dict[str, str]] = None,
    spatial_key: str = "spatial",
    slice_key: Optional[str] = None,
    candidate_slice_keys: Optional[Sequence[str]] = None,
    n_neighbors: int = 15,
    graph_metric: str = "euclidean",
    random_state: int = 0,
    embedding_pca_dim: int = 50,
    leiden_pca_dim: Optional[int] = None,
    normalize_before_pca: bool = True,
    target_sum: float = 1e4,
    max_iterations: int = 10,
    return_details: bool = False,
    copy_slice: bool = True,
    compute_umap: bool = False,
    save_plots: bool = False,
    output_dir: Optional[str] = None,
    save_method_results: bool = False,
    method_results_dir: Optional[str] = None,
    plot_ground_truth: bool = False,
    spatial_point_size: float = 8,
    umap_point_size: float = 8,
    selected_slices: Optional[Sequence[str]] = None,
    selected_methods: Optional[Sequence[str]] = None,
    selected_labels: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run clustering benchmark on full adata by looping over slice × method × label."""
    if method_embedding_keys is None:
        method_embedding_keys = {}

    label_keys = [str(k) for k in label_keys]
    methods = [str(m) for m in methods]

    selected_slices = _normalize_selection(selected_slices)
    selected_methods = _normalize_selection(selected_methods)
    selected_labels = _normalize_selection(selected_labels)

    if selected_methods is not None:
        methods = [m for m in methods if m in selected_methods]
    if selected_labels is not None:
        label_keys = [k for k in label_keys if k in selected_labels]

    if len(methods) == 0:
        raise ValueError("No methods left after filtering.")
    if len(label_keys) == 0:
        raise ValueError("No label_keys left after filtering.")

    for label_key_i in label_keys:
        if label_key_i not in adata.obs.columns:
            raise KeyError(f"{label_key_i} not found in adata.obs.")

    if spatial_key is not None and spatial_key not in adata.obsm:
        raise KeyError(
            f"{spatial_key} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}"
        )

    inferred_slice_key = _infer_slice_key(
        adata=adata,
        slice_key=slice_key,
        candidate_slice_keys=candidate_slice_keys,
    )

    slice_ids = sorted(adata.obs[inferred_slice_key].astype(str).unique().tolist())
    if selected_slices is not None:
        slice_ids = [sid for sid in slice_ids if sid in selected_slices]

    if len(slice_ids) == 0:
        raise ValueError("No slice_ids left after filtering.")

    all_results = []

    if verbose:
        print(f"Using slice key: {inferred_slice_key}")
        print(f"Number of slices: {len(slice_ids)}")
        print(f"Methods: {methods}")
        print(f"Labels: {label_keys}")

    if save_plots and output_dir is None:
        raise ValueError("output_dir must be provided when save_plots=True.")

    if save_method_results and method_results_dir is None:
        raise ValueError("method_results_dir must be provided when save_method_results=True.")

    if save_method_results:
        os.makedirs(method_results_dir, exist_ok=True)

    for slice_id in slice_ids:
        mask = adata.obs[inferred_slice_key].astype(str) == slice_id
        adata_slice = adata[mask].copy() if copy_slice else adata[mask]

        if verbose:
            print(f"\n===== Slice: {slice_id} | cells: {adata_slice.n_obs} =====")

        slice_plot_dir = None
        if save_plots:
            slice_plot_dir = os.path.join(output_dir, str(slice_id))
            os.makedirs(slice_plot_dir, exist_ok=True)

        gt_written = set()

        for method_name in methods:
            method_name_lower = method_name.lower()

            if verbose:
                print(f"  --- Method: {method_name} ---")

            if method_name_lower == "pca":
                embedding_key = None
            else:
                if method_name not in method_embedding_keys:
                    raise KeyError(
                        f"embedding key for method '{method_name}' not provided in method_embedding_keys."
                    )
                embedding_key = method_embedding_keys[method_name]
                if embedding_key not in adata_slice.obsm:
                    raise KeyError(
                        f"{embedding_key} not found in adata_slice.obsm for method '{method_name}'."
                    )

            for label_key_i in label_keys:
                plot_prefix = f"{method_name}_{label_key_i}"
                umap_key = f"X_umap_{method_name}_{label_key_i}" if compute_umap else None

                write_gt = False
                if plot_ground_truth:
                    gt_tag = (slice_id, label_key_i)
                    if gt_tag not in gt_written:
                        write_gt = True
                        gt_written.add(gt_tag)

                _, results = evaluate_clustering(
                    adata=adata_slice,
                    method_name=method_name,
                    label_key=label_key_i,
                    embedding_key=embedding_key,
                    spatial_key=spatial_key,
                    cluster_key=None,
                    target_n_clusters=None,
                    resolution=1.0,
                    n_neighbors=n_neighbors,
                    graph_metric=graph_metric,
                    random_state=random_state,
                    embedding_pca_dim=embedding_pca_dim,
                    leiden_pca_dim=leiden_pca_dim,
                    normalize_before_pca=normalize_before_pca,
                    target_sum=target_sum,
                    max_iterations=max_iterations,
                    return_details=return_details,
                    copy=False,
                    compute_umap=compute_umap,
                    umap_key=umap_key,
                    save_plots=save_plots,
                    plot_dir=slice_plot_dir,
                    plot_prefix=plot_prefix,
                    plot_ground_truth=write_gt,
                    spatial_point_size=spatial_point_size,
                    umap_point_size=umap_point_size,
                )

                results["slice_key"] = inferred_slice_key
                results["slice_id"] = slice_id
                all_results.append(results)

                if verbose:
                    ari_val = results.get("ari", np.nan)
                    neighbor_val = results.get("neighbor_agreement", np.nan)
                    print(
                        f"    label={label_key_i} | "
                        f"method={method_name} | "
                        f"ARI={ari_val:.4f} | "
                        f"neighbor_agreement={neighbor_val:.4f}"
                    )

    results_df = pd.DataFrame(all_results)

    preferred_cols = [
        "slice_key", "slice_id", "method", "embedding_key", "label_key", "cluster_key",
        "n_cells", "n_clusters_true", "n_clusters_pred", "resolution",
        "n_neighbors", "graph_metric", "random_state",
    ]
    metric_cols = [
        "ari", "nmi", "ami", "homogeneity", "completeness",
        "v_measure", "purity", "silhouette", "neighbor_agreement", "label_entropy",
    ]

    ordered_cols = [c for c in preferred_cols + metric_cols if c in results_df.columns]
    remaining_cols = [c for c in results_df.columns if c not in ordered_cols]
    results_df = results_df[ordered_cols + remaining_cols]

    if save_method_results:
        for method_name, subdf in results_df.groupby("method", sort=False):
            save_path = os.path.join(method_results_dir, f"{method_name}.csv")
            subdf.to_csv(save_path, index=False)
            if verbose:
                print(f"[Saved] {method_name}: {save_path}")

    return results_df


def collect_cached_method_results(
    methods,
    results_dir: str,
    force_rerun_methods=None,
    verbose: bool = True,
):
    """
    Collect cached per-method result CSV files and determine which methods still need to run.
    """
    os.makedirs(results_dir, exist_ok=True)

    if force_rerun_methods is None:
        force_rerun_methods = []
    force_rerun_methods = set(force_rerun_methods)

    cached_dfs = []
    methods_to_run = []

    for method in methods:
        result_csv = os.path.join(results_dir, f"{method}.csv")

        if os.path.exists(result_csv) and method not in force_rerun_methods:
            df = pd.read_csv(result_csv)
            cached_dfs.append(df)
            if verbose:
                print(f"[Cache] Use existing result for method: {method}")
        else:
            methods_to_run.append(method)
            if verbose:
                reason = "force rerun" if method in force_rerun_methods else "not found"
                print(f"[Run] Method: {method} ({reason})")

    return cached_dfs, methods_to_run


def merge_and_sort_method_results(
    cached_dfs=None,
    new_dfs=None,
    method_order=None,
    sort_keys=None,
):
    """
    Merge cached and new result DataFrames, then sort with optional keys.

    Parameters
    ----------
    cached_dfs : list[pd.DataFrame] or None
        Cached result DataFrames.
    new_dfs : list[pd.DataFrame] or None
        Newly generated result DataFrames.
    method_order : list[str] or None
        Desired method order. If provided, 'method' will follow this order.
    sort_keys : list[str] or None
        Columns used for sorting. Default is ['slice_id', 'method', 'label_key'].

    Returns
    -------
    pd.DataFrame
        Merged and sorted DataFrame.
    """
    cached_dfs = cached_dfs or []
    new_dfs = new_dfs or []

    all_dfs = [df for df in (cached_dfs + new_dfs) if df is not None and len(df) > 0]
    if len(all_dfs) == 0:
        return pd.DataFrame()

    df = pd.concat(all_dfs, axis=0, ignore_index=True)

    if method_order is not None and "method" in df.columns:
        df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    if sort_keys is None:
        sort_keys = ["slice_id", "method", "label_key"]

    valid_sort_keys = [k for k in sort_keys if k in df.columns]
    if len(valid_sort_keys) > 0:
        df = df.sort_values(valid_sort_keys).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    return df

def summarize_benchmark_results(
    df: pd.DataFrame,
    groupby_keys=None,
    metric_cols=None,
) -> pd.DataFrame:
    """
    Summarize benchmark results by mean/std across slices.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if groupby_keys is None:
        groupby_keys = ["method", "label_key"]

    if metric_cols is None:
        metric_cols = [
            "ari", "nmi", "ami", "homogeneity", "completeness",
            "v_measure", "purity", "silhouette",
            "neighbor_agreement", "label_entropy",
        ]

    metric_cols = [c for c in metric_cols if c in df.columns]
    if len(metric_cols) == 0:
        return pd.DataFrame()

    summary_df = (
        df.groupby(groupby_keys)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary_df.columns
    ]

    return summary_df