import numpy as np
import pandas as pd

from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.neighbors import NearestNeighbors


# =========================
# Basic utilities
# =========================

DEFAULT_CLUSTERING_METRICS = ["ari", "nmi"]
DEFAULT_EMBEDDING_METRICS = ["asw"]
DEFAULT_SPATIAL_METRICS = ["neighbor_agreement", "label_entropy"]
DEFAULT_SPATIAL_AUTOCORR_METRICS = ["moran_i", "local_moran_i", "geary_c"]


def _to_1d_array(x, name):
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, but got shape {arr.shape}.")
    return arr


def _to_2d_array(x, name):
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {arr.shape}.")
    return arr


def _check_same_length(a, b, a_name, b_name):
    if len(a) != len(b):
        raise ValueError(
            f"{a_name} and {b_name} must have the same length, "
            f"but got {len(a)} and {len(b)}."
        )


def _safe_float(x):
    if x is None:
        return np.nan
    return float(x)


def _compute_purity(y_true, y_pred):
    table = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
    )
    denom = table.to_numpy().sum()
    if denom == 0:
        return np.nan
    return table.max(axis=0).sum() / denom


# =========================
# Spatial graph utilities
# =========================

def _compute_spatial_knn_indices(spatial, n_neighbors=6, algorithm="auto"):
    """
    Build KNN indices from spatial coordinates.
    Returns shape (n_samples, k) excluding self-neighbor.
    """
    spatial = _to_2d_array(spatial, "spatial")
    n_samples = spatial.shape[0]
    if n_samples < 2:
        return None

    k = min(int(n_neighbors) + 1, n_samples)
    if k <= 1:
        return None

    nbrs = NearestNeighbors(
        n_neighbors=k,
        algorithm=algorithm,
    )
    nbrs.fit(spatial)
    indices = nbrs.kneighbors(return_distance=False)

    knn_indices = indices[:, 1:]
    if knn_indices.shape[1] == 0:
        return None

    return knn_indices


def precompute_spatial_graph(spatial, n_neighbors=6, algorithm="auto"):
    """
    Precompute reusable spatial graph cache.

    Returns
    -------
    cache : dict
        {
            "knn_indices": ndarray or None,
            "row_idx": ndarray or None,
            "col_idx": ndarray or None,
            "n_samples": int,
            "n_edges_directed": int,
            "n_neighbors": int,
        }
    """
    spatial = _to_2d_array(spatial, "spatial")
    knn_indices = _compute_spatial_knn_indices(
        spatial=spatial,
        n_neighbors=n_neighbors,
        algorithm=algorithm,
    )

    if knn_indices is None:
        return {
            "knn_indices": None,
            "row_idx": None,
            "col_idx": None,
            "n_samples": spatial.shape[0],
            "n_edges_directed": 0,
            "n_neighbors": int(n_neighbors),
        }

    n, k = knn_indices.shape
    row_idx = np.repeat(np.arange(n), k)
    col_idx = knn_indices.reshape(-1)

    return {
        "knn_indices": knn_indices,
        "row_idx": row_idx,
        "col_idx": col_idx,
        "n_samples": n,
        "n_edges_directed": int(len(row_idx)),
        "n_neighbors": int(n_neighbors),
    }


def _resolve_spatial_graph_cache(
    spatial=None,
    n_neighbors=6,
    knn_indices=None,
    spatial_graph_cache=None,
):
    """
    Resolve spatial graph inputs to a unified cache dict.
    """
    if spatial_graph_cache is not None:
        return spatial_graph_cache

    if knn_indices is not None:
        knn_indices = np.asarray(knn_indices)
        if knn_indices.ndim != 2:
            raise ValueError(
                f"knn_indices must be 2D, but got shape {knn_indices.shape}."
            )
        n, k = knn_indices.shape
        row_idx = np.repeat(np.arange(n), k)
        col_idx = knn_indices.reshape(-1)
        return {
            "knn_indices": knn_indices,
            "row_idx": row_idx,
            "col_idx": col_idx,
            "n_samples": n,
            "n_edges_directed": int(len(row_idx)),
            "n_neighbors": int(k),
        }

    if spatial is None:
        raise ValueError(
            "At least one of spatial, knn_indices, or spatial_graph_cache must be provided."
        )

    return precompute_spatial_graph(
        spatial=spatial,
        n_neighbors=n_neighbors,
        algorithm="auto",
    )


def _knn_indices_to_edge_array(knn_indices):
    """
    Convert KNN index matrix to undirected unique edges.
    Output shape: (n_edges, 2)
    """
    if knn_indices is None:
        return None

    rows = np.repeat(np.arange(knn_indices.shape[0]), knn_indices.shape[1])
    cols = knn_indices.reshape(-1)

    edges = np.stack([rows, cols], axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]

    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    if edges.shape[0] == 0:
        return None

    return edges


# =========================
# Spatial label metrics
# =========================

def _spatial_neighbor_agreement(y_pred, knn_indices):
    if knn_indices is None:
        return np.nan
    return float((y_pred[knn_indices] == y_pred[:, None]).mean())


def _spatial_label_entropy(y_pred, knn_indices):
    if knn_indices is None:
        return np.nan

    if len(np.unique(y_pred)) == 1:
        return 0.0

    entropies = []
    for idx in knn_indices:
        labels = y_pred[idx]
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropies.append(-(probs * np.log(probs + 1e-12)).sum())

    return float(np.mean(entropies))


def _spatial_fide(y_pred, knn_indices):
    """
    Edge-based spatial continuity score.

    Notes
    -----
    This implementation returns the fraction of spatial graph edges whose
    endpoints share the same predicted label:
        same-label edges / total edges
    """
    if knn_indices is None:
        return np.nan

    edges = _knn_indices_to_edge_array(knn_indices)
    if edges is None:
        return np.nan

    same = (y_pred[edges[:, 0]] == y_pred[edges[:, 1]]).astype(float)
    return float(same.mean())


# =========================
# Spatial autocorrelation metrics
# =========================

def _validate_spatial_autocorr_values(values, n_samples):
    values = _to_1d_array(values, "values")
    if len(values) != n_samples:
        raise ValueError(
            f"values and spatial must have the same length, "
            f"but got {len(values)} and {n_samples}."
        )
    return values.astype(float)


def _global_moran_i_from_cache(values, spatial_graph_cache):
    """
    Global Moran's I computed from directed KNN edges in O(nk).
    """
    if spatial_graph_cache is None:
        return np.nan

    row_idx = spatial_graph_cache["row_idx"]
    col_idx = spatial_graph_cache["col_idx"]
    n = spatial_graph_cache["n_samples"]
    s0 = spatial_graph_cache["n_edges_directed"]

    if row_idx is None or col_idx is None or s0 <= 0:
        return np.nan

    x = np.asarray(values, dtype=float)
    if len(x) != n:
        raise ValueError(f"values length {len(x)} != n_samples {n}.")

    if n < 2:
        return np.nan

    z = x - x.mean()
    denom = np.sum(z ** 2)
    if denom <= 1e-12:
        return 0.0

    num = np.sum(z[row_idx] * z[col_idx])
    return float((n / s0) * (num / denom))


def _local_moran_i_from_cache(values, spatial_graph_cache):
    """
    Local Moran's I for each sample, computed from KNN graph in O(nk).
    Returns an array of shape (n_samples,).
    """
    if spatial_graph_cache is None:
        return None

    row_idx = spatial_graph_cache["row_idx"]
    col_idx = spatial_graph_cache["col_idx"]
    n = spatial_graph_cache["n_samples"]

    if row_idx is None or col_idx is None:
        return None

    x = np.asarray(values, dtype=float)
    if len(x) != n:
        raise ValueError(f"values length {len(x)} != n_samples {n}.")

    if n < 2:
        return None

    z = x - x.mean()
    m2 = np.sum(z ** 2) / n
    if m2 <= 1e-12:
        return np.zeros(n, dtype=float)

    neighbor_sum = np.zeros(n, dtype=float)
    np.add.at(neighbor_sum, row_idx, z[col_idx])

    local_i = (z / m2) * neighbor_sum
    return local_i


def _global_geary_c_from_cache(values, spatial_graph_cache):
    """
    Global Geary's C computed from directed KNN edges in O(nk).
    Lower means stronger positive spatial autocorrelation.
    """
    if spatial_graph_cache is None:
        return np.nan

    row_idx = spatial_graph_cache["row_idx"]
    col_idx = spatial_graph_cache["col_idx"]
    n = spatial_graph_cache["n_samples"]
    s0 = spatial_graph_cache["n_edges_directed"]

    if row_idx is None or col_idx is None or s0 <= 0:
        return np.nan

    x = np.asarray(values, dtype=float)
    if len(x) != n:
        raise ValueError(f"values length {len(x)} != n_samples {n}.")

    if n < 2:
        return np.nan

    z = x - x.mean()
    denom = np.sum(z ** 2)
    if denom <= 1e-12:
        return 0.0

    diff_sq = (x[row_idx] - x[col_idx]) ** 2
    num = np.sum(diff_sq)

    return float(((n - 1) / (2 * s0)) * (num / denom))


def _labels_to_one_vs_rest_matrix(labels):
    """
    Convert discrete labels to one-vs-rest indicator matrix.
    Output shape: (n_samples, n_classes)
    """
    labels = _to_1d_array(labels, "labels")
    classes = np.unique(labels)
    mat = np.stack([(labels == c).astype(float) for c in classes], axis=1)
    return mat, classes


def _aggregate_autocorr_over_label_indicators(labels, spatial_graph_cache, metric="moran_i"):
    """
    Compute autocorrelation metrics on one-vs-rest label indicators
    and average across classes.
    """
    indicator_mat, classes = _labels_to_one_vs_rest_matrix(labels)

    scores = []
    for j in range(indicator_mat.shape[1]):
        values = indicator_mat[:, j]

        if np.all(values == values[0]):
            continue

        if metric == "moran_i":
            score = _global_moran_i_from_cache(values, spatial_graph_cache)
        elif metric == "local_moran_i":
            local_vals = _local_moran_i_from_cache(values, spatial_graph_cache)
            score = np.nan if local_vals is None else np.mean(local_vals)
        elif metric == "geary_c":
            score = _global_geary_c_from_cache(values, spatial_graph_cache)
        else:
            raise ValueError(f"Unsupported autocorrelation metric: {metric}")

        if not np.isnan(score):
            scores.append(score)

    if len(scores) == 0:
        return np.nan

    return float(np.mean(scores))


# =========================
# Clustering metrics
# =========================

def compute_clustering_metrics(y_true, y_pred, metrics=None):
    y_true = _to_1d_array(y_true, "y_true")
    y_pred = _to_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, "y_true", "y_pred")

    if metrics is None:
        metrics = DEFAULT_CLUSTERING_METRICS

    results = {}

    if "ari" in metrics:
        results["ari"] = float(adjusted_rand_score(y_true, y_pred))
    if "nmi" in metrics:
        results["nmi"] = float(normalized_mutual_info_score(y_true, y_pred))
    if "ami" in metrics:
        results["ami"] = float(adjusted_mutual_info_score(y_true, y_pred))
    if "homogeneity" in metrics:
        results["homogeneity"] = float(homogeneity_score(y_true, y_pred))
    if "completeness" in metrics:
        results["completeness"] = float(completeness_score(y_true, y_pred))
    if "v_measure" in metrics:
        results["v_measure"] = float(v_measure_score(y_true, y_pred))
    if "purity" in metrics:
        results["purity"] = float(_compute_purity(y_true, y_pred))

    return results


# =========================
# Embedding metrics
# =========================

def compute_embedding_metrics(X, labels, metrics=None):
    X = _to_2d_array(X, "X")
    labels = _to_1d_array(labels, "labels")
    _check_same_length(X, labels, "X", "labels")

    if metrics is None:
        metrics = DEFAULT_EMBEDDING_METRICS

    metrics = ["asw" if m == "silhouette" else m for m in metrics]

    results = {}
    n_unique = len(np.unique(labels))

    if "asw" in metrics:
        if n_unique < 2 or n_unique >= len(labels):
            results["asw"] = np.nan
        else:
            results["asw"] = float(silhouette_score(X, labels))

    return results


# =========================
# Spatial label metrics
# =========================

def compute_spatial_metrics(
    y_pred,
    spatial=None,
    metrics=None,
    n_neighbors=6,
    knn_indices=None,
    spatial_graph_cache=None,
):
    y_pred = _to_1d_array(y_pred, "y_pred")

    cache = _resolve_spatial_graph_cache(
        spatial=spatial,
        n_neighbors=n_neighbors,
        knn_indices=knn_indices,
        spatial_graph_cache=spatial_graph_cache,
    )
    knn_indices = cache["knn_indices"]

    if len(y_pred) != cache["n_samples"]:
        raise ValueError(
            f"y_pred and spatial graph must have the same length, "
            f"but got {len(y_pred)} and {cache['n_samples']}."
        )

    if metrics is None:
        metrics = DEFAULT_SPATIAL_METRICS

    results = {}

    if "fide" in metrics:
        results["fide"] = _spatial_fide(
            y_pred=y_pred,
            knn_indices=knn_indices,
        )

    if "neighbor_agreement" in metrics:
        results["neighbor_agreement"] = _spatial_neighbor_agreement(
            y_pred=y_pred,
            knn_indices=knn_indices,
        )

    if "label_entropy" in metrics:
        results["label_entropy"] = _spatial_label_entropy(
            y_pred=y_pred,
            knn_indices=knn_indices,
        )

    return results


# =========================
# Spatial autocorrelation metrics
# =========================

def compute_spatial_autocorr_metrics(
    spatial=None,
    metrics=None,
    values=None,
    labels=None,
    n_neighbors=6,
    knn_indices=None,
    spatial_graph_cache=None,
):
    """
    Compute spatial autocorrelation metrics.

    Two supported input modes:
    1. values + spatial
       Use continuous values directly.
    2. labels + spatial
       Convert discrete labels to one-vs-rest indicators, then average
       the metric across classes.
    """
    if metrics is None:
        metrics = DEFAULT_SPATIAL_AUTOCORR_METRICS

    if values is None and labels is None:
        raise ValueError("At least one of values or labels must be provided.")

    if values is not None and labels is not None:
        raise ValueError("Please provide only one of values or labels, not both.")

    cache = _resolve_spatial_graph_cache(
        spatial=spatial,
        n_neighbors=n_neighbors,
        knn_indices=knn_indices,
        spatial_graph_cache=spatial_graph_cache,
    )

    results = {}

    if values is not None:
        values = _validate_spatial_autocorr_values(values, cache["n_samples"])

        if "moran_i" in metrics:
            results["moran_i"] = _global_moran_i_from_cache(values, cache)

        if "local_moran_i" in metrics:
            local_vals = _local_moran_i_from_cache(values, cache)
            results["local_moran_i"] = (
                np.nan if local_vals is None else float(np.mean(local_vals))
            )

        if "geary_c" in metrics:
            results["geary_c"] = _global_geary_c_from_cache(values, cache)

    else:
        labels = _to_1d_array(labels, "labels")
        if len(labels) != cache["n_samples"]:
            raise ValueError(
                f"labels and spatial graph must have the same length, "
                f"but got {len(labels)} and {cache['n_samples']}."
            )

        if "moran_i" in metrics:
            results["moran_i"] = _aggregate_autocorr_over_label_indicators(
                labels=labels,
                spatial_graph_cache=cache,
                metric="moran_i",
            )

        if "local_moran_i" in metrics:
            results["local_moran_i"] = _aggregate_autocorr_over_label_indicators(
                labels=labels,
                spatial_graph_cache=cache,
                metric="local_moran_i",
            )

        if "geary_c" in metrics:
            results["geary_c"] = _aggregate_autocorr_over_label_indicators(
                labels=labels,
                spatial_graph_cache=cache,
                metric="geary_c",
            )

    return results


# =========================
# Unified metric selector
# =========================

def compute_selected_metrics(
    metrics,
    y_true=None,
    y_pred=None,
    X=None,
    spatial=None,
    n_neighbors=6,
    autocorr_values=None,
    autocorr_labels=None,
    knn_indices=None,
    spatial_graph_cache=None,
):
    """
    Unified entry point for selected metrics.
    """
    if metrics is None:
        raise ValueError("metrics must be provided.")

    metrics = list(metrics)
    results = {}

    clustering_metric_set = {
        "ari", "nmi", "ami", "homogeneity", "completeness", "v_measure", "purity"
    }
    embedding_metric_set = {"asw", "silhouette"}
    spatial_metric_set = {"fide", "neighbor_agreement", "label_entropy"}
    autocorr_metric_set = {"moran_i", "local_moran_i", "geary_c"}

    clustering_metrics = [m for m in metrics if m in clustering_metric_set]
    embedding_metrics = [m for m in metrics if m in embedding_metric_set]
    spatial_metrics = [m for m in metrics if m in spatial_metric_set]
    autocorr_metrics = [m for m in metrics if m in autocorr_metric_set]

    if len(clustering_metrics) > 0:
        if y_true is None or y_pred is None:
            raise ValueError(
                f"Metrics {clustering_metrics} require both y_true and y_pred."
            )
        results.update(
            compute_clustering_metrics(
                y_true=y_true,
                y_pred=y_pred,
                metrics=clustering_metrics,
            )
        )

    if len(embedding_metrics) > 0:
        if X is None or y_pred is None:
            raise ValueError(
                f"Metrics {embedding_metrics} require both X and labels "
                f"(here labels are usually y_pred)."
            )
        results.update(
            compute_embedding_metrics(
                X=X,
                labels=y_pred,
                metrics=embedding_metrics,
            )
        )

    if len(spatial_metrics) > 0:
        if y_pred is None:
            raise ValueError(
                f"Metrics {spatial_metrics} require y_pred."
            )
        results.update(
            compute_spatial_metrics(
                y_pred=y_pred,
                spatial=spatial,
                metrics=spatial_metrics,
                n_neighbors=n_neighbors,
                knn_indices=knn_indices,
                spatial_graph_cache=spatial_graph_cache,
            )
        )

    if len(autocorr_metrics) > 0:
        if autocorr_values is not None:
            results.update(
                compute_spatial_autocorr_metrics(
                    spatial=spatial,
                    metrics=autocorr_metrics,
                    values=autocorr_values,
                    labels=None,
                    n_neighbors=n_neighbors,
                    knn_indices=knn_indices,
                    spatial_graph_cache=spatial_graph_cache,
                )
            )
        else:
            labels_for_autocorr = autocorr_labels if autocorr_labels is not None else y_pred
            if labels_for_autocorr is None:
                raise ValueError(
                    f"Metrics {autocorr_metrics} require either autocorr_values "
                    f"or labels (autocorr_labels / y_pred)."
                )
            results.update(
                compute_spatial_autocorr_metrics(
                    spatial=spatial,
                    metrics=autocorr_metrics,
                    values=None,
                    labels=labels_for_autocorr,
                    n_neighbors=n_neighbors,
                    knn_indices=knn_indices,
                    spatial_graph_cache=spatial_graph_cache,
                )
            )

    return results


# =========================
# Summary utilities
# =========================

def summarize_metrics(results_df, groupby_cols=None):
    if not isinstance(results_df, pd.DataFrame):
        raise TypeError("results_df must be a pandas DataFrame.")

    if results_df.empty:
        return results_df.copy()

    if groupby_cols is None:
        groupby_cols = ["method"]

    missing_cols = [col for col in groupby_cols if col not in results_df.columns]
    if missing_cols:
        raise KeyError(f"Missing groupby columns in results_df: {missing_cols}")

    numeric_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
    metric_cols = [col for col in numeric_cols if col not in groupby_cols]

    if not metric_cols:
        return results_df[groupby_cols].drop_duplicates().reset_index(drop=True)

    summary = (
        results_df.groupby(groupby_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    return summary