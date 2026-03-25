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


def _compute_purity(y_true, y_pred):
    table = pd.crosstab(
        pd.Series(y_true, name="true"),
        pd.Series(y_pred, name="pred"),
    )
    return table.max(axis=0).sum() / table.to_numpy().sum()


def _compute_spatial_knn_indices(spatial, n_neighbors=6):
    n_samples = spatial.shape[0]
    if n_samples < 2:
        return None

    k = min(n_neighbors + 1, n_samples)
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(spatial)
    indices = nbrs.kneighbors(return_distance=False)

    knn_indices = indices[:, 1:]
    if knn_indices.shape[1] == 0:
        return None

    return knn_indices


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


def compute_clustering_metrics(y_true, y_pred, metrics=None):
    y_true = _to_1d_array(y_true, "y_true")
    y_pred = _to_1d_array(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, "y_true", "y_pred")

    if metrics is None:
        metrics = [
            "ari",
            "nmi",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
            "purity",
        ]

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


def compute_embedding_metrics(X, labels, metrics=None):
    X = _to_2d_array(X, "X")
    labels = _to_1d_array(labels, "labels")
    _check_same_length(X, labels, "X", "labels")

    if metrics is None:
        metrics = ["silhouette"]

    results = {}
    n_unique = len(np.unique(labels))

    if "silhouette" in metrics:
        if n_unique < 2 or n_unique >= len(labels):
            results["silhouette"] = np.nan
        else:
            results["silhouette"] = float(silhouette_score(X, labels))

    return results


def compute_spatial_metrics(y_pred, spatial, metrics=None, n_neighbors=6):
    y_pred = _to_1d_array(y_pred, "y_pred")
    spatial = _to_2d_array(spatial, "spatial")
    _check_same_length(y_pred, spatial, "y_pred", "spatial")

    if metrics is None:
        metrics = ["neighbor_agreement", "label_entropy"]

    knn_indices = _compute_spatial_knn_indices(
        spatial=spatial,
        n_neighbors=n_neighbors,
    )

    results = {}

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