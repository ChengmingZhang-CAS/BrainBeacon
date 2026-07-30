import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def label_sort_key(label):
    """Natural-ish label ordering for common cortical subclass names."""
    text = str(label)
    import re

    match = re.match(r"^L(\d+(?:/\d+)*)(.*)$", text)
    if match:
        nums = [int(x) for x in match.group(1).split("/")]
        return (1, nums[0], nums, match.group(2), text)
    priority = {
        "ASC": 0,
        "Ast": 0,
        "MG": 1,
        "OPC": 2,
        "OLG": 3,
        "VLMC": 4,
        "Vascular": 4,
        "EC": 4,
        "LAMP5": 50,
        "RELN": 51,
        "VIP": 52,
        "SST": 53,
        "PVALB": 54,
        "PV_CHC": 55,
        "PVALB_CHC": 55,
    }
    return (priority.get(text, 100), text)


def _stable_sort_component(value):
    if isinstance(value, (list, tuple)):
        return tuple(_stable_sort_component(x) for x in value)
    if isinstance(value, (int, np.integer)):
        return (0, int(value))
    if isinstance(value, (float, np.floating)):
        return (1, float(value))
    return (2, str(value))


def label_order(labels):
    labels = [str(x) for x in pd.unique(list(labels))]
    return sorted(labels, key=lambda label: _stable_sort_component(label_sort_key(label)))


def centroid_matrix(x, labels, order=None, l2_normalize=True):
    labels = np.asarray([str(x).strip() for x in labels])
    if order is None:
        order = label_order(labels)
    rows = []
    kept = []
    for label in order:
        idx = labels == label
        if idx.sum() == 0:
            continue
        if sparse.issparse(x):
            rows.append(np.asarray(x[idx].mean(axis=0)).ravel())
        else:
            rows.append(np.asarray(x[idx].mean(axis=0)).ravel())
        kept.append(label)
    out = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, x.shape[1]), dtype=np.float32)
    if l2_normalize and out.size:
        out = normalize(out, norm="l2")
    return out, kept


def l2_normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    out = np.zeros_like(x, dtype=np.float32)
    np.divide(x, norms, out=out, where=norms > 0)
    return out


def centered_l2_normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.nanmean(x, axis=1, keepdims=True)
    x[~np.isfinite(x)] = 0.0
    return l2_normalize_rows(x)


def rank_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return np.vstack([pd.Series(row).rank(method="average").to_numpy(dtype=np.float32) for row in x])


def profile_similarity(query_profiles, ref_profiles, metric="cosine"):
    """Compute label-profile similarity by cosine, Pearson correlation, or Spearman correlation."""
    metric = str(metric).lower()
    if metric == "cosine":
        return l2_normalize_rows(query_profiles) @ l2_normalize_rows(ref_profiles).T
    if metric == "pcc":
        return centered_l2_normalize_rows(query_profiles) @ centered_l2_normalize_rows(ref_profiles).T
    if metric == "spearman":
        return centered_l2_normalize_rows(rank_rows(query_profiles)) @ centered_l2_normalize_rows(rank_rows(ref_profiles)).T
    raise ValueError(f"Unsupported profile similarity metric: {metric}")


def l2_normalize_dense(x):
    return normalize(np.asarray(x, dtype=np.float32), norm="l2")


def top_pc_from_centered(x):
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] < 2:
        return None
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return vt[0].astype(np.float32)


def remove_pc(x, pc):
    if pc is None:
        return np.asarray(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    pc = np.asarray(pc, dtype=np.float32)
    pc = pc / (np.linalg.norm(pc) + 1e-12)
    return x - (x @ pc[:, None]) * pc[None, :]


def pca_transform_pair(x_ref, x_query, n_components=50, remove_first_pc=False, random_state=0):
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)
    n_components = int(min(n_components, x_ref.shape[1], max(1, x_ref.shape[0] - 1)))
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(x_ref)
    z_ref = pca.transform(x_ref).astype(np.float32)
    z_query = pca.transform(x_query).astype(np.float32)
    if remove_first_pc and z_ref.shape[1] > 1:
        z_ref = z_ref[:, 1:]
        z_query = z_query[:, 1:]
    return l2_normalize_dense(z_ref), l2_normalize_dense(z_query)


def posthoc_transform_pair(x_ref, x_query, mode="raw", pca_n_components=50, seed=0):
    """Apply pairwise posthoc transforms for label transfer readouts."""
    mode = str(mode)
    x_ref = np.asarray(x_ref, dtype=np.float32)
    x_query = np.asarray(x_query, dtype=np.float32)
    if mode in {"raw", "none"}:
        return x_ref, x_query
    if mode == "l2":
        return l2_normalize_dense(x_ref), l2_normalize_dense(x_query)
    if mode in {"global_center", "center"}:
        mu = np.vstack([x_ref, x_query]).mean(axis=0, keepdims=True)
        return x_ref - mu, x_query - mu
    if mode == "within_dataset_center":
        return x_ref - x_ref.mean(axis=0, keepdims=True), x_query - x_query.mean(axis=0, keepdims=True)
    if mode == "ref_center":
        mu = x_ref.mean(axis=0, keepdims=True)
        return x_ref - mu, x_query - mu
    if mode == "pca":
        return pca_transform_pair(x_ref, x_query, n_components=pca_n_components, remove_first_pc=False, random_state=seed)
    if mode == "pca_rmPC1":
        return pca_transform_pair(x_ref, x_query, n_components=pca_n_components, remove_first_pc=True, random_state=seed)
    if mode == "global_center_l2":
        a, b = posthoc_transform_pair(x_ref, x_query, "global_center")
        return l2_normalize_dense(a), l2_normalize_dense(b)
    if mode == "within_dataset_center_l2":
        a, b = posthoc_transform_pair(x_ref, x_query, "within_dataset_center")
        return l2_normalize_dense(a), l2_normalize_dense(b)
    if mode == "ref_center_l2":
        a, b = posthoc_transform_pair(x_ref, x_query, "ref_center")
        return l2_normalize_dense(a), l2_normalize_dense(b)
    raise ValueError(f"Unsupported posthoc mode: {mode}")


def alignment_metrics(sim, rows, cols, tag=None):
    """Rank-based alignment summary for a similarity matrix."""
    rows = [str(x) for x in rows]
    cols = [str(x) for x in cols]
    exact = []
    ranks = []
    margins = []
    diag = []
    offdiag_max = []
    for i, row in enumerate(rows):
        if row not in cols:
            continue
        j = cols.index(row)
        order = np.argsort(-sim[i])
        rank = int(np.where(order == j)[0][0]) + 1
        ranks.append(rank)
        exact.append(rank == 1)
        diag.append(float(sim[i, j]))
        others = np.delete(sim[i], j)
        off = float(np.max(others)) if others.size else np.nan
        offdiag_max.append(off)
        margins.append(float(sim[i, j] - off) if np.isfinite(off) else np.nan)
    return {
        "tag": tag,
        "n_rows": int(len(rows)),
        "n_cols": int(len(cols)),
        "exact_top1_frac": float(np.mean(exact)) if exact else np.nan,
        "mean_exact_rank": float(np.mean(ranks)) if ranks else np.nan,
        "mean_exact_margin": float(np.nanmean(margins)) if margins else np.nan,
        "mean_diagonal_similarity": float(np.nanmean(diag)) if diag else np.nan,
        "mean_offdiag_max": float(np.nanmean(offdiag_max)) if offdiag_max else np.nan,
    }


def expected_label_profile_metrics(sim, rows, cols, expected_labels):
    """Score whether expected same-name labels rank well in query-vs-reference profiles."""
    rows = [str(x) for x in rows]
    cols = [str(x) for x in cols]
    expected = [str(x) for x in expected_labels]
    row_to_i = {label: i for i, label in enumerate(rows)}
    col_to_j = {label: j for j, label in enumerate(cols)}
    ranks = []
    hits = []
    rank_scores = []
    for label in expected:
        if label not in row_to_i or label not in col_to_j:
            continue
        i = row_to_i[label]
        j = col_to_j[label]
        order = np.argsort(-sim[i])
        rank = int(np.where(order == j)[0][0]) + 1
        ranks.append(rank)
        hits.append(rank == 1)
        denom = max(1, len(cols) - 1)
        rank_scores.append(1.0 - (rank - 1) / denom)
    return {
        "expected_n_labels": int(len(expected)),
        "expected_present_n": int(len(ranks)),
        "SNhit_expected": float(np.mean(hits)) if hits else np.nan,
        "SNmean_rank_expected": float(np.mean(ranks)) if ranks else np.nan,
        "SNrank_expected": float(np.mean(rank_scores)) if rank_scores else np.nan,
    }


def prediction_distribution_metrics(query_labels, expected_labels=None):
    labels = pd.Series(np.asarray(query_labels, dtype=object).astype(str))
    if labels.empty:
        return {"max_pred_frac": np.nan, "effective_n_pred_labels": np.nan, "top_pred_label": ""}
    counts = labels.value_counts()
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    eff_n = float(np.exp(entropy))
    out = {
        "max_pred_frac": float(probs.iloc[0]),
        "effective_n_pred_labels": eff_n,
        "top_pred_label": str(counts.index[0]),
        "n_pred_labels": int(counts.shape[0]),
    }
    if expected_labels is not None:
        expected = set(map(str, expected_labels))
        out["pred_expected_frac"] = float(labels.isin(expected).mean())
    return out


def row_normalize_counts(mat):
    mat = np.asarray(mat, dtype=np.float64)
    denom = mat.sum(axis=1, keepdims=True)
    out = np.zeros_like(mat, dtype=np.float64)
    np.divide(mat, denom, out=out, where=denom > 0)
    return out


def row_normalize_shifted_similarity(sim, eps=1e-6):
    sim = np.asarray(sim, dtype=np.float64)
    if sim.size == 0:
        return sim
    row_min = np.nanmin(sim, axis=1, keepdims=True)
    shifted = sim - row_min
    shifted[~np.isfinite(shifted)] = 0.0
    shifted = shifted + eps
    denom = shifted.sum(axis=1, keepdims=True)
    out = np.zeros_like(shifted, dtype=np.float64)
    np.divide(shifted, denom, out=out, where=denom > 0)
    return out


def row_certainty(row_prob):
    row_prob = np.asarray(row_prob, dtype=np.float64)
    if row_prob.ndim != 2 or row_prob.shape[1] <= 1:
        return np.nan
    valid = row_prob.sum(axis=1) > 0
    if not np.any(valid):
        return np.nan
    p = row_prob[valid]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    certainty = 1.0 - entropy / np.log(p.shape[1])
    return float(np.mean(certainty))


def format_metric_value(value):
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def plot_row_normalized_heatmap(
    mat,
    rows,
    cols,
    title,
    out_path,
    xlabel="reference label",
    ylabel="query/predicted label",
    cbar_label="row-normalized value",
    cmap="viridis",
):
    _ensure_dir(os.path.dirname(out_path))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        mat,
        ax=ax,
        xticklabels=cols,
        yticklabels=rows,
        cmap=cmap,
        vmin=0.0,
        vmax=float(np.nanmax(mat)) if np.size(mat) else 1.0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_label_profile_metric_panels(panels, rows, cols, title, out_path, cmap="viridis"):
    _ensure_dir(os.path.dirname(out_path))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 7.2), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        mat = panel["matrix"]
        sns.heatmap(
            mat,
            ax=ax,
            xticklabels=cols,
            yticklabels=rows,
            cmap=cmap,
            vmin=0.0,
            vmax=float(np.nanmax(mat)) if np.size(mat) else 1.0,
            linewidths=0.2,
            linecolor="white",
            cbar_kws={"label": "row-normalized value"},
        )
        if np.size(mat):
            top_cols = np.nanargmax(mat, axis=1)
            col_to_j = {str(col): j for j, col in enumerate(cols)}
            for i, j in enumerate(top_cols):
                value = mat[i, j]
                if np.isfinite(value):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        format_metric_value(value),
                        ha="center",
                        va="center",
                        color="white" if value >= 0.55 else "black",
                        fontsize=7,
                        fontweight="bold",
                    )
            for i, row in enumerate(rows):
                j = col_to_j.get(str(row))
                if j is None:
                    continue
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", lw=1.3))
        ax.set_title(
            f"{panel['metric']}\n"
            f"SNrank={format_metric_value(panel.get('rank'))}; "
            f"SNhit={format_metric_value(panel.get('hit'))}; "
            f"Cert={format_metric_value(panel.get('certainty'))}"
        )
        ax.set_xlabel("reference label")
        ax.set_ylabel("query/predicted label")
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pred_reference_confusion(pred_labels, reference_labels, out_path, title=None, row_order=None, col_order=None):
    """Plot row-normalized predicted-label by reference-label counts."""
    pred_labels = pd.Series(np.asarray(pred_labels, dtype=object).astype(str), name="pred")
    reference_labels = pd.Series(np.asarray(reference_labels, dtype=object).astype(str), name="reference")
    if row_order is None:
        row_order = label_order(pred_labels)
    if col_order is None:
        col_order = label_order(reference_labels)
    counts = pd.crosstab(pred_labels, reference_labels).reindex(index=row_order, columns=col_order, fill_value=0)
    row_prob = row_normalize_counts(counts.to_numpy())
    cert = row_certainty(row_prob)
    plot_row_normalized_heatmap(
        row_prob,
        list(counts.index),
        list(counts.columns),
        title or f"predicted label vs reference label; cert={format_metric_value(cert)}",
        out_path,
        xlabel="reference label",
        ylabel="predicted label",
    )
    return {"conf_certainty": cert, "conf_n_rows": int(counts.shape[0]), "conf_n_cols": int(counts.shape[1])}


def evaluate_label_profile_similarity(
    query_profiles,
    query_labels,
    ref_profiles,
    ref_labels,
    metrics=("cosine", "pcc", "spearman"),
    expected_labels=None,
    tag=None,
    out_path=None,
    title=None,
):
    """Evaluate predicted/query label profiles against reference label profiles."""
    query_labels = np.asarray([str(x).strip() for x in query_labels])
    ref_labels = np.asarray([str(x).strip() for x in ref_labels])
    if expected_labels is None:
        expected_labels = label_order(set(query_labels) & set(ref_labels))
    eval_order = label_order(expected_labels)
    c_query, rows = centroid_matrix(query_profiles, query_labels, eval_order, l2_normalize=False)
    c_ref, cols = centroid_matrix(ref_profiles, ref_labels, eval_order, l2_normalize=False)

    metrics = [str(x).strip().lower() for x in metrics if str(x).strip()]
    if not metrics:
        raise ValueError("metrics cannot be empty.")
    bad = sorted(set(metrics) - {"cosine", "pcc", "spearman"})
    if bad:
        raise ValueError(f"Unsupported profile metrics: {bad}")

    result = {}
    panels = []
    for idx, metric in enumerate(metrics):
        sim = profile_similarity(c_query, c_ref, metric)
        align = alignment_metrics(sim, rows, cols, tag=tag)
        expected = expected_label_profile_metrics(sim, rows, cols, expected_labels)
        cert = row_certainty(row_normalize_shifted_similarity(sim))
        suffix = "" if idx == 0 else f"_{metric}"
        for key, value in {**align, **expected}.items():
            if key == "tag":
                continue
            result[f"{key}{suffix}"] = value
        result[f"profile_certainty{suffix}"] = cert
        panels.append(
            {
                "metric": metric,
                "matrix": row_normalize_shifted_similarity(sim),
                "rank": expected.get("SNrank_expected"),
                "hit": expected.get("SNhit_expected"),
                "certainty": cert,
            }
        )

    result.update(prediction_distribution_metrics(query_labels, expected_labels))
    result["profile_metrics"] = ";".join(metrics)
    if out_path is not None:
        plot_label_profile_metric_panels(
            panels,
            rows,
            cols,
            title or f"{tag or 'query'} label profiles vs reference",
            out_path,
        )
    return result
