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


def run_knn_voting(
    X_ref,
    y_ref,
    X_query,
    method="native",
    K=30,
    metric="euclidean",
    weight_mode="distance",
    unassigned_threshold=None,
    device="cpu",
):
    """Unified KNN voting interface for label transfer evaluation."""
    if method == "native":
        from sklearn.neighbors import KNeighborsClassifier

        knn = KNeighborsClassifier(n_neighbors=K, weights=weight_mode, metric=metric, n_jobs=-1)
        knn.fit(X_ref, y_ref)
        preds = knn.predict(X_query)

        if unassigned_threshold is not None:
            distances, _ = knn.kneighbors(X_query, n_neighbors=K)
            top_t = max(1, int(np.sqrt(K)))
            distances_t = distances[:, :top_t]
            if metric == "cosine":
                sims_t = 1 - distances_t
            else:
                d_min = distances.min()
                d_max = distances.max()
                sims_t = 1 - (distances_t - d_min) / (d_max - d_min + 1e-8)
            weights_t = 1.0 / (distances_t + 1e-8)
            weighted_mean_sims = (sims_t * weights_t).sum(axis=1) / weights_t.sum(axis=1)
            preds = np.where(weighted_mean_sims < unassigned_threshold, "unassigned", preds)
        return preds

    if method == "hnsw":
        import hnswlib

        n_cells, dim = X_ref.shape
        space_map = {"euclidean": "l2", "cosine": "cosine", "ip": "ip"}
        index = hnswlib.Index(space=space_map[metric], dim=dim)
        index.init_index(max_elements=n_cells, ef_construction=200, M=16, random_seed=42)
        index.add_items(X_ref, np.arange(n_cells))
        index.set_ef(max(50, K + 10))
        nn_index, _ = index.knn_query(X_query, k=K)
        return pd.DataFrame(np.asarray(y_ref)[nn_index]).apply(lambda row: row.value_counts().idxmax(), axis=1).tolist()

    if method == "faiss":
        import faiss

        dim = X_ref.shape[1]
        if metric == "euclidean":
            index = faiss.IndexFlatL2(dim)
        elif metric == "cosine":
            faiss.normalize_L2(X_ref)
            faiss.normalize_L2(X_query)
            index = faiss.IndexFlatIP(dim)
        else:
            raise ValueError(f"Unsupported metric {metric} for FAISS")
        if device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(X_ref.astype(np.float32))
        _, nn_index = index.search(X_query.astype(np.float32), K)
        return pd.DataFrame(np.asarray(y_ref)[nn_index]).apply(lambda row: row.value_counts().idxmax(), axis=1).tolist()

    raise ValueError(f"Unsupported KNN method: {method}")


def run_prototype_classifier(X_ref, y_ref, X_query, metric="euclidean"):
    """Prototype classifier for label transfer evaluation."""
    prototypes = {label: X_ref[y_ref == label].mean(axis=0) for label in np.unique(y_ref)}
    labels = list(prototypes.keys())
    centroids = np.vstack(list(prototypes.values()))
    if metric == "cosine":
        centroids = normalize(centroids, norm="l2")
        X_query = normalize(X_query, norm="l2")
        pred_idx = np.dot(X_query, centroids.T).argmax(axis=1)
    else:
        dists = np.linalg.norm(X_query[:, None, :] - centroids[None, :, :], axis=2)
        pred_idx = dists.argmin(axis=1)
    return [labels[i] for i in pred_idx]


def run_logreg_classifier(X_ref, y_ref, X_query, max_iter=200, C=1.0):
    """Logistic regression classifier for label transfer evaluation."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1, multi_class="auto")
    clf.fit(X_ref, y_ref)
    return clf.predict(X_query)


def run_label_transfer(X_ref, y_ref, X_query, method, **kwargs):
    """Dispatcher for common label transfer readouts."""
    if method in {"native", "hnsw", "faiss"}:
        return run_knn_voting(X_ref, y_ref, X_query, method=method, **kwargs)
    if method == "prototype":
        return run_prototype_classifier(X_ref, y_ref, X_query, metric=kwargs.get("metric", "euclidean"))
    if method == "logreg":
        return run_logreg_classifier(X_ref, y_ref, X_query, max_iter=kwargs.get("max_iter", 200), C=kwargs.get("C", 1.0))
    raise ValueError(f"Unsupported method: {method}")


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


def profile_eval_transform_label(transform):
    labels = {
        "lib_hmc": "lib_hmc",
        "lib_log1p": "lib_log1p",
        "lib": "lib",
    }
    return labels.get(str(transform), str(transform))


def profile_eval_gene_mode_label(mode):
    labels = {
        "direct_gene": "gene-matched",
        "homo_mean": "homo-mean",
    }
    return labels.get(str(mode), str(mode))


def _sf_normalize_for_profile_eval(x, target_sum=1e4):
    if sparse.issparse(x):
        x = x.astype(np.float32).tocsr(copy=True)
        totals = np.asarray(x.sum(axis=1)).ravel()
        scale = np.zeros_like(totals, dtype=np.float32)
        ok = totals > 0
        scale[ok] = float(target_sum) / totals[ok]
        return x.multiply(scale[:, None]).tocsr()

    x = np.asarray(x, dtype=np.float32).copy()
    totals = x.sum(axis=1, keepdims=True)
    np.divide(x * float(target_sum), totals, out=x, where=totals > 0)
    x[totals.ravel() <= 0] = 0.0
    return x


def _mean_vector_for_profile_eval(adata, gene_dict, mean_matrix):
    mean_by_gene = pd.Series(np.asarray(mean_matrix, dtype=np.float32).ravel(), index=gene_dict.var_names.astype(str))
    mean = pd.Index(adata.var_names.astype(str)).map(mean_by_gene).to_numpy(dtype=np.float32)
    return np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)


def expression_matrix_for_profile_eval(adata, gene_dict, mean_matrix, transform="lib_hmc"):
    """Build a profile-evaluation expression matrix.

    `lib_hmc` matches BB mean-corrected expression; `lib_log1p` is the
    conventional library-normalized log1p baseline; `lib` only normalizes depth.
    """
    transform = str(transform)
    if transform not in {"lib_hmc", "lib_log1p", "lib"}:
        raise ValueError(f"Unsupported profile eval transform: {transform}")

    x = _sf_normalize_for_profile_eval(adata.X)
    if transform == "lib_log1p":
        if sparse.issparse(x):
            x = x.copy()
            x.data = np.log1p(x.data)
            return x
        np.log1p(x, out=x)
        return x
    if transform == "lib":
        return x

    mean_vector = _mean_vector_for_profile_eval(adata, gene_dict, mean_matrix)
    if sparse.issparse(x):
        inv_mean = np.zeros_like(mean_vector, dtype=np.float32)
        ok = mean_vector > 0
        inv_mean[ok] = 1.0 / mean_vector[ok]
        return x.multiply(inv_mean).tocsr()

    ok = mean_vector > 0
    x[:, ok] = x[:, ok] / mean_vector[ok][None, :]
    x[:, ~ok] = 0.0
    return x


def direct_gene_profile_pair(ref, query, gene_dict, mean_matrix, transform="lib_hmc"):
    """Build ref/query profile matrices on exact shared gene IDs."""
    ref_genes = pd.Index(ref.var_names.astype(str))
    query_genes = pd.Index(query.var_names.astype(str))
    common = ref_genes.intersection(query_genes)
    if common.empty:
        raise ValueError("No exact shared genes between reference and query for profile evaluation.")

    ref_pos = ref_genes.get_indexer(common)
    query_pos = query_genes.get_indexer(common)
    x_ref = expression_matrix_for_profile_eval(ref, gene_dict, mean_matrix, transform)[:, ref_pos]
    x_query = expression_matrix_for_profile_eval(query, gene_dict, mean_matrix, transform)[:, query_pos]
    return x_ref, x_query, common.astype(str).tolist()


def homo_profile_matrix_from_expression(adata, x, homo_order, homo_col="homo_connect_id"):
    """Average gene-level expression into a fixed homo-group feature order."""
    homo = adata.var[homo_col].astype(str).to_numpy()
    homo_to_col = {str(h): i for i, h in enumerate(homo_order)}
    row_idx = []
    col_idx = []
    weights = []
    counts = pd.Series(homo).value_counts()
    for gene_idx, homo_id in enumerate(homo):
        col = homo_to_col.get(str(homo_id))
        if col is None:
            continue
        row_idx.append(gene_idx)
        col_idx.append(col)
        weights.append(1.0 / float(counts[homo_id]))
    mapper = sparse.csr_matrix(
        (np.asarray(weights, dtype=np.float32), (row_idx, col_idx)),
        shape=(adata.n_vars, len(homo_order)),
    )
    profile = x @ mapper
    return profile.tocsr() if sparse.issparse(profile) else sparse.csr_matrix(profile)


def homo_profile_pair(ref, query, gene_dict, mean_matrix, homo_order, transform="lib_hmc", homo_col="homo_connect_id"):
    """Build ref/query profile matrices after averaging genes by homo group."""
    x_ref = expression_matrix_for_profile_eval(ref, gene_dict, mean_matrix, transform)
    x_query = expression_matrix_for_profile_eval(query, gene_dict, mean_matrix, transform)
    return (
        homo_profile_matrix_from_expression(ref, x_ref, homo_order, homo_col=homo_col),
        homo_profile_matrix_from_expression(query, x_query, homo_order, homo_col=homo_col),
    )


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
    if mode == "within_center":
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
    if mode == "within_center_l2":
        a, b = posthoc_transform_pair(x_ref, x_query, "within_center")
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
    present = []
    for label in expected:
        if label not in col_to_j:
            continue
        if label not in row_to_i:
            ranks.append(len(cols))
            hits.append(False)
            rank_scores.append(0.0)
            present.append(False)
            continue
        i = row_to_i[label]
        j = col_to_j[label]
        order = np.argsort(-sim[i])
        rank = int(np.where(order == j)[0][0]) + 1
        ranks.append(rank)
        hits.append(rank == 1)
        denom = max(1, len(cols) - 1)
        rank_scores.append(1.0 - (rank - 1) / denom)
        present.append(True)
    return {
        "expected_n_labels": int(len(expected)),
        "expected_present_n": int(np.sum(present)),
        "SNcoverage_expected": float(np.mean(present)) if present else np.nan,
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
            col_to_j = {str(col): j for j, col in enumerate(cols)}
            for i, row in enumerate(rows):
                row_values = np.asarray(mat[i], dtype=float)
                finite = np.isfinite(row_values)
                if np.any(finite):
                    top_j = int(np.nanargmax(row_values))
                    ax.scatter(
                        top_j + 0.5,
                        i + 0.5,
                        s=20,
                        marker="o",
                        facecolors="white",
                        edgecolors="0.2",
                        linewidths=0.6,
                        zorder=5,
                    )
                expected_j = col_to_j.get(str(row))
                if expected_j is None:
                    continue
                ax.scatter(
                    expected_j + 0.5,
                    i + 0.5,
                    s=42,
                    marker="o",
                    facecolors="none",
                    edgecolors="0.75",
                    linewidths=1.0,
                    zorder=4,
                )
        ax.set_title(
            f"{panel['metric']}\n"
            f"Cov={panel.get('present_n', 'NA')}/{panel.get('expected_n', 'NA')}; "
            f"SNrank={format_metric_value(panel.get('rank'))}; "
            f"SNhit={format_metric_value(panel.get('hit'))}; "
            f"Cert={format_metric_value(panel.get('certainty'))}"
        )
        ax.set_xlabel("reference label")
        ax.set_ylabel("query/predicted label")
    fig.suptitle(title, y=1.02)
    fig.text(0.5, 0.01, "Open circle: same-label reference; filled circle: row-wise top match", ha="center", fontsize=9)
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
        coverage = expected.get("SNcoverage_expected", np.nan)
        cert_expected = cert * coverage if np.isfinite(cert) and np.isfinite(coverage) else np.nan
        suffix = "" if idx == 0 else f"_{metric}"
        for key, value in {**align, **expected}.items():
            if key == "tag":
                continue
            result[f"{key}{suffix}"] = value
        result[f"profile_certainty{suffix}"] = cert
        result[f"profile_certainty_expected{suffix}"] = cert_expected
        panels.append(
            {
                "metric": metric,
                "matrix": row_normalize_shifted_similarity(sim),
                "rank": expected.get("SNrank_expected"),
                "hit": expected.get("SNhit_expected"),
                "present_n": expected.get("expected_present_n"),
                "expected_n": expected.get("expected_n_labels"),
                "certainty": cert_expected,
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


def plot_spatial_comparison(
    adata,
    true_label_col,
    pred_label_col,
    output_path,
    spot_size=100,
    figsize=(18, 8),
    exclude_unassigned=False,
    save_confusion=True,
):
    """Plot true and predicted spatial labels with a shared color palette."""
    predefined_palette = {
        "ASC": "#e31a1c",
        "Ast": "#e31a1c",
        "MG": "#7f7f7f",
        "OPC": "#a0a0a0",
        "OLG": "#ffd54f",
        "VLMC": "#ef5350",
        "Vascular": "#ef5350",
        "EC": "#f46d43",
        "L2": "#1f77b4",
        "L2/3": "#2ca02c",
        "L2/3/4": "#4caf50",
        "L3": "#388e3c",
        "L3/4": "#9467bd",
        "L3/4/5": "#66bb6a",
        "L4": "purple",
        "L4/5": "#ff7f0e",
        "L4/5/6": "#ffa726",
        "L5": "#ffb347",
        "L5/6": "#ffcc80",
        "L6": "#d4ac0d",
        "LAMP5": "#a5d6a7",
        "LAMP5-RELN": "#c8e6c9",
        "RELN": "#4292c6",
        "VIP": "#6baed6",
        "VIP_RELN": "#9ecae1",
        "SST": "#81c784",
        "PVALB": "#b39ddb",
        "PV": "#b39ddb",
        "PV_CHC": "#c0a5e0",
        "PV-CHC": "#c0a5e0",
        "unassigned": "#d0d0d0",
    }
    mapping = {
        "Astrocytes": "ASC",
        "L2 IT neurons": "L2",
        "L2/3 IT neurons": "L2/3",
        "L3 IT neurons": "L3",
        "L3-6 IT neurons": "L3/4/5",
        "L4 IT neurons": "L4",
        "L5 ET neurons": "L4/5",
        "L5/6 IT neurons": "L5/6",
        "L5/6 NP neurons": "L5/6",
        "L6 CT neurons": "L6",
        "L6 CAR3 neurons": "L6",
        "L6b neurons": "L6",
        "LAMP5 neurons": "LAMP5",
        "Microglia": "MG",
        "Oligodendrocytes": "OLG",
        "Oligodendrocyte precursor cells": "OPC",
        "PVALB neurons": "PVALB",
        "PVALB Chandelier neurons": "PV_CHC",
        "SST neurons": "SST",
        "SST CHODL neurons": "SST",
        "VIP neurons": "VIP",
        "RELN neurons": "RELN",
        "Vascular cells": "VLMC",
    }
    predefined_palette.update({k: predefined_palette[v] for k, v in mapping.items() if v in predefined_palette})

    def make_palette(categories):
        import scanpy as sc

        base_colors = sc.pl.palettes.default_102
        palette = {cat: predefined_palette[cat] for cat in categories if cat in predefined_palette}
        unused_colors = [c for c in base_colors if c not in palette.values()]
        i = 0
        for cat in categories:
            if cat not in palette:
                palette[cat] = unused_colors[i % len(unused_colors)]
                i += 1
        return palette

    import scanpy as sc

    _ensure_dir(os.path.dirname(output_path))
    adata = adata.copy()
    if "spatial" not in adata.obsm:
        for cols in [("spatial1", "spatial2"), ("x", "y"), ("X", "Y"), ("rx", "ry"), ("coor_x", "coor_y")]:
            if all(col in adata.obs.columns for col in cols):
                adata.obsm["spatial"] = adata.obs.loc[:, list(cols)].to_numpy(dtype=float)
                break
    else:
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)

    if pred_label_col not in adata.obs:
        print(f"[ERROR] Predicted label column '{pred_label_col}' not found. Skip plotting.")
        return

    for col in [true_label_col, pred_label_col]:
        if col in adata.obs and not pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].astype("category")

    if exclude_unassigned and "unassigned" in list(adata.obs[pred_label_col].cat.categories):
        adata = adata[adata.obs[pred_label_col] != "unassigned"].copy()

    has_true = true_label_col in adata.obs
    true_cats = list(adata.obs[true_label_col].cat.categories) if has_true else []
    pred_cats = list(adata.obs[pred_label_col].cat.categories)
    palette_map = make_palette(sorted(set(true_cats) | set(pred_cats)))

    n_panels = 1 + int(has_true)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    if has_true:
        sc.pl.spatial(
            adata,
            color=true_label_col,
            spot_size=spot_size,
            palette=[palette_map[c] for c in adata.obs[true_label_col].cat.categories],
            ax=axes[0],
            show=False,
        )
        axes[0].set_title(f"True Labels ({true_label_col})")

    sc.pl.spatial(
        adata,
        color=pred_label_col,
        spot_size=spot_size,
        palette=[palette_map[c] for c in adata.obs[pred_label_col].cat.categories],
        ax=axes[-1],
        show=False,
    )
    axes[-1].set_title(f"Predicted Labels ({pred_label_col})")

    suffix = adata.uns.get("suffix", "")
    query_name = adata.uns.get("query_name", "")
    if suffix or query_name:
        fig.suptitle(f"Query: {query_name}\n{suffix}", fontsize=12, y=0.98)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    if save_confusion and has_true:
        cm_recall = pd.crosstab(adata.obs[true_label_col], adata.obs[pred_label_col], normalize="index")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_recall, annot=False, linewidths=0.2, cmap="viridis", ax=ax)
        ax.set_title(f"Confusion Matrix\n(True={true_label_col}, Pred={pred_label_col})")
        cm_path = str(output_path).replace("spatial.png", "confusion.png")
        fig.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
