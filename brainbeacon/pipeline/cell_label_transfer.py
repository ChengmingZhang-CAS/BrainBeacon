import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, adjusted_rand_score
from scipy.sparse import csr_matrix
from anndata import AnnData

from sklearn.preprocessing import normalize

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for reproducibility with FAISS + CUDA

from typing import Literal
from typing import Dict, List, Optional


def run_knn_voting(
    X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray,
    method: str = "native",
    K: int = 30,
    metric: str = "euclidean",
    weight_mode: str = "distance",
    unassigned_threshold: float = None,
    device: str = "cpu"
) -> List[str]:
    """Unified KNN voting interface."""
    if method == "native":
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=K, weights=weight_mode, metric=metric, n_jobs=-1)
        knn.fit(X_ref, y_ref)
        preds = knn.predict(X_query)

        # --- if threshold is enabled, apply unassigned filtering ---
        if unassigned_threshold is not None:
            distances, indices = knn.kneighbors(X_query, n_neighbors=K)
            # --- only use top-T neighbors for threshold ---
            T = max(1, int(np.sqrt(K)))
            distances_T = distances[:, :T]
            sims_T = None

            if metric == "cosine":
                sims_T = 1 - distances_T
            else:
                D_min = distances.min()
                D_max = distances.max()
                sims_T = 1 - (distances_T - D_min) / (D_max - D_min + 1e-8)

            # --- weighted mean similarity for threshold ---
            weights_T = 1.0 / (distances_T + 1e-8)
            weighted_mean_sims = (sims_T * weights_T).sum(axis=1) / weights_T.sum(axis=1)

            preds = np.where(weighted_mean_sims < unassigned_threshold, "unassigned", preds)

            # --- report unassigned ratio ---
            unassigned_count = np.sum(preds == "unassigned")
            ratio_unassigned = unassigned_count / len(preds) * 100
            print(f"Unassigned: {ratio_unassigned:.2f}%  (threshold={unassigned_threshold}, metric={metric}, top-T={T}/{K})")

        return preds

    elif method == "hnsw":
        import hnswlib
        N, dim = X_ref.shape
        space_map = {"euclidean": "l2", "cosine": "cosine", "ip": "ip"}
        index = hnswlib.Index(space=space_map[metric], dim=dim)
        index.init_index(max_elements=N, ef_construction=200, M=16, random_seed=42)
        index.add_items(X_ref, np.arange(N))
        index.set_ef(max(50, K + 10))
        nnIndex, _ = index.knn_query(X_query, k=K)
        preds = pd.DataFrame(np.array(y_ref)[nnIndex]).apply(lambda row: row.value_counts().idxmax(), axis=1).tolist()
        return preds

    elif method == "faiss":
        import faiss
        d = X_ref.shape[1]
        if metric == "euclidean":
            index = faiss.IndexFlatL2(d)
        elif metric == "cosine":
            faiss.normalize_L2(X_ref)
            faiss.normalize_L2(X_query)
            index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unsupported metric {metric} for FAISS")

        if device == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(X_ref.astype(np.float32))
        _, nnIndex = index.search(X_query.astype(np.float32), K)
        preds = pd.DataFrame(np.array(y_ref)[nnIndex]).apply(lambda row: row.value_counts().idxmax(), axis=1).tolist()
        return preds

    else:
        raise ValueError(f"Unsupported KNN method: {method}")


def run_prototype_classifier(X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray, metric: str = "euclidean") -> List[str]:
    """Prototype (Nearest Centroid) classifier."""
    prototypes = {label: X_ref[y_ref == label].mean(axis=0) for label in np.unique(y_ref)}
    labels = list(prototypes.keys())
    centroids = np.vstack(list(prototypes.values()))

    if metric == "cosine":
        centroids = normalize(centroids, norm="l2")
        X_query = normalize(X_query, norm="l2")
        sims = np.dot(X_query, centroids.T)
        pred_idx = sims.argmax(axis=1)
    else:
        dists = np.linalg.norm(X_query[:, None, :] - centroids[None, :, :], axis=2)
        pred_idx = dists.argmin(axis=1)

    return [labels[i] for i in pred_idx]


def run_logreg_classifier(X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray, max_iter: int = 200, C: float = 1.0) -> List[str]:
    """Logistic Regression classifier."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1, multi_class="auto")
    clf.fit(X_ref, y_ref)
    return clf.predict(X_query)

def run_label_transfer(X_ref: np.ndarray, y_ref: np.ndarray, X_query: np.ndarray, method: str, **kwargs) -> List[str]:
    """Dispatcher for different classification methods."""
    if method in ["native", "hnsw", "faiss"]:
        return run_knn_voting(X_ref, y_ref, X_query, method=method, **kwargs)
    elif method == "prototype":
        return run_prototype_classifier(X_ref, y_ref, X_query, metric=kwargs.get("metric", "euclidean"))
    elif method == "logreg":
        return run_logreg_classifier(X_ref, y_ref, X_query, max_iter=kwargs.get("max_iter", 200), C=kwargs.get("C", 1.0))
    else:
        raise ValueError(f"Unsupported method: {method}")


def build_marker_dict(csv_path, class_col="SubClass", cutoff=1.0, top_n=None):
    """
    Build marker dict {class: {gene: logFC}} directly from marker CSV.
    """
    df_marker = pd.read_csv(csv_path, index_col=0)
    df_marker = df_marker[df_marker["avg_log2FC"] > cutoff]

    marker_dict = {}
    for cls, df_cls in df_marker.groupby(class_col):
        if top_n is not None:
            df_cls = df_cls.sort_values("avg_log2FC", ascending=False).head(top_n)
        marker_dict[cls] = dict(zip(df_cls["gene"], df_cls["avg_log2FC"]))
    counts = {cls: len(genes) for cls, genes in marker_dict.items()}
    all_counts = list(counts.values())
    print(
        f"[INFO] Marker gene stats: min={min(all_counts)}, max={max(all_counts)}, mean={np.mean(all_counts):.1f}, total_classes={len(all_counts)}")

    assert len(marker_dict) > 0, f"[ERROR] No valid markers found in {csv_path}"
    return marker_dict

def manual_spatial_smooth(
    adata: AnnData,
    n_neighbors: int = 6,
    layer_key: str = 'smoothed',
    spatial_key: str = 'spatial',
    inplace: bool = True,
    sigma_mode: str = "median",   # "median", "mean", "fixed"
    fixed_sigma: float = 50.0,    # only if sigma_mode="fixed"
) -> Optional[np.ndarray]:
    """
    Spatial smoothing with Gaussian kernel.
    sigma_mode:
        - "median": median neighbor distance
        - "mean": mean neighbor distance
        - "fixed": user-defined fixed_sigma
    """
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key} not found in adata.obsm.")

    n_spots = adata.n_obs
    X = adata.X

    # Build KNN graph
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree')
    nn.fit(adata.obsm[spatial_key])
    dists, indices = nn.kneighbors(adata.obsm[spatial_key])

    # Choose sigma
    if sigma_mode == "median":
        sigma = np.median(dists[:, 1:].flatten())
    elif sigma_mode == "mean":
        sigma = np.mean(dists[:, 1:].flatten())
    elif sigma_mode == "fixed":
        sigma = fixed_sigma
    else:
        raise ValueError("sigma_mode must be 'median', 'mean', or 'fixed'.")

    # Gaussian weights
    rows = np.repeat(np.arange(n_spots), n_neighbors + 1)
    cols = indices.flatten()
    weights = np.exp(- (dists.flatten() ** 2) / (2 * sigma ** 2))

    # Row normalize
    A = csr_matrix((weights, (rows, cols)), shape=(n_spots, n_spots))
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    normA = csr_matrix((1.0 / row_sums, (np.arange(n_spots), np.arange(n_spots))), shape=(n_spots, n_spots))
    smoothed_X = (normA @ A) @ X

    if inplace:
        adata.layers[layer_key] = smoothed_X
        return None
    else:
        return smoothed_X


# helper: make HVG mask and merge with add_genes BEFORE slicing
def apply_hvg_with_add(
        adata_full: AnnData,
        n_top: int,
        add_genes_list: Optional[List[str]],
        prefer_target_gene: bool,
        hvg_flavor: str = "seurat_v3",
) -> AnnData:
    tmp = adata_full.copy()
    if hvg_flavor == "seurat_v3":
        # seurat_v3 expects raw counts / count-like input
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat_v3")

    elif hvg_flavor == "seurat":
        # seurat expects log-normalized input
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")
    else:
        raise ValueError(f"Unsupported hvg_flavor: {hvg_flavor}")

    hvg_mask = tmp.var.highly_variable.copy()

    if add_genes_list:
        gene_col = "target_gene" if (prefer_target_gene and "target_gene" in tmp.var.columns) else "genenames"
        extra_mask = tmp.var[gene_col].isin(add_genes_list).fillna(False)

        if extra_mask.any():
            newly_added = int((extra_mask & ~hvg_mask).sum())
            total_hits = int(extra_mask.sum())
            keep_mask = hvg_mask | extra_mask
            print(
                f"[INFO] Forcing {newly_added} newly added genes "
                f"(total hits={total_hits}, col={gene_col})"
            )
        else:
            keep_mask = hvg_mask
            print("[WARN] None of the add_genes were found in this namespace; HVG unchanged.")
    else:
        keep_mask = hvg_mask

    return adata_full[:, keep_mask].copy()

def flatten_prior_marker_genes(prior_marker_dict):
    """Flatten all prior marker genes for HVG forcing."""
    if prior_marker_dict is None:
        return []

    genes = []
    for marker_genes in prior_marker_dict.values():
        if isinstance(marker_genes, dict):
            genes.extend(marker_genes.keys())
        else:
            genes.extend(marker_genes)

    return sorted(set(map(str, genes)))


def build_marker_df_from_adata(
    adata: sc.AnnData,
    label_col: str,
    cutoff: float = 0.5,
    top_n: int = 50,
    gene_col: str = "genenames",
    method: str = "wilcoxon",
    prior_marker_dict: Optional[dict] = None,
    global_marker_key: str = "global",
) -> pd.DataFrame:
    """
    Build marker dataframe from AnnData before HVG selection.

    Auto markers are selected by DE ranking.
    Class-specific prior markers are merged using real positive logFC.
    Global prior markers are skipped here and only used for HVG forcing.

    Returns a long-format dataframe with columns:
        class, gene, scores, logfoldchange, pvals, pvals_adj, rank, weight, source
    """
    if label_col not in adata.obs.columns:
        raise KeyError(f"{label_col} not found in adata.obs.")

    tmp = adata.copy()

    if gene_col in tmp.var.columns:
        tmp.var_names = tmp.var[gene_col].astype(str)
        tmp.var_names_make_unique()
    else:
        tmp.var_names = tmp.var_names.astype(str)
        tmp.var_names_make_unique()

    tmp.obs[label_col] = tmp.obs[label_col].astype(str).astype("category")

    sc.pp.normalize_total(tmp, target_sum=1e4)
    sc.pp.log1p(tmp)

    sc.tl.rank_genes_groups(
        tmp,
        groupby=label_col,
        method=method,
        use_raw=False,
        n_genes=tmp.n_vars,
    )

    df_all = sc.get.rank_genes_groups_df(tmp, group=None).copy()
    df_all = df_all.rename(
        columns={
            "group": "class",
            "names": "gene",
            "logfoldchanges": "logfoldchange",
        }
    )
    df_all["class"] = df_all["class"].astype(str)
    df_all["gene"] = df_all["gene"].astype(str)

    df_marker = df_all[df_all["logfoldchange"] > cutoff].copy()
    df_marker = df_marker.sort_values(["class", "scores"], ascending=[True, False])
    df_marker["rank"] = df_marker.groupby("class").cumcount() + 1

    if top_n is not None:
        df_marker = df_marker[df_marker["rank"] <= top_n].copy()

    df_marker["source"] = "auto"

    if prior_marker_dict is not None:
        prior_rows = []
        skipped = 0

        for cls, marker_genes in prior_marker_dict.items():
            cls = str(cls)
            if cls == global_marker_key:
                continue

            genes = list(marker_genes.keys()) if isinstance(marker_genes, dict) else list(marker_genes)

            for gene in genes:
                gene = str(gene)
                matched = df_all[(df_all["class"] == cls) & (df_all["gene"] == gene)]

                if len(matched) == 0:
                    skipped += 1
                    continue

                row = matched.iloc[0].copy()
                logfc = float(row["logfoldchange"])

                if not np.isfinite(logfc) or logfc <= 0:
                    skipped += 1
                    continue

                row["source"] = "prior"
                prior_rows.append(row)

        if prior_rows:
            df_prior = pd.DataFrame(prior_rows)
            df_marker = pd.concat([df_marker, df_prior], axis=0, ignore_index=True)
            df_marker = (
                df_marker
                .sort_values(["class", "source", "logfoldchange"], ascending=[True, True, False])
                .drop_duplicates(subset=["class", "gene"], keep="first")
                .copy()
            )

        print(f"[INFO] Prior markers merged into marker_df: {len(prior_rows)}")
        if skipped > 0:
            print(f"[WARN] Prior markers skipped from marker_df: {skipped}")

    df_marker["weight"] = df_marker["logfoldchange"].astype(float).clip(lower=cutoff, upper=5.0)
    df_marker = df_marker.sort_values(["class", "source", "weight"], ascending=[True, True, False]).copy()
    df_marker["rank"] = df_marker.groupby("class").cumcount() + 1
    df_marker = df_marker.reset_index(drop=True)

    counts = df_marker.groupby("class").size()
    if len(counts) == 0:
        raise ValueError(
            "No marker genes were found. "
            "Try lowering marker_cutoff or checking label_col."
        )

    print(
        f"[INFO] Marker gene stats: "
        f"min={counts.min()}, max={counts.max()}, "
        f"mean={counts.mean():.1f}, total_classes={len(counts)}"
    )

    print(f"[INFO] Marker source counts: {df_marker['source'].value_counts().to_dict()}")

    return df_marker


def preprocess_one_adata(
    adata: AnnData,
    info: Dict,
    n_hvg: int,
    target_species: str = "macaque",
    convert_id: bool = True,
    smooth_st: bool = True,
    smooth_k: int = 25,
    use_marker: bool = False,
    add_marker_genes: bool = False,
    marker_label_col: Optional[str] = None,
    marker_topn: int = 50,
    marker_cutoff: float = 0.5,
    marker_gene_col: str = "genenames",
    add_genes: Optional[List[str]] = None,
    hvg_flavor: str = "seurat",
    homology_df: Optional[pd.DataFrame] = None,
    prior_marker_dict: Optional[dict] = None,
    global_marker_key: str = "global",
) -> AnnData:
    """Preprocess a single AnnData with optional smoothing, ID conversion, HVG, and forcing extra genes into HVG set."""
    print(f"[INFO] Preprocessing {info['data_name']}...")
    adata.obs_names = adata.obs_names.astype(str)
    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    # 1) ensure gene name column
    if "genenames" not in adata.var.columns:
        adata.var["genenames"] = adata.var["gene_symbol"]

    # 2) spatial coords
    if "spatial" in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)
        print(f"[INFO] Spatial coordinates already exist in {info['data_name']}.")
    elif info["assay"] == "snrna":
        print(f"[INFO] {info['data_name']} is snRNA-seq. Skip spatial coordinate setup.")
    else:
        if "rx" in adata.obs.columns and "ry" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["rx", "ry"]].values.astype(float)
            print(f"[INFO] Spatial coordinates added from rx/ry for {info['data_name']}.")
        elif "x" in adata.obs.columns and "y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values.astype(float)
            print(f"[INFO] Spatial coordinates added from x/y for {info['data_name']}.")
        elif "coor_x" in adata.obs.columns and "coor_y" in adata.obs.columns:
            adata.obsm["spatial"] = adata.obs[["coor_x", "coor_y"]].values.astype(float)
            print(f"[INFO] Spatial coordinates added from coor_x/coor_y for {info['data_name']}.")
        else:
            warnings.warn(f"Spatial coordinates not found for {info['data_name']}. Skipping smoothing.")

    if "spatial" in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)
        valid_idx = ~np.isnan(adata.obsm["spatial"]).any(axis=1)
        adata = adata[valid_idx].copy()

    # 3) optional spatial smoothing before HVG
    if smooth_st and info["assay"] != "snrna" and "spatial" in adata.obsm:
        manual_spatial_smooth(adata, layer_key="smooth", n_neighbors=smooth_k)
        adata.X = adata.layers["smooth"]

    # Light QC
    min_gene_threshold = 0.01
    dynamic_min_genes = min(200, int(adata.n_vars * min_gene_threshold))
    dynamic_min_genes = max(dynamic_min_genes, 1)
    print(f"[INFO] Using min_genes={dynamic_min_genes} (based on {adata.n_vars} genes)")
    sc.pp.filter_cells(adata, min_genes=dynamic_min_genes)
    sc.pp.filter_genes(adata, min_cells=3)

    marker_genes = []

    # 4) optional auto marker calculation before HVG
    if use_marker:
        if marker_label_col is None:
            raise ValueError("marker_label_col must be provided when use_marker=True.")

        # Prior markers are merged into marker_df only when add_marker_genes=True.
        marker_prior_dict = prior_marker_dict if add_marker_genes else None

        marker_df = build_marker_df_from_adata(
            adata,
            label_col=marker_label_col,
            cutoff=marker_cutoff,
            top_n=marker_topn,
            gene_col=marker_gene_col,
            prior_marker_dict=marker_prior_dict,
            global_marker_key=global_marker_key,
        )
        adata.uns["marker_df"] = marker_df

        # marker_df genes are sampling marker genes and should be kept once computed.
        marker_genes = marker_df["gene"].astype(str).unique().tolist()
        if add_genes is None:
            add_genes = marker_genes
        else:
            add_genes = sorted(set(map(str, add_genes)) | set(marker_genes))

        print(f"[INFO] Added marker_df genes for HVG forcing: {len(marker_genes)}")

    # 5) optional prior marker genes forcing into HVG
    if add_marker_genes:
        prior_genes = flatten_prior_marker_genes(prior_marker_dict)

        if len(prior_genes) > 0:
            if add_genes is None:
                add_genes = prior_genes
            else:
                add_genes = sorted(set(map(str, add_genes)) | set(prior_genes))

            print(f"[INFO] Added prior marker genes for HVG forcing: {len(prior_genes)}")
        else:
            print("[INFO] add_marker_genes=True, but no prior marker genes were available for HVG forcing.")

    # 6) same-species or no-conversion branch
    if info["species"] == target_species or not convert_id:
        if info["species"] != target_species and not convert_id and add_genes:
            warnings.warn("[WARN] convert_id=False and species differ from target; add_genes may not match current gene namespace.")

        if not convert_id:
            # Ensure Ensembl IDs for BrainBeacon input
            if not adata.var_names.str.startswith("ENS").all():
                print(f"[WARN] {info['data_name']} gene IDs not in Ensembl format, running ensure_ensembl_ids()...")
                from brainbeacon.tokenizer import ensure_ensembl_ids
                adata = ensure_ensembl_ids(adata, species=info["species"])

        adata.var_names_make_unique()
        adata = apply_hvg_with_add(
            adata,
            n_hvg,
            add_genes,
            hvg_flavor=hvg_flavor,
            prefer_target_gene=False,
        )
        adata.obs_names_make_unique()
        return adata

    # 7) cross-species branch: map homologs first, then HVG + add_genes in target namespace
    adata = map_homologs(
        adata,
        homology_df,
        source_species=info["species"],
        target_species=target_species,
        source_gene_col="genenames",
    )

    species_list = ["macaque", "marmoset", "human", "mouse"]
    if target_species in species_list:
        from brainbeacon.tokenizer import ensure_ensembl_ids
        adata = ensure_ensembl_ids(adata, species=target_species)
    else:
        warnings.warn(f"Unknown species '{target_species}'. Skipping Ensembl ID conversion.")

    adata = adata[:, ~adata.var["genenames"].duplicated()].copy()
    adata = apply_hvg_with_add(
        adata,
        n_hvg,
        add_genes,
        hvg_flavor=hvg_flavor,
        prefer_target_gene=True,
    )
    adata.obs_names_make_unique()
    return adata


def marker_df_to_dict(marker_df, class_col="class", gene_col="gene", weight_col="weight"):
    if marker_df is None or marker_df.empty:
        return None
    required_cols = {class_col, gene_col, weight_col}
    missing_cols = required_cols - set(marker_df.columns)
    if missing_cols:
        raise KeyError(f"marker_df missing required columns: {missing_cols}")
    return {
        str(cls): dict(zip(df_cls[gene_col].astype(str), df_cls[weight_col].astype(float)))
        for cls, df_cls in marker_df.groupby(class_col)
    }

def align_ref_query_genes(
    adata_ref,
    adata_query,
    mode="gene",
    gene_dict=None,
    homo_col="homo_connect_id",
):
    """
    Align reference and query genes by direct gene intersection or shared homologous groups.

    Parameters
    ----------
    adata_ref : AnnData
        Reference AnnData after preprocessing.
    adata_query : AnnData
        Query AnnData after preprocessing.
    mode : str
        "gene": direct gene intersection using var_names.
        "homo": homo group intersection using gene_dict.var.index and gene_dict.var[homo_col].
    gene_dict : AnnData, optional
        Gene dictionary AnnData. Required when mode="homo".
    homo_col : str
        Column name in gene_dict.var for homologous group IDs.

    Returns
    -------
    adata_ref : AnnData
        Filtered reference AnnData.
    adata_query : AnnData
        Filtered query AnnData.
    align_info : dict
        Minimal alignment summary for debugging.
    """
    mode = str(mode).lower()
    n_ref_before = adata_ref.n_vars
    n_query_before = adata_query.n_vars

    if mode == "gene":
        common_genes = adata_ref.var_names.intersection(adata_query.var_names)

        align_info = {
            "mode": "gene",
            "n_ref_before": n_ref_before,
            "n_query_before": n_query_before,
            "n_common_genes": len(common_genes),
            "n_ref_after": len(common_genes),
            "n_query_after": len(common_genes),
        }

        print(
            f"[INFO] Gene alignment mode=gene | "
            f"ref {n_ref_before}->{len(common_genes)}, "
            f"query {n_query_before}->{len(common_genes)}, "
            f"common_genes={len(common_genes)}"
        )

        return adata_ref[:, common_genes].copy(), adata_query[:, common_genes].copy(), align_info

    if mode == "homo":
        if gene_dict is None:
            raise ValueError("gene_dict is required when mode='homo'.")
        if homo_col not in gene_dict.var.columns:
            raise KeyError(f"'{homo_col}' not found in gene_dict.var.")

        gene_dict_var = gene_dict.var

        # Strict mode: only use gene_dict.var.index as the gene ID key.
        gene_to_homo = {
            str(gene): homo
            for gene, homo in gene_dict_var[homo_col].items()
            if not pd.isna(homo)
        }

        ref_gene_to_homo = {
            str(g): gene_to_homo[str(g)]
            for g in adata_ref.var_names
            if str(g) in gene_to_homo
        }

        query_gene_to_homo = {
            str(g): gene_to_homo[str(g)]
            for g in adata_query.var_names
            if str(g) in gene_to_homo
        }

        common_homo = set(ref_gene_to_homo.values()).intersection(set(query_gene_to_homo.values()))

        ref_keep = [
            g
            for g in adata_ref.var_names
            if ref_gene_to_homo.get(str(g)) in common_homo
        ]

        query_keep = [
            g
            for g in adata_query.var_names
            if query_gene_to_homo.get(str(g)) in common_homo
        ]

        align_info = {
            "mode": "homo",
            "homo_col": homo_col,
            "n_ref_before": n_ref_before,
            "n_query_before": n_query_before,
            "n_ref_mapped": len(ref_gene_to_homo),
            "n_query_mapped": len(query_gene_to_homo),
            "n_common_homo": len(common_homo),
            "n_ref_after": len(ref_keep),
            "n_query_after": len(query_keep),
        }

        print(
            f"[INFO] Gene alignment mode=homo | "
            f"ref {n_ref_before}->{len(ref_keep)} mapped={len(ref_gene_to_homo)}, "
            f"query {n_query_before}->{len(query_keep)} mapped={len(query_gene_to_homo)}, "
            f"common_homo={len(common_homo)}"
        )

        return adata_ref[:, ref_keep].copy(), adata_query[:, query_keep].copy(), align_info

    raise ValueError("Unsupported mode. Expected mode='gene' or mode='homo'.")

def align_ref_query_list_genes(
    adata_ref,
    adata_query_list,
    mode="homo",
    gene_dict=None,
    query_names=None,
    homo_col="homo_connect_id",
):
    """
    Align one reference AnnData and multiple query AnnData objects to a shared gene/homo space.

    mode="gene":
        Keep genes shared by ref and all queries using var_names intersection.

    mode="homo":
        Keep genes whose homo_connect_id is shared by ref and all queries.
        This avoids over-restricting cross-species data by direct var_names intersection.

    Parameters
    ----------
    adata_ref : AnnData
        Reference AnnData.
    adata_query_list : list[AnnData]
        Query AnnData objects.
    mode : str
        "gene" or "homo".
    gene_dict : AnnData or None
        Gene dictionary used when mode="homo".
    query_names : list[str] or None
        Names of query datasets for logging and align_info.
    homo_col : str
        Column name in gene_dict.var for homologous group IDs.

    Returns
    -------
    adata_ref : AnnData
        Aligned reference AnnData.
    aligned_query_list : list[AnnData]
        Aligned query AnnData objects.
    align_info : dict
        Alignment information.
    """
    if not isinstance(adata_query_list, (list, tuple)):
        raise TypeError("adata_query_list must be a list or tuple of AnnData objects.")

    if len(adata_query_list) == 0:
        raise ValueError("adata_query_list must contain at least one query AnnData.")

    if query_names is None:
        query_names = [f"query_{i}" for i in range(len(adata_query_list))]

    if len(query_names) != len(adata_query_list):
        raise ValueError(
            f"query_names and adata_query_list must have the same length, "
            f"got {len(query_names)} and {len(adata_query_list)}."
        )

    mode = str(mode).lower()
    all_names = ["ref"] + list(query_names)
    all_adata = [adata_ref] + list(adata_query_list)
    n_vars_before = {name: int(adata.n_vars) for name, adata in zip(all_names, all_adata)}

    if mode == "gene":
        common_genes = adata_ref.var_names.copy()

        for adata_query in adata_query_list:
            common_genes = common_genes.intersection(adata_query.var_names)

        if len(common_genes) == 0:
            raise ValueError("No common genes remain after multi-query gene alignment.")

        adata_ref = adata_ref[:, common_genes].copy()
        aligned_query_list = [
            adata_query[:, common_genes].copy()
            for adata_query in adata_query_list
        ]

        align_info = {
            "mode": "gene",
            "n_queries": len(aligned_query_list),
            "query_names": list(query_names),
            "n_vars_before": n_vars_before,
            "n_common_genes": int(len(common_genes)),
            "n_vars_after": {
                name: int(adata.n_vars)
                for name, adata in zip(all_names, [adata_ref] + aligned_query_list)
            },
        }

        print(
            f"[INFO] Multi-query gene alignment mode=gene | "
            f"common_genes={len(common_genes)}"
        )

        return adata_ref, aligned_query_list, align_info

    if mode == "homo":
        if gene_dict is None:
            raise ValueError("gene_dict is required when mode='homo'.")

        if homo_col not in gene_dict.var.columns:
            raise KeyError(f"'{homo_col}' not found in gene_dict.var.")

        gene_to_homo = {
            str(gene): homo
            for gene, homo in gene_dict.var[homo_col].items()
            if not pd.isna(homo)
        }

        adata_gene_to_homo_list = []
        adata_homo_sets = []

        for adata in all_adata:
            gene_to_homo_this = {
                str(g): gene_to_homo[str(g)]
                for g in adata.var_names
                if str(g) in gene_to_homo
            }
            adata_gene_to_homo_list.append(gene_to_homo_this)
            adata_homo_sets.append(set(gene_to_homo_this.values()))

        common_homo = set.intersection(*adata_homo_sets)

        if len(common_homo) == 0:
            raise ValueError("No common homo groups remain after multi-query homo alignment.")

        aligned_adata_list = []
        n_mapped = {}
        n_vars_after = {}
        n_unique_homo_after = {}

        for name, adata, gene_to_homo_this in zip(all_names, all_adata, adata_gene_to_homo_list):
            keep_genes = [
                g
                for g in adata.var_names
                if gene_to_homo_this.get(str(g)) in common_homo
            ]

            aligned = adata[:, keep_genes].copy()
            aligned_adata_list.append(aligned)

            kept_homo = {
                gene_to_homo_this[str(g)]
                for g in keep_genes
                if str(g) in gene_to_homo_this
            }

            n_mapped[name] = int(len(gene_to_homo_this))
            n_vars_after[name] = int(aligned.n_vars)
            n_unique_homo_after[name] = int(len(kept_homo))

        adata_ref = aligned_adata_list[0]
        aligned_query_list = aligned_adata_list[1:]

        align_info = {
            "mode": "homo",
            "homo_col": homo_col,
            "n_queries": len(aligned_query_list),
            "query_names": list(query_names),
            "n_vars_before": n_vars_before,
            "n_mapped": n_mapped,
            "n_common_homo": int(len(common_homo)),
            "n_vars_after": n_vars_after,
            "n_unique_homo_after": n_unique_homo_after,
        }

        print(
            f"[INFO] Multi-query gene alignment mode=homo | "
            f"common_homo={len(common_homo)}, "
            f"n_vars_after={n_vars_after}"
        )

        return adata_ref, aligned_query_list, align_info

    raise ValueError("Unsupported mode. Expected mode='gene' or mode='homo'.")


def map_homologs(
    adata: sc.AnnData,
    homology_df: pd.DataFrame,
    source_species: Literal['human', 'macaque', 'marmoset', 'mouse'],
    target_species: Literal['human', 'macaque', 'marmoset', 'mouse'],
    source_gene_col: str,
    delete_tmp: bool = False,
) -> sc.AnnData:
    # Step 0: same species, just return a copy
    if source_species == target_species:
        print("[WARNING] Source and target species are the same. Returning a copy.")
        return adata.copy()

    # Step 1: pick source/target columns from homology table
    species_to_col = {
        'human': 'humanGene',
        'macaque': 'macaqueGene',
        'marmoset': 'marmosetGene',
        'mouse': 'mouseGene'
    }
    source_col_map = species_to_col[source_species]
    target_col_map = species_to_col[target_species]
    print(f"[INFO] Mapping from {source_species} → {target_species}")

    homology_map_clean = homology_df[[source_col_map, target_col_map]].dropna()
    homology_map_clean = homology_map_clean.drop_duplicates(subset=[source_col_map], keep='first')
    homology_map_clean = homology_map_clean.drop_duplicates(subset=[target_col_map], keep='first')

    mapping_dict = pd.Series(
        homology_map_clean[target_col_map].values,
        index=homology_map_clean[source_col_map]
    ).to_dict()
    print(f"[INFO] Created mapping for {len(mapping_dict)} genes.")

    if source_gene_col == 'index':
        source_genes = adata.var.index
    else:
        if source_gene_col not in adata.var.columns:
            raise ValueError(f"[ERROR] Column {source_gene_col} not found in var.")
        source_genes = adata.var[source_gene_col]

    # Step 2: keep only mappable genes
    mappable_mask = source_genes.isin(mapping_dict.keys())
    adata_mappable = adata[:, mappable_mask].copy()
    print(f"[INFO] Found {adata_mappable.n_vars} mappable genes.")

    # Step 3: map to target gene names
    if source_gene_col == 'index':
        target_genes = adata_mappable.var.index.map(mapping_dict)
    else:
        target_genes = adata_mappable.var[source_gene_col].map(mapping_dict)

    non_null_mask = target_genes.notna()
    if not non_null_mask.all():
        print(f"[WARN] Dropping {(~non_null_mask).sum()} unmapped genes.")
        adata_mappable = adata_mappable[:, non_null_mask].copy()
        target_genes = target_genes[non_null_mask]

    adata_mappable.var['target_gene'] = target_genes

    # Step 4: handle duplicates
    if adata_mappable.var['target_gene'].duplicated().any():
        print("[INFO] Duplicate mappings found. Aggregating by summing counts.")
        grouped = adata_mappable.to_df().groupby(adata_mappable.var['target_gene'], axis=1).sum()

        new_var = adata_mappable.var.drop_duplicates(subset=['target_gene'], keep='first')
        new_var = new_var.set_index('target_gene')
        new_var = new_var.reindex(index=grouped.columns)

        # --- rebuild AnnData but keep obs/obsm/uns ---
        adata_final = sc.AnnData(
            X=grouped.values,
            obs=adata_mappable.obs.copy(),
            var=new_var.copy()
        )
        adata_final.obsm = adata_mappable.obsm.copy()
        adata_final.uns = adata_mappable.uns.copy()
        adata_final.var_names = grouped.columns.tolist()

    else:
        print("[INFO] All mappings unique. No aggregation needed.")
        adata_final = adata_mappable.copy()
        adata_final.var.index = adata_final.var['target_gene']
        adata_final.var_names = adata_final.var['target_gene'].tolist()

    if 'target_gene' in adata_final.var.columns and delete_tmp:
        adata_final.var = adata_final.var.drop(columns=['target_gene'])

    print(f"[INFO] Mapping complete. Final AnnData has {adata_final.n_vars} genes.")
    return adata_final


def compute_marker_scores(adata, cells, marker_dict_cls, gene_col="genenames"):
    """
    Compute weighted marker scores for a set of cells.
    Steps: normalize library size, log1p, then aggregate marker genes.
    """
    # Map gene names to indices
    var_genes = adata.var[gene_col].values
    gene_to_idx = {g: i for i, g in enumerate(var_genes)}

    # Get cell × gene matrix
    X_all = adata[cells, :].X
    if not isinstance(X_all, np.ndarray):
        X_all = X_all.toarray()

    # Library size normalization
    libsize = X_all.sum(axis=1, keepdims=True)
    libsize[libsize == 0] = 1
    X_all = X_all / libsize * 1e4

    # Log1p
    X_all = np.log1p(X_all)

    # Select marker genes present in adata
    valid = [(gene_to_idx[g], w) for g, w in marker_dict_cls.items() if g in gene_to_idx]
    if not valid:
        return np.zeros(len(cells), dtype=np.float32)

    idx, w = zip(*valid)
    X = X_all[:, list(idx)]  # take only marker genes
    w = np.array(w, dtype=np.float32)

    # Weighted average score
    scores = (X * w).sum(axis=1) / (w.sum() + 1e-8)
    return scores


def subsample_adata(
    adata: sc.AnnData,
    class_col: str,
    mode: str,
    min_cells_per_class: int,
    rate: float,
    alpha: float,
    marker_dict: dict = None,
    pool_factor: int = 1,
    oversample_dict: dict = None,
) -> sc.AnnData:
    """Subsamples an AnnData object based on the specified mode.
    If marker_dict is provided, cells are ranked by marker scores instead of random sampling.
    """
    if mode == "none":
        return adata

    print(f"[INFO] ref: Subsampling with mode='{mode}' "
          f"{'(marker-guided)' if marker_dict is not None else '(random)'}")

    sampled_idx = []

    if mode == "fix":
        for cls, df_cls in adata.obs.groupby(class_col):
            n_target = min(min_cells_per_class, len(df_cls))
            if oversample_dict and cls in oversample_dict:
                factor = oversample_dict[cls]
                n_target = min(len(df_cls), n_target * factor)
                print(f"[INFO] Oversampling {cls}: factor={factor}, target={n_target}")

            if marker_dict is not None and cls in marker_dict:
                scores = compute_marker_scores(adata, df_cls.index, marker_dict[cls])
                df_cls = df_cls.copy()
                df_cls["marker_score"] = scores

                n_pool = min(len(df_cls), n_target * pool_factor)
                candidates = df_cls.nlargest(n_pool, "marker_score")
                top_cells = candidates.sample(n=n_target, random_state=42).index
            else:
                top_cells = df_cls.sample(n=n_target, random_state=42).index
            sampled_idx.extend(top_cells)

    elif mode == "prop":
        N = len(adata)
        class_sizes = adata.obs[class_col].value_counts()
        weights = (class_sizes / N) ** alpha
        weights = weights / weights.sum()

        for cls, df_cls in adata.obs.groupby(class_col):
            n_target = int(weights[cls] * N * rate)
            n_target = min(n_target, len(df_cls))
            if oversample_dict and cls in oversample_dict:
                factor = oversample_dict[cls]
                n_target = min(len(df_cls), n_target * factor)
                print(f"[INFO] Oversampling {cls}: factor={factor}, target={n_target}")

            if marker_dict is not None and cls in marker_dict:
                scores = compute_marker_scores(adata, df_cls.index, marker_dict[cls])
                df_cls = df_cls.copy()
                df_cls["marker_score"] = scores
                n_pool = min(len(df_cls), n_target * pool_factor)
                candidates = df_cls.nlargest(n_pool, "marker_score")
                top_cells = candidates.sample(n=n_target, random_state=42).index
            else:
                top_cells = df_cls.sample(n=n_target, random_state=42).index
            sampled_idx.extend(top_cells)

    else:
        raise ValueError(f"Unknown sample_mode: {mode}")

    adata = adata[sampled_idx, :].copy()
    print(f"[INFO] ref: Shape after subsampling {adata.shape}")
    return adata


def plot_spatial_comparison(
    adata,
    true_label_col: str,
    pred_label_col: str,
    output_path: str,
    spot_size: int = 100,
    figsize: tuple = (18, 8),
    exclude_unassigned: bool = False
):
    """
    Plot spatial plots of true and/or predicted labels.
    - Use predefined colors for known subclasses.
    - Fallback to Scanpy default_102 palette for others.
    - If both true and predicted labels exist, plot side by side.
    """

    predefined_palette = {
        # 蓝色系（L2, RELN, VIP 等）
        "L2": "#1f77b4",  # 深蓝
        "RELN": "#4292c6",  # 中蓝
        "VIP": "#6baed6",  # 浅蓝
        "VIP_RELN": "#9ecae1",  # 最浅蓝

        # 绿色系（L2/3, L3/4/5, SST, LAMP5 等）
        "L2/3": "#2ca02c",  # 深绿
        "L2/3/4": "#4caf50",  # 中绿
        "L3": "#388e3c",  # 新增：介于 L2/3 和 L3/4
        "L3/4/5": "#66bb6a",  # 中浅绿
        "SST": "#81c784",  # 浅绿
        "LAMP5": "#a5d6a7",  # 最浅绿
        "LAMP5-RELN": "#c8e6c9",  # 放在绿色系，比 LAMP5 (#a5d6a7) 更浅

        # 紫色系（L3/4, L4, PVALB, PV_CHC 等）
        "L3/4": "#9467bd",  # 深紫
        "L4": "purple",  # 中紫
        "PVALB": "#b39ddb",  # 浅紫
        "PV": "#b39ddb",  # 浅紫
        "PV_CHC": "#c0a5e0",  # 最浅紫
        "PV-CHC": "#c0a5e0",  # 最浅紫

        # 橙色系（L4/5, L4/5/6, ASC 等）
        "L4/5": "#ff7f0e",  # 深橙
        "L4/5/6": "#ffa726",  # 中橙
        "L5": "#ffb347",  # 新增：介于 L4/5 和 L5/6
        "L5/6": "#ffcc80",  # 浅橙

        # 红色系（L5/6, VLMC 等）
        "ASC": "#e31a1c",  # 深红
        "Ast": "#e31a1c",  # 深红
        "VLMC": "#ef5350",  # 浅红

        # 黄色系（L6, OLG 等）
        "L6": "#d4ac0d",  # 深黄
        "OLG": "#ffd54f",  # 浅黄

        # 灰色系（MG, OPC, EC 等）
        "MG": "#7f7f7f",  # 深灰
        "OPC": "#a0a0a0",  # 中灰
        "EC": "#f46d43",  # 珊瑚橙红
        "unassigned": "#d0d0d0",  # 浅灰
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

    predefined_palette.update({
        k: predefined_palette[v] for k, v in mapping.items() if v in predefined_palette
    })

    def make_palette(categories, predefined):
        """Build palette: predefined first, then fallback to default_102."""
        import scanpy as sc
        base_colors = sc.pl.palettes.default_102
        palette = {}

        # Assign predefined colors
        for cat in categories:
            if cat in predefined:
                palette[cat] = predefined[cat]

        # Assign fallback colors
        unused_colors = [c for c in base_colors if c not in palette.values()]
        i = 0
        for cat in categories:
            if cat not in palette:
                palette[cat] = unused_colors[i % len(unused_colors)]
                i += 1

        # Report how many used fallback
        n_fallback = len([c for c in categories if c not in predefined])
        if n_fallback > 0:
            print(f"[INFO] {n_fallback} categories used fallback colors.")

        return palette

    # --- Make a working copy to avoid modifying the original ---
    adata = adata.copy()
    if "spatial" not in adata.obsm:
        for cols in [("spatial1", "spatial2"), ("x", "y"), ("X", "Y"), ("rx", "ry"), ("coor_x", "coor_y")]:
            if all(col in adata.obs.columns for col in cols):
                adata.obsm["spatial"] = adata.obs.loc[:, list(cols)].to_numpy(dtype=float)
                break
    elif "spatial" in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)

    if exclude_unassigned and "unassigned" in adata.obs[pred_label_col].cat.categories:
        before = adata.n_obs
        adata = adata[adata.obs[pred_label_col] != "unassigned"].copy()
        after = adata.n_obs
        print(f"[INFO] Excluded 'unassigned' cells ({before - after} removed, {after} remaining).")

    # --- Check available columns ---
    has_true = true_label_col in adata.obs.columns
    has_pred = pred_label_col in adata.obs.columns
    if not has_pred:
        print(f"[ERROR] Predicted label column '{pred_label_col}' not found. Skip plotting.")
        return

    # --- Collect categories ---
    true_cats = list(adata.obs[true_label_col].cat.categories) if has_true else []
    pred_cats = list(adata.obs[pred_label_col].cat.categories) if has_pred else []
    all_cats = sorted(set(true_cats) | set(pred_cats))

    # --- Build unified palette ---
    palette_map = make_palette(all_cats, predefined_palette)

    # --- Setup figure ---
    n_panels = 1 + int(has_true)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # --- Plot true labels ---
    if has_true:
        sc.pl.spatial(
            adata,
            color=true_label_col,
            spot_size=spot_size,
            palette=[palette_map[c] for c in adata.obs[true_label_col].cat.categories],
            ax=axes[0],
            show=False
        )
        axes[0].set_title(f'True Labels ({true_label_col})')

    # --- Plot predicted labels ---
    sc.pl.spatial(
        adata,
        color=pred_label_col,
        spot_size=spot_size,
        palette=[palette_map[c] for c in adata.obs[pred_label_col].cat.categories],
        ax=axes[-1],
        show=False
    )
    axes[-1].set_title(f'Predicted Labels ({pred_label_col})')

    # --- Add common title ---
    suffix = adata.uns.get("suffix", "")
    query_name = adata.uns.get("query_name", "")
    if suffix or query_name:
        fig.suptitle(f"Query: {query_name}\n{suffix}", fontsize=12, y=0.98)

    # --- Save figure ---
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[INFO] Saved spatial comparison to {output_path}")

    # --- Also plot confusion matrix if both labels exist ---
    if has_true and has_pred:
        cm_recall = pd.crosstab(
            adata.obs[true_label_col],
            adata.obs[pred_label_col],
            normalize="index"
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_recall, annot=False, linewidths=0.2, cmap="viridis")
        plt.title(f"Confusion Matrix\n(True={true_label_col}, Pred={pred_label_col})")
        cm_path = output_path.replace("spatial.png", "confusion.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[INFO] Saved confusion matrix to {cm_path}")

def plot_label_proportion_comparison(
    pred_df: pd.DataFrame,
    output_path: str,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    title: str = "True vs Predicted Label Proportions",
    show_values: bool = True,
    value_fmt: str = "{:.1%}",
):
    """Plot true vs predicted label proportions as a grouped bar plot."""
    pred_df = pred_df.copy()

    if pred_col not in pred_df.columns:
        raise KeyError(f"{pred_col} not found in pred_df.")

    pred_df[pred_col] = pred_df[pred_col].astype(str)

    if true_col in pred_df.columns:
        pred_df[true_col] = pred_df[true_col].astype(str)

        true_props = pred_df[true_col].value_counts(normalize=True).sort_index()
        pred_props = pred_df[pred_col].value_counts(normalize=True).sort_index()

        props_df = pd.DataFrame({
            "True": true_props,
            "Predicted": pred_props,
        }).fillna(0)

        props_df["Pred_minus_True"] = props_df["Predicted"] - props_df["True"]
        props_df = props_df.sort_values("True", ascending=False)

        plot_cols = ["True", "Predicted"]
    else:
        pred_props = pred_df[pred_col].value_counts(normalize=True).sort_index()

        props_df = pd.DataFrame({
            "Predicted": pred_props,
        }).fillna(0)

        props_df = props_df.sort_values("Predicted", ascending=False)
        plot_cols = ["Predicted"]

    props_df.index.name = "label"

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.splitext(output_path)[0] + ".csv"
    props_df.to_csv(csv_path)

    ax = props_df[plot_cols].plot(
        kind="bar",
        figsize=(12, max(7, 0.35 * len(props_df))),
        width=0.75,
    )

    ax.set_title(title)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Label")
    plt.xticks(rotation=45, ha="right")

    # Add proportion values above each bar.
    if show_values:
        max_height = 0.0

        for container in ax.containers:
            for bar in container:
                height = bar.get_height()
                max_height = max(max_height, height)

                # Skip zero-height bars to reduce visual clutter.
                if height <= 0:
                    continue

                ax.annotate(
                    value_fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        # Leave extra space for text labels.
        upper = max(0.05, max_height * 1.25)
        ax.set_ylim(0, min(1.05, upper))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Saved label proportion plot: {output_path}")
    print(f"[INFO] Saved label proportion table: {csv_path}")

    return csv_path


def dev_sum(df: pd.DataFrame):
    """
    Normalize a DataFrame by columns, dividing each value by the sum of its column.
    """
    v = df.values.copy()  # Use a copy to avoid modifying the original data
    col_sums = np.sum(v, axis=0)
    # Prevent division by zero
    v[:, col_sums > 0] /= col_sums[col_sums > 0]
    return pd.DataFrame(v, index=df.index, columns=df.columns)

def run_prediction_pipeline(
        adata: sc.AnnData,
        pretrained_model: pd.DataFrame,
        marker_gene_dict: dict,
        output_folder: str,
        true_label_col: str = 'SubClass',
        study_col: str = 'slice',
        layer_col: str = 'layer',
        pred_col_name: str = 'subclass_pre'
):
    """
    Run the full MetaNeighbor-based evaluation and visualization pipeline.

    Args:
        adata (sc.AnnData): Input AnnData object with gene expression and metadata.
        pretrained_model (pd.DataFrame): Pretrained MetaNeighbor reference model_raw.
        marker_gene_dict (dict): Marker gene dictionary for dotplot visualization.
        output_folder (str): Directory for saving outputs.
        true_label_col (str): Column name of true labels in adata.obs.
        study_col (str): Column name of study/sample in adata.obs.
        layer_col (str): Column name for spatial layer information in adata.obs.
        pred_col_name (str): Column name for predicted labels to be stored in adata.obs.
    """
    # --- 0. Preparation ---
    print(f"--- Pipeline started. Output will be saved to: {output_folder} ---")
    os.makedirs(output_folder, exist_ok=True)
    adata = adata.copy()

    # Ensure correct datatypes
    adata.obs[true_label_col] = adata.obs[true_label_col].astype("category")
    adata.obs[study_col] = adata.obs[study_col].astype("category")
    if layer_col in adata.obs:
        adata.obs[layer_col] = adata.obs[layer_col].astype("category")
    if "genenames" in adata.var.columns:
        adata.var_names = adata.var["genenames"].astype(str)
        adata.var_names_make_unique()
    elif "gene_symbol" in adata.var.columns:
        adata.var_names = adata.var["gene_symbol"].astype(str)
        adata.var_names_make_unique()

    # Remove duplicated gene symbols
    adata = adata[:, ~adata.var_names.duplicated()].copy()

    # --- 1. Run MetaNeighborUS ---
    print("--- 1. Running MetaNeighborUS to get predictions ---")
    import pymn
    pymn.MetaNeighborUS(
        adata,
        study_col=study_col,
        ct_col=pred_col_name,  # use predicted labels as input for AUROC calculation
        trained_model=pretrained_model,
        one_vs_best=True
    )

    auroc_results = adata.uns['MetaNeighborUS_1v1']
    print(f"Predictions stored in 'adata.obs[{pred_col_name}]'.")

    # AUROC heatmap
    pymn.plotMetaNeighborUS_pretrained(
        adata, cmap="coolwarm", mn_key='MetaNeighborUS_1v1',
        figsize=(10, 10), show=False
    )
    plt.savefig(os.path.join(output_folder, '0_MetaNeighborUS_AUROC_heatmap.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # --- 2. Evaluation & Visualization ---
    print("\n--- 2. Starting evaluation and visualization ---")

    # a. Marker gene dotplot
    print("Saving marker gene dotplot...")
    # check if all marker genes exist in adata.var_names
    all_genes = {g for genes in marker_gene_dict.values() for g in genes}
    missing = all_genes - set(adata.var_names)

    if missing:
        print(f"[WARN] {len(missing)} marker genes not found in adata, filtering...")
        marker_gene_dict = {
            ct: [g for g in genes if g in adata.var_names]
            for ct, genes in marker_gene_dict.items()
        }
        marker_gene_dict = {ct: genes for ct, genes in marker_gene_dict.items() if genes}

    sc.pl.dotplot(adata, marker_gene_dict, groupby=pred_col_name, use_raw=False, show=False)
    plt.savefig(os.path.join(output_folder, '1a_dotplot_predicted_labels.png'), bbox_inches='tight')
    plt.close()
    sc.pl.dotplot(adata, marker_gene_dict, groupby=true_label_col, use_raw=False, show=False)
    plt.savefig(os.path.join(output_folder, '1b_dotplot_true_labels.png'), bbox_inches='tight')
    plt.close()

    # b. Distribution across spatial layers
    if layer_col in adata.obs:
        print("Analyzing distribution in layers...")
        layer_dist = pd.crosstab(adata.obs[layer_col], adata.obs[pred_col_name])
        layer_dist_norm_row = layer_dist.div(layer_dist.sum(axis=1), axis=0)
        dfwide = dev_sum(layer_dist_norm_row)
        plt.figure(figsize=(12, 4))
        sns.heatmap(dfwide, annot=False, cmap="viridis", linewidths=0.1)
        plt.title('Distribution of Predicted Cell Types across Layers')
        plt.savefig(os.path.join(output_folder, '2_layer_distribution_heatmap.png'),
                    bbox_inches='tight')
        plt.close()

    # c. Cell type proportions
    print("Comparing cell type proportions...")
    true_props = adata.obs[true_label_col].value_counts(normalize=True).sort_index()
    pred_props = adata.obs[pred_col_name].value_counts(normalize=True).sort_index()
    props_df = pd.DataFrame({'True': true_props, 'Predicted': pred_props})
    props_df.plot(kind='bar', figsize=(12, 7), position=0.5, width=0.4)
    plt.title('Proportion of Cell Types: True vs. Predicted')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(output_folder, '3_proportion_comparison.png'), bbox_inches='tight')
    plt.close()

    # d. Confusion matrix
    print("Generating confusion matrices...")
    cm_recall = pd.crosstab(
        adata.obs[true_label_col], adata.obs[pred_col_name], normalize='index'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_recall, annot=True, fmt='.2f', cmap='viridis')

    plt.title('Confusion Matrix (Normalized by True Label -> Recall)')
    plt.savefig(os.path.join(output_folder, '4a_confusion_matrix_recall.png'), bbox_inches='tight')
    plt.close()

    # e. Spatial plots
    print("Generating spatial plots...")

    # --- Unified palette (align plot_spatial_comparison) ---
    predefined_palette = {
        "L2": "#1f77b4", "RELN": "#4292c6", "VIP": "#6baed6", "VIP_RELN": "#9ecae1",
        "L2/3": "#2ca02c", "L2/3/4": "#4caf50", "L3/4/5": "#66bb6a", "SST": "#81c784", "LAMP5": "#a5d6a7",
        "L3/4": "#9467bd", "L4": "purple", "PVALB": "#b39ddb", "PV_CHC": "#c0a5e0",
        "L4/5": "#ff7f0e", "L4/5/6": "#ffa726", "L5/6": "#ffcc80",
        "ASC": "#e31a1c", "VLMC": "#ef5350",
        "L6": "#d4ac0d", "OLG": "#ffd54f",
        "MG": "#7f7f7f", "OPC": "#a0a0a0", "EC": "#f46d43",
        "unassigned": "#d0d0d0",  # 浅灰
    }

    def make_palette(categories, predefined):
        import scanpy as sc
        base_colors = sc.pl.palettes.default_102
        palette = {}
        for cat in categories:
            if cat in predefined:
                palette[cat] = predefined[cat]
        unused_colors = [c for c in base_colors if c not in palette.values()]
        i = 0
        for cat in categories:
            if cat not in palette:
                palette[cat] = unused_colors[i % len(unused_colors)]
                i += 1
        return palette

    # unified palette for both true and predicted labels
    all_categories = sorted(set(adata.obs[true_label_col].dropna().unique()) |
                            set(adata.obs[pred_col_name].dropna().unique()))
    palette_map = make_palette(all_categories, predefined_palette)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sc.pl.spatial(
        adata,
        color=true_label_col,
        spot_size=100,
        palette=[palette_map[c] for c in adata.obs[true_label_col].cat.categories],
        ax=axes[0],
        show=False
    )
    axes[0].set_title(f'True Labels ({true_label_col})')

    sc.pl.spatial(
        adata,
        color=pred_col_name,
        spot_size=100,
        palette=[palette_map[c] for c in adata.obs[pred_col_name].cat.categories],
        ax=axes[1],
        show=False
    )
    axes[1].set_title(f'Predicted Labels ({pred_col_name})')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, '5_spatial_comparison.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

    # f. Classification metrics
    print("Calculating classification metrics...")
    report = classification_report(adata.obs[true_label_col], adata.obs[pred_col_name], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_folder, '6a_classification_report.csv'))

    ari_score = adjusted_rand_score(adata.obs[true_label_col], adata.obs[pred_col_name])
    with open(os.path.join(output_folder, '6b_ari_score.txt'), 'w') as f:
        f.write(f"Adjusted Rand Index (ARI): {ari_score:.4f}\n")

    print(f"\n--- Pipeline finished successfully. All outputs are in {output_folder} ---")
    return adata
