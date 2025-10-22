import os
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import argparse
import warnings
import seaborn as sns
from sklearn.metrics import classification_report, adjusted_rand_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from anndata import AnnData
from typing import Literal, Dict, List, Tuple, Optional
from sklearn.metrics import classification_report, adjusted_rand_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 固定配置，无需修改

from typing import Literal

import sys

from config.config_cdniche import GENE_DICT_PATH
from config.config_train_cdniche import config_train
from model.utils import get_gene_mean_path, spatial_expression_imputation, ensure_ensembl_ids
from model.pipeline.cell_embedding import run_bbcellformer_pipeline, run_bbcellformer_recon
from model.pipeline.cell_label_transfer import train_encoder_on_multi_adata, run_prediction_pipeline

from typing import Dict, List, Tuple, Optional

# homology_df = pd.read_csv("/cluster/home/yangyiwen/mart_export.humanMacaqeMarmosetMouse.oneToOneOrth.ensembl91.20220428.csv")
base_dir = "/raid/zhangchengming/BrainBeacon-master"
homology_df = pd.read_csv(os.path.join(base_dir, "prior_knowledge", "mart_export.humanMacaqeMarmosetMouse.oneToOneOrth.ensembl91.20220428.csv"))


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

def _preprocess_one_adata(
    adata: AnnData,
    info: Dict,
    n_hvg: int,
    target_species: str = "macaque",
    convert_id: bool = True,
    smooth_st: bool = True,
    smooth_k: int = 25,
    add_genes: Optional[List[str]] = None,
) -> AnnData:
    """Preprocess a single AnnData with optional smoothing, ID conversion, HVG, and forcing extra genes into HVG set."""
    print(f"[INFO] Preprocessing {info['data_name']}...")

    # 1) ensure gene name column
    if "genenames" not in adata.var.columns:
        adata.var["genenames"] = adata.var['gene_symbol']

    # 2) spatial coords
    if "spatial" not in adata.obsm and info["assay"] != "snrna":
        if 'rx' in adata.obs and 'ry' in adata.obs:
            adata.obsm["spatial"] = adata.obs[["rx", "ry"]].values
        elif 'x' in adata.obs and 'y' in adata.obs:
            adata.obsm["spatial"] = adata.obs[["x", "y"]].values
        else:
            warnings.warn(f"Spatial coordinates not found for {info['data_name']}. Skipping smoothing.")
    else:
        print(f"[INFO] Spatial coordinates already exist in {info['data_name']}.")

    if "spatial" in adata.obsm:
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=float)
        valid_idx = ~np.isnan(adata.obsm["spatial"]).any(axis=1)
        adata = adata[valid_idx].copy()

    # 3) optional spatial smoothing (before HVG)
    if smooth_st and "spatial" in adata.obsm:
        manual_spatial_smooth(adata, layer_key='smooth', n_neighbors=smooth_k)
        adata.X = adata.layers["smooth"]

    # 4) light QC
    min_gene_threshold = 0.01
    dynamic_min_genes = min(200, int(adata.n_vars * min_gene_threshold))
    dynamic_min_genes = max(dynamic_min_genes, 1)
    print(f"[INFO] Using min_genes={dynamic_min_genes} (based on {adata.n_vars} genes)")
    sc.pp.filter_cells(adata, min_genes=dynamic_min_genes)
    sc.pp.filter_genes(adata, min_cells=3)

    # helper: make HVG mask and merge with add_genes BEFORE slicing
    def _apply_hvg_with_add(
            adata_full: AnnData,
            n_top: int,
            add_genes_list: Optional[List[str]],
            prefer_target_gene: bool
    ) -> AnnData:
        tmp = adata_full.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat_v3")
        hvg_mask = tmp.var.highly_variable.copy()

        if add_genes_list:
            gene_col = "target_gene" if (prefer_target_gene and "target_gene" in tmp.var.columns) else "genenames"
            extra_mask = tmp.var[gene_col].isin(add_genes_list).fillna(False)

            if extra_mask.any():
                newly_added = int((extra_mask & ~hvg_mask).sum())
                total_hits = int(extra_mask.sum())
                keep_mask = hvg_mask | extra_mask
                print(f"[INFO] Forcing {newly_added} newly added genes "
                      f"(total hits={total_hits}, col={gene_col})")
            else:
                keep_mask = hvg_mask
                print("[WARN] None of the add_genes were found in this namespace; HVG unchanged.")
        else:
            keep_mask = hvg_mask

        return adata_full[:, keep_mask].copy()

    # 5) same-species (or no-conversion) branch
    if info["specie"] == target_species or not convert_id:
        # note: if species != target and convert_id=False, namespaces may differ -> add_genes may not match
        if info["specie"] != target_species and not convert_id and add_genes:
            warnings.warn("[WARN] convert_id=False and species differ from target; add_genes may not match current gene namespace.")

        if not convert_id:
            # Ensure Ensembl IDs for BrainBeacon input
            if not adata.var_names.str.startswith("ENS").all():
                print(f"[WARN] {info['data_name']} gene IDs not in Ensembl format, running ensure_ensembl_ids()...")
                adata = ensure_ensembl_ids(adata, species=info["specie"])

        # dedup before HVG
        adata = adata[:, ~adata.var["genenames"].duplicated()].copy()
        adata = _apply_hvg_with_add(adata, n_hvg, add_genes, prefer_target_gene=False)
        adata.obs_names_make_unique()
        return adata

    # 6) cross-species branch: map homologs first, then HVG + add_genes in target namespace
    adata = map_homologs(
        adata, homology_df,
        source_species=info['specie'], target_species=target_species,
        source_gene_col="genenames"
    )
    species_list = ["macaque", "marmoset", "human", "mouse"]
    if target_species in species_list:
        adata = ensure_ensembl_ids(adata, species=target_species)
    else:
        warnings.warn(f"Unknown species '{target_species}'. Skipping Ensembl ID conversion.")

    # dedup before HVG
    adata = adata[:, ~adata.var["genenames"].duplicated()].copy()
    adata = _apply_hvg_with_add(adata, n_hvg, add_genes, prefer_target_gene=True)
    adata.obs_names_make_unique()
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
        sns.heatmap(cm_recall, annot=True, fmt=".2f", cmap="viridis")
        plt.title(f"Confusion Matrix\n(True={true_label_col}, Pred={pred_label_col})")
        cm_path = output_path.replace("spatial.png", "confusion.png")
        plt.savefig(cm_path, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[INFO] Saved confusion matrix to {cm_path}")

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


def _subsample_adata(
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
