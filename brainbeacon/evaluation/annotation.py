# brainbeacon/evaluation/annotation.py

import os
import json
import warnings
from contextlib import nullcontext
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from brainbeacon.pipeline.cell_label_transfer import run_label_transfer

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

warnings.filterwarnings(
    "ignore",
    message="'multi_class' was deprecated.*",
    category=FutureWarning,
)

METRIC_COLS = ["Accuracy", "F1_macro", "Precision_macro", "Recall_macro"]

META_COLS = [
    "method",
    "variant",
    "transfer",
    "transfer_method",
    "transfer_params",
    "case",
    "slice",
    "label_key",
    "emb_key",
    "n_train",
    "n_test",
    "n_labels",
]

COL_ORDER = META_COLS + METRIC_COLS

SUPPORTED_VARIANTS = {"l1", "l2", "pca", "zscore"}
REUSABLE_VARIANTS = {"l1", "l2"}


def limited_thread_context(max_threads=None):
    """Limit native BLAS/OpenMP thread pools during heavy benchmark calls."""
    if max_threads is None:
        max_threads = int(os.environ.get("BRAINBEACON_BENCHMARK_THREADS", "8"))

    if threadpool_limits is None or max_threads <= 0:
        return nullcontext()

    return threadpool_limits(limits=max_threads)


def split_reusable_variants(variants):
    """Split variants into case-invariant and case-dependent groups."""
    reusable = []
    case_dependent = []

    for variant in variants:
        if variant in REUSABLE_VARIANTS:
            reusable.append(variant)
        else:
            case_dependent.append(variant)

    return reusable, case_dependent


def validate_embedding_variants(variants):
    """Validate benchmark embedding variants."""
    invalid = [v for v in variants if v not in ["", "raw"] and v not in SUPPORTED_VARIANTS]
    if invalid:
        raise ValueError(
            "Unsupported embedding variants: "
            f"{invalid}. Supported variants are raw, l1, l2, pca, zscore."
        )


def make_result_key(method, variant, transfer, label_key, emb_key):
    """Build the resume key for one benchmark unit."""
    return (
        str(method),
        str(variant),
        str(transfer),
        str(label_key),
        str(emb_key),
    )


def get_completed_result_keys(case_df):
    """Return completed model/variant/transfer/label keys from existing case results."""
    if case_df is None or case_df.empty:
        return set()

    required_cols = ["method", "variant", "transfer", "label_key", "emb_key", "slice"]
    if any(col not in case_df.columns for col in required_cols):
        return set()

    overall_df = case_df[case_df["slice"].astype(str) == "overall"].copy()
    return {
        make_result_key(
            row["method"],
            row["variant"],
            row["transfer"],
            row["label_key"],
            row["emb_key"],
        )
        for _, row in overall_df.iterrows()
    }


def merge_case_results(existing_df, new_df):
    """Merge old and new case results, preferring newly computed rows on key collisions."""
    if existing_df is None or existing_df.empty:
        return new_df.reindex(columns=COL_ORDER)
    if new_df is None or new_df.empty:
        return existing_df.reindex(columns=COL_ORDER)

    key_cols = ["method", "variant", "transfer", "case", "slice", "label_key", "emb_key"]
    combined = pd.concat([existing_df, new_df], axis=0, ignore_index=True).reindex(columns=COL_ORDER)
    combined = combined.drop_duplicates(subset=key_cols, keep="last")
    return combined.reindex(columns=COL_ORDER)


def safe_result_name(value):
    """Make a filesystem-safe result name."""
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def get_pair_csv_path(out_dir, variant, transfer):
    """Return the pair-level result CSV path for one variant/transfer pair."""
    pair_dir = os.path.join(out_dir, "results_by_pair")
    pair_name = f"{safe_result_name(variant)}__{safe_result_name(transfer)}.csv"
    return os.path.join(pair_dir, pair_name)


def validate_pair_result_complete(pair_df, expected_keys, pair_csv):
    """Validate that an existing pair CSV contains every expected model/label result."""
    required_cols = ["method", "label_key", "slice"]
    missing_cols = [col for col in required_cols if col not in pair_df.columns]
    if missing_cols:
        raise ValueError(f"Existing pair CSV is malformed: {pair_csv}, missing columns={missing_cols}")

    overall_df = pair_df[pair_df["slice"].astype(str) == "overall"]
    observed_keys = {
        (str(row["method"]), str(row["label_key"]))
        for _, row in overall_df.iterrows()
    }

    missing_keys = sorted(expected_keys - observed_keys)
    if missing_keys:
        preview = missing_keys[:20]
        raise ValueError(
            f"Existing pair CSV is incomplete: {pair_csv}. "
            f"Missing method/label overall rows={preview}"
            + (" ..." if len(missing_keys) > len(preview) else "")
        )


def add_hvg_baseline_embedding(adata, n_hvg=1000, emb_key="baseline_emb"):
    """Add HVG expression baseline embedding to adata.obsm."""
    X_raw = adata.X.copy()
    adata_tmp = adata.copy()

    sc.pp.normalize_total(adata_tmp, target_sum=1e4)
    sc.pp.log1p(adata_tmp)
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=n_hvg, flavor="seurat_v3")

    hvg_mask = adata_tmp.var["highly_variable"].values
    X_hvg = adata_tmp.X[:, hvg_mask]
    adata.obsm[emb_key] = X_hvg.toarray() if hasattr(X_hvg, "toarray") else np.asarray(X_hvg)
    adata.X = X_raw

    print(f"Baseline embedding added: {emb_key}, HVG={n_hvg}, shape={adata.obsm[emb_key].shape}")
    return adata


def load_embeddings(
    adata,
    embeddings_dir,
    dataset_name,
    models,
    rename_map=None,
    file_map=None,
    baseline=True,
    n_hvg=1000,
):
    """Load model embeddings and optional HVG baseline embedding.

    rename_map controls the folder name.
    file_map controls the filename key before '_embeddings.npz'.
    """
    if rename_map is None:
        rename_map = {}
    if file_map is None:
        file_map = {}

    if baseline:
        adata = add_hvg_baseline_embedding(adata, n_hvg=n_hvg)

    for model in models:
        model_key = rename_map.get(model, model)
        file_key = file_map.get(model, model_key)

        emb_path = os.path.join(
            embeddings_dir,
            model_key,
            f"{dataset_name}_{file_key}_embeddings.npz",
        )

        if not os.path.exists(emb_path):
            print(f"[WARN] Embedding file not found, skip {model}: {emb_path}")
            continue

        embs = np.load(emb_path)["embeddings"]

        if embs.shape[0] != adata.n_obs:
            print(
                f"[WARN] Embedding shape mismatch, skip {model}: "
                f"embedding_n={embs.shape[0]}, adata_n={adata.n_obs}"
            )
            continue

        adata.obsm[f"{model}_emb"] = embs
        print(f"Loaded {model} embedding: {embs.shape}")

    return adata


def add_embedding_variants(adata, models, variants=None, pca_dim=50, include_baseline=True):
    """Add benchmark embedding variants."""
    if variants is None:
        variants = ["pca"]
    validate_embedding_variants(variants)

    model_list = list(models)
    if include_baseline:
        model_list = ["baseline"] + model_list

    for model in model_list:
        emb_key = f"{model}_emb"
        if emb_key not in adata.obsm:
            continue

        X = adata.obsm[emb_key]

        for variant in variants:
            if variant in ["", "raw"]:
                continue

            out_key = f"{model}_{variant}_emb"
            if out_key in adata.obsm:
                continue

            if variant.startswith("pca"):
                dim_str = variant.replace("pca", "")
                dim = int(dim_str) if dim_str else pca_dim
                pca = PCA(n_components=min(dim, X.shape[1]))
                with limited_thread_context():
                    adata.obsm[out_key] = pca.fit_transform(X)

            elif variant == "l2":
                adata.obsm[out_key] = normalize(X, norm="l2")

            elif variant == "l1":
                adata.obsm[out_key] = normalize(X, norm="l1")

            elif variant == "zscore":
                scaler = StandardScaler()
                adata.obsm[out_key] = scaler.fit_transform(X)

            else:
                raise ValueError(f"Unsupported embedding variant: {variant}")

            print(f"Added {variant} variant for {model}: {adata.obsm[out_key].shape}")

    return adata


def set_split(
    adata,
    mode="random",
    test_frac=0.2,
    slice_key="slice",
    train_slices=None,
    test_slices=None,
    label_key="cell_label",
    random_state=0,
    fold_key=None,
    test_fold=None,
    filter_common_labels=True,
):
    """Set train/test split in adata.obs['split'].

    Notes
    -----
    mode="random":
        Stratified random train/test split using label_key.

    mode="slice":
        Train/test split based on train_slices and test_slices.

    mode="fold":
        Train/test split based on a precomputed fold column in adata.obs.
        Cells with adata.obs[fold_key] == test_fold are used as test;
        all remaining cells are used as train.

    filter_common_labels=True keeps the original single-label behavior:
    cells with labels not shared by train and test are removed immediately.

    For multi-label benchmarks, set filter_common_labels=False and let
    run_annotation_case filter common labels separately for each label_key.
    This avoids one label_key accidentally pre-filtering cells for another label_key.
    """
    if mode == "random":
        if label_key not in adata.obs:
            raise ValueError(f"{label_key} not found in adata.obs.")

        idx = np.arange(adata.n_obs)
        y = adata.obs[label_key].astype(str).values

        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_frac,
            random_state=random_state,
            stratify=y,
        )

        adata.obs["split"] = "unsplit"
        adata.obs.iloc[train_idx, adata.obs.columns.get_loc("split")] = "train"
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc("split")] = "test"

    elif mode == "slice":
        if slice_key not in adata.obs:
            raise ValueError(f"{slice_key} not found in adata.obs.")

        keep_slices = []
        if train_slices is not None:
            keep_slices.extend(train_slices)
        if test_slices is not None:
            keep_slices.extend(test_slices)

        if keep_slices:
            adata = adata[adata.obs[slice_key].isin(keep_slices)].copy()

        adata.obs["split"] = "train"

        if test_slices is not None:
            adata.obs.loc[adata.obs[slice_key].isin(test_slices), "split"] = "test"

        if train_slices is not None:
            adata.obs.loc[adata.obs[slice_key].isin(train_slices), "split"] = "train"

    elif mode == "fold":
        if fold_key is None:
            raise ValueError("fold_key must be provided when mode='fold'.")
        if test_fold is None:
            raise ValueError("test_fold must be provided when mode='fold'.")
        if fold_key not in adata.obs:
            raise ValueError(f"{fold_key} not found in adata.obs.")

        adata.obs["split"] = "train"
        adata.obs.loc[adata.obs[fold_key].astype(str) == str(test_fold), "split"] = "test"

    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    if filter_common_labels:
        train_labels = set(adata.obs.loc[adata.obs["split"] == "train", label_key].astype(str))
        test_labels = set(adata.obs.loc[adata.obs["split"] == "test", label_key].astype(str))
        common_labels = train_labels.intersection(test_labels)

        before_n = adata.n_obs
        adata = adata[adata.obs[label_key].astype(str).isin(common_labels)].copy()
        after_n = adata.n_obs

        if before_n != after_n:
            print(f"Filtered {before_n - after_n} cells to keep labels shared by train and test.")

    print(f"Split done. Train={(adata.obs['split'] == 'train').sum()}, Test={(adata.obs['split'] == 'test').sum()}")
    return adata


def get_emb_key(model, variant):
    """Return embedding key from model name and variant name."""
    if variant in ["", "raw"]:
        return f"{model}_emb"
    return f"{model}_{variant}_emb"


def get_variant_name(variant):
    """Normalize variant name for output tables."""
    return "raw" if variant in ["", "raw"] else variant


def default_transfer_configs():
    """Default behavior keeps the original logreg-only annotation workflow."""
    return [
        {
            "method": "logreg",
            "params": {
                "max_iter": 1000,
                "C": 1.0,
            },
        }
    ]


def build_transfer_name(method, params):
    """Build a compact transfer name for output tables and filenames."""
    if method in ["native", "hnsw", "faiss"]:
        return f"{method}_k{params.get('K', 30)}_{params.get('metric', 'euclidean')}_{params.get('weight_mode', 'uniform')}"

    if method == "prototype":
        return f"prototype_{params.get('metric', 'euclidean')}"

    if method == "logreg":
        return f"logreg_C{params.get('C', 1.0)}"

    return method


def get_transfer_kwargs(method, params):
    """Convert transfer config to keyword arguments for run_label_transfer."""
    if method in ["native", "hnsw", "faiss"]:
        kwargs = {
            "K": params.get("K", 30),
            "metric": params.get("metric", "euclidean"),
        }

        if method == "native":
            kwargs["weight_mode"] = params.get("weight_mode", "distance")
            if params.get("unassigned_threshold", None) is not None:
                kwargs["unassigned_threshold"] = params["unassigned_threshold"]

        if method == "faiss":
            kwargs["device"] = params.get("device", "cpu")

        return kwargs

    if method == "prototype":
        return {
            "metric": params.get("metric", "euclidean"),
        }

    if method == "logreg":
        return {
            "max_iter": params.get("max_iter", 1000),
            "C": params.get("C", 1.0),
        }

    raise ValueError(f"Unsupported transfer method: {method}")


def compute_metrics(y_true, y_pred):
    """Compute annotation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, labels, title, output_file, annot=False):
    """Save confusion matrix as PNG and PDF."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="d",
        cmap="Blues",
        linewidths=0.2,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def save_spatial_plot(adata, label_key, title, output_file, metrics=None, spot_size=1):
    """Save spatial plot as PNG and PDF."""
    if "spatial" in adata.obsm and not isinstance(adata.obsm["spatial"], np.ndarray):
        adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"])

    if metrics is not None:
        title = f"{title}\nAcc={metrics['Accuracy']:.2f}, F1={metrics['F1_macro']:.2f}"

    sc.pl.spatial(
        adata,
        color=label_key,
        title=title,
        spot_size=spot_size,
        show=False,
        save=None,
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()


def add_prediction_to_obs(adata, pred_col, test_indices, y_pred):
    """Store predictions in adata.obs with object dtype."""
    adata.obs[pred_col] = pd.Series(index=adata.obs.index, dtype="object")
    adata.obs.iloc[
        test_indices,
        adata.obs.columns.get_loc(pred_col),
    ] = np.asarray(y_pred).astype(str)


def _short_transfer_name(name):
    """Shorten transfer names for cleaner legends."""
    name = str(name)
    name = name.replace("prototype_cosine", "proto_cos")
    name = name.replace("prototype_euclidean", "proto_euc")
    name = name.replace("native_k10_cosine_distance", "knn10_cos")
    name = name.replace("native_k30_cosine_distance", "knn30_cos")
    name = name.replace("native_k50_cosine_distance", "knn50_cos")
    name = name.replace("native_k30_euclidean_distance", "knn30_euc")
    name = name.replace("logreg_C1.0", "logreg")
    return name


def plot_case_metric_barplot(case_df, output_dir, prefix, case_name, metric_cols=None):
    """Plot metric barplots for one case.

    If the case contains one label_key, save one case-level figure:
        {prefix}_{case_name}_metrics.png

    If the case contains multiple label_keys, save one figure per case-label:
        {prefix}_{case_name}_{label_key}_metrics.png
    """
    if metric_cols is None:
        metric_cols = METRIC_COLS

    os.makedirs(output_dir, exist_ok=True)

    overall_df = case_df[case_df["slice"] == "overall"].copy()
    if overall_df.empty:
        print(f"[WARN] No overall results available for plotting case={case_name}.")
        return

    label_keys = list(overall_df["label_key"].drop_duplicates())
    multi_label = len(label_keys) > 1

    for label_key in label_keys:
        plot_df = overall_df[overall_df["label_key"] == label_key].copy()
        if plot_df.empty:
            continue

        plot_df["display_variant"] = (
            plot_df["variant"].astype(str)
            + " | "
            + plot_df["transfer"].map(_short_transfer_name).astype(str)
        )

        group_cols = ["method", "variant", "transfer", "display_variant"]
        plot_df = (
            plot_df
            .groupby(group_cols, as_index=False)[metric_cols]
            .mean()
        )

        methods = list(plot_df["method"].drop_duplicates())
        variants = list(plot_df["display_variant"].drop_duplicates())

        n_metrics = len(metric_cols)
        n_cols = 2
        n_rows = int(np.ceil(n_metrics / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(12, len(methods) * 0.8), 4.2 * n_rows),
            squeeze=False,
        )
        axes = axes.flatten()

        handles, labels = None, None

        for i, metric in enumerate(metric_cols):
            ax = axes[i]

            sns.barplot(
                data=plot_df,
                x="method",
                y=metric,
                hue="display_variant",
                order=methods,
                hue_order=variants,
                errorbar=None,
                ax=ax,
            )

            ax.set_title(metric)
            ax.set_xlabel("")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)

            for tick in ax.get_xticklabels():
                tick.set_ha("right")

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        for j in range(n_metrics, len(axes)):
            fig.delaxes(axes[j])

        if multi_label:
            fig.suptitle(f"{prefix} | {case_name} | {label_key}", fontsize=14, y=0.98)
        else:
            fig.suptitle(f"{prefix} | {case_name}", fontsize=14, y=0.98)

        if handles is not None and labels is not None:
            fig.legend(
                handles,
                labels,
                title="Variant | Transfer",
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                fontsize=9,
                title_fontsize=10,
            )

        fig.tight_layout(rect=[0, 0, 0.82, 0.95])

        safe_case_name = str(case_name).replace("/", "_").replace(" ", "_")
        safe_label_key = str(label_key).replace("/", "_").replace(" ", "_")

        name_parts = [str(prefix)] if str(prefix) else []
        name_parts.append(safe_case_name)
        if multi_label:
            name_parts.append(safe_label_key)

        out_png = os.path.join(output_dir, f"{'_'.join(name_parts)}_metrics.png")

        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)

        print(f"Saved metric plot to {out_png}")


def run_annotation_case(
    adata,
    dataset_name,
    case_name,
    case_cfg,
    models,
    label_keys,
    selected_variants,
    out_dir,
    transfer_configs=None,
    slice_key="slice",
    pca_dim=50,
    include_baseline=True,
    save_metric_plots=True,
    save_confusion_matrices=True,
    save_spatial_plots=False,
    save_adata=False,
    save_plots=None,
    verbose=False,
    resume_existing=False,
    force=False,
):
    """Run one annotation case."""
    os.makedirs(out_dir, exist_ok=True)
    case_csv = os.path.join(out_dir, f"{case_name}_annotation_results.csv")
    legacy_case_csv = os.path.join(out_dir, f"{dataset_name}_{case_name}_annotation_results.csv")
    resume_case_csv = case_csv if os.path.exists(case_csv) else legacy_case_csv
    existing_case_df = None
    existing_pair_dfs = []
    completed_pairs = set()

    if resume_existing and not force and os.path.exists(resume_case_csv):
        if save_adata or save_spatial_plots or save_confusion_matrices:
            print(
                "[WARN] resume_existing reuses metric CSV rows only. "
                "Use force=True when regenerating predictions, spatial plots, or confusion matrices."
            )
        existing_case_df = pd.read_csv(resume_case_csv)
        print(f"Loaded legacy/case-level results for compatibility: {resume_case_csv}")

    if transfer_configs is None:
        transfer_configs = default_transfer_configs()

    if save_plots is not None:
        save_confusion_matrices = save_plots
        save_spatial_plots = save_plots

    if case_cfg.get("mode", "slice") == "slice" and slice_key in adata.obs:
        keep_slices = []
        if case_cfg.get("train_slices") is not None:
            keep_slices.extend(case_cfg.get("train_slices"))
        if case_cfg.get("test_slices") is not None:
            keep_slices.extend(case_cfg.get("test_slices"))

        if keep_slices:
            keep_slices = set(map(str, keep_slices))
            keep_mask = adata.obs[slice_key].astype(str).isin(keep_slices).values
            adata_case = adata[keep_mask].copy()
        else:
            adata_case = adata.copy()
    else:
        adata_case = adata.copy()

    adata_case = set_split(
        adata_case,
        mode=case_cfg.get("mode", "slice"),
        test_frac=case_cfg.get("test_frac", 0.2),
        slice_key=slice_key,
        train_slices=case_cfg.get("train_slices"),
        test_slices=case_cfg.get("test_slices"),
        label_key=label_keys[0],
        random_state=case_cfg.get("random_state", 0),
        fold_key=case_cfg.get("fold_key"),
        test_fold=case_cfg.get("test_fold"),
        filter_common_labels=False,
    )

    variant_to_add = [v for v in selected_variants if v not in ["", "raw"]]
    adata_case = add_embedding_variants(
        adata_case,
        models=models,
        variants=variant_to_add,
        pca_dim=pca_dim,
        include_baseline=include_baseline,
    )

    model_list = list(models)
    if include_baseline:
        model_list = ["baseline"] + model_list

    case_results = []

    train_mask = (adata_case.obs["split"] == "train").values
    test_mask = (adata_case.obs["split"] == "test").values

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print(f"[WARN] Skip {case_name}: empty train or test split.")
        return adata_case, pd.DataFrame(columns=COL_ORDER)

    valid_label_keys = []
    for label_key in label_keys:
        if label_key not in adata_case.obs:
            print(f"[WARN] Missing label_key: {label_key}, skip.")
            continue

        y_all = adata_case.obs[label_key].astype(str).values
        y_ref = y_all[train_mask]
        y_true = y_all[test_mask]
        common_labels = set(y_ref).intersection(set(y_true))

        if len(common_labels) == 0:
            print(f"[WARN] No common labels for case={case_name}, label={label_key}.")
            continue

        valid_label_keys.append(label_key)

    if len(valid_label_keys) == 0:
        print(f"[WARN] Skip {case_name}: no valid label keys.")
        return adata_case, pd.DataFrame(columns=COL_ORDER)

    for variant in selected_variants:
        variant_name = get_variant_name(variant)
        available_models = []
        for model in model_list:
            emb_key = get_emb_key(model, variant_name)
            if emb_key in adata_case.obsm:
                available_models.append(model)

        if len(available_models) == 0:
            continue

        expected_keys = {
            (str(model), str(label_key))
            for model in available_models
            for label_key in valid_label_keys
        }

        for transfer_cfg in transfer_configs:
            transfer_name = build_transfer_name(
                transfer_cfg["method"],
                transfer_cfg.get("params", {}),
            )
            pair_key = (variant_name, transfer_name)
            pair_csv = get_pair_csv_path(out_dir, variant_name, transfer_name)

            if resume_existing and not force and os.path.exists(pair_csv):
                pair_df = pd.read_csv(pair_csv)
                validate_pair_result_complete(pair_df, expected_keys, pair_csv)
                existing_pair_dfs.append(pair_df.reindex(columns=COL_ORDER))
                completed_pairs.add(pair_key)
                print(f"Loaded complete pair results: {pair_csv}")

    for model in model_list:
        for variant in selected_variants:
            print("Running model:", model)
            variant_name = get_variant_name(variant)
            emb_key = get_emb_key(model, variant_name)

            if emb_key not in adata_case.obsm:
                print(f"[WARN] Missing embedding: {emb_key}, skip.")
                continue

            X = adata_case.obsm[emb_key]
            X_ref = X[train_mask]
            X_query = X[test_mask]

            for label_key in label_keys:
                print(f"Processing label_key: {label_key}")
                if label_key not in adata_case.obs:
                    print(f"[WARN] Missing label_key: {label_key}, skip.")
                    continue

                y_all = adata_case.obs[label_key].astype(str).values
                y_ref = y_all[train_mask]
                y_true = y_all[test_mask]

                common_labels = sorted(set(y_ref).intersection(set(y_true)))
                if len(common_labels) == 0:
                    print(
                        f"[WARN] No common labels: "
                        f"case={case_name}, method={model}, variant={variant_name}, label={label_key}."
                    )
                    continue

                ref_keep = np.isin(y_ref, common_labels)
                query_keep = np.isin(y_true, common_labels)

                X_ref_use = X_ref[ref_keep]
                y_ref_use = y_ref[ref_keep]
                X_query_use = X_query[query_keep]
                y_true_use = y_true[query_keep]
                test_indices = np.where(test_mask)[0][query_keep]

                for transfer_cfg in transfer_configs:
                    transfer_method = transfer_cfg["method"]
                    transfer_params = transfer_cfg.get("params", {})
                    transfer_name = build_transfer_name(transfer_method, transfer_params)
                    transfer_kwargs = get_transfer_kwargs(transfer_method, transfer_params)
                    pair_key = (variant_name, transfer_name)

                    if pair_key in completed_pairs:
                        if verbose:
                            print(
                                f"[SKIP] Complete pair CSV exists: case={case_name} | "
                                f"variant={variant_name} | transfer={transfer_name}"
                            )
                        continue

                    if verbose:
                        print(
                            f"[RUN] case={case_name} | method={model} | variant={variant_name} | "
                            f"label={label_key} | transfer={transfer_name} | "
                            f"train={X_ref_use.shape[0]} | test={X_query_use.shape[0]}"
                        )

                    try:
                        with limited_thread_context():
                            y_pred = run_label_transfer(
                                X_ref=X_ref_use,
                                y_ref=y_ref_use,
                                X_query=X_query_use,
                                method=transfer_method,
                                **transfer_kwargs,
                            )
                    except Exception as e:
                        print(
                            f"[ERROR] Failed: case={case_name}, method={model}, variant={variant_name}, "
                            f"transfer={transfer_name}, error={repr(e)}"
                        )
                        continue

                    y_pred = np.asarray(y_pred).astype(str)
                    metrics = compute_metrics(y_true_use, y_pred)

                    row = {
                        "method": model,
                        "variant": variant_name,
                        "transfer": transfer_name,
                        "transfer_method": transfer_method,
                        "transfer_params": json.dumps(transfer_params, sort_keys=True),
                        "case": case_name,
                        "slice": "overall",
                        "label_key": label_key,
                        "emb_key": emb_key,
                        "n_train": int(X_ref_use.shape[0]),
                        "n_test": int(X_query_use.shape[0]),
                        "n_labels": int(len(common_labels)),
                    }
                    row.update(metrics)
                    case_results.append(row)

                    pred_col = f"{model}_{variant_name}_{label_key}_{transfer_name}_pred"

                    if save_adata or save_spatial_plots:
                        add_prediction_to_obs(
                            adata=adata_case,
                            pred_col=pred_col,
                            test_indices=test_indices,
                            y_pred=y_pred,
                        )

                    if save_confusion_matrices:
                        cm_dir = os.path.join(out_dir, "confusion_matrices")
                        os.makedirs(cm_dir, exist_ok=True)
                        cm_path = os.path.join(
                            cm_dir,
                            f"cm_{model}_{variant_name}_{label_key}_{transfer_name}_overall.png",
                        )
                        save_confusion_matrix(
                            y_true=y_true_use,
                            y_pred=y_pred,
                            labels=common_labels,
                            title=f"{model} {variant_name} {transfer_name}",
                            output_file=cm_path,
                            annot=False,
                        )

                    if slice_key in adata_case.obs:
                        test_obs = adata_case.obs.iloc[test_indices].copy()

                        for slice_id in sorted(test_obs[slice_key].astype(str).unique()):
                            slice_mask = test_obs[slice_key].astype(str).values == str(slice_id)
                            if slice_mask.sum() == 0:
                                continue

                            y_true_slice = y_true_use[slice_mask]
                            y_pred_slice = y_pred[slice_mask]
                            slice_metrics = compute_metrics(y_true_slice, y_pred_slice)

                            slice_row = {
                                "method": model,
                                "variant": variant_name,
                                "transfer": transfer_name,
                                "transfer_method": transfer_method,
                                "transfer_params": json.dumps(transfer_params, sort_keys=True),
                                "case": case_name,
                                "slice": slice_id,
                                "label_key": label_key,
                                "emb_key": emb_key,
                                "n_train": int(X_ref_use.shape[0]),
                                "n_test": int(slice_mask.sum()),
                                "n_labels": int(len(set(y_true_slice))),
                            }
                            slice_row.update(slice_metrics)
                            case_results.append(slice_row)

                            if save_confusion_matrices:
                                slice_cm_dir = os.path.join(out_dir, "confusion_matrices", f"slice_{slice_id}")
                                os.makedirs(slice_cm_dir, exist_ok=True)
                                cm_path = os.path.join(
                                    slice_cm_dir,
                                    f"cm_{model}_{variant_name}_{label_key}_{transfer_name}_{slice_id}.png",
                                )
                                save_confusion_matrix(
                                    y_true=y_true_slice,
                                    y_pred=y_pred_slice,
                                    labels=sorted(set(y_true_slice)),
                                    title=f"{model} {variant_name} {transfer_name} {slice_id}",
                                    output_file=cm_path,
                                    annot=False,
                                )

                            if save_spatial_plots:
                                slice_spatial_dir = os.path.join(out_dir, "spatial_plots", f"slice_{slice_id}")
                                os.makedirs(slice_spatial_dir, exist_ok=True)

                                mask_global = np.zeros(adata_case.n_obs, dtype=bool)
                                mask_global[test_indices[slice_mask]] = True

                                spatial_path = os.path.join(
                                    slice_spatial_dir,
                                    f"spatial_{model}_{variant_name}_{label_key}_{transfer_name}_{slice_id}.png",
                                )
                                save_spatial_plot(
                                    adata=adata_case[mask_global].copy(),
                                    label_key=pred_col,
                                    title=f"{model} ({variant_name}, {transfer_name})",
                                    output_file=spatial_path,
                                    metrics=slice_metrics,
                                )

                                true_path = os.path.join(slice_spatial_dir, f"spatial_true_{label_key}_{slice_id}.png")
                                if not os.path.exists(true_path):
                                    save_spatial_plot(
                                        adata=adata_case[mask_global].copy(),
                                        label_key=label_key,
                                        title=f"True {label_key}",
                                        output_file=true_path,
                                    )

    new_case_df = pd.DataFrame(case_results).reindex(columns=COL_ORDER)

    if len(new_case_df) > 0:
        pair_dir = os.path.join(out_dir, "results_by_pair")
        os.makedirs(pair_dir, exist_ok=True)

        for (variant_name, transfer_name), pair_df in new_case_df.groupby(["variant", "transfer"]):
            pair_csv = get_pair_csv_path(out_dir, variant_name, transfer_name)
            pair_df = pair_df.reindex(columns=COL_ORDER)
            pair_df.to_csv(pair_csv, index=False)
            print(f"Saved pair results to {pair_csv}")

    case_parts = existing_pair_dfs
    if len(new_case_df) > 0:
        case_parts = case_parts + [new_case_df]

    if len(case_parts) > 0:
        case_df = pd.concat(case_parts, axis=0, ignore_index=True).reindex(columns=COL_ORDER)
    elif existing_case_df is not None:
        case_df = existing_case_df.reindex(columns=COL_ORDER)
    else:
        case_df = pd.DataFrame(columns=COL_ORDER)

    case_df.to_csv(case_csv, index=False)

    if save_metric_plots:
        plot_case_metric_barplot(
            case_df=case_df,
            output_dir=out_dir,
            prefix="",
            case_name=case_name,
        )

    if save_adata:
        adata_out = os.path.join(out_dir, f"{dataset_name}_{case_name}_with_predictions.h5ad")
        adata_case.write(adata_out)
        print(f"Saved adata with predictions to {adata_out}")

    print(f"Saved case results to {case_csv}")
    return adata_case, case_df


def run_annotation_benchmark(
    adata,
    dataset_name,
    cases,
    models,
    label_keys,
    selected_variants,
    base_out_dir,
    transfer_configs=None,
    slice_key="slice",
    pca_dim=50,
    include_baseline=True,
    save_metric_plots=True,
    save_confusion_matrices=True,
    save_spatial_plots=False,
    save_adata=False,
    save_plots=None,
    verbose=False,
    precompute_variants=True,
    resume_existing=False,
    force=False,
    save_avg_results=False,
):
    """Run annotation benchmark across multiple cases."""
    os.makedirs(base_out_dir, exist_ok=True)

    if transfer_configs is None:
        transfer_configs = default_transfer_configs()

    if save_plots is not None:
        save_confusion_matrices = save_plots
        save_spatial_plots = save_plots

    validate_embedding_variants(selected_variants)

    variant_to_add = [v for v in selected_variants if v not in ["", "raw"]]
    reusable_variants, case_dependent_variants = split_reusable_variants(variant_to_add)

    if precompute_variants and len(reusable_variants) > 0:
        print(f"Precomputing reusable embedding variants once: {reusable_variants}")
        adata = add_embedding_variants(
            adata,
            models=models,
            variants=reusable_variants,
            pca_dim=pca_dim,
            include_baseline=include_baseline,
        )

    if len(case_dependent_variants) > 0:
        print(f"Case-dependent variants will be computed inside each case: {case_dependent_variants}")

    all_case_results = []

    for case_name, case_cfg in cases.items():
        print(f"\n=== Running case: {case_name} ===")
        out_dir = os.path.join(base_out_dir, case_name)

        _, case_df = run_annotation_case(
            adata=adata,
            dataset_name=dataset_name,
            case_name=case_name,
            case_cfg=case_cfg,
            models=models,
            label_keys=label_keys,
            selected_variants=selected_variants,
            out_dir=out_dir,
            transfer_configs=transfer_configs,
            slice_key=slice_key,
            pca_dim=pca_dim,
            include_baseline=include_baseline,
            save_metric_plots=save_metric_plots,
            save_confusion_matrices=save_confusion_matrices,
            save_spatial_plots=save_spatial_plots,
            save_adata=save_adata,
            verbose=verbose,
            resume_existing=resume_existing,
            force=force,
        )

        if len(case_df) > 0:
            all_case_results.append(case_df)

    if len(all_case_results) == 0:
        raise RuntimeError("No valid annotation results were generated.")

    all_results_df = pd.concat(all_case_results, axis=0, ignore_index=True).reindex(columns=COL_ORDER)

    all_results_csv = os.path.join(base_out_dir, f"{dataset_name}_all_annotation_results_with_slices.csv")
    all_results_df.to_csv(all_results_csv, index=False)

    overall_df = all_results_df[all_results_df["slice"] == "overall"].copy()
    overall_csv = os.path.join(base_out_dir, f"{dataset_name}_all_annotation_results.csv")
    overall_df.to_csv(overall_csv, index=False)

    print(f"Saved all results with slices to {all_results_csv}")
    print(f"Saved overall results to {overall_csv}")

    avg_df = None
    if save_avg_results:
        avg_df = (
            overall_df
            .groupby(["method", "variant", "transfer", "transfer_method", "label_key"], as_index=False)[METRIC_COLS]
            .mean()
        )
        avg_csv = os.path.join(base_out_dir, f"{dataset_name}_avg_annotation_across_cases.csv")
        avg_df.to_csv(avg_csv, index=False)
        print(f"Saved averaged results to {avg_csv}")

    return overall_df, avg_df
