# brainbeacon/evaluation/annotation.py

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


METRIC_COLS = ["Accuracy", "F1_macro", "Precision_macro", "Recall_macro"]
META_COLS = ["method", "variant", "case", "slice", "label_key"]
COL_ORDER = META_COLS + METRIC_COLS


def add_hvg_baseline_embedding(adata, n_hvg=1000, emb_key="baseline_emb"):
    """Add raw HVG expression baseline embedding to adata.obsm."""
    X_raw = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")

    hvg_mask = adata.var["highly_variable"].values
    X_hvg = adata.X[:, hvg_mask]
    adata.obsm[emb_key] = X_hvg.toarray() if hasattr(X_hvg, "toarray") else np.asarray(X_hvg)

    adata.X = X_raw
    print(f"Baseline embedding added: {emb_key}, HVG={n_hvg}, shape={adata.obsm[emb_key].shape}")
    return adata


def load_embeddings(adata, embeddings_dir, dataset_name, models, rename_map=None, baseline=True, n_hvg=1000):
    """Load model embeddings and optional HVG baseline embedding."""
    if baseline:
        adata = add_hvg_baseline_embedding(adata, n_hvg=n_hvg)

    for model in models:
        model_key = rename_map.get(model, model) if rename_map else model
        emb_path = os.path.join(embeddings_dir, model_key, f"{dataset_name}_{model_key}_embeddings.npz")

        if not os.path.exists(emb_path):
            print(f"Warning: {emb_path} not found, skipping {model}")
            continue

        embs = np.load(emb_path)["embeddings"]
        adata.obsm[f"{model}_emb"] = embs
        print(f"Loaded {model} embedding: {embs.shape}")

    return adata


def add_embedding_variants(adata, models, variants=None, pca_dim=50, include_baseline=True):
    """Add PCA / L2 / Z-score / MinMax variants for existing embeddings."""
    if variants is None:
        variants = ["pca"]

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

            if variant.startswith("pca"):
                dim_str = variant.replace("pca", "")
                dim = int(dim_str) if dim_str else pca_dim
                pca = PCA(n_components=min(dim, X.shape[1]))
                adata.obsm[f"{model}_{variant}_emb"] = pca.fit_transform(X)

            elif variant == "l2":
                adata.obsm[f"{model}_l2_emb"] = normalize(X, norm="l2")

            elif variant == "zscore":
                scaler = StandardScaler()
                adata.obsm[f"{model}_zscore_emb"] = scaler.fit_transform(X)

            elif variant == "minmax":
                scaler = MinMaxScaler()
                adata.obsm[f"{model}_minmax_emb"] = scaler.fit_transform(X)

            elif variant == "zscore_pca":
                scaler = StandardScaler()
                X_z = scaler.fit_transform(X)
                pca = PCA(n_components=min(pca_dim, X_z.shape[1]))
                adata.obsm[f"{model}_zscore_pca_emb"] = pca.fit_transform(X_z)

            elif variant == "pca_l2":
                pca = PCA(n_components=min(pca_dim, X.shape[1]))
                X_pca = pca.fit_transform(X)
                adata.obsm[f"{model}_pca_l2_emb"] = normalize(X_pca, norm="l2")

            else:
                raise ValueError(f"Unsupported variant: {variant}")

            print(f"Added {variant} variant for {model}: {adata.obsm[f'{model}_{variant}_emb'].shape}")

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
):
    """Set train/test split in adata.obs['split']."""
    if mode == "random":
        if label_key not in adata.obs:
            raise ValueError(f"{label_key} not in adata.obs")

        idx = np.arange(adata.n_obs)
        y = adata.obs[label_key].values

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
            raise ValueError(f"{slice_key} not in adata.obs")

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

    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    train_labels = set(adata.obs.loc[adata.obs["split"] == "train", label_key])
    test_labels = set(adata.obs.loc[adata.obs["split"] == "test", label_key])
    common_labels = train_labels.intersection(test_labels)

    before_n = adata.n_obs
    adata = adata[adata.obs[label_key].isin(common_labels)].copy()
    after_n = adata.n_obs

    if before_n != after_n:
        print(f"Filtered {before_n - after_n} cells to keep common labels.")

    print(f"Split done. Train={(adata.obs['split'] == 'train').sum()}, Test={(adata.obs['split'] == 'test').sum()}")
    return adata


def get_emb_key(model, variant):
    """Return embedding key from model and variant."""
    if variant in ["", "raw"]:
        return f"{model}_emb"
    return f"{model}_{variant}_emb"


def compute_metrics(y_true, y_pred):
    """Compute annotation metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_logreg(adata, emb_key, label_key="cell_label", split_mask=None, max_iter=1000, random_state=42):
    """Train logistic regression and return metrics, y_true, y_pred."""
    if emb_key not in adata.obsm:
        return None

    if split_mask is None:
        split_mask = np.ones(adata.n_obs, dtype=bool)

    train_idx = (adata.obs["split"] == "train").values & split_mask
    test_idx = (adata.obs["split"] == "test").values & split_mask

    if train_idx.sum() == 0 or test_idx.sum() == 0:
        return None

    X = adata.obsm[emb_key]
    y = adata.obs[label_key].values

    clf = LogisticRegression(max_iter=max_iter, random_state=random_state)
    clf.fit(X[train_idx], y[train_idx])

    y_pred = clf.predict(X[test_idx])
    y_true = y[test_idx]

    metrics = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, method, variant, label_key, output_file):
    """Save confusion matrix as PNG and PDF."""
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {method} ({variant}, {label_key})")
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


def plot_metric_barplot(results_df, output_dir, prefix, metric_cols=None):
    """Plot metric barplots across methods and cases."""
    if metric_cols is None:
        metric_cols = METRIC_COLS

    os.makedirs(output_dir, exist_ok=True)

    plot_df = results_df.copy()
    plot_df = plot_df[plot_df["slice"] == "overall"].copy()

    for metric in metric_cols:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=plot_df,
            x="case",
            y=metric,
            hue="method",
            errorbar=None,
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title(f"{metric} across cases")
        plt.tight_layout()

        out_png = os.path.join(output_dir, f"{prefix}_{metric}_by_case.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

    print(f"Saved case-level metric plots to {output_dir}")


def plot_avg_metric_barplot(avg_df, output_dir, prefix, metric_cols=None):
    """Plot averaged metric barplots across methods."""
    if metric_cols is None:
        metric_cols = METRIC_COLS

    os.makedirs(output_dir, exist_ok=True)

    for metric in metric_cols:
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=avg_df,
            x="method",
            y=metric,
            hue="variant",
            errorbar=None,
        )
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title(f"Average {metric} across cases")
        plt.tight_layout()

        out_png = os.path.join(output_dir, f"{prefix}_avg_{metric}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

    print(f"Saved averaged metric plots to {output_dir}")


def run_annotation_case(
    adata,
    dataset_name,
    case_name,
    case_cfg,
    models,
    label_keys,
    selected_variants,
    out_dir,
    slice_key="slice",
    pca_dim=50,
    include_baseline=True,
    save_plots=True,
    save_adata=True,
):
    """Run one annotation case."""
    os.makedirs(out_dir, exist_ok=True)

    adata_case = adata.copy()
    adata_case = set_split(
        adata_case,
        mode=case_cfg.get("mode", "slice"),
        test_frac=case_cfg.get("test_frac", 0.2),
        slice_key=slice_key,
        train_slices=case_cfg.get("train_slices"),
        test_slices=case_cfg.get("test_slices"),
        label_key=label_keys[0],
    )

    variant_to_add = [v for v in selected_variants if v not in ["", "raw"]]
    adata_case = add_embedding_variants(
        adata_case,
        models=models,
        variants=variant_to_add,
        pca_dim=pca_dim,
        include_baseline=include_baseline,
    )

    case_results = []
    overall_results = []

    model_list = list(models)
    if include_baseline:
        model_list = ["baseline"] + model_list

    for model in model_list:
        for variant in selected_variants:
            variant_name = "raw" if variant == "" else variant
            emb_key = get_emb_key(model, variant_name)

            if emb_key not in adata_case.obsm:
                continue

            for label_key in label_keys:
                result = evaluate_logreg(
                    adata_case,
                    emb_key=emb_key,
                    label_key=label_key,
                    split_mask=np.ones(adata_case.n_obs, dtype=bool),
                )

                if result is None:
                    continue

                metrics, y_true, y_pred = result
                metrics.update({
                    "method": model,
                    "variant": variant_name,
                    "case": case_name,
                    "slice": "overall",
                    "label_key": label_key,
                })

                case_results.append(metrics)
                overall_results.append(metrics)

                pred_col = f"{model}_{variant_name}_{label_key}_pred"
                test_mask = adata_case.obs["split"] == "test"
                adata_case.obs.loc[test_mask, pred_col] = y_pred

                if save_plots:
                    cm_path = os.path.join(out_dir, f"cm_{model}_{variant_name}_{label_key}_overall.png")
                    save_confusion_matrix(y_true, y_pred, model, variant_name, label_key, cm_path)

                if slice_key not in adata_case.obs:
                    continue

                for slice_id in adata_case.obs[slice_key].unique():
                    mask = (adata_case.obs["split"] == "test") & (adata_case.obs[slice_key] == slice_id)
                    if mask.sum() == 0:
                        continue

                    slice_dir = os.path.join(out_dir, f"slice_{slice_id}")
                    os.makedirs(slice_dir, exist_ok=True)

                    y_true_slice = adata_case.obs.loc[mask, label_key]
                    y_pred_slice = adata_case.obs.loc[mask, pred_col]

                    slice_metrics = compute_metrics(y_true_slice, y_pred_slice)
                    slice_metrics.update({
                        "method": model,
                        "variant": variant_name,
                        "case": case_name,
                        "slice": slice_id,
                        "label_key": label_key,
                    })
                    case_results.append(slice_metrics)

                    if save_plots:
                        cm_path = os.path.join(slice_dir, f"cm_{model}_{variant_name}_{label_key}_{slice_id}.png")
                        save_confusion_matrix(y_true_slice, y_pred_slice, model, variant_name, label_key, cm_path)

                        spatial_path = os.path.join(slice_dir, f"spatial_{model}_{variant_name}_{label_key}_{slice_id}.png")
                        save_spatial_plot(
                            adata_case[mask].copy(),
                            pred_col,
                            title=f"{model} ({variant_name})",
                            output_file=spatial_path,
                            metrics=slice_metrics,
                        )

                        true_path = os.path.join(slice_dir, f"spatial_true_{label_key}_{slice_id}.png")
                        if not os.path.exists(true_path):
                            save_spatial_plot(
                                adata_case[mask].copy(),
                                label_key,
                                title=f"True {label_key}",
                                output_file=true_path,
                            )

    case_df = pd.DataFrame(case_results).reindex(columns=COL_ORDER)
    case_csv = os.path.join(out_dir, f"{dataset_name}_{case_name}_logreg_results.csv")
    case_df.to_csv(case_csv, index=False)

    if save_adata:
        adata_case.write(os.path.join(out_dir, f"{dataset_name}_{case_name}_with_predictions.h5ad"))

    print(f"Saved case results to {case_csv}")
    return adata_case, case_df, pd.DataFrame(overall_results).reindex(columns=COL_ORDER)


def run_annotation_benchmark(
    adata,
    dataset_name,
    cases,
    models,
    label_keys,
    selected_variants,
    base_out_dir,
    slice_key="slice",
    pca_dim=50,
    include_baseline=True,
    save_plots=True,
    save_adata=True,
    save_metric_plots=True,
):
    """Run annotation benchmark across cases."""
    os.makedirs(base_out_dir, exist_ok=True)

    all_case_results = []
    all_overall_results = {}

    for case_name, case_cfg in cases.items():
        print(f"\n=== Running case: {case_name} ===")
        out_dir = os.path.join(base_out_dir, case_name)

        _, case_df, overall_df = run_annotation_case(
            adata=adata,
            dataset_name=dataset_name,
            case_name=case_name,
            case_cfg=case_cfg,
            models=models,
            label_keys=label_keys,
            selected_variants=selected_variants,
            out_dir=out_dir,
            slice_key=slice_key,
            pca_dim=pca_dim,
            include_baseline=include_baseline,
            save_plots=save_plots,
            save_adata=save_adata,
        )

        all_case_results.append(case_df)
        all_overall_results[case_name] = overall_df

    all_results_df = pd.concat(all_case_results, axis=0, ignore_index=True).reindex(columns=COL_ORDER)
    all_results_csv = os.path.join(base_out_dir, f"{dataset_name}_all_logreg_results_with_slices.csv")
    all_results_df.to_csv(all_results_csv, index=False)

    overall_df = pd.concat(all_overall_results.values(), axis=0, ignore_index=True).reindex(columns=COL_ORDER)
    overall_csv = os.path.join(base_out_dir, f"{dataset_name}_all_logreg_results.csv")
    overall_df.to_csv(overall_csv, index=False)

    avg_df = overall_df.groupby(["method", "variant"])[METRIC_COLS].mean().reset_index()
    avg_csv = os.path.join(base_out_dir, f"{dataset_name}_avg_across_cases.csv")
    avg_df.to_csv(avg_csv, index=False)

    if save_metric_plots:
        plot_dir = os.path.join(base_out_dir, "plots")
        plot_metric_barplot(
            overall_df,
            output_dir=plot_dir,
            prefix=dataset_name,
        )
        plot_avg_metric_barplot(
            avg_df,
            output_dir=plot_dir,
            prefix=dataset_name,
        )

    print(f"Saved all results with slices to {all_results_csv}")
    print(f"Saved overall results to {overall_csv}")
    print(f"Saved averaged results to {avg_csv}")

    return overall_df, avg_df