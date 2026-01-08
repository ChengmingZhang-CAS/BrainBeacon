import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# ========== Utility Functions ==========

def load_embeddings(adata, embeddings_dir, dataset_name, models, rename_map=None, baseline=True):
    """Load embeddings into adata.obsm"""
    if baseline:
        n_hvg=1000  # number of highly variable genes for baseline
        mat = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X.copy()

        # log-normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # === HVG selection ===
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
        hvg_mask = adata.var["highly_variable"].values  # ← 注意这里加 .values
        adata.obsm["baseline_emb"] = adata.X[:, hvg_mask].toarray() if hasattr(adata.X, "toarray") else adata.X[
            :, hvg_mask]

        print(f"Baseline embedding added with HVG={n_hvg}, shape: {adata.obsm['baseline_emb'].shape}")

        # restore adata.X
        adata.X = mat

    for model in models:
        model_key = rename_map.get(model, model) if rename_map else model
        emb_path = os.path.join(embeddings_dir, model_key, f"{dataset_name}_{model_key}_embeddings.npz")
        if os.path.exists(emb_path):
            embs = np.load(emb_path)["embeddings"]
            adata.obsm[f"{model}_emb"] = embs
            print(f"Loaded {model} embedding: {embs.shape}")
        else:
            print(f"Warning: {emb_path} not found, skipping {model}")
    return adata


def add_variants(adata, models, variants=["pca", "l2", "zscore", "minmax", "zscore_pca", "pca_l2"], pca_dim=50):
    """Generate PCA / L2-normalized / Z-score / MinMax variants of embeddings"""
    for model in ["baseline"] + models:
        key = f"{model}_emb"
        if key not in adata.obsm:
            continue
        X = adata.obsm[key]

        # PCA
        for v in variants:
            if v.startswith("pca"):  # 支持 pca, pca10, pca20, ...
                parts = v.split("pca")
                dim = int(parts[1]) if len(parts) > 1 and parts[1] != "" else pca_dim
                pca = PCA(n_components=min(dim, X.shape[1]))
                adata.obsm[f"{model}_{v}_emb"] = pca.fit_transform(X)
                print(f"Added {v} variant for {model}: {adata.obsm[f'{model}_{v}_emb'].shape}")

        # L2 normalization
        if "l2" in variants:
            adata.obsm[f"{model}_l2_emb"] = normalize(X, norm="l2")
            print(f"Added L2-normalized variant for {model}: {adata.obsm[f'{model}_l2_emb'].shape}")

        # Z-score standardization
        if "zscore" in variants:
            scaler = StandardScaler()
            adata.obsm[f"{model}_zscore_emb"] = scaler.fit_transform(X)
            print(f"Added Z-score variant for {model}: {adata.obsm[f'{model}_zscore_emb'].shape}")

        # Min-Max scaling
        if "minmax" in variants:
            scaler = MinMaxScaler()
            adata.obsm[f"{model}_minmax_emb"] = scaler.fit_transform(X)
            print(f"Added MinMax variant for {model}: {adata.obsm[f'{model}_minmax_emb'].shape}")

        # Z-score + PCA
        if "zscore_pca" in variants:
            scaler = StandardScaler()
            X_z = scaler.fit_transform(X)
            pca = PCA(n_components=min(pca_dim, X_z.shape[1]))
            adata.obsm[f"{model}_zscore_pca_emb"] = pca.fit_transform(X_z)
            print(f"Added Z-score+PCA variant for {model}: {adata.obsm[f'{model}_zscore_pca_emb'].shape}")

        # PCA + L2
        if "pca_l2" in variants:
            pca = PCA(n_components=min(pca_dim, X.shape[1]))
            X_pca = pca.fit_transform(X)
            X_pca_l2 = normalize(X_pca, norm="l2")
            adata.obsm[f"{model}_pca_l2_emb"] = X_pca_l2
            print(f"Added PCA+L2 variant for {model}: {adata.obsm[f'{model}_pca_l2_emb'].shape}")

    return adata


def set_split(
    adata,
    mode="random",
    test_frac=0.2,
    slice_key="slice",
    train_slices=None,
    test_slices=None,
    label_key="cell_label"
):
    """Add a split column into adata.obs.
    - random: stratified split based on label_key
    - slice: use provided train/test slice lists
    After splitting, filter to keep only labels present in both train and test.
    """
    n = adata.n_obs

    if mode == "random":
        if label_key not in adata.obs:
            raise ValueError(f"{label_key} not in adata.obs for stratified split")

        y = adata.obs[label_key].values
        idx = np.arange(n)

        train_idx, test_idx = train_test_split(
            idx,
            test_size=test_frac,
            random_state=0,
            stratify=y
        )

        # use .iloc to avoid KeyError
        adata.obs["split"] = "unsplit"
        adata.obs.iloc[train_idx, adata.obs.columns.get_loc("split")] = "train"
        adata.obs.iloc[test_idx, adata.obs.columns.get_loc("split")] = "test"

    elif mode == "slice":
        if slice_key not in adata.obs:
            raise ValueError(f"{slice_key} not in adata.obs for slice-based split")
        # keep only specified slices
        keep_slices = []
        if train_slices is not None:
            keep_slices.extend(train_slices)
        if test_slices is not None:
            keep_slices.extend(test_slices)
        if keep_slices:  # 避免空 list
            adata = adata[adata.obs[slice_key].isin(keep_slices)].copy()

        adata.obs["split"] = "train"
        if test_slices is not None:
            adata.obs.loc[adata.obs[slice_key].isin(test_slices), "split"] = "test"
        if train_slices is not None:
            adata.obs.loc[adata.obs[slice_key].isin(train_slices), "split"] = "train"
    else:
        raise ValueError(f"Unsupported split mode: {mode}")

    # assure split contains train/test
    if label_key in adata.obs:
        train_labels = set(adata.obs.loc[adata.obs["split"] == "train", label_key])
        test_labels = set(adata.obs.loc[adata.obs["split"] == "test", label_key])
        common_labels = train_labels.intersection(test_labels)

        before_n = adata.n_obs
        adata = adata[adata.obs[label_key].isin(common_labels)].copy()
        after_n = adata.n_obs
        if before_n != after_n:
            print(f"Filtered {before_n - after_n} cells to keep only common labels between train/test.")

    print(f"Split done. Train={sum(adata.obs['split']=='train')}, Test={sum(adata.obs['split']=='test')}")
    return adata


def evaluate_logreg(adata, emb_key, label_key="cell_label", split_mask=None):
    """Train logistic regression on emb_key embedding"""
    if emb_key not in adata.obsm:
        raise KeyError(f"{emb_key} not found in adata.obsm")

    X = adata.obsm[emb_key]
    y = adata.obs[label_key].values

    if split_mask is None:
        raise ValueError("split_mask must be provided")

    # boolean mask for train/test
    train_idx = (adata.obs["split"] == "train") & split_mask
    test_idx = (adata.obs["split"] == "test") & split_mask

    if sum(test_idx) == 0 or sum(train_idx) == 0:
        return None  # skip empty slice

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # train classifier
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # compute metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1_macro": f1_score(y_test, y_pred, average="macro"),
        "Precision_macro": precision_score(y_test, y_pred, average="macro"),
        "Recall_macro": recall_score(y_test, y_pred, average="macro"),
    }

    return metrics, y_test, y_pred


def save_confusion_matrix(y_true, y_pred, method, variant, label_key, output_file):
    """Save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {method} ({variant}, {label_key})")

    # save PNG
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # save PDF
    pdf_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()


def save_spatial_plot(adata, label_key, title, output_file, metrics=None):
    """Save spatial plot (only test cells)"""
    if "spatial" in adata.obsm and not isinstance(adata.obsm["spatial"], np.ndarray):
        adata.obsm["spatial"] = np.array(adata.obsm["spatial"])

    if metrics is not None:
        title = f"{title}\nAcc={metrics['Accuracy']:.2f}, F1={metrics['F1_macro']:.2f}"

    sc.pl.spatial(
        adata,
        color=label_key,
        title=title,
        spot_size=80,
        show=False,
        save=None
    )

    # save PNG
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    # save PDF
    pdf_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_file, bbox_inches="tight")
    plt.close()


def main():
    BASE_DIR = "/raid/zhangchengming/BrainBeacon-master"
    dataset_name = "heffel2024"

    # Input raw data
    input_data_file = os.path.join(
        BASE_DIR, "data", "MERFISH_Human_Heffel2024Temporally3D", "processed",
        "Heffel2024Temporally3D.h5ad"
    )
    adata = sc.read_h5ad(input_data_file)
    print(f"Loaded AnnData: {adata.shape}")

    # Embedding dirs
    embeddings_dir = os.path.join(BASE_DIR, "downstream_tasks", "cell_clustering", "outputs", dataset_name)

    # Models
    models = ["geneformer", "cellplm", "scgpt", "nicheformer", "uce", "brainbeacon"]
    # models = ["brainbeacon"]
    rename_map = {"brainbeacon": "bbcell_cf99_epoch_0_step_800000_0.33B_hvg1000_cd0.02"}

    # Label keys
    label_keys = ["cell_label"]

    # Step 1: load embeddings
    adata = load_embeddings(adata, embeddings_dir, dataset_name, models, rename_map=rename_map, baseline=True)

    # Step 2: define experimental cases
    cases = {
        "random_split": {
            "mode": "random",
            "test_frac": 0.2
        },
        # Example of slice mode
        # "mtg18_case": {
        #     "mode": "slice",
        #     "train_slices": ["H18.06.006.MTG.4000.expand.rep1", "H18.06.006.MTG.4000.expand.rep2"],
        #     "test_slices": ["H18.06.006.MTG.4000.expand.rep3"],
        # }
    }

    # Step 3: define embedding variants
    selected_variants = ["", "pca", "l2", "zscore", "zscore_pca"]  # "" = raw, others will be added automatically
    selected_variants = ["pca", ]  # "" = raw, others will be added automatically
    # selected_variants = [""]  # "" = raw, others will be added automatically

    # Step 4: evaluation loop
    base_out_dir = os.path.join(BASE_DIR, "downstream_tasks", "cell_annotation", "outputs", dataset_name)
    os.makedirs(base_out_dir, exist_ok=True)

    all_results = []  # 收集 overall + slice-level
    overall_results = []  # 专门存 overall，用于全局 CSV

    # 列顺序
    meta_cols = ["method", "variant", "case", "slice", "label_key"]
    metric_cols = ["Accuracy", "F1_macro", "Precision_macro", "Recall_macro"]
    col_order = meta_cols + metric_cols

    for case_name, case_cfg in cases.items():
        print(f"\n=== Running case: {case_name} ===")
        adata_case = adata.copy()
        case_results = []  # 当前 case 的结果（包含 overall + slice）

        # Split train/test according to case config
        adata_case = set_split(
            adata_case,
            mode=case_cfg["mode"],
            test_frac=case_cfg.get("test_frac", 0.2),
            train_slices=case_cfg.get("train_slices"),
            test_slices=case_cfg.get("test_slices")
        )

        # Add PCA / L2 variants after split
        adata_case = add_variants(
            adata_case,
            models,
            variants=[v for v in selected_variants if v != ""],
            pca_dim=50
        )

        # Create directory for current case
        out_dir = os.path.join(base_out_dir, case_name)
        os.makedirs(out_dir, exist_ok=True)

        # Run evaluation for each model_raw and variant
        for model in ["baseline"] + models:
            for variant in selected_variants:
                suffix = "_emb" if variant == "" else f"_{variant}_emb"
                emb_key = f"{model}{suffix}"
                if emb_key not in adata_case.obsm:
                    continue

                variant_name = "raw" if variant == "" else variant

                for label_key in label_keys:
                    # === Overall evaluation ===
                    split_mask = np.ones(adata_case.n_obs, dtype=bool)
                    result = evaluate_logreg(adata_case, emb_key, label_key=label_key, split_mask=split_mask)
                    if result:
                        metrics, y_true, y_pred = result
                        metrics.update({
                            "method": model, "variant": variant_name,
                            "slice": "overall", "label_key": label_key,
                            "case": case_name
                        })
                        all_results.append(metrics)
                        case_results.append(metrics)
                        overall_results.append(metrics)  # 单独存 overall

                        # save confusion matrix
                        cm_path = os.path.join(out_dir, f"cm_{model}_{variant_name}_{label_key}_overall.png")
                        save_confusion_matrix(y_true, y_pred, model, variant_name, label_key, cm_path)

                        # add predictions to adata_case.obs
                        pred_col = f"{model}_{variant_name}_{label_key}_pred"
                        mask = adata_case.obs["split"] == "test"
                        adata_case.obs.loc[mask, pred_col] = y_pred

                    # === Slice-level evaluation & visualization ===
                    if "slice" in adata_case.obs:
                        for slice_id in adata_case.obs["slice"].unique():
                            # 只对 test 部分画图和算指标
                            mask = (adata_case.obs["split"] == "test") & (adata_case.obs["slice"] == slice_id)
                            if mask.sum() == 0:
                                continue

                            slice_dir = os.path.join(out_dir, f"slice_{slice_id}")
                            os.makedirs(slice_dir, exist_ok=True)

                            y_true_slice = adata_case.obs.loc[mask, label_key]
                            y_pred_slice = adata_case.obs.loc[mask, pred_col]

                            # === 保存 slice-level 指标 ===
                            slice_metrics = {
                                "method": model,
                                "variant": variant_name,
                                "case": case_name,
                                "slice": slice_id,
                                "label_key": label_key,
                                "Accuracy": accuracy_score(y_true_slice, y_pred_slice),
                                "F1_macro": f1_score(y_true_slice, y_pred_slice, average="macro"),
                                "Precision_macro": precision_score(y_true_slice, y_pred_slice, average="macro"),
                                "Recall_macro": recall_score(y_true_slice, y_pred_slice, average="macro"),
                            }
                            case_results.append(slice_metrics)
                            all_results.append(slice_metrics)

                            # === save confusion matrix ===
                            cm_path = os.path.join(slice_dir,
                                                   f"cm_{model}_{variant_name}_{label_key}_{slice_id}.png")
                            save_confusion_matrix(y_true_slice, y_pred_slice,
                                                  model, variant_name, label_key, cm_path)

                            # === save spatial plot ===
                            spatial_path = os.path.join(slice_dir,
                                                        f"spatial_{model}_{variant_name}_{label_key}_{slice_id}.png")
                            save_spatial_plot(
                                adata_case[mask].copy(),
                                pred_col,
                                title=f"{model} ({variant_name})",
                                metrics=slice_metrics,
                                output_file=spatial_path
                            )

                            # === save spatial plot (True label, only once) ===
                            true_spatial_path = os.path.join(slice_dir,
                                                             f"spatial_true_{label_key}_{slice_id}.png")
                            if not os.path.exists(true_spatial_path):  # 避免重复
                                save_spatial_plot(
                                    adata_case[mask].copy(),
                                    label_key,
                                    title=f"True {label_key}",
                                    metrics=None,
                                    output_file=true_spatial_path
                                )

        # Save AnnData for current case
        adata_case_path = os.path.join(out_dir, f"{dataset_name}_{case_name}_with_predictions.h5ad")
        adata_case.write(adata_case_path)
        print(f"Saved AnnData for case {case_name} to {adata_case_path}")

        # Save results for this case (overall + slice)，统一列顺序
        results_df_case = pd.DataFrame(case_results).reindex(columns=col_order)
        results_case_path = os.path.join(out_dir, f"{dataset_name}_{case_name}_logreg_results.csv")
        results_df_case.to_csv(results_case_path, index=False)
        print(f"Saved results for case {case_name} to {results_case_path}")

    # === Save combined results (只保留 overall) ===
    results_df_overall = pd.DataFrame(overall_results).reindex(columns=col_order)
    results_df_overall.to_csv(os.path.join(base_out_dir, f"{dataset_name}_all_logreg_results.csv"), index=False)
    print(f"Saved results to {os.path.join(base_out_dir, f'{dataset_name}_all_logreg_results.csv')}")

    # Save global AnnData (只含 embeddings，不含 split/variants)
    adata.write(os.path.join(base_out_dir, f"{dataset_name}_with_predictions.h5ad"))
    print("Updated AnnData saved (global).")

    # === Save averaged results across cases (基于 overall) ===
    avg_results = (
        results_df_overall
        .groupby(["method", "variant"])
        [metric_cols]
        .mean()
        .reset_index()
    )
    avg_outfile = os.path.join(base_out_dir, f"{dataset_name}_avg_across_cases.csv")
    avg_results.to_csv(avg_outfile, index=False)
    print(f"Saved averaged results across cases to {avg_outfile}")


if __name__ == "__main__":
    main()