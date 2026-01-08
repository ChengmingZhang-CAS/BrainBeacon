import os
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Function to save spatial plots using Scanpy only
def save_spatial_plots(adata, pred_column, method, label_key, ari_score, output_dir):
    spatial_file_path = os.path.join(output_dir, f"{method}_{label_key}_spatial.png")
    plot_spatial_scanpy(adata, pred_column, f"{method} (ARI: {ari_score:.2f})", spatial_file_path)

    umap_file_path = os.path.join(output_dir, f"{method}_{label_key}_umap.png")
    plot_umap(adata, pred_column, f"{method} (ARI: {ari_score:.2f})", umap_file_path)
    # ===== Save UMAP coordinates after plotting =====
    if "X_umap" in adata.obsm:
        umap_key = f"X_umap_{method}_{label_key}"
        adata.obsm[umap_key] = adata.obsm["X_umap"].copy()
        print(f"Saved UMAP coords to adata.obsm['{umap_key}']")

# Function to plot spatial scatter plot using Scanpy's built-in embedding
def plot_spatial_scanpy(adata, label_key, title, output_file, show_legend=True):
    if isinstance(adata.obsm["spatial"], pd.DataFrame):
        adata.obsm["spatial"] = adata.obsm["spatial"].to_numpy()

    # check nan and dtype
    if adata.obs[label_key].isna().sum() > 0:
        adata.obs[label_key].fillna("Unknown", inplace=True)
    if adata.obs[label_key].dtype != "category":
        adata.obs[label_key] = adata.obs[label_key].astype("category")

    num_categories = adata.obs[label_key].nunique()
    cmap = "Blues" if num_categories == 1 else None

    sc.pl.spatial(
        adata,
        color=label_key,
        cmap=cmap,
        legend_loc="right margin" if show_legend else "none",
        spot_size=40,
        title=title,
        show=False
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    pdf_output_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_output_file, dpi=300, bbox_inches="tight")

    plt.close()


def plot_umap(adata, label_key, title, output_file):
    if "X_umap" in adata.obsm:
        del adata.obsm["X_umap"]
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=label_key, title=title, show=False, size=30)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    pdf_output_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_output_file, dpi=300, bbox_inches="tight")

    plt.close()


def plot_umap(adata, label_key, title, output_file):
    if "X_umap" in adata.obsm:
        del adata.obsm["X_umap"]
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=label_key, title=title, show=False, size=30)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    pdf_output_file = output_file.replace(".png", ".pdf")
    plt.savefig(pdf_output_file, dpi=300, bbox_inches="tight")

    plt.close()


def perform_pca_on_embedding(embedding, pca_dim=50):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim)
    reduced_embedding = pca.fit_transform(embedding)
    print(f"Reduced embedding shape: {reduced_embedding.shape}")
    return reduced_embedding


def leiden_clustering(adata, embedding_key, n_neighbors=30, n_clusters=None, pca_dim=50, method_name="method",
                      use_rapids=True, max_iterations=10):
    embedding = adata.obsm.get(embedding_key)
    if embedding is None:
        raise KeyError(f"Embedding '{embedding_key}' not found in adata.obsm.")

    if embedding.shape[1] > pca_dim:
        print(f"Reducing dimensions from {embedding.shape[1]} to {pca_dim} using PCA...")
        embedding = perform_pca_on_embedding(embedding, pca_dim=pca_dim)

    adata.obsm["X_emb"] = embedding

    if adata.n_obs > 100000 and use_rapids:
        print(f"Using RAPIDS for neighbor graph construction due to large cell count ({adata.n_obs})...")
        sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors, method="rapids")
    else:
        sc.pp.neighbors(adata, use_rep="X_emb", n_neighbors=n_neighbors)

    if n_clusters is not None:
        print(f"\nClustering using embedding: '{embedding_key}', target clusters: {n_clusters}")
        init_res = max(0.1, n_clusters / 10.0)
        sc.tl.leiden(adata, resolution=init_res, random_state=0)
        init_clusters = adata.obs["leiden"].nunique()
        init_diff = abs(init_clusters - n_clusters)
        print(f"Initial resolution: {init_res:.4f}, initial clusters: {init_clusters}, diff={init_diff}")

        best_res, best_diff = init_res, init_diff
        best_leiden = adata.obs["leiden"].copy()
        if init_clusters < n_clusters:
            low = init_res
            high = min(10.0, init_res * 5.0)
        else:
            low = max(0.01, init_res / 5.0)
            high = init_res

        tolerance = max(1, int(np.ceil(n_clusters * 0.05)))
        same_count, last_count = 0, init_clusters

        if best_diff > tolerance or init_clusters <= 1:
            for i in range(max_iterations):
                mid = (low + high) / 2
                sc.tl.leiden(adata, resolution=mid, random_state=0)
                n = adata.obs["leiden"].nunique()
                diff = abs(n - n_clusters)
                if diff < best_diff:
                    best_res, best_diff = mid, diff
                    best_leiden = adata.obs["leiden"].copy()
                if n == last_count:
                    same_count += 1
                    if same_count >= 3:
                        print(f"Cluster count stabilized at {n} for {same_count} iterations. Early stopping.")
                        break
                else:
                    same_count = 0
                last_count = n
                if n < n_clusters:
                    low = mid
                else:
                    high = mid
                print(f"Iter {i+1}: res={mid:.4f}, clusters={n}, diff={diff}")
                if diff <= tolerance:
                    print("Found clusters within tolerance.")
                    break
        else:
            print(f"Initial clustering within tolerance ({tolerance}), skipping search.")

        adata.obs["leiden"] = best_leiden.copy()
        print(f"Optimal resolution: {best_res:.4f} with {len(best_leiden.unique())} clusters.")

    else:
        sc.tl.leiden(adata, resolution=1.0, random_state=0)
        print("No target n_clusters provided. Performed Leiden clustering with resolution=1.0.")

    adata.obs[f"leiden_{method_name}"] = adata.obs["leiden"]
    return adata.obs[f"leiden_{method_name}"]


def load_embeddings(adata, embedding_dirs, force_recompute_pca=True):
    for method, embedding_file in embedding_dirs.items():
        embedding_key = f"X_{method}"

        # rerun pca
        if method == "pca":
            if embedding_key in adata.obsm and not force_recompute_pca:
                print(f"PCA embedding already exists. Skipping.")
                continue

            print("Computing PCA embedding...")
            mat = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X.copy()
            cell_sums = mat.sum(axis=1, keepdims=True)
            mat = mat / (cell_sums + 1e-8) * 1e4  # Normalize
            mat = np.log1p(mat)  # log1p
            adata.obsm["X_pca"] = perform_pca_on_embedding(mat, pca_dim=50)

        elif embedding_key not in adata.obsm:
            if embedding_file and os.path.exists(embedding_file):
                embs = np.load(embedding_file)["embeddings"]
                adata.obsm[embedding_key] = embs
                print(f"{method} embeddings loaded with shape: {embs.shape}")
            else:
                print(f"Embedding for {method} not found or not provided. Skipping {method}.")


def process_and_evaluate(adata, method, label_key, embedding_key, n_clusters, use_rapids=True, output_dir=None, slice_id=None):
    true_labels = adata.obs[label_key]
    pred_labels = leiden_clustering(adata, embedding_key=embedding_key, n_clusters=n_clusters, method_name=method, use_rapids=use_rapids)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    silhouette = silhouette_score(adata.obsm[embedding_key], pred_labels)
    n_clusters_gt = true_labels.nunique()

    pred_column = f"{method}_{label_key}_pred"
    adata.obs[pred_column] = pred_labels

    print(f"{method} - ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {silhouette:.4f} for label {label_key}")

    # save spatial plots
    if output_dir:
        save_spatial_plots(adata, pred_column, method, label_key, ari, output_dir)

    return {"method": method, "label_key": label_key, "n_clusters_gt": n_clusters_gt,"slice": slice_id, "ARI": ari, "NMI": nmi, "Silhouette": silhouette}


def main():
    BASE_DIR = "/raid/zhangchengming/BrainBeacon-master"
    data_dir = os.path.join(BASE_DIR, "data")
    dataset_name = "heffel2024"
    input_data_file = os.path.join(data_dir, "MERFISH_Human_Heffel2024Temporally3D", "processed",
                                   "Heffel2024Temporally3D.h5ad")
    output_dir = os.path.join(BASE_DIR, "downstream_tasks", "cell_clustering", "outputs", dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    adata = sc.read_h5ad(input_data_file).copy()
    print(f"Loaded AnnData with shape: {adata.shape}")

    # embedding_dirs = {"pca": None}  # special for pca
    embedding_dirs = {}  # special for pca
    # other model_raw
    models = [
        "geneformer",
        "cellplm",
        "scgpt",
        "nicheformer",
        "uce",
        # "bbcell_epoch_6_hv_hvg1000_cd0.02",
        # "bbcellformer_epoch_6_hv_hvg1000_cd0.02",
        # "bbcellformer_epoch_0_step_800000_hvg1000_cd0.02",
        # "bbcell_cf80_epoch_0_step_800000_0.33B_hvg1000_cd0.02",
        "bbcell_cf99_epoch_0_step_800000_0.33B_hvg1000_cd0.02",
        # "bbcell_cf100_epoch_0_step_800000_0.33B_hvg1000_cd0.02",
        # "bbcell_epoch_0_step_800000_0.33B_hvg1000_cd0.02_fit10",
        # "nicheformer_enshuman",
        # "nicheformer_debug",
        # "brainbeacon",
        # "brainbeacon_full",
        # "brainbeacon_epoch_1",
        # "brainbeacon_epoch_3",
        # "brainbeacon_small_epoch_1",
        # "brainbeacon_small_epoch_1_step_10000",
        # "brainbeacon_small_epoch_0_step_100000_hybrid",
        # "brainbeacon_small_epoch_0_step_200000_hybrid",
        # "brainbeacon_epoch_1_cdandniche",
        # "brainbeacon_epoch_2_cdandniche",
        # "brainbeacon_epoch_6_cdandniche",
        # "brainbeacon_epoch_6_cdandniche_expr_cd0.01",
        # "brainbeacon_epoch_6_cdandniche_expr_cd0.02",
        # "brainbeacon_epoch_6_cdandniche_nogid_expr_cd0.02",
        # "brainbeacon_epoch_6_cdandniche_expr_cd0.05",
        # "brainbeacon_epoch_6_cdandniche_expr_cd0.1",
        # "brainbeacon_epoch_6_cdandniche_lin",
        # "brainbeacon_epoch_6_hv_expr_cd0.02",
        # "brainbeacon_epoch_6_hv_nogid_expr_cd0.02",
    ]
    for model in models:
        embedding_dirs[model] = os.path.join(output_dir, model, f"{dataset_name}_{model}_embeddings.npz")
        # Add raw embedding path with clear full prefix
        # if model_raw.startswith("bbcellformer"):
        #     raw_model_key = model_raw.replace("bbcellformer", "bbcellformer_raw", 1)
        #     raw_model_path = embedding_dirs[model_raw].replace("_embeddings.npz", "_bb_embeddings.npz")
        #     embedding_dirs[raw_model_key] = raw_model_path
        #
        # elif model_raw.startswith("bbcell"):
        #     raw_model_key = model_raw.replace("bbcell", "bbcell_raw", 1)
        #     raw_model_path = embedding_dirs[model_raw].replace("_embeddings.npz", "_bb_embeddings.npz")
        #     embedding_dirs[raw_model_key] = raw_model_path

    # Default label keys if none provided
    label_keys = ["cell_label"]

    # Load existing results only for querying, do not modify
    results_csv_file = os.path.join(output_dir, "clustering_evaluation_existingxxx.csv")
    if os.path.exists(results_csv_file):
        existing_results = pd.read_csv(results_csv_file)
    else:
        existing_results = pd.DataFrame(columns=["method", "label_key", "slice", "ARI", "NMI", "Silhouette"])

    # **Step 2: Load embeddings for the whole adata**
    load_embeddings(adata, embedding_dirs)

    results = []

    # # **Step 3: Evaluate the whole dataset for all label_keys**
    # print("\n--- Evaluating the full dataset ---")
    # for label_key in label_keys:
    #     n_clusters = adata.obs[label_key].nunique()
    #     for method in embedding_dirs.keys():
    #         if f"X_{method}" in adata.obsm:
    #             # Check if this method-label_key has already been computed in CSV
    #             existing_result_mask = (
    #                     (existing_results["method"] == method) &
    #                     (existing_results["label_key"] == label_key) &
    #                     (existing_results["slice"] == dataset_name)
    #             )
    #
    #             if existing_result_mask.any():
    #                 print(f"Using previous results for {method} on {label_key} (full dataset).")
    #                 previous_result = existing_results[existing_result_mask].iloc[0].to_dict()
    #                 results.append(previous_result)
    #             else:
    #                 results.append(process_and_evaluate(
    #                     adata, method, label_key, embedding_key=f"X_{method}", n_clusters=n_clusters,
    #                     output_dir=output_dir, slice_id=dataset_name
    #                 ))
    #
    # adata_file_path = os.path.join(output_dir, "full_results.h5ad")
    # adata.write(adata_file_path)
    # print(f"Saved combined results for the full dataset to {adata_file_path}.")

    # **Step 4: Process each slice individually for all label_keys**
    if "slice" in adata.obs:
        for slice_id in adata.obs["slice"].unique():
            print(f"\n--- Processing slice {slice_id} ---")
            slice_adata = adata[adata.obs["slice"] == slice_id].copy()
            slice_output_dir = os.path.join(output_dir, f"{slice_id}")
            os.makedirs(slice_output_dir, exist_ok=True)

            # preprocess each label key
            for label_key in label_keys:
                ground_truth_file = os.path.join(slice_output_dir, f"ground_truth_{slice_id}_{label_key}.png")
                plot_spatial_scanpy(slice_adata, label_key, f"Ground Truth ({label_key})", ground_truth_file)

                n_clusters = slice_adata.obs[label_key].nunique()
                for method in embedding_dirs.keys():
                    if f"X_{method}" in slice_adata.obsm:
                        # Check if this method-label_key-slice has already been computed in CSV
                        existing_result_mask = (
                                (existing_results["method"] == method) &
                                (existing_results["label_key"] == label_key) &
                                (existing_results["slice"] == slice_id)  # 只检查该 slice
                        )

                        if existing_result_mask.any():
                            print(f"Using previous results for {method} on {label_key} (slice: {slice_id}).")
                            previous_result = existing_results[existing_result_mask].iloc[0].to_dict()
                            results.append(previous_result)
                        else:
                            results.append(process_and_evaluate(
                                slice_adata, method, label_key, embedding_key=f"X_{method}", n_clusters=n_clusters,
                                output_dir=slice_output_dir, slice_id=slice_id
                            ))

            slice_file_path = os.path.join(slice_output_dir, f"{slice_id}_results.h5ad")
            slice_adata.write(slice_file_path)
            print(f"Saved combined results for slice {slice_id} to {slice_file_path}.")

            # Add per-slice detailed result (with label_key)
            slice_results_df = pd.DataFrame([r for r in results if r["slice"] == slice_id])
            slice_results_csv = os.path.join(slice_output_dir, f"{slice_id}_clustering_results.csv")
            slice_results_df.to_csv(slice_results_csv, index=False)
            print(f"Saved detailed clustering results for slice {slice_id} to {slice_results_csv}")

            # Plot horizontal grouped barplot for each metric
            for metric in ["ARI", "NMI", "Silhouette"]:
                plt.figure(figsize=(10, 8))

                # Pivot the data to method (row) × label_key (column)
                pivot_df = slice_results_df.pivot(index="method", columns="label_key", values=metric)
                pivot_df = pivot_df.reindex(embedding_dirs.keys())  # keep the order of methods

                ax = pivot_df.plot(kind="barh", figsize=(12, 6), width=0.75)
                plt.xlabel(metric)
                plt.ylabel("Method")
                plt.title(f"{metric} comparison across methods and labels (Slice {slice_id})")
                plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')

                # Add value labels to each bar
                for container in ax.containers:
                    for bar in container:
                        width = bar.get_width()
                        y = bar.get_y() + bar.get_height() / 2
                        ax.text(width + 0.01, y, f"{width:.2f}", va="center", ha="left", fontsize=8)

                plt.tight_layout()

                # Save both to slice folder and dataset folder before closing the plot
                plot_path_slice = os.path.join(slice_output_dir, f"{slice_id}_{metric}_grouped_barplot.png")
                plot_path_upper = os.path.join(output_dir, f"{slice_id}_{metric}_grouped_barplot.png")
                plt.savefig(plot_path_slice, dpi=300)
                plt.savefig(plot_path_upper, dpi=300)

                plt.close()
                print(f"Saved horizontal grouped barplot for {metric} to {plot_path_slice} and {plot_path_upper}")

    # **Step 5: Save clustering evaluation metrics**
    results_df = pd.DataFrame(results)
    results_csv_file = os.path.join(output_dir, "clustering_evaluation.csv")
    results_df.to_csv(results_csv_file, index=False)
    print(f"Clustering evaluation results saved to {results_csv_file}.")

    # **Compute summary statistics (excluding dataset_name)**
    summary_df = results_df[results_df["slice"] != dataset_name].groupby("method")[["ARI", "NMI", "Silhouette"]].agg(
        ["mean", "median"]).reset_index()
    summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]  # Flatten column names
    summary_df.rename(columns={"method_": "method"}, inplace=True)  # Fix method column name

    # Save summary statistics separately
    summary_csv_file = os.path.join(output_dir, "clustering_summary.csv")
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary statistics saved to {summary_csv_file}.")

    # **Step 6: Separate boxplots for each metric and label_key across different slices (excluding dataset_name)**
    metrics = ["ARI", "NMI", "Silhouette"]
    for metric in metrics:
        for label_key in label_keys:
            plt.figure(figsize=(8, 6))

            # Filter results for the current label_key, excluding 'fang2022'
            subset_df = results_df[(results_df["label_key"] == label_key) & (results_df["slice"] != dataset_name)]

            # Plot boxplot
            sns.boxplot(data=subset_df, x="method", y=metric, palette="Set2")
            plt.xticks(rotation=45, ha="right")  # Rotate method names for better readability
            plt.title(f"{metric} Distribution for {label_key}")
            plt.xlabel("Method")
            plt.ylabel(metric)

            # Save boxplot
            boxplot_file = os.path.join(output_dir, f"clustering_metrics_{metric}_{label_key}.png")
            plt.savefig(boxplot_file, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Boxplot of {metric} for {label_key} saved to {boxplot_file}.")


if __name__ == "__main__":
    main()
