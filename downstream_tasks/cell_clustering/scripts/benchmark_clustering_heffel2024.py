import os
import scanpy as sc

from brainbeacon.evaluation.clustering import (
    run_clustering_benchmark,
    collect_cached_method_results,
    merge_and_sort_method_results,
    summarize_benchmark_results,
    attach_embeddings_to_adata,
)
from brainbeacon.evaluation.plotting import (
    plot_all_metric_comparisons,
    plot_metric_subplots,
    get_default_metric_list,
)


def main():
    base_dir = "/raid/zhangchengming/BrainBeacon-master"
    benchmark_root = os.path.join(base_dir, "downstream_tasks", "cell_clustering", "outputs")
    data_dir = os.path.join(base_dir, "data")

    dataset_name = "heffel2024"
    input_data_file = os.path.join(data_dir, "MERFISH_Human_Heffel2024Temporally3D", "processed",
                                   "Heffel2024Temporally3D.h5ad")

    output_dir = os.path.join(benchmark_root, dataset_name)
    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    adata = sc.read_h5ad(input_data_file).copy()
    print(f"loaded adata: {adata.shape}")

    label_keys = ["cell_label"]
    methods = [
        "geneformer",
        "cellplm",
        "scgpt",
        "nicheformer",
        "uce",
        "stofm",
        "scgpt_spatial",
        "bb_abl1",
        "bb_abl1_bb",
        "bb_abl1_mean",
        "bb_abl1_mean_bb",
        # "bb_abl2",
        # "bb_abl2_bb",
        "bb_base",
    ]

    embedding_paths = {
        method: os.path.join(
            output_dir,
            method[:-3] if method.startswith("bb_abl") and method.endswith("_bb") else method,
            f"{dataset_name}_{method}_embeddings.npz",
        )
        for method in methods if method != "pca"
    }
    embedding_paths["bb_base"] = os.path.join(
        output_dir,
        "bbcell_cf99_epoch_0_step_800000_0.33B_hvg1000_cd0.02",
        f"{dataset_name}_bbcell_cf99_epoch_0_step_800000_0.33B_hvg1000_cd0.02_embeddings.npz",
    )
    method_embedding_keys = {
        method: f"X_{method}"
        for method in methods if method != "pca"
    }

    if embedding_paths:
        attach_embeddings_to_adata(
            adata=adata,
            method_npz_paths=embedding_paths,
            method_embedding_keys=method_embedding_keys,
            array_key=None,
        )

    force_rerun_methods = None
    # force_rerun_methods = ["bbcell_cf99_epoch_0_step_800000_0.33B_hvg1000_cd0.02"]

    cached_dfs, methods_to_run = collect_cached_method_results(
        methods=methods,
        results_dir=results_dir,
        force_rerun_methods=force_rerun_methods,
        verbose=True,
    )

    new_dfs = []

    if methods_to_run:
        print(f"\n===== running methods: {methods_to_run} =====")
        df_new = run_clustering_benchmark(
            adata=adata,
            label_keys=label_keys,
            methods=methods_to_run,
            method_embedding_keys=method_embedding_keys,
            spatial_key="spatial",
            slice_key=None,
            n_neighbors=15,
            graph_metric="euclidean",
            random_state=0,
            embedding_pca_dim=50,
            leiden_pca_dim=None,
            normalize_before_pca=True,
            target_sum=1e4,
            max_iterations=10,
            return_details=False,
            copy_slice=True,
            compute_umap=True,
            save_plots=True,
            output_dir=plots_dir,
            save_method_results=True,
            method_results_dir=results_dir,
            plot_ground_truth=True,
            verbose=True,
        )
        new_dfs.append(df_new)

    all_results_df = merge_and_sort_method_results(
        cached_dfs=cached_dfs,
        new_dfs=new_dfs,
        method_order=methods,
        sort_keys=["slice_id", "method", "label_key"],
    )

    results_all_path = os.path.join(output_dir, "results_all.csv")
    all_results_df.to_csv(results_all_path, index=False)
    print(f"\nsaved: {results_all_path}")

    summary_df = summarize_benchmark_results(df=all_results_df, groupby_keys=["method", "label_key"])
    summary_path = os.path.join(output_dir, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"saved: {summary_path}")

    metrics = get_default_metric_list(all_results_df)

    plot_all_metric_comparisons(results_df=all_results_df, metrics=metrics, output_dir=plots_dir, x="method", kind="bar")
    plot_metric_subplots(
        results_df=all_results_df,
        metrics=metrics,
        x="method",
        kind="bar",
        ncols=3,
        suptitle="Clustering Benchmark Metrics",
        save_path=os.path.join(plots_dir, "metrics_subplots.png"),
    )

    print("\nall done.")


if __name__ == "__main__":
    main()