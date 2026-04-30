from typing import Optional, Sequence, Tuple, List

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# =============================================================================
# Basic helpers
# =============================================================================

def _to_numpy_1d(x, name: str) -> np.ndarray:
    """Convert input to a 1D NumPy array."""
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, but got shape {arr.shape}.")
    return arr


def _to_numpy_2d(x, name: str) -> np.ndarray:
    """Convert input to a 2D NumPy array."""
    arr = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, but got shape {arr.shape}.")
    return arr


def _ensure_parent_dir(save_path: Optional[str]):
    """Create parent directory for save_path if needed."""
    if save_path is None:
        return
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_figure(fig, save_path: Optional[str], dpi: int = 300):
    """Save figure to file and also export PDF when saving PNG."""
    if save_path is None:
        return

    _ensure_parent_dir(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    root, ext = os.path.splitext(save_path)
    if ext.lower() == ".png":
        fig.savefig(root + ".pdf", bbox_inches="tight")


def _prepare_ax(ax=None, figsize=(6, 5)):
    """Create figure/axis if ax is None."""
    if ax is not None:
        return ax.figure, ax, False
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax, True


def _build_label_to_int(labels: np.ndarray):
    """Map labels to integer ids for plotting."""
    labels = labels.astype(str)
    unique_labels = pd.unique(labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    values = np.array([label_to_int[label] for label in labels], dtype=int)
    return unique_labels, label_to_int, values


def _format_method_title(method_name: str, ari: Optional[float] = None) -> str:
    """Format plot title using method name and optional ARI."""
    if ari is None or pd.isna(ari):
        return str(method_name)
    return f"{method_name} (ARI: {ari:.3f})"


def _get_discrete_cmap(n_labels: int):
    """Return a discrete colormap for categorical labels."""
    if n_labels <= 10:
        base = cm.get_cmap("tab10", max(n_labels, 1))
    elif n_labels <= 20:
        base = cm.get_cmap("tab20", max(n_labels, 1))
    else:
        base = cm.get_cmap("gist_ncar", max(n_labels, 1))
    colors = [base(i) for i in range(max(n_labels, 1))]
    return ListedColormap(colors)


def _sanitize_filename(text: str) -> str:
    """Convert text to a safe filename fragment."""
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _infer_plot_kind(results_df: pd.DataFrame, kind: Optional[str] = None) -> str:
    """Infer plot kind automatically when not specified."""
    if kind is not None:
        if kind not in {"bar", "box"}:
            raise ValueError("kind must be 'bar' or 'box'.")
        return kind

    if "slice" in results_df.columns and results_df["slice"].nunique() > 1:
        return "box"
    if "slice_id" in results_df.columns and results_df["slice_id"].nunique() > 1:
        return "box"

    return "bar"


def _is_summary_metric(metric: str) -> bool:
    """Check whether a metric column looks like a summarized metric."""
    return str(metric).endswith("_mean")


def _get_summary_std_col(metric: str) -> Optional[str]:
    """Return paired std column for a summary metric."""
    if _is_summary_metric(metric):
        return metric[:-5] + "_std"
    return None


# =============================================================================
# Spatial and UMAP plotting
# =============================================================================

def plot_spatial(
    coord,
    labels,
    title: Optional[str] = None,
    ax=None,
    figsize=(6, 5),
    point_size: float = 8,
    alpha: float = 0.9,
    show_legend: bool = True,
    legend_loc: str = "best",
    save_path: Optional[str] = None,
):
    """
    Plot spatial coordinates colored by labels.
    """
    coord = _to_numpy_2d(coord, "coord")
    labels = _to_numpy_1d(labels, "labels").astype(str)

    if coord.shape[1] < 2:
        raise ValueError(f"coord must have at least 2 columns, but got shape {coord.shape}.")

    fig, ax, created = _prepare_ax(ax=ax, figsize=figsize)

    unique_labels, _, label_values = _build_label_to_int(labels)
    cmap = _get_discrete_cmap(len(unique_labels))

    ax.scatter(
        coord[:, 0],
        coord[:, 1],
        c=label_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
    )

    ax.set_xlabel("spatial_1")
    ax.set_ylabel("spatial_2")
    ax.set_title(title if title is not None else "Spatial Plot")
    ax.invert_yaxis()

    if show_legend:
        handles = []
        for i, label in enumerate(unique_labels):
            color = cmap(i)
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=max(4, math.sqrt(point_size)),
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=str(label),
                )
            )
        ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8)

    _save_figure(fig, save_path)

    if created:
        return fig, ax
    return fig, ax


def plot_umap(
    umap,
    labels,
    title: Optional[str] = None,
    ax=None,
    figsize=(6, 5),
    point_size: float = 8,
    alpha: float = 0.9,
    show_legend: bool = True,
    legend_loc: str = "best",
    save_path: Optional[str] = None,
):
    """
    Plot UMAP coordinates colored by labels.
    """
    umap = _to_numpy_2d(umap, "umap")
    labels = _to_numpy_1d(labels, "labels").astype(str)

    if umap.shape[1] < 2:
        raise ValueError(f"umap must have at least 2 columns, but got shape {umap.shape}.")

    fig, ax, created = _prepare_ax(ax=ax, figsize=figsize)

    unique_labels, _, label_values = _build_label_to_int(labels)
    cmap = _get_discrete_cmap(len(unique_labels))

    ax.scatter(
        umap[:, 0],
        umap[:, 1],
        c=label_values,
        cmap=cmap,
        s=point_size,
        alpha=alpha,
    )

    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title if title is not None else "UMAP Plot")

    if show_legend:
        handles = []
        for i, label in enumerate(unique_labels):
            color = cmap(i)
            handles.append(
                plt.Line2D(
                    [0], [0],
                    marker="o",
                    linestyle="",
                    markersize=max(4, math.sqrt(point_size)),
                    markerfacecolor=color,
                    markeredgecolor=color,
                    label=str(label),
                )
            )
        ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8)

    _save_figure(fig, save_path)

    if created:
        return fig, ax
    return fig, ax


def save_method_plots(
    adata,
    label_key: str,
    cluster_key: str,
    plot_dir: str,
    plot_prefix: str,
    spatial_key: str = "spatial",
    umap_key: Optional[str] = None,
    method_name: Optional[str] = None,
    ari: Optional[float] = None,
    plot_ground_truth: bool = False,
    spatial_point_size: float = 8,
    umap_point_size: float = 8,
):
    """
    Save spatial / UMAP plots for one method on one slice.
    """
    os.makedirs(plot_dir, exist_ok=True)

    if method_name is None:
        method_name = plot_prefix

    method_title = _format_method_title(method_name, ari=ari)

    if plot_ground_truth:
        if spatial_key in adata.obsm and label_key in adata.obs:
            gt_spatial_path = os.path.join(plot_dir, f"ground_truth_{label_key}_spatial.png")
            plot_spatial(
                coord=adata.obsm[spatial_key],
                labels=adata.obs[label_key].to_numpy(),
                title=f"Ground Truth: {label_key}",
                point_size=spatial_point_size,
                save_path=gt_spatial_path,
            )
            plt.close("all")

    if spatial_key in adata.obsm and cluster_key in adata.obs:
        pred_spatial_path = os.path.join(plot_dir, f"{plot_prefix}_spatial.png")
        plot_spatial(
            coord=adata.obsm[spatial_key],
            labels=adata.obs[cluster_key].to_numpy(),
            title=method_title,
            point_size=spatial_point_size,
            save_path=pred_spatial_path,
        )
        plt.close("all")

    if umap_key is not None and umap_key in adata.obsm and cluster_key in adata.obs:
        pred_umap_path = os.path.join(plot_dir, f"{plot_prefix}_umap.png")
        plot_umap(
            umap=adata.obsm[umap_key],
            labels=adata.obs[cluster_key].to_numpy(),
            title=method_title,
            point_size=umap_point_size,
            save_path=pred_umap_path,
        )
        plt.close("all")


# =============================================================================
# Metric plotting
# =============================================================================

def _check_metric_exists(results_df: pd.DataFrame, metric: str):
    """Validate metric column exists."""
    if metric not in results_df.columns:
        raise KeyError(f"{metric} not found in results_df.columns.")


def plot_metric_comparison(
    results_df: pd.DataFrame,
    metric: str,
    x: str = "method",
    kind: str = "bar",
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax=None,
    figsize=(6, 5),
    rotation: float = 45,
    save_path: Optional[str] = None,
    error_col: Optional[str] = None,
    use_errorbar: bool = True,
    capsize: float = 3.0,
):
    """
    Plot one metric comparison across groups.
    """
    if x not in results_df.columns:
        raise KeyError(f"{x} not found in results_df.columns.")
    _check_metric_exists(results_df, metric)

    fig, ax, created = _prepare_ax(ax=ax, figsize=figsize)

    base_cols = [x, metric]
    if error_col is not None and error_col in results_df.columns:
        base_cols.append(error_col)

    df = results_df[base_cols].dropna(subset=[x, metric]).copy()
    if len(df) == 0:
        raise ValueError(f"No valid rows available for metric={metric} and x={x}.")

    if kind == "bar":
        if error_col is None and use_errorbar:
            inferred_error_col = _get_summary_std_col(metric)
            if inferred_error_col in results_df.columns:
                error_col = inferred_error_col
                if error_col not in df.columns:
                    df = results_df[[x, metric, error_col]].dropna(subset=[x, metric]).copy()

        if error_col is not None and error_col in df.columns:
            plot_df = (
                df.groupby(x, sort=False, as_index=False)
                .agg(**{
                    metric: (metric, "mean"),
                    error_col: (error_col, "mean"),
                })
            )
            ax.bar(
                plot_df[x].astype(str),
                plot_df[metric].to_numpy(),
                yerr=plot_df[error_col].to_numpy() if use_errorbar else None,
                capsize=capsize if use_errorbar else 0,
            )
        else:
            grouped = df.groupby(x, sort=False)[metric].mean()
            ax.bar(grouped.index.astype(str), grouped.values)

    elif kind == "box":
        groups = []
        labels = []
        for group_name, subdf in df.groupby(x, sort=False):
            groups.append(subdf[metric].to_numpy())
            labels.append(str(group_name))
        ax.boxplot(groups, tick_labels=labels)

    else:
        raise ValueError("kind must be 'bar' or 'box'.")

    ax.set_xlabel(x)
    ax.set_ylabel(ylabel if ylabel is not None else metric)
    ax.set_title(title if title is not None else f"{metric} by {x}")
    ax.tick_params(axis="x", rotation=rotation)

    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")

    _save_figure(fig, save_path)

    if created:
        return fig, ax
    return fig, ax


def plot_metric_distribution(
    results_df: pd.DataFrame,
    metric: str,
    title: Optional[str] = None,
    bins: int = 20,
    ax=None,
    figsize=(6, 5),
    save_path: Optional[str] = None,
):
    """
    Plot histogram distribution for one metric.
    """
    _check_metric_exists(results_df, metric)

    fig, ax, created = _prepare_ax(ax=ax, figsize=figsize)

    values = results_df[metric].dropna().to_numpy()
    if len(values) == 0:
        raise ValueError(f"No valid values available for metric={metric}.")

    ax.hist(values, bins=bins)
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")
    ax.set_title(title if title is not None else f"Distribution of {metric}")

    _save_figure(fig, save_path)

    if created:
        return fig, ax
    return fig, ax


def plot_all_metric_comparisons(
    results_df: pd.DataFrame,
    metrics: Sequence[str],
    output_dir: Optional[str] = None,
    x: str = "method",
    kind: Optional[str] = None,
    rotation: float = 45,
    prefix: str = "",
    split_by: Optional[str] = "label_key",
    use_errorbar: bool = True,
    capsize: float = 3.0,
):
    """
    Plot each metric as a separate figure, optionally split by one column.
    """
    figures = []

    metrics = [metric for metric in metrics if metric in results_df.columns]
    if len(metrics) == 0:
        raise ValueError("No valid metrics found in results_df.")

    if x not in results_df.columns:
        raise KeyError(f"{x} not found in results_df.columns.")

    if split_by is not None and split_by not in results_df.columns:
        raise KeyError(f"{split_by} not found in results_df.columns.")

    if split_by is None:
        grouped_data = [("all", results_df.copy())]
    else:
        grouped_data = []
        for group_name, subdf in results_df.groupby(split_by, sort=False):
            subdf = subdf.copy()
            if len(subdf) > 0:
                grouped_data.append((str(group_name), subdf))

    for group_name, subdf in grouped_data:
        current_kind = _infer_plot_kind(subdf, kind=kind)

        for metric in metrics:
            save_path = None
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                if split_by is None:
                    filename = f"{prefix}{metric}_{current_kind}_by_{x}.png"
                else:
                    safe_group_name = _sanitize_filename(group_name)
                    filename = f"{prefix}{safe_group_name}_{metric}_{current_kind}_by_{x}.png"
                save_path = os.path.join(output_dir, filename)

            error_col = None
            if current_kind == "bar":
                inferred_error_col = _get_summary_std_col(metric)
                if inferred_error_col in subdf.columns:
                    error_col = inferred_error_col

            title = f"{metric} by {x}" if split_by is None else f"{group_name}: {metric} by {x}"

            fig, ax = plot_metric_comparison(
                results_df=subdf,
                metric=metric,
                x=x,
                kind=current_kind,
                title=title,
                rotation=rotation,
                save_path=save_path,
                error_col=error_col,
                use_errorbar=use_errorbar,
                capsize=capsize,
            )
            figures.append((group_name, metric, fig, ax))

    return figures


def plot_metric_subplots(
    results_df: pd.DataFrame,
    metrics: Sequence[str],
    x: str = "method",
    kind: Optional[str] = None,
    ncols: int = 3,
    figsize_per_panel: Tuple[float, float] = (5, 4),
    rotation: float = 45,
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None,
    split_by: Optional[str] = None,
    output_dir: Optional[str] = None,
    prefix: str = "",
    use_errorbar: bool = True,
    capsize: float = 3.0,
):
    """
    Plot multiple metrics in one subplot figure.
    """
    metrics = [m for m in metrics if m in results_df.columns]
    if len(metrics) == 0:
        raise ValueError("No valid metrics found in results_df.")

    if x not in results_df.columns:
        raise KeyError(f"{x} not found in results_df.columns.")

    if split_by is not None and split_by not in results_df.columns:
        raise KeyError(f"{split_by} not found in results_df.columns.")

    def _plot_one_subplot_figure(
        subdf: pd.DataFrame,
        current_suptitle: Optional[str],
        current_save_path: Optional[str],
    ):
        current_kind = _infer_plot_kind(subdf, kind=kind)

        n_panels = len(metrics)
        nrows = math.ceil(n_panels / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        )

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, metric in enumerate(metrics):
            error_col = None
            if current_kind == "bar":
                inferred_error_col = _get_summary_std_col(metric)
                if inferred_error_col in subdf.columns:
                    error_col = inferred_error_col

            plot_metric_comparison(
                results_df=subdf,
                metric=metric,
                x=x,
                kind=current_kind,
                rotation=rotation,
                ax=axes[i],
                save_path=None,
                error_col=error_col,
                use_errorbar=use_errorbar,
                capsize=capsize,
            )
            axes[i].set_title(metric)

        for j in range(len(metrics), len(axes)):
            axes[j].axis("off")

        if current_suptitle is not None:
            fig.suptitle(current_suptitle)

        fig.tight_layout()
        _save_figure(fig, current_save_path)
        return fig, axes

    if split_by is None:
        return _plot_one_subplot_figure(
            subdf=results_df,
            current_suptitle=suptitle,
            current_save_path=save_path,
        )

    grouped_outputs = []
    for group_name, subdf in results_df.groupby(split_by, sort=False):
        subdf = subdf.copy()
        if len(subdf) == 0:
            continue

        current_save_path = None
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            safe_group_name = _sanitize_filename(group_name)
            filename = f"{prefix}metrics_subplots_{safe_group_name}.png"
            current_save_path = os.path.join(output_dir, filename)
        elif save_path is not None:
            root, ext = os.path.splitext(save_path)
            safe_group_name = _sanitize_filename(group_name)
            current_save_path = f"{root}_{safe_group_name}{ext}"

        current_suptitle = suptitle if suptitle is not None else f"Metrics - {group_name}"
        if suptitle is not None:
            current_suptitle = f"{suptitle} - {group_name}"

        fig, axes = _plot_one_subplot_figure(
            subdf=subdf,
            current_suptitle=current_suptitle,
            current_save_path=current_save_path,
        )
        grouped_outputs.append((str(group_name), fig, axes))

    return grouped_outputs


# =============================================================================
# Default metric helpers
# =============================================================================

def get_default_metric_list(results_df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Return default raw metric list for clustering benchmark plotting.
    """
    metric_list = [
        "ari",
        "nmi",
        "asw",
        "neighbor_agreement",
        "label_entropy",
        "moran_i",
        "local_moran_i",
        "geary_c",
    ]

    if results_df is None:
        return metric_list

    return [m for m in metric_list if m in results_df.columns]


def get_summary_metric_list(results_df: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Return default summary metric list, using *_mean columns.
    """
    metric_list = [
        "ari_mean",
        "nmi_mean",
        "asw_mean",
        "neighbor_agreement_mean",
        "label_entropy_mean",
        "moran_i_mean",
        "local_moran_i_mean",
        "geary_c_mean",
    ]

    if results_df is None:
        return metric_list

    return [m for m in metric_list if m in results_df.columns]


def convert_raw_metrics_to_summary(metrics: Sequence[str]) -> List[str]:
    """
    Convert raw metric names to summary metric names by appending '_mean'.
    """
    return [f"{m}_mean" for m in metrics]


def get_available_metric_list(
    results_df: pd.DataFrame,
    prefer_summary: bool = False,
) -> List[str]:
    """
    Return available metric columns automatically.
    """
    if prefer_summary:
        summary_metrics = get_summary_metric_list(results_df)
        if len(summary_metrics) > 0:
            return summary_metrics

    raw_metrics = get_default_metric_list(results_df)
    if len(raw_metrics) > 0:
        return raw_metrics

    summary_metrics = get_summary_metric_list(results_df)
    return summary_metrics