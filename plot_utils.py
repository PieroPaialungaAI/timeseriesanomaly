import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os
from constants import IMAGES_DIR


def save_figure(fig: plt.Figure, filename: str, images_dir: str = IMAGES_DIR) -> Dict[str, str]:
    os.makedirs(images_dir, exist_ok=True)
    
    # Remove extension if provided
    base_name = os.path.splitext(filename)[0]
    
    # Save PNG
    png_path = os.path.join(images_dir, f"{base_name}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    
    # Save SVG
    svg_path = os.path.join(images_dir, f"{base_name}.svg")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    
    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    
    return {"png": png_path, "svg": svg_path}


def plot_single_ts(x: np.ndarray, y: np.ndarray, metadata: Dict = None, 
                   ax: plt.Axes = None, title: str = None, color: str = "steelblue",
                   save_as: str = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(x, y, color=color, linewidth=0.8)
    
    # Highlight anomaly regions if metadata provided
    if metadata and metadata.get("anomalies"):
        for anomaly in metadata["anomalies"]:
            if anomaly["type"] == "point_anomaly":
                idx = anomaly["anomaly_idx"]
                ax.scatter([x[idx]], [y[idx]], color="red", s=50, zorder=5, label="Point anomaly")
            elif anomaly["type"] == "level_shift":
                # Uniform level shift - just note it in title, no region to highlight
                pass
            elif anomaly["type"] == "volatility":
                # Uniform volatility - just note it in title, no region to highlight
                pass
            elif anomaly["type"] == "trend":
                # Single linear trend - just note it in title, no special visualization
                pass
    
    if title:
        ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(ax.figure, save_as)
    
    return ax


def plot_multiple_ts(x: np.ndarray, y: np.ndarray, indices: List[int], 
                     metadata: List[Dict] = None, ncols: int = 1,
                     save_as: str = None) -> plt.Figure:
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, idx in enumerate(indices):
        meta = metadata[idx] if metadata else None
        anomaly_types = [a["type"] for a in meta["anomalies"]] if meta else []
        title = f"Series {idx}: {anomaly_types}" if anomaly_types else f"Series {idx}"
        plot_single_ts(x, y[idx], meta, ax=axes[i], title=title)
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


def plot_trend_detection(x: np.ndarray, y: np.ndarray, trend_result: Dict,
                         ax: plt.Axes = None, title: str = None,
                         save_as: str = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot original series
    ax.plot(x, y, color="steelblue", linewidth=0.8, alpha=0.7, label="Original")
    
    # Plot fitted trend
    best_model = trend_result["best_model"]
    if best_model == "linear":
        predictions = trend_result["linear_result"]["predictions"]
        slope = trend_result["linear_result"]["slope"]
        label = f"Linear (slope={slope:.4f})"
    else:
        predictions = trend_result["polynomial_result"]["predictions"]
        degree = trend_result["polynomial_result"]["degree"]
        label = f"Polynomial (degree={degree})"
    
    ax.plot(x, predictions, color="red", linewidth=2, label=label)
    
    # Add title with stats
    p_val = trend_result["p_value"]
    r_sq = trend_result["r_squared"]
    has_trend = trend_result["has_trend"]
    status = "TREND DETECTED" if has_trend else "No significant trend"
    
    if title:
        full_title = f"{title}\n{status} | p={p_val:.4f} | R²={r_sq:.3f}"
    else:
        full_title = f"{status} | p={p_val:.4f} | R²={r_sq:.3f}"
    
    ax.set_title(full_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(ax.figure, save_as)
    
    return ax


def plot_multiple_trends(x: np.ndarray, y: np.ndarray, trend_results: List[Dict],
                         indices: List[int] = None, ncols: int = 1,
                         save_as: str = None) -> plt.Figure:
    if indices is None:
        indices = [r["series_idx"] for r in trend_results]
    
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    # Create lookup for results
    results_dict = {r["series_idx"]: r for r in trend_results}
    
    for i, idx in enumerate(indices):
        result = results_dict.get(idx)
        if result:
            plot_trend_detection(x, y[idx], result, ax=axes[i], title=f"Series {idx}")
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


def plot_by_anomaly_type(x: np.ndarray, y: np.ndarray, metadata: List[Dict], 
                         anomaly_type: str, num_plot: int = 5,
                         save_as: str = None) -> plt.Figure:
    # Find series with this anomaly type
    indices = []
    for meta in metadata:
        for anomaly in meta["anomalies"]:
            if anomaly["type"] == anomaly_type:
                indices.append(meta["series_idx"])
                break
    
    if not indices:
        print(f"No series found with anomaly type: {anomaly_type}")
        return None
    
    # Sample random indices
    num_plot = min(num_plot, len(indices))
    selected = np.random.choice(indices, num_plot, replace=False)
    
    fig = plot_multiple_ts(x, y, selected, metadata)
    fig.suptitle(f"Time Series with {anomaly_type}", fontsize=14, y=1.02)
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


# ============ VOLATILITY PLOTTING ============

def plot_volatility_detection(x: np.ndarray, y: np.ndarray, volatility_result: Dict,
                               ax: plt.Axes = None, title: str = None,
                               save_as: str = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot detrended series (residuals)
    residuals = volatility_result["residuals"]
    ax.plot(x, residuals, color="steelblue", linewidth=0.8)
    
    # Add horizontal lines at ±std
    vol = volatility_result["volatility"]
    ax.axhline(y=vol, color="red", linestyle="--", alpha=0.7, label=f"+1σ = {vol:.3f}")
    ax.axhline(y=-vol, color="red", linestyle="--", alpha=0.7, label=f"-1σ = {-vol:.3f}")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    
    # Fill between ±std
    ax.fill_between(x, -vol, vol, alpha=0.1, color="red")
    
    # Title with stats
    is_outlier = volatility_result.get("is_outlier", False)
    z_score = volatility_result.get("z_score", 0)
    trend_type = volatility_result.get("trend_type", "linear")
    status = "HIGH VOLATILITY" if is_outlier else "Normal volatility"
    
    if title:
        full_title = f"{title} | Detrended with {trend_type}\n{status} | σ={vol:.3f} | z={z_score:.2f}"
    else:
        full_title = f"Detrended with {trend_type}\n{status} | σ={vol:.3f} | z={z_score:.2f}"
    
    ax.set_title(full_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Detrended Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(ax.figure, save_as)
    
    return ax


def plot_multiple_volatilities(x: np.ndarray, y: np.ndarray, volatility_results: List[Dict],
                                indices: List[int] = None, ncols: int = 1,
                                save_as: str = None) -> plt.Figure:
    if indices is None:
        indices = [r["series_idx"] for r in volatility_results]
    
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    # Create lookup for results
    results_dict = {r["series_idx"]: r for r in volatility_results}
    
    for i, idx in enumerate(indices):
        result = results_dict.get(idx)
        if result:
            plot_volatility_detection(x, y[idx], result, ax=axes[i], title=f"Series {idx}")
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


def plot_volatility_distribution(volatility_results: List[Dict], stats: Dict,
                                  save_as: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    volatilities = [r["volatility"] for r in volatility_results]
    is_outlier = stats["is_outlier"]
    
    # Separate outliers and normal
    normal_vols = [v for v, o in zip(volatilities, is_outlier) if not o]
    outlier_vols = [v for v, o in zip(volatilities, is_outlier) if o]
    
    # Histogram
    bins = 30
    ax.hist(normal_vols, bins=bins, alpha=0.7, color="steelblue", label="Normal", edgecolor="white")
    if outlier_vols:
        ax.hist(outlier_vols, bins=bins, alpha=0.7, color="red", label="High Volatility", edgecolor="white")
    
    # Add threshold lines
    ax.axvline(stats["median_volatility"], color="green", linestyle="-", linewidth=2, 
               label=f"Median = {stats['median_volatility']:.3f}")
    ax.axvline(stats["percentile_threshold"], color="orange", linestyle="--", linewidth=2,
               label=f"75th pctl = {stats['percentile_threshold']:.3f}")
    
    ax.set_xlabel("Volatility (Detrended Std)")
    ax.set_ylabel("Count")
    ax.set_title(f"Volatility Distribution | {len(outlier_vols)} outliers out of {len(volatilities)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


# ============ POINT ANOMALY PLOTTING ============

def plot_point_anomaly_detection(x: np.ndarray, y: np.ndarray, point_result: Dict,
                                  ax: plt.Axes = None, title: str = None,
                                  save_as: str = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the series
    ax.plot(x, y, color="steelblue", linewidth=0.8, label="Series")
    
    # Highlight anomalous points
    anomaly_indices = point_result["anomaly_indices"]
    if len(anomaly_indices) > 0:
        ax.scatter(x[anomaly_indices], y[anomaly_indices], 
                   color="red", s=80, zorder=5, label=f"Anomalies ({len(anomaly_indices)})")
    
    # Title with stats
    num_anomalies = point_result["num_anomalies"]
    z_threshold = point_result["z_threshold"]
    status = f"{num_anomalies} POINT ANOMALIES" if num_anomalies > 0 else "No point anomalies"
    
    if title:
        full_title = f"{title}\n{status} | Z threshold = {z_threshold}"
    else:
        full_title = f"{status} | Z threshold = {z_threshold}"
    
    ax.set_title(full_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(ax.figure, save_as)
    
    return ax


def plot_multiple_point_anomalies(x: np.ndarray, y: np.ndarray, point_results: List[Dict],
                                   indices: List[int] = None, ncols: int = 1,
                                   save_as: str = None) -> plt.Figure:
    if indices is None:
        indices = [r["series_idx"] for r in point_results]
    
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    # Create lookup for results
    results_dict = {r["series_idx"]: r for r in point_results}
    
    for i, idx in enumerate(indices):
        result = results_dict.get(idx)
        if result:
            plot_point_anomaly_detection(x, y[idx], result, ax=axes[i], title=f"Series {idx}")
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


# ============ DATASET-LEVEL ANOMALY PLOTTING ============

def plot_dataset_level_detection(x: np.ndarray, y: np.ndarray, series_result: Dict,
                                  ax: plt.Axes = None, title: str = None,
                                  save_as: str = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot the series
    ax.plot(x, y, color="steelblue", linewidth=0.8, label="Series")
    
    # Plot baseline (median) as horizontal line
    baseline = series_result["baseline"]
    median_baseline = series_result["median_baseline"]
    
    ax.axhline(y=baseline, color="red", linestyle="-", linewidth=2, 
               label=f"Series baseline = {baseline:.2f}")
    ax.axhline(y=median_baseline, color="green", linestyle="--", linewidth=2,
               label=f"Dataset median = {median_baseline:.2f}")
    
    # Title with stats
    z_score = series_result["z_score"]
    is_anomaly = series_result["is_anomaly"]
    status = "DATASET-LEVEL ANOMALY" if is_anomaly else "Normal baseline"
    
    if title:
        full_title = f"{title}\n{status} | z={z_score:.2f}"
    else:
        full_title = f"{status} | z={z_score:.2f}"
    
    ax.set_title(full_title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(ax.figure, save_as)
    
    return ax


def plot_multiple_dataset_level(x: np.ndarray, y: np.ndarray, dataset_result: Dict,
                                 indices: List[int] = None, ncols: int = 1,
                                 save_as: str = None) -> plt.Figure:
    if indices is None:
        indices = list(dataset_result["anomaly_indices"])
    
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    for i, idx in enumerate(indices):
        series_result = {
            "baseline": dataset_result["baselines"][idx],
            "z_score": dataset_result["z_scores"][idx],
            "is_anomaly": dataset_result["is_anomaly"][idx],
            "median_baseline": dataset_result["median_baseline"],
            "series_idx": idx
        }
        plot_dataset_level_detection(x, y[idx], series_result, ax=axes[i], title=f"Series {idx}")
    
    # Hide unused subplots
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


def plot_baseline_distribution(dataset_result: Dict, save_as: str = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    
    baselines = dataset_result["baselines"]
    is_anomaly = dataset_result["is_anomaly"]
    
    # Separate normal and anomalous
    normal_baselines = [b for b, a in zip(baselines, is_anomaly) if not a]
    anomaly_baselines = [b for b, a in zip(baselines, is_anomaly) if a]
    
    # Histogram
    bins = 30
    ax.hist(normal_baselines, bins=bins, alpha=0.7, color="steelblue", label="Normal", edgecolor="white")
    if anomaly_baselines:
        ax.hist(anomaly_baselines, bins=bins, alpha=0.7, color="red", label="Anomalous", edgecolor="white")
    
    # Add threshold lines
    ax.axvline(dataset_result["median_baseline"], color="green", linestyle="-", linewidth=2,
               label=f"Median = {dataset_result['median_baseline']:.2f}")
    ax.axvline(dataset_result["mean_baseline"], color="orange", linestyle="--", linewidth=2,
               label=f"Mean = {dataset_result['mean_baseline']:.2f}")
    
    ax.set_xlabel("Baseline (Median of Series)")
    ax.set_ylabel("Count")
    ax.set_title(f"Baseline Distribution | {len(anomaly_baselines)} anomalies out of {len(baselines)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


# ============ ALL ANOMALIES PLOTTING ============

def plot_all_anomalies_single(x: np.ndarray, y: np.ndarray,
                               trend_result: Dict, vol_result: Dict,
                               point_result: Dict, dataset_result: Dict,
                               title: str = None, save_as: str = None) -> plt.Figure:
    """Plot a single series with all anomaly information."""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Plot series
    ax.plot(x, y, color="steelblue", linewidth=1, label="Series")
    
    # Highlight point anomalies
    if point_result and point_result["num_anomalies"] > 0:
        anomaly_indices = point_result["anomaly_indices"]
        ax.scatter(x[anomaly_indices], y[anomaly_indices], 
                   color="red", s=100, zorder=5, marker="o", label=f"Point anomalies ({len(anomaly_indices)})")
    
    # Show baseline vs dataset median
    if dataset_result:
        ax.axhline(y=dataset_result["baseline"], color="orange", linestyle="-", linewidth=1.5, alpha=0.7,
                   label=f"Baseline = {dataset_result['baseline']:.2f}")
        ax.axhline(y=dataset_result["median_baseline"], color="green", linestyle="--", linewidth=1.5, alpha=0.7,
                   label=f"Dataset median = {dataset_result['median_baseline']:.2f}")
    
    # Build status summary
    status_parts = []
    if trend_result and trend_result["has_trend"]:
        status_parts.append("TREND")
    if vol_result and vol_result.get("is_outlier", False):
        status_parts.append("HIGH VOLATILITY")
    if point_result and point_result["num_anomalies"] > 0:
        status_parts.append(f"POINT ANOMALY ({point_result['num_anomalies']})")
    if dataset_result and dataset_result["is_anomaly"]:
        status_parts.append("DATASET-LEVEL")
    
    status = " | ".join(status_parts) if status_parts else "No anomalies detected"
    
    if title:
        full_title = f"{title}\n{status}"
    else:
        full_title = status
    
    ax.set_title(full_title, fontsize=11)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig


def plot_multiple_all_anomalies(x: np.ndarray, y: np.ndarray, results: Dict,
                                 indices: List[int], ncols: int = 1,
                                 save_as: str = None) -> plt.Figure:
    """Plot multiple series showing all anomaly types."""
    num_plots = len(indices)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    
    # Create lookups
    trend_dict = {r["series_idx"]: r for r in results["all_trends"]}
    vol_dict = {r["series_idx"]: r for r in results["all_volatilities"]}
    point_dict = {r["series_idx"]: r for r in results["all_point_anomalies"]}
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Get results for this series
        trend_result = trend_dict.get(idx)
        vol_result = vol_dict.get(idx)
        point_result = point_dict.get(idx)
        dataset_result = {
            "baseline": results["dataset_level"]["baselines"][idx],
            "z_score": results["dataset_level"]["z_scores"][idx],
            "is_anomaly": results["dataset_level"]["is_anomaly"][idx],
            "median_baseline": results["dataset_level"]["median_baseline"]
        }
        
        # Plot series
        ax.plot(x, y[idx], color="steelblue", linewidth=0.8, label="Series")
        
        # Highlight point anomalies
        if point_result and point_result["num_anomalies"] > 0:
            anomaly_indices = point_result["anomaly_indices"]
            ax.scatter(x[anomaly_indices], y[idx][anomaly_indices], 
                       color="red", s=60, zorder=5, marker="o")
        
        # Build status
        status_parts = []
        if trend_result and trend_result["has_trend"]:
            status_parts.append("T")
        if vol_result and vol_result.get("is_outlier", False):
            status_parts.append("V")
        if point_result and point_result["num_anomalies"] > 0:
            status_parts.append(f"P({point_result['num_anomalies']})")
        if dataset_result["is_anomaly"]:
            status_parts.append("D")
        
        status = " | ".join(status_parts) if status_parts else "None"
        ax.set_title(f"Series {idx}: [{status}]", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    # Hide unused
    for j in range(num_plots, len(axes)):
        axes[j].set_visible(False)
    
    # Add legend explanation
    fig.text(0.5, -0.02, "Legend: T=Trend, V=High Volatility, P(n)=Point Anomalies, D=Dataset-level", 
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_as:
        save_figure(fig, save_as)
    
    return fig

