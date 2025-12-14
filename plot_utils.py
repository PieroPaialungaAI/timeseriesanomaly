import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import os

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")


def save_figure(fig: plt.Figure, filename: str, images_dir: str = IMAGES_DIR) -> str:
    os.makedirs(images_dir, exist_ok=True)
    filepath = os.path.join(images_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    print(f"Saved: {filepath}")
    return filepath


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

