import numpy as np
from typing import Tuple, List, Dict
import json
import os
import csv

def generate_base_signal(x: np.ndarray, noise_level: float, config: Dict) -> np.ndarray:
    """
    Generate a base sinusoidal signal with noise.
    
    Args:
        x: Time axis array
        noise_level: Standard deviation of Gaussian noise
        config: Base signal config dict
    
    Returns:
        Base signal array
    """
    frequency = np.random.uniform(config["frequency_min"], config["frequency_max"])
    phase = np.random.uniform(0, 2 * np.pi)
    amplitude = np.random.uniform(config["amplitude_min"], config["amplitude_max"])
    
    signal = amplitude * np.sin(frequency * x + phase)
    noise = np.random.normal(0, noise_level, len(x))
    
    return signal + noise


def add_trend(signal: np.ndarray, x: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Add a polynomial trend (linear or higher degree) from start to end.
    
    Args:
        signal: Input signal array
        x: Time axis array
        config: Trend config dict with max_degree, magnitude_min, magnitude_max
    
    Returns:
        Signal with trend added, metadata dict
    """
    n = len(signal)
    
    # Randomly choose degree (1 = linear, 2 = quadratic, 3 = cubic, etc.)
    degree = np.random.randint(1, config["max_degree"] + 1)
    
    # Normalize x to [0, 1] range
    x_norm = np.linspace(0, 1, n)
    
    # Total magnitude of trend at the end (relative to signal std)
    signal_std = np.std(signal)
    direction = np.random.choice([-1, 1])
    magnitude = direction * np.random.uniform(config["magnitude_min"], config["magnitude_max"]) * signal_std
    
    # Generate polynomial trend
    if degree == 1:
        # Linear: goes from 0 to magnitude
        trend = magnitude * x_norm
        coefficients = [magnitude]
    elif degree == 2:
        # Quadratic: curved, reaches magnitude at end
        trend = magnitude * (x_norm ** 2)
        coefficients = [magnitude]
    else:
        # Cubic: S-curve-ish, reaches magnitude at end
        trend = magnitude * (x_norm ** 3)
        coefficients = [magnitude]
    
    metadata = {
        "type": "trend",
        "degree": degree,
        "magnitude": magnitude,
        "coefficients": coefficients
    }
    
    return signal + trend, metadata


def add_volatility(signal: np.ndarray, x: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Add uniform high volatility across the entire time series.
    
    Args:
        signal: Input signal array
        x: Time axis array
        config: Volatility config dict
    
    Returns:
        Signal with increased volatility, metadata dict
    """
    n = len(signal)
    result = signal.copy()
    
    volatility_multiplier = np.random.uniform(config["multiplier_min"], config["multiplier_max"])
    
    # Add extra noise across the entire series
    extra_noise = np.random.normal(0, config["base_noise"] * volatility_multiplier, n)
    result += extra_noise
    
    metadata = {
        "type": "volatility",
        "multiplier": volatility_multiplier
    }
    
    return result, metadata


def add_point_anomaly(signal: np.ndarray, x: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Add a single point anomaly (spike).
    
    Args:
        signal: Input signal array
        x: Time axis array
        config: Point anomaly config dict
    
    Returns:
        Signal with point anomaly, metadata dict
    """
    n = len(signal)
    result = signal.copy()
    
    # Random location for the spike (avoid edges)
    margin = int(n * config["margin_frac"])
    anomaly_idx = np.random.randint(margin, n - margin)
    
    # Spike magnitude (significant deviation)
    signal_std = np.std(signal)
    magnitude_mult = np.random.uniform(config["magnitude_min"], config["magnitude_max"])
    spike_magnitude = np.random.choice([-1, 1]) * magnitude_mult * signal_std
    
    result[anomaly_idx] += spike_magnitude
    
    metadata = {
        "type": "point_anomaly",
        "anomaly_idx": anomaly_idx,
        "magnitude": spike_magnitude
    }
    
    return result, metadata


def add_level_shift(signal: np.ndarray, x: np.ndarray, config: Dict) -> Tuple[np.ndarray, Dict]:
    """
    Add a uniform level shift across the entire time series (higher or lower than normal).
    
    Args:
        signal: Input signal array
        x: Time axis array
        config: Level shift config dict
    
    Returns:
        Signal with level shift, metadata dict
    """
    result = signal.copy()
    
    # Shift magnitude (can be positive or negative)
    signal_std = np.std(signal)
    shift_magnitude = np.random.choice([-1, 1]) * np.random.uniform(config["magnitude_min"], config["magnitude_max"]) * signal_std
    
    result += shift_magnitude
    
    metadata = {
        "type": "level_shift",
        "magnitude": shift_magnitude
    }
    
    return result, metadata


def assign_anomaly_types(num_series: int, max_anomalies_per_series: int = 2) -> List[List[str]]:
    """
    Assign anomaly types to each series. Every series gets at least one anomaly.
    25% distribution for each type, can overlap up to max_anomalies_per_series.
    
    Args:
        num_series: Total number of series
        max_anomalies_per_series: Maximum anomaly types per series
    
    Returns:
        List of anomaly type lists for each series
    """
    anomaly_types = ["trend", "volatility", "point_anomaly", "level_shift"]
    assignments = [[] for _ in range(num_series)]
    
    # First pass: ensure every series has at least one anomaly
    for i in range(num_series):
        assignments[i].append(np.random.choice(anomaly_types))
    
    # Second pass: assign additional anomalies to reach ~25% per type
    num_per_type = num_series // 4
    
    for anomaly_type in anomaly_types:
        # Count how many already have this type
        current_count = sum(1 for a in assignments if anomaly_type in a)
        needed = max(0, num_per_type - current_count)
        
        if needed > 0:
            # Get eligible series (those with fewer than max anomalies and don't have this type)
            eligible = [i for i in range(num_series) 
                        if len(assignments[i]) < max_anomalies_per_series 
                        and anomaly_type not in assignments[i]]
            
            # Randomly select series for this anomaly type
            if eligible:
                selected = np.random.choice(eligible, min(needed, len(eligible)), replace=False)
                for idx in selected:
                    assignments[idx].append(anomaly_type)
    
    return assignments


ANOMALY_FUNCTIONS = {
    "trend": add_trend,
    "volatility": add_volatility,
    "point_anomaly": add_point_anomaly,
    "level_shift": add_level_shift
}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return json.load(f)


# Export functions

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def export_to_npy(x: np.ndarray, y: np.ndarray, metadata: List[Dict], 
                  output_dir: str, data_filename: str, metadata_filename: str,
                  include_time_axis: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    if include_time_axis:
        # Save with time axis as first row
        data_with_time = np.vstack([x, y])
        np.save(os.path.join(output_dir, f"{data_filename}.npy"), data_with_time)
    else:
        np.save(os.path.join(output_dir, f"{data_filename}.npy"), y)
    
    # Save metadata as JSON (npy doesn't support dicts)
    with open(os.path.join(output_dir, f"{metadata_filename}.json"), "w") as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)


def export_to_json(x: np.ndarray, y: np.ndarray, metadata: List[Dict],
                   output_dir: str, data_filename: str, metadata_filename: str,
                   include_time_axis: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    data_dict = {
        "time_axis": x.tolist() if include_time_axis else None,
        "series": y.tolist()
    }
    
    with open(os.path.join(output_dir, f"{data_filename}.json"), "w") as f:
        json.dump(data_dict, f, indent=2)
    
    with open(os.path.join(output_dir, f"{metadata_filename}.json"), "w") as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)


def export_to_csv(x: np.ndarray, y: np.ndarray, metadata: List[Dict],
                  output_dir: str, data_filename: str, metadata_filename: str,
                  include_time_axis: bool = True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, f"{data_filename}.csv")
    
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header row
        if include_time_axis:
            header = ["time"] + [f"series_{i}" for i in range(y.shape[0])]
        else:
            header = [f"series_{i}" for i in range(y.shape[0])]
        writer.writerow(header)
        
        # Data rows (transposed: each row is a time point)
        for t_idx in range(len(x)):
            if include_time_axis:
                row = [x[t_idx]] + [y[s_idx, t_idx] for s_idx in range(y.shape[0])]
            else:
                row = [y[s_idx, t_idx] for s_idx in range(y.shape[0])]
            writer.writerow(row)
    
    # Metadata as JSON
    with open(os.path.join(output_dir, f"{metadata_filename}.json"), "w") as f:
        json.dump(convert_numpy_types(metadata), f, indent=2)


EXPORT_FUNCTIONS = {
    "npy": export_to_npy,
    "json": export_to_json,
    "csv": export_to_csv
}