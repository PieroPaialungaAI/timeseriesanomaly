import json
import os
import numpy as np
from typing import Dict, List
from data_utils import *
from constants import *
from plot_utils import *


class DataGenerator:
        
    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        self.config = load_config(config_path)
        data_cfg = self.config["data_generation"]
        self.set_config(data_cfg)
        self.metadata: List[Dict] = []
    

    def set_config(self, data_cfg: Dict):
        self.num_points = data_cfg["num_points"]
        self.num_series = data_cfg["num_series"]
        self.noise_level = data_cfg["noise_level"]
        self.x = np.linspace(data_cfg["start_time"], data_cfg["end_time"], self.num_points)
        self.y = np.zeros((self.num_series, self.num_points))


    def generate_data(self) -> np.ndarray:
        assignments = assign_anomaly_types(
            self.num_series, 
            self.config["assignment"]["max_anomalies_per_series"]
        )
        for i in range(self.num_series):
            signal = generate_base_signal(self.x, self.noise_level, self.config["base_signal"])
            series_metadata = {"series_idx": i, "anomalies": []}   
            for anomaly_type in assignments[i]:
                anomaly_fn = ANOMALY_FUNCTIONS[anomaly_type]
                signal, anomaly_metadata = anomaly_fn(signal, self.x, self.config[anomaly_type])
                series_metadata["anomalies"].append(anomaly_metadata)
            self.y[i] = signal
            self.metadata.append(series_metadata)        
        return self.y
    

    def get_metadata(self) -> List[Dict]:
        return self.metadata
    

    def get_series_by_anomaly_type(self, anomaly_type: str) -> np.ndarray:
        indices = []
        for meta in self.metadata:
            for anomaly in meta["anomalies"]:
                if anomaly["type"] == anomaly_type:
                    indices.append(meta["series_idx"])
                    break
        return np.array(indices)


    def export_data(self, output_dir: str = None, format: str = None) -> None:
        export_cfg = self.config["export"]
        output_dir = output_dir or OUTPUT_DIR
        format = format or export_cfg["format"]
        export_fn = EXPORT_FUNCTIONS[format]
        export_fn(
            x=self.x,
            y=self.y,
            metadata=self.metadata,
            output_dir=output_dir,
            data_filename=export_cfg["data_filename"],
            metadata_filename=export_cfg["metadata_filename"],
            include_time_axis=export_cfg["include_time_axis"]
        )


    def random_ts_plot(self, idx: int = None, save: bool = True):
        if idx is None:
            idx = np.random.randint(0, self.num_series)
        meta = self.metadata[idx] if self.metadata else None
        anomaly_types = [a["type"] for a in meta["anomalies"]] if meta else []
        title = f"Series {idx}: {anomaly_types}" if anomaly_types else f"Series {idx}"
        save_as = f"random_ts_{idx}.png" if save else None
        return plot_single_ts(self.x, self.y[idx], meta, title=title, save_as=save_as)


    def plot_multiple(self, num_plot: int = 5, save: bool = True):
        indices = np.random.choice(self.num_series, min(num_plot, self.num_series), replace=False)
        save_as = f"multiple_ts_{num_plot}.png" if save else None
        return plot_multiple_ts(self.x, self.y, indices, self.metadata, save_as=save_as)


    def plot_by_anomaly(self, anomaly_type: str, num_plot: int = 5, save: bool = True):
        save_as = f"anomaly_{anomaly_type}_{num_plot}.png" if save else None
        return plot_by_anomaly_type(self.x, self.y, self.metadata, anomaly_type, num_plot, save_as=save_as)
