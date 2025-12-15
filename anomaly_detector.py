import numpy as np
from typing import Dict, List
from anomaly_detector_utils import *
from plot_utils import (plot_trend_detection, plot_multiple_trends,
                        plot_volatility_detection, plot_multiple_volatilities, 
                        plot_volatility_distribution,
                        plot_point_anomaly_detection, plot_multiple_point_anomalies,
                        plot_dataset_level_detection, plot_multiple_dataset_level,
                        plot_baseline_distribution,
                        plot_all_anomalies_single, plot_multiple_all_anomalies)
from data_utils import load_config
from constants import *


class AnomalyDetector:
    
    def __init__(self, x: np.ndarray, y: np.ndarray, config_path: str = ANOMALY_CONFIG_PATH):
        self.x = x
        self.y = y
        self.config = load_config(config_path)
        self.results: Dict = {}
    

    def detect_trend_anomaly(self, series_idx: int = None, 
                             p_threshold: float = None,
                             mse_threshold_factor: float = None,
                             poly_degree: int = None) -> Dict:
        cfg = self.config["trend_detection"]
        p_threshold = p_threshold or cfg["p_threshold"]
        mse_threshold_factor = mse_threshold_factor or cfg["mse_threshold_factor"]
        poly_degree = poly_degree or cfg["poly_degree"]
        if series_idx is not None:
            y = self.y[series_idx]
        else:
            y = self.y
        
        result = detect_trend(
            self.x, y,
            p_threshold=p_threshold,
            mse_threshold_factor=mse_threshold_factor,
            poly_degree=poly_degree
        )
        
        self.results["trend"] = result
        return result
    

    def detect_all_trends(self, p_threshold: float = None,
                          mse_threshold_factor: float = None,
                          poly_degree: int = None) -> List[Dict]:
        cfg = self.config["trend_detection"]
        p_threshold = p_threshold or cfg["p_threshold"]
        mse_threshold_factor = mse_threshold_factor or cfg["mse_threshold_factor"]
        poly_degree = poly_degree or cfg["poly_degree"]
        
        if self.y.ndim == 1:
            return [self.detect_trend_anomaly(p_threshold=p_threshold)]
        
        results = []
        for i in range(self.y.shape[0]):
            result = detect_trend(
                self.x, self.y[i],
                p_threshold=p_threshold,
                mse_threshold_factor=mse_threshold_factor,
                poly_degree=poly_degree
            )
            result["series_idx"] = i
            results.append(result)
        
        self.results["all_trends"] = results
        return results
    

    def get_series_with_trend(self, p_threshold: float = None) -> np.ndarray:
        cfg = self.config["trend_detection"]
        p_threshold = p_threshold or cfg["p_threshold"]
        
        if "all_trends" not in self.results:
            self.detect_all_trends(p_threshold=p_threshold)
        
        indices = [r["series_idx"] for r in self.results["all_trends"] if r["has_trend"]]
        return np.array(indices)


    def plot_trend_anomaly(self, series_idx: int, save: bool = True):
        if "all_trends" in self.results:
            results_dict = {r["series_idx"]: r for r in self.results["all_trends"]}
            result = results_dict.get(series_idx)
        else:
            result = None
        
        if result is None:
            result = detect_trend(self.x, self.y[series_idx], 
                                  p_threshold=self.config["trend_detection"]["p_threshold"])
            result["series_idx"] = series_idx
        
        save_as = f"trend_series_{series_idx}.png" if save else None
        return plot_trend_detection(self.x, self.y[series_idx], result, 
                                    title=f"Series {series_idx}", save_as=save_as)


    def plot_trend_anomalies(self, num_plot: int = 5, only_with_trend: bool = True, save: bool = True):
        if "all_trends" not in self.results:
            self.detect_all_trends()
        
        if only_with_trend:
            indices = [r["series_idx"] for r in self.results["all_trends"] if r["has_trend"]]
        else:
            indices = [r["series_idx"] for r in self.results["all_trends"]]
        
        if not indices:
            print("No series found with trend anomalies")
            return None
        
        # Sample random indices
        num_plot = min(num_plot, len(indices))
        selected = list(np.random.choice(indices, num_plot, replace=False))
        
        save_as = f"trend_anomalies_{num_plot}.png" if save else None
        return plot_multiple_trends(self.x, self.y, self.results["all_trends"], 
                                    indices=selected, save_as=save_as)


    def detect_volatility_anomaly(self, series_idx: int) -> Dict:
        result = compute_detrended_volatility(self.x, self.y[series_idx])
        result["series_idx"] = series_idx
        return result


    def detect_all_volatilities(self, threshold_percentile: float = None,
                                 threshold_factor: float = None) -> Dict:
        cfg = self.config["volatility_detection"]
        threshold_percentile = threshold_percentile or cfg["threshold_percentile"]
        threshold_factor = threshold_factor or cfg["threshold_factor"]
        volatility_results = []
        volatilities = []        
        for i in range(self.y.shape[0]):
            result = compute_detrended_volatility(self.x, self.y[i])
            result["series_idx"] = i
            volatility_results.append(result)
            volatilities.append(result["volatility"])
        volatilities = np.array(volatilities)
        outlier_result = detect_volatility_outliers(
            volatilities, 
            threshold_percentile=threshold_percentile,
            threshold_factor=threshold_factor
        )
        for i, result in enumerate(volatility_results):
            result["is_outlier"] = outlier_result["is_outlier"][i]
            result["z_score"] = outlier_result["z_scores"][i]        
        self.results["all_volatilities"] = volatility_results
        self.results["volatility_stats"] = outlier_result
        return {
            "series_results": volatility_results,
            "stats": outlier_result
        }


    def get_series_with_high_volatility(self, threshold_percentile: float = None,
                                         threshold_factor: float = None) -> np.ndarray:
        if "volatility_stats" not in self.results:
            self.detect_all_volatilities(threshold_percentile, threshold_factor)
        
        return self.results["volatility_stats"]["outlier_indices"]


    def plot_volatility_anomaly(self, series_idx: int, save: bool = True):
        # Get or compute volatility result
        if "all_volatilities" in self.results:
            results_dict = {r["series_idx"]: r for r in self.results["all_volatilities"]}
            result = results_dict.get(series_idx)
        else:
            result = None
        
        if result is None:
            result = compute_detrended_volatility(self.x, self.y[series_idx])
            result["series_idx"] = series_idx
        
        save_as = f"volatility_series_{series_idx}" if save else None
        return plot_volatility_detection(self.x, self.y[series_idx], result,
                                         title=f"Series {series_idx}", save_as=save_as)


    def plot_volatility_anomalies(self, num_plot: int = 5, only_high_volatility: bool = True, save: bool = True):
        if "all_volatilities" not in self.results:
            self.detect_all_volatilities()
        
        if only_high_volatility:
            indices = list(self.results["volatility_stats"]["outlier_indices"])
        else:
            indices = [r["series_idx"] for r in self.results["all_volatilities"]]
        
        if not indices:
            print("No series found with high volatility")
            return None
        
        num_plot = min(num_plot, len(indices))
        selected = list(np.random.choice(indices, num_plot, replace=False))
        
        save_as = f"volatility_anomalies_{num_plot}" if save else None
        return plot_multiple_volatilities(self.x, self.y, self.results["all_volatilities"],
                                          indices=selected, save_as=save_as)


    def plot_volatility_distribution(self, save: bool = True):
        if "all_volatilities" not in self.results:
            self.detect_all_volatilities()
        
        save_as = "volatility_distribution" if save else None
        return plot_volatility_distribution(self.results["all_volatilities"],
                                            self.results["volatility_stats"],
                                            save_as=save_as)


    # ============ POINT ANOMALY DETECTION ============

    def detect_point_anomaly(self, series_idx: int, 
                             window_size: int = None,
                             z_threshold: float = None) -> Dict:
        cfg = self.config["point_anomaly_detection"]
        window_size = window_size or cfg["window_size"]
        z_threshold = z_threshold or cfg["z_threshold"]
        
        result = detect_point_anomalies(self.y[series_idx], window_size, z_threshold)
        result["series_idx"] = series_idx
        return result


    def detect_all_point_anomalies(self, window_size: int = None,
                                    z_threshold: float = None) -> List[Dict]:
        cfg = self.config["point_anomaly_detection"]
        window_size = window_size or cfg["window_size"]
        z_threshold = z_threshold or cfg["z_threshold"]
        
        results = []
        for i in range(self.y.shape[0]):
            result = detect_point_anomalies(self.y[i], window_size, z_threshold)
            result["series_idx"] = i
            results.append(result)
        
        self.results["all_point_anomalies"] = results
        return results


    def get_series_with_point_anomalies(self, window_size: int = None,
                                         z_threshold: float = None) -> np.ndarray:
        if "all_point_anomalies" not in self.results:
            self.detect_all_point_anomalies(window_size, z_threshold)
        
        indices = [r["series_idx"] for r in self.results["all_point_anomalies"] 
                   if r["num_anomalies"] > 0]
        return np.array(indices)


    def plot_point_anomaly(self, series_idx: int, save: bool = True):
        # Get or compute result
        if "all_point_anomalies" in self.results:
            results_dict = {r["series_idx"]: r for r in self.results["all_point_anomalies"]}
            result = results_dict.get(series_idx)
        else:
            result = None
        
        if result is None:
            cfg = self.config["point_anomaly_detection"]
            result = detect_point_anomalies(self.y[series_idx], cfg["window_size"], cfg["z_threshold"])
            result["series_idx"] = series_idx
        
        save_as = f"point_anomaly_series_{series_idx}" if save else None
        return plot_point_anomaly_detection(self.x, self.y[series_idx], result,
                                            title=f"Series {series_idx}", save_as=save_as)


    def plot_point_anomalies(self, num_plot: int = 5, only_with_anomalies: bool = True, save: bool = True):
        if "all_point_anomalies" not in self.results:
            self.detect_all_point_anomalies()
        
        if only_with_anomalies:
            indices = [r["series_idx"] for r in self.results["all_point_anomalies"] 
                       if r["num_anomalies"] > 0]
        else:
            indices = [r["series_idx"] for r in self.results["all_point_anomalies"]]
        
        if not indices:
            print("No series found with point anomalies")
            return None
        
        num_plot = min(num_plot, len(indices))
        selected = list(np.random.choice(indices, num_plot, replace=False))
        
        save_as = f"point_anomalies_{num_plot}" if save else None
        return plot_multiple_point_anomalies(self.x, self.y, self.results["all_point_anomalies"],
                                             indices=selected, save_as=save_as)


    # ============ DATASET-LEVEL ANOMALY DETECTION ============

    def detect_dataset_level_anomalies(self, z_threshold: float = None) -> Dict:
        cfg = self.config["dataset_level_detection"]
        z_threshold = z_threshold or cfg["z_threshold"]
        
        result = detect_dataset_level_anomalies(self.y, z_threshold)
        self.results["dataset_level"] = result
        return result


    def get_dataset_level_anomalies(self, z_threshold: float = None) -> np.ndarray:
        if "dataset_level" not in self.results:
            self.detect_dataset_level_anomalies(z_threshold)
        
        return self.results["dataset_level"]["anomaly_indices"]


    def plot_dataset_level_anomaly(self, series_idx: int, save: bool = True):
        if "dataset_level" not in self.results:
            self.detect_dataset_level_anomalies()
        
        result = self.results["dataset_level"]
        baseline = result["baselines"][series_idx]
        z_score = result["z_scores"][series_idx]
        is_anomaly = result["is_anomaly"][series_idx]
        
        series_result = {
            "baseline": baseline,
            "z_score": z_score,
            "is_anomaly": is_anomaly,
            "median_baseline": result["median_baseline"],
            "series_idx": series_idx
        }
        
        save_as = f"dataset_level_series_{series_idx}" if save else None
        return plot_dataset_level_detection(self.x, self.y[series_idx], series_result,
                                            title=f"Series {series_idx}", save_as=save_as)


    def plot_dataset_level_anomalies(self, num_plot: int = 5, save: bool = True):
        if "dataset_level" not in self.results:
            self.detect_dataset_level_anomalies()
        
        result = self.results["dataset_level"]
        indices = list(result["anomaly_indices"])
        
        if not indices:
            print("No dataset-level anomalies found")
            return None
        
        num_plot = min(num_plot, len(indices))
        selected = list(np.random.choice(indices, num_plot, replace=False))
        
        save_as = f"dataset_level_anomalies_{num_plot}" if save else None
        return plot_multiple_dataset_level(self.x, self.y, result, indices=selected, save_as=save_as)


    def plot_baseline_distribution(self, save: bool = True):
        if "dataset_level" not in self.results:
            self.detect_dataset_level_anomalies()
        
        save_as = "baseline_distribution" if save else None
        return plot_baseline_distribution(self.results["dataset_level"], save_as=save_as)


    # ============ ALL ANOMALIES ============

    def detect_all_anomalies(self) -> Dict:
        """Run all 4 anomaly detection methods and store results."""
        print("Detecting trends...")
        self.detect_all_trends()
        
        print("Detecting volatility...")
        self.detect_all_volatilities()
        
        print("Detecting point anomalies...")
        self.detect_all_point_anomalies()
        
        print("Detecting dataset-level anomalies...")
        self.detect_dataset_level_anomalies()
        
        print("Done!")
        return self.results


    def get_summary_df(self):
        """Get a summary DataFrame with anomaly flags for each series."""
        import pandas as pd
        
        # Ensure all detections have run
        if "all_trends" not in self.results:
            self.detect_all_anomalies()
        
        num_series = self.y.shape[0]
        
        # Build summary
        summary = {
            "series_idx": list(range(num_series)),
            "has_trend": [False] * num_series,
            "has_high_volatility": [False] * num_series,
            "has_point_anomaly": [False] * num_series,
            "is_dataset_anomaly": [False] * num_series,
            "num_point_anomalies": [0] * num_series
        }
        
        # Fill in trend
        for r in self.results["all_trends"]:
            summary["has_trend"][r["series_idx"]] = r["has_trend"]
        
        # Fill in volatility
        for r in self.results["all_volatilities"]:
            summary["has_high_volatility"][r["series_idx"]] = r.get("is_outlier", False)
        
        # Fill in point anomalies
        for r in self.results["all_point_anomalies"]:
            summary["has_point_anomaly"][r["series_idx"]] = r["num_anomalies"] > 0
            summary["num_point_anomalies"][r["series_idx"]] = r["num_anomalies"]
        
        # Fill in dataset-level
        for i in range(num_series):
            summary["is_dataset_anomaly"][i] = self.results["dataset_level"]["is_anomaly"][i]
        
        return pd.DataFrame(summary)


    def plot_all_anomalies(self, series_idx: int, save: bool = True):
        """Plot a single series showing all detected anomalies."""
        if "all_trends" not in self.results:
            self.detect_all_anomalies()
        
        save_as = f"all_anomalies_series_{series_idx}" if save else None
        
        # Get all results for this series
        trend_result = next((r for r in self.results["all_trends"] if r["series_idx"] == series_idx), None)
        vol_result = next((r for r in self.results["all_volatilities"] if r["series_idx"] == series_idx), None)
        point_result = next((r for r in self.results["all_point_anomalies"] if r["series_idx"] == series_idx), None)
        dataset_result = {
            "baseline": self.results["dataset_level"]["baselines"][series_idx],
            "z_score": self.results["dataset_level"]["z_scores"][series_idx],
            "is_anomaly": self.results["dataset_level"]["is_anomaly"][series_idx],
            "median_baseline": self.results["dataset_level"]["median_baseline"]
        }
        
        return plot_all_anomalies_single(self.x, self.y[series_idx], 
                                         trend_result, vol_result, point_result, dataset_result,
                                         title=f"Series {series_idx}", save_as=save_as)


    def plot_random_sample_all(self, num_plot: int = 5, save: bool = True):
        """Plot random sample of series showing all anomaly types."""
        if "all_trends" not in self.results:
            self.detect_all_anomalies()
        
        indices = list(np.random.choice(self.y.shape[0], min(num_plot, self.y.shape[0]), replace=False))
        
        save_as = f"all_anomalies_sample_{num_plot}" if save else None
        return plot_multiple_all_anomalies(self.x, self.y, self.results, indices, save_as=save_as)
