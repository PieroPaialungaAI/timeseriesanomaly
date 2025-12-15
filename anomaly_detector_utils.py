import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


def fit_linear_trend(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit a linear regression to the time series and compute statistics.
    
    Args:
        x: Time axis array
        y: Signal values
    
    Returns:
        Dict with slope, intercept, r_squared, p_value, mse, predictions
    """
    X = x.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    
    # Compute R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Compute p-value for the slope using t-test
    n = len(y)
    slope = model.coef_[0]
    intercept = model.intercept_
    
    # Standard error of the slope
    se_slope = np.sqrt(ss_res / (n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
    t_stat = slope / se_slope if se_slope > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    
    return {
        "model_type": "linear",
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "p_value": p_value,
        "mse": mse,
        "predictions": y_pred
    }


def fit_polynomial_trend(x: np.ndarray, y: np.ndarray, degree: int = 2) -> Dict:
    """
    Fit a polynomial regression to the time series.
    
    Args:
        x: Time axis array
        y: Signal values
        degree: Polynomial degree (default 2)
    
    Returns:
        Dict with coefficients, r_squared, p_value, mse, predictions
    """
    X = x.reshape(-1, 1)
    
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, y)
    
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    
    # Compute R-squared
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # F-test for overall significance of polynomial fit
    n = len(y)
    p = degree + 1  # number of parameters
    
    if ss_tot > 0 and n > p:
        f_stat = (r_squared / (p - 1)) / ((1 - r_squared) / (n - p))
        p_value = 1 - stats.f.cdf(f_stat, p - 1, n - p)
    else:
        p_value = 1.0
    
    return {
        "model_type": "polynomial",
        "degree": degree,
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_,
        "r_squared": r_squared,
        "p_value": p_value,
        "mse": mse,
        "predictions": y_pred
    }


def detect_trend(x: np.ndarray, y: np.ndarray, 
                 p_threshold: float = 0.05,
                 mse_threshold_factor: float = 1.5,
                 poly_degree: int = 2) -> Dict:
    """
    Detect trend anomaly using linear regression first, then polynomial if needed.
    
    Args:
        x: Time axis array
        y: Signal values
        p_threshold: P-value threshold for significance (default 0.05)
        mse_threshold_factor: If poly MSE is this much better, prefer polynomial
        poly_degree: Degree for polynomial regression
    
    Returns:
        Dict with trend detection results
    """
    # First try linear fit
    linear_result = fit_linear_trend(x, y)
    
    # Try polynomial fit
    poly_result = fit_polynomial_trend(x, y, degree=poly_degree)
    
    # Decide which model is better
    # Use polynomial if its MSE is significantly lower
    use_polynomial = poly_result["mse"] < linear_result["mse"] / mse_threshold_factor
    
    best_result = poly_result if use_polynomial else linear_result
    
    # Determine if trend is significant
    has_significant_trend = best_result["p_value"] < p_threshold
    
    return {
        "has_trend": has_significant_trend,
        "best_model": best_result["model_type"],
        "linear_result": linear_result,
        "polynomial_result": poly_result,
        "p_value": best_result["p_value"],
        "r_squared": best_result["r_squared"],
        "mse": best_result["mse"]
    }



def compute_detrended_volatility(x: np.ndarray, y: np.ndarray, 
                                  poly_degree: int = 2,
                                  mse_threshold_factor: float = 1.5) -> Dict:
    """
    Compute volatility after removing best-fit trend (linear or polynomial).
    
    Args:
        x: Time axis array
        y: Signal values
        poly_degree: Max polynomial degree to try
        mse_threshold_factor: Use polynomial if MSE is this much better
    
    Returns:
        Dict with volatility metrics
    """
    # Try both linear and polynomial
    linear_result = fit_linear_trend(x, y)
    poly_result = fit_polynomial_trend(x, y, degree=poly_degree)
    
    # Use polynomial if significantly better fit
    use_polynomial = poly_result["mse"] < linear_result["mse"] / mse_threshold_factor
    
    if use_polynomial:
        best_result = poly_result
        trend_type = f"polynomial (degree {poly_degree})"
    else:
        best_result = linear_result
        trend_type = "linear"
    
    # Get residuals (detrended signal)
    residuals = y - best_result["predictions"]
    
    # Compute volatility as std of residuals
    volatility = np.std(residuals)
    
    return {
        "volatility": volatility,
        "residuals": residuals,
        "trend_type": trend_type,
        "trend_predictions": best_result["predictions"],
        "trend_mse": best_result["mse"],
        "trend_r_squared": best_result["r_squared"]
    }


def detect_volatility_outliers(volatilities: np.ndarray, 
                                threshold_percentile: float = 75,
                                threshold_factor: float = 1.5) -> Dict:
    """
    Detect which series are volatility outliers compared to the dataset.
    
    Args:
        volatilities: Array of volatility values for each series
        threshold_percentile: Percentile above which to flag (default 75th)
        threshold_factor: Or flag if > factor * median
    
    Returns:
        Dict with outlier detection results
    """
    median_vol = np.median(volatilities)
    mean_vol = np.mean(volatilities)
    std_vol = np.std(volatilities)
    percentile_threshold = np.percentile(volatilities, threshold_percentile)
    
    # Z-scores
    z_scores = (volatilities - mean_vol) / std_vol if std_vol > 0 else np.zeros_like(volatilities)
    
    # Flag as outlier if above percentile OR above factor * median
    is_outlier = (volatilities > percentile_threshold) | (volatilities > threshold_factor * median_vol)
    
    return {
        "median_volatility": median_vol,
        "mean_volatility": mean_vol,
        "std_volatility": std_vol,
        "percentile_threshold": percentile_threshold,
        "z_scores": z_scores,
        "is_outlier": is_outlier,
        "outlier_indices": np.where(is_outlier)[0]
    }


# ============ POINT ANOMALY DETECTION ============

def compute_rolling_zscore(y: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Compute Z-score for each point using a rolling window.
    
    Args:
        y: Signal values
        window_size: Size of rolling window for local mean/std
    
    Returns:
        Array of Z-scores for each point
    """
    n = len(y)
    z_scores = np.zeros(n)
    
    half_window = window_size // 2
    
    for i in range(n):
        # Define window bounds
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        # Exclude the current point from window stats
        window = np.concatenate([y[start:i], y[i+1:end]])
        
        if len(window) > 1:
            mean = np.mean(window)
            std = np.std(window)
            z_scores[i] = (y[i] - mean) / std if std > 0 else 0
        else:
            z_scores[i] = 0
    
    return z_scores


def detect_point_anomalies(y: np.ndarray, window_size: int = 50, 
                           z_threshold: float = 3.0) -> Dict:
    """
    Detect single point anomalies using rolling Z-score.
    
    Args:
        y: Signal values
        window_size: Size of rolling window
        z_threshold: Z-score threshold to flag as anomaly
    
    Returns:
        Dict with point anomaly detection results
    """
    z_scores = compute_rolling_zscore(y, window_size)
    
    # Points where |z| > threshold are anomalies
    is_anomaly = np.abs(z_scores) > z_threshold
    anomaly_indices = np.where(is_anomaly)[0]
    
    return {
        "z_scores": z_scores,
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices,
        "num_anomalies": len(anomaly_indices),
        "z_threshold": z_threshold,
        "window_size": window_size
    }


# ============ DATASET-LEVEL ANOMALY DETECTION ============

def compute_baseline(y: np.ndarray) -> float:
    """Compute baseline (median) of a time series."""
    return np.median(y)


def detect_dataset_level_anomalies(y_all: np.ndarray, z_threshold: float = 3.0) -> Dict:
    """
    Detect time series with anomalous baseline compared to the dataset.
    
    Args:
        y_all: 2D array of shape (num_series, num_points)
        z_threshold: Z-score threshold to flag as anomaly
    
    Returns:
        Dict with dataset-level anomaly detection results
    """
    # Compute baseline for each series
    baselines = np.array([compute_baseline(y_all[i]) for i in range(y_all.shape[0])])
    
    # Dataset statistics
    median_baseline = np.median(baselines)
    mean_baseline = np.mean(baselines)
    std_baseline = np.std(baselines)
    
    # Z-score for each series baseline
    z_scores = (baselines - mean_baseline) / std_baseline if std_baseline > 0 else np.zeros_like(baselines)
    
    # Flag anomalies
    is_anomaly = np.abs(z_scores) > z_threshold
    anomaly_indices = np.where(is_anomaly)[0]
    
    # Separate high and low anomalies
    high_anomaly_indices = np.where(z_scores > z_threshold)[0]
    low_anomaly_indices = np.where(z_scores < -z_threshold)[0]
    
    return {
        "baselines": baselines,
        "median_baseline": median_baseline,
        "mean_baseline": mean_baseline,
        "std_baseline": std_baseline,
        "z_scores": z_scores,
        "z_threshold": z_threshold,
        "is_anomaly": is_anomaly,
        "anomaly_indices": anomaly_indices,
        "high_anomaly_indices": high_anomaly_indices,
        "low_anomaly_indices": low_anomaly_indices,
        "num_anomalies": len(anomaly_indices)
    }
