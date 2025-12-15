import os
# Time axis defaults
START_TIME = 0
END_TIME = 100

# Base signal generation
FREQUENCY_MIN = 0.5
FREQUENCY_MAX = 2.0
AMPLITUDE_MIN = 0.5
AMPLITUDE_MAX = 1.5
DEFAULT_NOISE_LEVEL = 0.1

# Trend parameters (polynomial trend, degree 1 to max_degree)
TREND_MAX_DEGREE = 3
TREND_MAGNITUDE_MIN = 5      # Multiplied by std
TREND_MAGNITUDE_MAX = 15

# Volatility parameters (uniform across entire series)
VOLATILITY_MULTIPLIER_MIN = 3.0
VOLATILITY_MULTIPLIER_MAX = 6.0
VOLATILITY_BASE_NOISE = 0.1

# Point anomaly parameters
POINT_ANOMALY_MARGIN_FRAC = 0.1     # Avoid edges by n // 10
POINT_ANOMALY_MAGNITUDE_MIN = 5     # Multiplied by std
POINT_ANOMALY_MAGNITUDE_MAX = 10

# Level shift parameters (uniform across entire series, can be + or -)
LEVEL_SHIFT_MAGNITUDE_MIN = 15      # Multiplied by std
LEVEL_SHIFT_MAGNITUDE_MAX = 50

# Assignment parameters
MAX_ANOMALIES_PER_SERIES = 2

# Trend detection parameters
TREND_P_THRESHOLD = 0.05
TREND_MSE_THRESHOLD_FACTOR = 1.5
TREND_POLY_DEGREE = 2

# Volatility detection parameters
VOLATILITY_THRESHOLD_PERCENTILE = 75
VOLATILITY_THRESHOLD_FACTOR = 1.5

# Point anomaly detection parameters
POINT_ANOMALY_WINDOW_SIZE = 50
POINT_ANOMALY_Z_THRESHOLD = 3.0

# Dataset-level anomaly detection parameters
DATASET_LEVEL_Z_THRESHOLD = 3.0

# Directory paths
CONFIG_FILE_DIR = "config_files"
CONFIG_FILE_NAME = "default_config.json"
ANOMALY_CONFIG_FILE_NAME = "anomaly_config.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILE_DIR, CONFIG_FILE_NAME)
ANOMALY_CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILE_DIR, ANOMALY_CONFIG_FILE_NAME)
