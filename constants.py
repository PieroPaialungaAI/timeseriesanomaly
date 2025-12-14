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

# Trend parameters (single linear trend from start to end)
TREND_SLOPE_MIN = -0.05
TREND_SLOPE_MAX = 0.05

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
CONFIG_FILE_DIR = "config_files"
CONFIG_FILE_NAME = "default_config.json"

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), CONFIG_FILE_DIR, CONFIG_FILE_NAME)
