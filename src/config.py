"""
src/config.py
Tunable configuration parameters and path constants for the IPL Markov Chase project.
"""
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ipl_male_csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORT_DIR = PROJECT_ROOT / "report"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Train / Test Split
TRAIN_MAX_SEASON = 2022  # Seasons <= 2022 are Train, >= 2023 are Test
TEST_MIN_SEASON = 2023

# Simulation & Statistical Parameters
RANDOM_SEED = 42
N_BOOT = 1000  # Number of bootstrap iterations for confidence intervals
N_SIM = 10000  # Number of Monte Carlo simulation runs per starting state

# State Space Definitions
# Wickets buckets: W_Low (0-2), W_Mid (3-5), W_High (6-9)
# Note: 10 wickets lost triggers immediate absorbing LOSS
WICKET_BUCKETS = {
    "W_Low": (0, 2),
    "W_Mid": (3, 5),
    "W_High": (6, 9),
}

# Pressure buckets:
# P1 (< -2), P2 (-2 to 0), P3 (0 to 2), P4 (2 to 4), P5 (> 4)
# Intervals are [lower, upper) except P5 which is [4, inf)
PRESSURE_CUTOFFS = [-2.0, 0.0, 2.0, 4.0]
PRESSURE_LABELS = ["P1", "P2", "P3", "P4", "P5"]

# 15 Transient states ordered consistently:
# (W_Low, P1), (W_Low, P2), ..., (W_High, P5)
WICKET_ORDER = ["W_Low", "W_Mid", "W_High"]
TRANSIENT_STATES = [
    f"{w}_{p}" for w in WICKET_ORDER for p in PRESSURE_LABELS
]

# Absorbing states
ABSORBING_STATES = ["WIN", "LOSS"]
ALL_STATES = TRANSIENT_STATES + ABSORBING_STATES

def get_wicket_bucket(w: int) -> str:
    """Classify wickets fallen into W_Low, W_Mid, or W_High."""
    if w <= 2:
        return "W_Low"
    elif w <= 5:
        return "W_Mid"
    elif w <= 9:
        return "W_High"
    else:
        return "ALL_OUT"

def get_pressure_bucket(pressure: float) -> str:
    """Classify pressure (RRR - CRR) into P1..P5."""
    if pressure < -2.0:
        return "P1"
    elif pressure < 0.0:
        return "P2"
    elif pressure < 2.0:
        return "P3"
    elif pressure < 4.0:
        return "P4"
    else:
        return "P5"

def get_state(wickets: int, pressure: float) -> str:
    """Return combined state name e.g. W_Low_P2."""
    w_bucket = get_wicket_bucket(wickets)
    p_bucket = get_pressure_bucket(pressure)
    return f"{w_bucket}_{p_bucket}"
