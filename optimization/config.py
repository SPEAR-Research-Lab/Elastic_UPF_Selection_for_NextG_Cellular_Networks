#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for UPF optimization
Contains all network parameters and constants
"""

import numpy as np

# === Network Topology Parameters ===
AREA_KM2 = 25000 
SURFACE_WIDTH = int(np.sqrt(AREA_KM2))

# Predefined DC positions (for reproducibility)
DC_POSITIONS = [
    (SURFACE_WIDTH * 0.25, SURFACE_WIDTH * 0.75),
    (SURFACE_WIDTH * 0.75, SURFACE_WIDTH * 0.75),
    (SURFACE_WIDTH * 0.5, SURFACE_WIDTH * 0.25)
]

# Device counts
DEVICE_COUNTS = {
    "DC": 2,
    "BigRouter": 10,
    "SmallRouter": 100,
    "gNodeB": 500 
}

# Device capacities (Mbps)
DEVICE_CAPACITIES = {
    "DC": 150000, #M cores
    "BigRouter": 12500, #8 cores
    "SmallRouter": 1000 #4 cores
}

# Fixed UPF Latencies (ms)
UPF_LATENCIES = {
    "SmallRouter": 0.25,
    "BigRouter": 0.04,
    "DC": 0.05
}

# === Traffic Parameters ===
TRAFFIC_DISTRIBUTION = {
    "TRAFFIC_TO_DC": 0.25,
    "TRAFFIC_SAME_SUBNET": 0.25,
    "TRAFFIC_SAME_BIG_ROUTER": 0.25,
    "TRAFFIC_TO_ROUTER": 0.25
}

THROUGHPUT_RANGE = [1, 100]
LATENCY_OPTIONS = [0, 1, 5, 10]

# === Power Coefficients per UPF Type ===
POWER_COEFFICIENTS = {
    "SmallRouter": {"w": 4.20, "p": 0.0054},
    "BigRouter": {"w": 16.05, "p": 0.0075},
    "DC": {"w": 248.46, "p": 0.0051}
}

# === Simulation Parameters ===
NUM_SESSIONS = 3500  # (unused by main_comparison.py; kept for standalone scripts)
RANDOM_SEED = 4 #1 #99 #10
CAPACITY_THRESHOLD = 0.8  # Use 80% of UPF capacity
RANDOM_SEEDS = [1, 4, 10, 42, 99, 123, 456, 789, 2024, 2025]  # 10 different seeds

# === Optimization Parameters ===
# Alpha-Gamma parameter sweep
ALPHA_VALUES = [1]
GAMMA_VALUES = [0.01, 0.1, 1, 10, 100]

# Solver timeout (seconds)
SOLVER_TIMEOUT = 600  # seconds; matches the paper's 600s solver limit

# === NEW: Multi-dimensional parameter sweep ===
# Different session counts to test
NUM_SESSIONS_VALUES = [100, 1000, 5000]  # (unused; main_comparison.py sweeps [50,100,500,1000,2500,5000])

# Different timeout values to test (in seconds)
SOLVER_TIMEOUT_VALUES = [600]  # (unused; main_comparison.py uses TIMEOUT=600)

# Other optimization constants
M = 1e6
EPSILON = 1e-4