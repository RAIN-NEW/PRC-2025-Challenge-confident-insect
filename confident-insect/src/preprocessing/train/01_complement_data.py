#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair a single PRC-2025 trajectory parquet and write to the output directory.

Key Features:
- Upsampling: Inserts missing time steps to ensure continuity.
- Interpolation: Fills gaps using two-sided linear interpolation (handles Longitude wrapping).
- Extrapolation: Fills edges using a trend-based approach, degrading to constant value if needed.
- Constraints: Applies physical limits (Altitude, Speed, etc.) and handles IDL (International Date Line) crossing.
- Optimized for batch processing on Ubuntu 22.04 LTS.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import os  # Added for cpu_count
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================== Parameters ==================

# Upsampling Configuration
INSERT_GAP_SEC = 1.5       # Gap threshold to trigger insertion (seconds)
INSERT_STEP_SEC = 1.0      # Time step for insertion (supports float, e.g., 0.5)

# Extrapolation Configuration (Safety Tuning)
# N_seconds: Window size of historical data used for trend fitting (5-10s recommended)
# trend_max: Maximum duration allowed for linear trend extrapolation
# const_max: Maximum duration allowed for constant value extrapolation (after trend fails/expires)
EXTRAP_FIT_WIN_SEC = 10.0      
EXTRAP_TREND_MAX_SEC = 10.0    
EXTRAP_CONST_MAX_SEC = 5.0     

# Smoothing (Reserved for future use, currently disabled in main pipeline)
SMOOTH_MID_SEC = 5 

# Physical Constraints
GS_MIN, GS_MAX = 0.0, 1000.0           # Ground Speed (knots)
ALT_MIN, ALT_MAX = -2000.0, 70000.0    # Altitude (ft), lower bound relaxed for pressure fluctuations
VR_MIN, VR_MAX = -15000.0, 15000.0     # Vertical Rate (ft/min)

RAD = np.pi / 180.0

# ================== Utilities ==================

def ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'timestamp' column is datetime64 and sorted."""
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)

def ts_to_s(ts: pd.Series) -> np.ndarray:
    """Convert tz-aware timestamp series to float seconds (Unix epoch)."""
    return ts.astype("int64", copy=False) / 1e9

def clamp(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Clip values between lower and upper bounds."""
    return s.clip(lower=lo, upper=hi)

# ----------------- Phase 1: Upsampling -----------------

def insert_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert missing timestamps where the gap is larger than INSERT_GAP_SEC.
    Uses INSERT_STEP_SEC to determine the interval.
    """
    ts = df["timestamp"].to_numpy()
    to_insert = []
    
    # Use pd.to_timedelta to support float seconds steps
    step_delta = pd.to_timedelta(INSERT_STEP_SEC, unit='s')
    
    for i in range(1, len(ts)):
        # Calculate gap in seconds
        dt = (ts[i] - ts[i-1]) / np.timedelta64(1, "s")
        
        if dt > INSERT_GAP_SEC:
            # Start from the left endpoint + step
            curr_t = ts[i-1] + step_delta
            
            # Keep inserting while strictly less than the right endpoint
            while curr_t < ts[i]:
                to_insert.append(curr_t)
                curr_t += step_delta

    if not to_insert:
        return df

    # Merge original and inserted timestamps
    base_cols = df.columns
    add = pd.DataFrame(index=to_insert)
    add.index.name = "timestamp"
    # Reindex to keep column structure, filling new rows with NaN
    add = add.reindex(columns=base_cols.drop("timestamp", errors="ignore"))
    
    df2 = pd.concat(
        [df.set_index("timestamp"), add],
        axis=0
    ).sort_index().reset_index().rename(columns={"index": "timestamp"})
    return df2

# ----------------- Phase 2: Imputation Logic -----------------

def _neighbors_indices(mask_na: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find indices of the nearest valid values on the left and right for each position.
    Returns: (left_indices, right_indices)
    """
    n = len(mask_na)
    left = np.full(n, -1, dtype=int)
    right = np.full(n, -1, dtype=int)

    # Forward pass for left neighbors
    last = -1
    for i in range(n):
        if not mask_na[i]:
            last = i
        left[i] = last
        
    # Backward pass for right neighbors
    nxt = -1
    for i in range(n-1, -1, -1):
        if not mask_na[i]:
            nxt = i
        right[i] = nxt
    return left, right

# --- Interpolation (Two-sided) ---

def _time_interp_two_sided(series: pd.Series, left_idx: np.ndarray, right_idx: np.ndarray) -> pd.Series:
    """Standard linear interpolation based on time."""
    s = series.copy()
    ts = ts_to_s(s.index.to_series())
    na = s.isna().to_numpy()
    
    for i in np.where(na)[0]:
        l, r = left_idx[i], right_idx[i]
        if l >= 0 and r >= 0 and l < r:
            tl, tr = ts.iloc[l], ts.iloc[r]
            if tr > tl:
                wl = (tr - ts.iloc[i]) / (tr - tl)
                wr = 1.0 - wl
                s.iat[i] = wl * s.iat[l] + wr * s.iat[r]
    return s

def _lon_difference(a, b):
    """Calculate the shortest signed difference between two longitudes (b - a)."""
    diff = b - a
    while diff <= -180: diff += 360.0
    while diff > 180:   diff -= 360.0
    return diff

def _time_interp_two_sided_lon(series: pd.Series, left_idx: np.ndarray, right_idx: np.ndarray) -> pd.Series:
    """
    Linear interpolation for Longitude.
    Handles the date line crossing (e.g., -179 to 179) via shortest arc.
    """
    s = series.copy()
    ts = ts_to_s(s.index.to_series())
    na = s.isna().to_numpy()
    
    for i in np.where(na)[0]:
        l, r = left_idx[i], right_idx[i]
        if l >= 0 and r >= 0 and l < r:
            tl, tr = ts.iloc[l], ts.iloc[r]
            if tr > tl:
                w = (ts.iloc[i] - tl) / (tr - tl)
                val_l, val_r = float(s.iat[l]), float(s.iat[r])
                
                # Interpolate shortest angular difference
                diff = _lon_difference(val_l, val_r)
                res = val_l + w * diff
                # Wrap result to [-180, 180)
                s.iat[i] = (res + 180.0) % 360.0 - 180.0
    return s

def _fit_trend(ts_s: np.ndarray, vs: np.ndarray) -> tuple[float, float]:
    """
    Weighted linear regression: v = a + b*t
    Weights decay exponentially for older data points.
    """
    t = ts_s - ts_s.min()
    # Weight: Higher weight for recent points to reduce impact of distant noise
    # max(..., 1e-3) prevents division by zero
    w = np.exp(-(t.max() - t) / max(10.0, 1e-3)) 
    
    X = np.c_[np.ones_like(t), t]
    W = np.diag(w)
    
    try:
        # Solve using Moore-Penrose pseudo-inverse: (X.T W X)^-1 X.T W y
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ vs)
        return float(beta[0]), float(beta[1])
    except:
        return np.nan, 0.0 # Fallback to constant if fitting fails

# --- Extrapolation (One-sided) ---

def _extrapolate_one_side(
    series: pd.Series,
    side: str,
    N_seconds: float,         # Fitting window size
    trend_max_span_sec: float,# Max duration for trend extrapolation
    const_max_span_sec: float,# Max duration for constant extrapolation
    is_lon: bool = False      # Flag for longitude unwrapping
) -> pd.Series:
    """
    One-sided extrapolation logic:
    1. Try Linear Trend Extrapolation.
    2. If limits reached or fitting fails, degrade to Constant Extrapolation.
    3. If 'is_lon' is True, handles unwrapping for IDL crossing.
    """
    s = series.copy()
    ts = s.index.to_series()
    ts_s = ts.astype("int64", copy=False) / 1e9

    def _try_trend_fill(idx_missing: int, j_lo: int, j_hi: int) -> bool:
        """Attempt to fill using linear trend from window [j_lo, j_hi]."""
        window = s.iloc[j_lo:j_hi+1].dropna()
        if len(window) < 2: return False
        
        w_ts = ts_s.loc[window.index].to_numpy()
        w_vals = window.to_numpy(dtype=float)
        
        # Core Logic: Unwrap longitude before fitting
        # e.g., [179, -179] -> [179, 181] for correct slope calculation
        if is_lon:
            w_vals_rad = np.deg2rad(w_vals)
            w_vals_unwrapped = np.rad2deg(np.unwrap(w_vals_rad))
            a, b = _fit_trend(w_ts, w_vals_unwrapped)
        else:
            a, b = _fit_trend(w_ts, w_vals)
            
        if np.isnan(a): return False

        # Predict
        dt = float(ts_s.iloc[idx_missing] - ts_s.iloc[j_lo])
        pred = a + b * dt
        
        s.iat[idx_missing] = pred # Fill raw prediction; wrapping happens later
        return True

    def _try_const_fill(idx_missing: int, ref_idx: int) -> bool:
        """Fill using the value from a reference index (nearest valid)."""
        s.iat[idx_missing] = float(s.iat[ref_idx])
        return True

    n = len(s)

    if side == "left":
        # Head missing: fill from inside (k) outwards (0)
        # 1. Find first valid index k
        k = 0
        while k < n and np.isnan(s.iat[k]): k += 1
        if k == n: return s 

        # 2. Iterate backwards through missing points
        for i in range(k - 1, -1, -1):
            dt = abs(float(ts_s.iloc[i] - ts_s.iloc[k]))
            
            filled = False
            # Priority 1: Trend Extrapolation
            if dt <= trend_max_span_sec:
                # Find valid window [k, j] on the right
                j = k
                while j < n and (ts_s.iloc[j] - ts_s.iloc[k] <= N_seconds):
                    j += 1
                # Valid window: k to j-1
                if k <= j - 1:
                    filled = _try_trend_fill(i, k, j - 1)
            
            # Priority 2: Constant Extrapolation (Degradation)
            if not filled and dt <= (trend_max_span_sec + const_max_span_sec):
                 # Use the nearest filled value (i+1)
                 filled = _try_const_fill(i, i + 1)

    else: # side == "right"
        # Tail missing: fill from inside (k) outwards (n-1)
        # 1. Find last valid index k
        k = n - 1
        while k >= 0 and np.isnan(s.iat[k]): k -= 1
        if k == -1: return s

        # 2. Iterate forwards through missing points
        for i in range(k + 1, n):
            dt = abs(float(ts_s.iloc[i] - ts_s.iloc[k]))
            
            filled = False
            # Priority 1: Trend Extrapolation
            if dt <= trend_max_span_sec:
                # Find valid window [j, k] on the left
                j = k
                while j >= 0 and (ts_s.iloc[k] - ts_s.iloc[j] <= N_seconds):
                    j -= 1
                # Valid window: j+1 to k
                if j + 1 <= k:
                    filled = _try_trend_fill(i, j + 1, k)
            
            # Priority 2: Constant Extrapolation
            if not filled and dt <= (trend_max_span_sec + const_max_span_sec):
                filled = _try_const_fill(i, i - 1)

    return s

# ----------------- Main Repair Function -----------------

def repair_one_file(in_path: Path, out_dir: Path) -> Path:
    try:
        df = pd.read_parquet(in_path)
        df = ensure_ts(df)

        # 1. Deduplication: Remove rows with dt=0
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        if len(df) > 1:
            ts_numeric = df["timestamp"].astype("int64") / 1e9
            dt = np.diff(ts_numeric)
            zero_dt_mask = np.isclose(dt, 0, atol=1e-6)
            rows_to_drop = np.where(zero_dt_mask)[0] + 1
            if len(rows_to_drop) > 0:
                keep_mask = np.ones(len(df), dtype=bool)
                keep_mask[rows_to_drop] = False
                df = df[keep_mask].reset_index(drop=True)

        # Identify metadata/text columns to preserve
        text_cols = [c for c in ["flight_id", "typecode", "source"] if c in df.columns]

        # 2. Phase 1: Upsampling (Insert points)
        df = insert_points(df)
        df = df.set_index("timestamp")

        # 3. Phase 2: Repair Data (Altitude, Latitude, Longitude)
        for c in ["altitude", "latitude", "longitude"]:
            if c not in df.columns: continue
            
            s = df[c].astype(float)
            is_lon = (c == "longitude")

            # --- Step A: Fill gaps (Interpolation) ---
            na = s.isna().to_numpy()
            left, right = _neighbors_indices(na)
            
            if is_lon:
                s = _time_interp_two_sided_lon(s, left, right)
            else:
                s = _time_interp_two_sided(s, left, right)

            # --- Step B: Fill edges (Extrapolation) ---
            extrap_kwargs = {
                "N_seconds": EXTRAP_FIT_WIN_SEC,
                "trend_max_span_sec": EXTRAP_TREND_MAX_SEC,
                "const_max_span_sec": EXTRAP_CONST_MAX_SEC,
                "is_lon": is_lon
            }

            head_na = s.isna()
            if head_na.any() and head_na.iloc[0]:
                s = _extrapolate_one_side(s, side="left", **extrap_kwargs)
            
            tail_na = s.isna()
            if tail_na.any() and tail_na.iloc[-1]:
                s = _extrapolate_one_side(s, side="right", **extrap_kwargs)

            # --- Step C: Clamp limits and Wrap longitude ---
            if c == "altitude":
                s = clamp(s, ALT_MIN, ALT_MAX)
                s = s.round(4)
            elif c == "latitude":
                s = clamp(s, -90.0, 90.0)
                s = s.round(6)
            elif c == "longitude":
                # Wrap longitude to [-180, 180)
                s = (s + 180.0) % 360.0 - 180.0
                s = s.round(6)

            df[c] = s

        # 4. Fill metadata columns (Forward/Backward fill)
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].ffill().bfill()

        # 5. Cleanup: Remove rows where essential data is still missing
        df = df.reset_index()
        target_cols = ["altitude", "latitude", "longitude"]
        existing_cols = df.columns.intersection(target_cols)
        if not existing_cols.empty:
            df = df.dropna(subset=existing_cols).reset_index(drop=True)
        
        # Save output
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / in_path.name
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception as e:
        raise RuntimeError(f"Error processing {in_path.name}: {e}")

def repair_batch(input_dir: Path, out_root_dir: Path, max_workers: int):
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"Warning: No parquet files found in {input_dir}")
        return
    
    print(f"Starting batch processing: {len(parquet_files)} files using {max_workers} cores")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(repair_one_file, p, out_root_dir): p for p in parquet_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", unit="file"):
            f = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed to process {f.name}: {e}")
    
    print(f"Completed. Results saved to: {out_root_dir}")

# ----------------- CLI -----------------

def parse_args():
    ap = argparse.ArgumentParser(description="Batch repair trajectory Parquet files.")
    
    # Dynamically locate project root
    # Assumes script is at: confident-insect/src/preprocessing/train/01_complement_data.py
    current_file = Path(__file__).resolve()
    try:
        project_root = current_file.parents[3]
    except IndexError:
        project_root = Path.cwd()

    # Define paths relative to project root
    default_input = project_root / "data" / "raw" / "prc-2025-datasets" / "flights_train"
    default_output = project_root / "data" / "intermediate" / "01_complement_data" / "flights_train"
    
    # Determine default workers: half of available CPUs (at least 1)
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)
    
    ap.add_argument("--input_dir", type=str, default=str(default_input), 
                    help=f"Input directory path (default: {default_input})")
    ap.add_argument("--out_dir", type=str, default=str(default_output), 
                    help=f"Output root directory path (default: {default_output})")
    ap.add_argument("--max_workers", type=int, default=default_workers, 
                    help=f"Number of parallel workers (default: {default_workers} [half of system CPUs])")
    
    return ap.parse_args()

def main():
    args = parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.out_dir)
    
    print(f"Data Root: {input_path}")
    print(f"Output Root: {output_path}")

    repair_batch(input_path, output_path, args.max_workers)

if __name__ == "__main__":
    main()