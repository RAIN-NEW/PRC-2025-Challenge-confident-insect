#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Samples By Aircraft Type (Rank/Test Version).

Features:
- **Inference Mode**: Adapted for Ranking/Test phase (No fuel label dependencies).
- **Streaming Write**: Periodically flushes data to disk to prevent Out-Of-Memory (OOM) errors.
- **Single File Append**: Uses ParquetWriter to efficiently append row groups.
- **Robust I/O**: Ensures Parquet files are closed properly, even when errors occur.
- **Strict Validation**: Enforces the existence of the 'idx' column (crucial for submission alignment).
- **Optimized**: Removed dead code and redundant calculations.
"""

import warnings
import argparse
import shutil
import os  # Added for cpu_count
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pymap3d
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/preprocessing/rank/02_build_script.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[3]
except IndexError:
    project_root = Path.cwd()

# Define paths relative to project root
# Input: Previous step's output (trajectories) + Metadata (fuel_rank_submission, flightlist_rank)
DEFAULT_ROOT = project_root / "data" / "intermediate" / "01_complement_data"

# Output: Current step's output directory
DEFAULT_OUT = project_root / "data" / "intermediate" / "02_build_script" / "rank"

FLUSH_THRESHOLD = 200_000  # Flush to disk every N trajectory points

# ===============================================

def parse_args():
    ap = argparse.ArgumentParser(description="Generate rank/test samples from trajectory data.")
    
    # Determine default workers: half of available CPUs (at least 1)
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)

    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                    help=f"Input directory containing flights_rank/ and metadata. Default: {DEFAULT_ROOT}")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help=f"Output directory for generated samples. Default: {DEFAULT_OUT}")
    ap.add_argument("--max_workers", type=int, default=default_workers,
                    help=f"Number of parallel workers. Default: {default_workers}")
    return ap.parse_args()

def load_traj(fid, root_dir):
    """Load trajectory file for a given flight ID (from flights_rank)."""
    f = root_dir / "flights_rank" / f"{fid}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        return None

def align_interval(traj, start, end):
    """
    Extract and process trajectory segment between start and end timestamps.
    Returns calculated kinematics and statistics.
    """
    mask = (traj["timestamp"] >= start) & (traj["timestamp"] < end)
    sub = traj[mask].copy()
    
    # Placeholder for empty results (14 elements matches return signature)
    empty_res = (None, None, 0.0, 0.0, (end - start).total_seconds(), 
                 None, None, None, None, None, None, None, None, None)
    
    if len(sub) < 2:
        return empty_res
    
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    
    # Filter high frequency points (timestamp duplicates or extremely close points)
    ts_values = sub["timestamp"].values.astype("int64")
    keep_mask = np.zeros(len(sub), dtype=bool)
    keep_mask[0] = True
    last_idx = 0
    for i in range(1, len(sub)):
        if ts_values[i] - ts_values[last_idx] >= 100_000_000: # 100ms threshold
            keep_mask[i] = True
            last_idx = i
    filtered_sub = sub[keep_mask].reset_index(drop=True)
    
    if len(filtered_sub) < 2:
        return empty_res

    # Extract bounds
    min_lat, max_lat = filtered_sub["latitude"].min(), filtered_sub["latitude"].max()
    min_lon, max_lon = filtered_sub["longitude"].min(), filtered_sub["longitude"].max()
    min_alt, max_alt = filtered_sub["altitude"].min(), filtered_sub["altitude"].max()
    
    # Prepare data for kinematic calculation
    lats = filtered_sub["latitude"].to_numpy()
    lons = filtered_sub["longitude"].to_numpy()
    alts = filtered_sub["altitude"].to_numpy() * 0.3048 # Convert ft to meters
    
    lat1, lon1, alt1 = lats[:-1], lons[:-1], alts[:-1]
    lat2, lon2, alt2 = lats[1:], lons[1:], alts[1:]
    ts_ns = filtered_sub["timestamp"].values.astype("int64")
    
    dt_sec = (ts_ns[1:] - ts_ns[:-1]) / 1e9
    dt_sec = np.where(dt_sec < 1e-3, 1e-3, dt_sec) # Prevent division by zero
    
    # Calculate ENU coordinates and velocities
    e, n_dist, u = pymap3d.geodetic2enu(lat2, lon2, alt2, lat1, lon1, alt1)
    Vn, Ve = n_dist / dt_sec, e / dt_sec
    
    vertical_rate = np.round(u / dt_sec, 1)
    dt_sec = np.round(dt_sec, 2)
    
    covered_time = (filtered_sub["timestamp"].iloc[-1] - filtered_sub["timestamp"].iloc[0]).total_seconds()
    total_time = float((end - start).total_seconds())
    cov = covered_time / total_time if total_time > 0 else 0.0
    
    return (filtered_sub.iloc[:-1], dt_sec, covered_time, cov, total_time,
            vertical_rate, Vn, Ve,
            min_lat, max_lat, min_lon, max_lon, min_alt, max_alt)

def process_single_row(row, id2type, root_dir):
    """Worker function to process a single flight segment."""
    fid = row.flight_id
    ac_type = id2type.get(fid, "UNKNOWN") or "UNKNOWN"
    
    # [Rank/Test Update] Removed fuel_kg check as it might not be present or valid in test data
    # if row.fuel_kg <= 0: return None, None, None
    
    traj = load_traj(fid, root_dir)
    if traj is None:
        return None, None, None
    
    (sub, dt, covered_time, cov, total_time,
     vertical_rate, Vn, Ve,
     min_lat, max_lat, min_lon, max_lon, min_alt, max_alt) = align_interval(traj, row.start, row.end)
    
    if sub is None:
        return None, None, None
    
    need_len = len(Vn)
    
    # Select relevant columns
    cols = ["timestamp", "latitude", "longitude", "altitude", "track", "groundspeed"]
    cols = [c for c in cols if c in sub.columns]
    
    need = sub[cols].iloc[:need_len].copy()
    
    # Append calculated features
    need["vertical_rate"] = vertical_rate
    need["dt"] = dt
    need["gs_vn"], need["gs_ve"] = Vn, Ve
    need["flight_id"] = fid
    need["aircraft_type"] = ac_type
    
    if len(need) < 1:
        return None, None, None
    
    # [Rank/Test Update] Removed fuel_rate_warm calculation
    
    start_ts = need["timestamp"].iloc[0]
    end_ts = need["timestamp"].iloc[-1] + pd.Timedelta(seconds=need["dt"].iloc[-1])
    
    # Compute Segment Statistics
    stat = {
        "flight_id": fid, "start": start_ts, "end": end_ts,
        # "fuel_kg": float(row.fuel_kg), # [Rank/Test Update] Removed fuel_kg
        "covered_time": covered_time,
        "coverage_ratio": cov, "total_time": total_time, "sample_count": len(need),
        "min_latitude": float(min_lat), "max_latitude": float(max_lat),
        "min_altitude": float(min_alt), "max_altitude": float(max_alt),
        "min_longitude": float(min_lon), "max_longitude": float(max_lon),
        
        "ac_type": ac_type,
        "idx_orin": row["idx"], # Strict check for submission alignment

        "is_cross_idl": False,
        "lon_min_east": None, "lon_max_east": None,
        "lon_min_west": None, "lon_max_west": None
    }
    
    # Handle International Date Line (IDL) crossing logic
    if (max_lon - min_lon) > 180.0:
        stat["is_cross_idl"] = True
        mask_e = need["longitude"] >= 0
        if mask_e.any():
            stat["lon_min_east"] = float(need.loc[mask_e, "longitude"].min())
            stat["lon_max_east"] = float(need.loc[mask_e, "longitude"].max())
        mask_w = need["longitude"] < 0
        if mask_w.any():
            stat["lon_min_west"] = float(need.loc[mask_w, "longitude"].min())
            stat["lon_max_west"] = float(need.loc[mask_w, "longitude"].max())

    return ac_type, need, stat

def flush_buffer_single_file(ac_type, buffer_samples, out_dir, writers):
    """Writes buffered samples to Parquet files, organizing by aircraft type."""
    if not buffer_samples:
        return
    
    ac_dir = out_dir / ac_type
    ac_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.concat(buffer_samples, ignore_index=True)
    
    # Enforce Schema ([Rank/Test Update] Removed fuel_rate_warm)
    target_cols = [
        "timestamp", "latitude", "longitude", "altitude", 
        "groundspeed", "track", "vertical_rate",
        "dt", "gs_vn", "gs_ve", 
        "interval_id", "flight_id", "aircraft_type"
    ]
    
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan
            
    df = df[target_cols]
    
    df["interval_id"] = df["interval_id"].astype("int64")
    # [Rank/Test Update] Removed fuel_rate_warm
    float_cols = ["latitude", "longitude", "altitude", "groundspeed", "track", "vertical_rate", "gs_vn", "gs_ve", "dt"]
    for c in float_cols:
        df[c] = df[c].astype("float64")

    table = pa.Table.from_pandas(df, preserve_index=False)
    
    # Initialize ParquetWriter if not already present
    if ac_type not in writers:
        out_path = ac_dir / "samples.parquet"
        writers[ac_type] = pq.ParquetWriter(out_path, table.schema, compression="zstd")
    
    try:
        writers[ac_type].write_table(table)
    except ValueError as e:
        print(f"\n[Error] Schema mismatch for {ac_type}!")
        raise e
    
    del df
    del table

def main():
    args = parse_args()
    ROOT = Path(args.root)
    OUT = Path(args.out)
    
    # Warn instead of deleting
    if OUT.exists():
        print(f"Warning: Output directory {OUT} already exists. Existing files will be overwritten/appended.")
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Load metadata ([Rank/Test Update] Use fuel_rank_submission and flightlist_rank)
    rank_file = ROOT / "fuel_rank_submission.parquet"
    list_file = ROOT / "flightlist_rank.parquet"
    
    print(f"Reading rank data: {rank_file}")
    try:
        fuel_df = pd.read_parquet(rank_file)
        flightlist = pd.read_parquet(list_file)[["flight_id", "aircraft_type"]]
        id2type = dict(zip(flightlist.flight_id.values, flightlist.aircraft_type.values))
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {e}")
        return
    
    # Ensure datetime format
    for col in ["start", "end"]:
        if not np.issubdtype(fuel_df[col].dtype, np.datetime64):
            fuel_df[col] = pd.to_datetime(fuel_df[col], utc=True)
    
    print(f"Processing {len(fuel_df)} tasks with Stream Flushing...")
    print(f"Input Root: {ROOT}")
    print(f"Output Root: {OUT}")
    print(f"Workers: {args.max_workers}")
    
    sample_buffers = defaultdict(list)
    buffer_row_counts = defaultdict(int)
    global_stats = defaultdict(list)
    id_counters = defaultdict(int)
    writers = {}
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_single_row, row, id2type, ROOT)
                for _, row in fuel_df.iterrows()
            ]
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Streaming"):
                try:
                    res = future.result()
                    if res[0] is None:
                        continue
                    
                    ac_type, sample_df, stat_dict = res
                    
                    # Assign Interval ID
                    current_id = id_counters[ac_type]
                    sample_df["interval_id"] = current_id
                    stat_dict["interval_id"] = current_id
                    id_counters[ac_type] += 1
                    
                    global_stats[ac_type].append(stat_dict)
                    sample_buffers[ac_type].append(sample_df)
                    buffer_row_counts[ac_type] += len(sample_df)
                    
                    # Flush if buffer is full
                    if buffer_row_counts[ac_type] >= FLUSH_THRESHOLD:
                        flush_buffer_single_file(ac_type, sample_buffers[ac_type], OUT, writers)
                        sample_buffers[ac_type] = []
                        buffer_row_counts[ac_type] = 0
                        
                except Exception as e:
                    if isinstance(e, KeyError) and "'idx'" in str(e):
                        print(f"\nCRITICAL ERROR: 'idx' column missing in input data!")
                        raise e 
                    print(f"[Warning] Flight processing failed: {e}")

        print("Flushing remaining buffers...")
        for ac_type in tqdm(global_stats.keys(), desc="Finalizing"):
            if buffer_row_counts[ac_type] > 0:
                flush_buffer_single_file(ac_type, sample_buffers[ac_type], OUT, writers)
            
            # Save Statistics
            stats_df = pd.DataFrame(global_stats[ac_type])
            stats_out = OUT / ac_type / "interval_stats.parquet"
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            stats_df.to_parquet(stats_out, index=False)
            
    finally:
        print("Closing all parquet writers...")
        for writer in writers.values():
            try:
                writer.close()
            except Exception:
                pass
        print(f"Done! Data saved to {OUT}")

if __name__ == "__main__":
    main()