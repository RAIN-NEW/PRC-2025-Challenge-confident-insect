#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add Airspeed Components with ERA5 Data (Multi-Core Parallel Version) for Rank Data.

Key Features:
- **Optimization**: CHUNK_SIZE adjusted for memory efficiency (targeting 32GB RAM systems).
- **Parallelization**: Fully utilizes available cores for processing.
- **Robustness**: Handles correct groupby logic for large datasets to prevent splitting trajectories.
- **Physics**: Converts Altitude (ft) to Meters, calculates Airspeed vectors from Groundspeed and Wind.

Directory Structure Assumption:
    weather_root/grid_folder/date_folder/filename.nc
"""

import warnings
import argparse
import math
import os
import gc
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/preprocessing/rank/03_add_airspeed.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[3]
except IndexError:
    project_root = Path.cwd()

# Define paths relative to project root (Rank Dataset)
INPUT_ROOT = project_root / "data" / "intermediate" / "02_build_script" / "rank"
OUTPUT_ROOT = project_root / "data" / "intermediate" / "03_add_airspeed" / "rank"
WEATHER_ROOT = project_root / "data" / "intermediate" / "climate_datasets" / "era5_rank"

# Grid size for ERA5 directory structure (degrees)
GRID_SIZE = 20

# Chunk size for processing
# Adjusted based on memory profiling (approx 26GB available).
# 300,000 rows per chunk balances memory usage (peak ~10-14GB with 8 workers) and throughput.
CHUNK_SIZE = 300_000

# ===============================================

def parse_args():
    ap = argparse.ArgumentParser(description="Add ERA5 wind and airspeed components to rank samples.")
    
    # Determine default workers: half of available CPUs (at least 1)
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)

    ap.add_argument("--input", type=str, default=str(INPUT_ROOT), help=f"Path to input samples. Default: {INPUT_ROOT}")
    ap.add_argument("--out", type=str, default=str(OUTPUT_ROOT), help=f"Path to save processed data. Default: {OUTPUT_ROOT}")
    ap.add_argument("--weather", type=str, default=str(WEATHER_ROOT), help=f"Path to ERA5 root. Default: {WEATHER_ROOT}")
    ap.add_argument("--workers", type=int, default=default_workers, help=f"Number of parallel worker processes. Default: {default_workers}")
    return ap.parse_args()

def alt_m_to_hpa(alt_m):
    """
    Convert Altitude (meters) to Pressure (hPa).
    Formula: Standard Atmosphere (Troposphere).
    """
    alt_m = np.clip(alt_m, 0, 70000)
    return 1013.25 * (1 - 2.25577e-5 * alt_m) ** 5.25588

def get_grid_dir_name(lat, lon):
    """
    Calculate the corresponding grid folder name based on latitude and longitude.
    """
    # 1. Base calculation (floor to nearest GRID_SIZE)
    s = math.floor(lat / GRID_SIZE) * GRID_SIZE
    w = math.floor(lon / GRID_SIZE) * GRID_SIZE
    
    # 2. Calculate upper bounds
    n = s + GRID_SIZE
    e = w + GRID_SIZE
    
    # 3. Boundary Clamping
    if s < -90: s = -90
    if n > 90:  n = 90
    
    if w < -180: w = -180
    if e > 180:  e = 180 

    # Special handling for 180 degree longitude
    if w >= 180:
         w = 160
         e = 180

    return f"N({int(n)})W({int(w)})S({int(s)})E({int(e)})"

class ERA5WindProvider:
    def __init__(self, root_dir):
        self.root = Path(root_dir)

    def process_batch(self, df_batch):
        """
        Interpolate wind fields for a batch of DataFrames.
        """
        n_samples = len(df_batch)
        if n_samples == 0:
            return np.array([]), np.array([])

        first_row = df_batch.iloc[0]
        grid_name = first_row['grid_name']
        date_str = first_row['date_str'] 
        
        file_name = f"{grid_name}-{date_str}.nc"
        nc_path = self.root / grid_name / date_str / file_name
        
        if not nc_path.exists():
            return np.full(n_samples, np.nan), np.full(n_samples, np.nan)

        try:
            with xr.open_dataset(nc_path) as ds:
                # ================= Dimension Renaming (Defensive) =================
                rename_map = {}
                if 'time' in ds.dims: rename_map['time'] = 'valid_time'
                if 'level' in ds.dims: rename_map['level'] = 'pressure_level'
                if 'isobaricInhPa' in ds.dims: rename_map['isobaricInhPa'] = 'pressure_level'
                if rename_map:
                    ds = ds.rename(rename_map)

                # Extract query coordinates
                times = df_batch['timestamp'].values
                lats = df_batch['latitude'].values
                lons = df_batch['longitude'].values
                
                # IMPORTANT: Convert Altitude (ft) to Meters (m) for pressure conversion
                alt_m = df_batch['altitude'].values * 0.3048 
                levels = alt_m_to_hpa(alt_m)

                t_da = xr.DataArray(times, dims="track")
                y_da = xr.DataArray(lats, dims="track")
                x_da = xr.DataArray(lons, dims="track")
                z_da = xr.DataArray(levels, dims="track")

                # 4D Linear Interpolation
                u_interp = ds['u'].interp(
                    valid_time=t_da, latitude=y_da, longitude=x_da, pressure_level=z_da, method='linear'
                ).values

                v_interp = ds['v'].interp(
                    valid_time=t_da, latitude=y_da, longitude=x_da, pressure_level=z_da, method='linear'
                ).values

                # Physical Plausibility Check
                # Mask values > 200 m/s as NaN (likely interpolation artifacts or errors)
                mask_invalid = (np.abs(u_interp) > 200) | (np.abs(v_interp) > 200)
                u_interp[mask_invalid] = np.nan
                v_interp[mask_invalid] = np.nan

                return u_interp, v_interp

        except Exception:
            return np.full(n_samples, np.nan), np.full(n_samples, np.nan)

def fill_missing_winds(df, group_col='interval_id'):
    """
    Completion strategy for missing wind data based on trajectory segments.
    """
    # Determine grouping key
    if group_col not in df.columns:
        if 'flight_id' in df.columns:
            group_col = 'flight_id'
        else:
            group_col = None
    
    def _impute_group(group):
        # Interpolate if at least one valid value exists
        if group['wind_u'].notna().any():
            group['wind_u'] = group['wind_u'].interpolate(method='linear', limit_direction='both')
            group['wind_v'] = group['wind_v'].interpolate(method='linear', limit_direction='both')
        
        # Fill edges
        group['wind_u'] = group['wind_u'].ffill().bfill()
        group['wind_v'] = group['wind_v'].ffill().bfill()
        
        # Fallback to 0.0 if entirely missing
        group['wind_u'] = group['wind_u'].fillna(0.0)
        group['wind_v'] = group['wind_v'].fillna(0.0)
        return group

    if group_col:
        return df.groupby(group_col, group_keys=False).apply(_impute_group)
    else:
        return _impute_group(df)

def process_chunk(df_chunk, wind_provider, group_key):
    """
    Core logic for processing a single data chunk.
    Returns:
        - Processed DataFrame
        - n_total: Total points in chunk
        - n_valid_wind: Points with valid ERA5 wind data (before imputation)
    """
    if len(df_chunk) == 0:
        return None, 0, 0

    # ================= Step 0: Force Clipping =================
    df_chunk['latitude'] = df_chunk['latitude'].clip(-90, 90)
    df_chunk['longitude'] = df_chunk['longitude'].clip(-180, 180)
    df_chunk['altitude'] = df_chunk['altitude'].clip(lower=0)
    
    # Ensure UTC
    if pd.api.types.is_datetime64tz_dtype(df_chunk['timestamp']):
        df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)

    # ================= Step 1: Generate Indices =================
    df_chunk['date_str'] = df_chunk['timestamp'].dt.strftime('%Y-%m-%d')
    df_chunk['grid_name'] = df_chunk.apply(lambda row: get_grid_dir_name(row['latitude'], row['longitude']), axis=1)

    df_chunk['wind_u'] = np.nan
    df_chunk['wind_v'] = np.nan

    # ================= Step 2: Batch Interpolation =================
    # Group by Grid and Date to minimize file IO
    for (grid_name, date_str), sub_df in df_chunk.groupby(['grid_name', 'date_str']):
        indices = sub_df.index
        w_u, w_v = wind_provider.process_batch(sub_df)
        df_chunk.loc[indices, 'wind_u'] = w_u
        df_chunk.loc[indices, 'wind_v'] = w_v

    # Statistics: Count valid winds before imputation
    n_total = len(df_chunk)
    valid_mask = df_chunk['wind_u'].notna() & df_chunk['wind_v'].notna()
    n_valid_wind = int(valid_mask.sum())

    # ================= Step 3: Missing Value Imputation =================
    df_chunk = fill_missing_winds(df_chunk, group_col=group_key)

    # ⭐ Step 3.5: Convert Altitude to Meters (Permanent Change)
    df_chunk['altitude'] = df_chunk['altitude'].astype(np.float32) * 0.3048

    # ================= Step 4: Calculate Airspeed =================
    # Airspeed = GroundSpeed_Vector - Wind_Vector
    df_chunk['as_u'] = df_chunk['gs_ve'] - df_chunk['wind_u']
    df_chunk['as_v'] = df_chunk['gs_vn'] - df_chunk['wind_v']

    # ================= Step 5: Cleanup & Type Conversion =================
    cols_to_drop = ['date_str', 'grid_name', 'wind_u', 'wind_v']
    final_df = df_chunk.drop(columns=[c for c in cols_to_drop if c in df_chunk.columns])

    target_cols = ['gs_ve', 'gs_vn', 'as_u', 'as_v', 'altitude', 'vertical_rate', 'dt', 'fuel_rate_warm']
    for c in target_cols:
        if c in final_df.columns:
            final_df[c] = final_df[c].astype(np.float32)
            
    return final_df, n_total, n_valid_wind

# ================= Worker Function =================

def process_aircraft_type_task(args):
    """
    Worker function to process a single aircraft type directory.
    Args: (ac_type_dir, out_root, weather_root)
    """
    ac_type_dir, out_root, weather_root = args
    
    # Instantiate provider inside process to avoid pickling complex objects
    wind_provider = ERA5WindProvider(weather_root)
    
    input_path = ac_type_dir / "samples.parquet"
    if not input_path.exists(): 
        return f"Skipped {ac_type_dir.name} (no input)"

    ac_type = ac_type_dir.name
    out_dir = out_root / ac_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "samples.parquet"

    try:
        pf = pq.ParquetFile(input_path)
        total_rows = pf.metadata.num_rows
        
        schema_names = pf.schema.names
        group_key = 'interval_id' if 'interval_id' in schema_names else 'flight_id'
        if group_key not in schema_names:
            group_key = None
        
        writer = None
        rows_processed = 0
        buffer_df = None
        
        # Statistics Counters
        total_points = 0
        matched_points = 0

        iterator = pf.iter_batches(batch_size=CHUNK_SIZE)
        
        for batch in iterator:
            # 1. Read
            df_current = batch.to_pandas()
            
            # 2. Concat Buffer
            if buffer_df is not None:
                df_processing = pd.concat([buffer_df, df_current], ignore_index=True)
            else:
                df_processing = df_current
            
            # 3. Split by Group (Ensure segments aren't split across chunks)
            if group_key in df_processing.columns and len(df_processing) > 0:
                last_id = df_processing[group_key].iloc[-1]
                mask_last = df_processing[group_key] == last_id
                
                if not mask_last.all():
                    df_to_process = df_processing[~mask_last].copy()
                    buffer_df = df_processing[mask_last].copy()
                else:
                    # Protection against huge groups exceeding memory
                    if len(df_processing) > 2 * CHUNK_SIZE:
                        df_to_process = df_processing
                        buffer_df = None
                    else:
                        df_to_process = None
                        buffer_df = df_processing
            else:
                df_to_process = df_processing
                buffer_df = None
            
            # 4. Process Chunk
            if df_to_process is not None and len(df_to_process) > 0:
                processed_chunk, n_total, n_valid = process_chunk(df_to_process, wind_provider, group_key)

                # Update Stats
                total_points += n_total
                matched_points += n_valid
                
                if processed_chunk is not None:
                    table = pa.Table.from_pandas(processed_chunk)
                    if writer is None:
                        writer = pq.ParquetWriter(out_file, table.schema, compression='zstd')
                    writer.write_table(table)
                    rows_processed += len(processed_chunk)
                
                del df_to_process, processed_chunk
                if 'table' in locals(): del table
            
            del df_current, df_processing
            gc.collect()

        # 5. Process Remaining Buffer
        if buffer_df is not None and len(buffer_df) > 0:
            processed_chunk, n_total, n_valid = process_chunk(buffer_df, wind_provider, group_key)
            total_points += n_total
            matched_points += n_valid

            if processed_chunk is not None:
                table = pa.Table.from_pandas(processed_chunk)
                if writer is None:
                    writer = pq.ParquetWriter(out_file, table.schema, compression='zstd')
                writer.write_table(table)
                rows_processed += len(processed_chunk)
        
        if writer:
            writer.close()
            
        # Copy interval_stats.parquet if it exists
        stats_src = ac_type_dir / "interval_stats.parquet"
        if stats_src.exists():
            try:
                pd.read_parquet(stats_src).to_parquet(out_dir / "interval_stats.parquet", index=False)
            except Exception:
                pass

        # ==========================
        # Report: ERA5 Coverage
        # ==========================
        coverage_ratio = matched_points / total_points if total_points > 0 else 0.0

        info_str = (
            f"[Done] {ac_type}: {rows_processed} rows, "
            f"ERA5 wind matched {matched_points}/{total_points} "
            f"({coverage_ratio:.2%})"
        )

        print(info_str)

        # Save coverage stats
        try:
            with open(out_dir / "wind_coverage.txt", "w") as f:
                f.write(info_str + "\n")
        except Exception:
            pass

        return info_str

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"[Error] {ac_type}: {str(e)}"

def main():
    args = parse_args()
    IN = Path(args.input)
    OUT = Path(args.out)
    WEATHER_ROOT = Path(args.weather)
    WORKERS = args.workers
    
    if not IN.exists():
        raise FileNotFoundError(f"Input directory {IN} does not exist.")
    
    OUT.mkdir(parents=True, exist_ok=True)
    
    subdirs = [d for d in IN.iterdir() if d.is_dir()]
    tasks = [(d, OUT, WEATHER_ROOT) for d in subdirs]
    
    print(f"Starting parallel processing with {WORKERS} workers for {len(tasks)} aircraft types.")
    print(f"Input: {IN}")
    print(f"Output: {OUT}")
    print(f"Weather: {WEATHER_ROOT}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(process_aircraft_type_task, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Total Progress"):
            result = future.result()

    print(f"\nAll Done. Output: {OUT}")

if __name__ == "__main__":
    main()