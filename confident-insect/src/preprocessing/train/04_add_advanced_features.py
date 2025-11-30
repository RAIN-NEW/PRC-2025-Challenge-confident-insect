#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel Feature Engineering for Aircraft Samples (Train Data).

Features Added:
1. **Kinematics**: Synthetic Groundspeed & Airspeed magnitudes.
2. **Energy**: Specific Energy (Es) calculation based on altitude (m) and airspeed (m/s).
3. **Dynamics**: Derivatives (Accelerations) for speed components and vertical rate.
4. **Power**: Energy Rate (Specific Excess Power, Ps).

Input: Parquet files from '03_add_airspeed' step.
Output: Final processed samples ready for model training/inference.
"""

import warnings
import argparse
import shutil
import os
import gc
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/preprocessing/train/04_add_advanced_features.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[3]
except IndexError:
    project_root = Path.cwd()

# Define paths relative to project root (Train Dataset)
DEFAULT_SRC = project_root / "data" / "intermediate" / "03_add_airspeed" / "train"
DEFAULT_DST = project_root / "data" / "intermediate" / "04_add_advanced_features" / "train"

# Memory management: Total memory = BATCH_SIZE * max_workers * row_size
DEFAULT_BATCH_SIZE = 200_000 

# ===============================================

def parse_args():
    ap = argparse.ArgumentParser(description="Parallel Feature Engineering for Aircraft Samples (with Energy)")
    
    # Determine default workers: half of available CPUs (at least 1)
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)

    ap.add_argument("--src", type=str, default=str(DEFAULT_SRC), help=f"Source directory. Default: {DEFAULT_SRC}")
    ap.add_argument("--dst", type=str, default=str(DEFAULT_DST), help=f"Destination directory. Default: {DEFAULT_DST}")
    ap.add_argument("--workers", type=int, default=default_workers, help=f"Number of parallel processes. Default: {default_workers}")
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Rows per batch per worker. Default: {DEFAULT_BATCH_SIZE}")
    return ap.parse_args()

def process_chunk(df_chunk):
    """
    Core Calculation Logic:
    1. Calculate Groundspeed Magnitude
    2. Calculate Airspeed Magnitude
    3. [New] Calculate Specific Energy (Es)
    4. Calculate Accelerations (dt-related features)
    5. [New] Calculate Energy Rate (Ps)
    6. Rounding for storage efficiency
    """
    # 1. Type conversion and sorting
    if not np.issubdtype(df_chunk["timestamp"].dtype, np.datetime64):
        df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'])
    
    # Ensure correct order for differentiation
    df_chunk = df_chunk.sort_values(by=['interval_id', 'timestamp'])

    # ================= [Velocity Magnitudes] =================
    # Re-calculate/Verify groundspeed
    if 'gs_vn' in df_chunk.columns and 'gs_ve' in df_chunk.columns:
        df_chunk['groundspeed'] = np.hypot(df_chunk['gs_vn'], df_chunk['gs_ve'])
    
    # Calculate airspeed magnitude
    if 'as_u' in df_chunk.columns and 'as_v' in df_chunk.columns:
        df_chunk['airspeed'] = np.hypot(df_chunk['as_u'], df_chunk['as_v'])
    else:
        df_chunk['airspeed'] = 0.0

    # ================= [Specific Energy] =================
    # Formula: Es = h + V^2 / 2g
    # Unit Assumption: Altitude in meters (m), Airspeed in m/s
    g = 9.80665
    
    # Use Airspeed for kinetic energy (standard for aerodynamic performance)
    if 'altitude' in df_chunk.columns:
        df_chunk['specific_energy'] = df_chunk['altitude'] + (df_chunk['airspeed'] ** 2) / (2 * g)
    else:
        df_chunk['specific_energy'] = 0.0

    # ================= [Accelerations & Rates] =================
    # Get dt, replace 0 with NaN to avoid division errors
    dt = df_chunk['dt'].replace(0, np.nan)
    
    # Helper for grouped differentiation
    # Formula: (Next_Val - Curr_Val) / dt
    # -diff(-1) is equivalent to (Curr - Next) * -1 => (Next - Curr)
    def calc_acc(col_name):
        return -df_chunk.groupby('interval_id')[col_name].diff(-1) / dt

    # Groundspeed component accelerations
    df_chunk['gs_ve_dt'] = calc_acc('gs_ve')
    df_chunk['gs_vn_dt'] = calc_acc('gs_vn')
    
    # Vertical acceleration
    df_chunk['vertical_rate_dt'] = calc_acc('vertical_rate')

    # Airspeed component accelerations
    if 'as_u' in df_chunk.columns:
        df_chunk['as_u_dt'] = calc_acc('as_u')
    if 'as_v' in df_chunk.columns:
        df_chunk['as_v_dt'] = calc_acc('as_v')

    # [New] Energy Rate (Specific Excess Power, Ps = dEs/dt)
    df_chunk['energy_rate'] = calc_acc('specific_energy')

    # ================= [Cleaning & Filling] =================
    fill_cols = [
        'gs_ve_dt', 'gs_vn_dt', 'vertical_rate_dt', 
        'as_u_dt', 'as_v_dt', 
        'groundspeed', 'airspeed',
        'specific_energy', 'energy_rate'
    ]
    
    # Only fill columns that exist
    actual_fill_cols = [c for c in fill_cols if c in df_chunk.columns]
    
    # Forward fill to handle NaN at segment boundaries
    df_chunk[actual_fill_cols] = df_chunk[actual_fill_cols].ffill()
    
    # Handle infinite values
    df_chunk[actual_fill_cols] = df_chunk[actual_fill_cols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ================= [Rounding] =================
    
    # 1. Magnitudes/Energy (1 decimal)
    cols_1_dec = ['groundspeed', 'airspeed', 'vertical_rate', 'specific_energy']
    for col in cols_1_dec:
        if col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].round(1)

    # 2. Components/Rates (3 decimals)
    cols_3_dec = ['gs_vn', 'gs_ve', 'as_u', 'as_v', 'energy_rate']
    for col in cols_3_dec:
        if col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].round(3)

    # 3. Accelerations (6 decimals)
    cols_6_dec = ['vertical_rate_dt', 'gs_ve_dt', 'gs_vn_dt', 'as_u_dt', 'as_v_dt']
    for col in cols_6_dec:
        if col in df_chunk.columns:
            df_chunk[col] = df_chunk[col].round(6)

    return df_chunk

def process_single_aircraft_task(ac_type_dir, dst_root, batch_size):
    """
    Worker function: Process a single aircraft type directory.
    """
    ac_type = ac_type_dir.name
    src_sample_path = ac_type_dir / "samples.parquet"
    src_stats_path = ac_type_dir / "interval_stats.parquet"

    # Skip if source missing
    if not src_sample_path.exists():
        return f"Skipped (No Data): {ac_type}"

    # Prepare destination
    dst_dir = Path(dst_root) / ac_type
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_sample_path = dst_dir / "samples.parquet"
    dst_stats_path = dst_dir / "interval_stats.parquet"

    # === Streaming Processing ===
    pf = pq.ParquetFile(src_sample_path)
    writer = None
    last_interval_rows = None 
    
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            df = batch.to_pandas()
            
            # Concat residue from previous batch
            if last_interval_rows is not None:
                df = pd.concat([last_interval_rows, df], ignore_index=True)
                last_interval_rows = None
            
            if not df.empty:
                # Ensure we don't split an interval_id across batches
                last_id = df['interval_id'].iloc[-1]
                mask_remain = df['interval_id'] == last_id
                
                # Extreme case: entire batch is one interval
                if mask_remain.all():
                    last_interval_rows = df
                    continue
                
                last_interval_rows = df[mask_remain].copy()
                df_to_process = df[~mask_remain].copy()
            else:
                df_to_process = df
                
            # Process and Write
            if not df_to_process.empty:
                df_processed = process_chunk(df_to_process)
                # Drop pandas index for clean parquet
                table = pa.Table.from_pandas(df_processed, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(dst_sample_path, table.schema, compression='zstd')
                writer.write_table(table)
                
                del df_processed, table, df_to_process
                gc.collect()

        # Process Final Residue
        if last_interval_rows is not None and not last_interval_rows.empty:
            df_processed = process_chunk(last_interval_rows)
            table = pa.Table.from_pandas(df_processed, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(dst_sample_path, table.schema, compression='zstd')
            writer.write_table(table)

    except Exception as e:
        return f"Error in {ac_type}: {str(e)}"
    
    finally:
        if writer:
            writer.close()

    # Copy stats file
    if src_stats_path.exists():
        try:
            shutil.copy2(src_stats_path, dst_stats_path)
        except Exception as e:
            return f"Error copying stats for {ac_type}: {str(e)}"

    return f"Success: {ac_type}"

def main():
    args = parse_args()
    
    src_root = Path(args.src)
    dst_root = Path(args.dst)
    
    if not src_root.exists():
        print(f"Source directory not found: {src_root}")
        return

    # Scan for tasks (subdirectories)
    subdirs = [d for d in src_root.iterdir() if d.is_dir()]
    total_tasks = len(subdirs)
    
    print(f"Found {total_tasks} aircraft types to process.")
    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Starting pool with {args.workers} workers. Batch size: {args.batch_size}")
    print("-" * 50)

    results = {"success": [], "error": [], "skipped": []}

    # === Parallel Execution ===
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_ac = {
            executor.submit(
                process_single_aircraft_task, 
                subdir, 
                dst_root, 
                args.batch_size
            ): subdir.name for subdir in subdirs
        }
        
        for future in tqdm(as_completed(future_to_ac), total=total_tasks, desc="Total Progress"):
            ac_name = future_to_ac[future]
            try:
                status_msg = future.result()
                
                if status_msg.startswith("Success"):
                    results["success"].append(ac_name)
                elif status_msg.startswith("Error"):
                    results["error"].append(f"{ac_name}: {status_msg}")
                    print(f"\n[!] {status_msg}")
                else:
                    results["skipped"].append(ac_name)
                    
            except Exception as exc:
                error_msg = f"{ac_name} generated an exception: {exc}"
                print(f"\n[!] {error_msg}")
                traceback.print_exc()
                results["error"].append(error_msg)

    # === Summary ===
    print("\n" + "="*50)
    print("Processing Complete Summary:")
    print(f"  Success: {len(results['success'])}")
    print(f"  Skipped: {len(results['skipped'])}")
    print(f"  Errors : {len(results['error'])}")
    
    if results["error"]:
        print("\nErrors Details:")
        for err in results["error"]:
            print(f"  - {err}")
    
    print(f"\nData saved to: {dst_root}")

if __name__ == "__main__":
    main()