#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate ERA5 Weather Data Download List (Train Dataset).

Purpose:
    Scans trajectory statistics (interval_stats.parquet) to determine which
    ERA5 grid tiles, dates, hours, and pressure levels are required.
    Outputs a optimized JSON task list for batch downloading.

Key Features:
    - **Grid Aggregation**: Groups requests by spatial grid (default 20x20 degrees) to minimize API calls.
    - **IDL Support**: Handles trajectories crossing the International Date Line (split into East/West bounding boxes).
    - **Pressure Level Selection**: dynamically selects standard pressure levels based on flight altitude range.
    - **Parallel Processing**: Uses multiprocessing to scan statistic files efficiently.

Input: interval_stats.parquet files from '02_build_script/train'
Output: JSON file containing download instructions.
"""

import argparse
import json
import math
import multiprocessing
import os
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from ambiance import Atmosphere
from tqdm import tqdm

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/tools/generate_era5_jobs_train.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[2]
except IndexError:
    project_root = Path.cwd()

# Define paths relative to project root
DEFAULT_ROOT = project_root / "data" / "intermediate" / "02_build_script" / "train"
DEFAULT_OUT = project_root / "data" / "intermediate" / "climate_datasets" / "download_lists" / "era5_download_train_GRID.json"

# Grid size for aggregation (degrees)
DEFAULT_GRID_SIZE = 20.0
# Padding around trajectory bounds (degrees)
DEFAULT_PADDING = 0.25

# ===============================================

# --- Global Helper Functions (Pickle-able) ---
def new_hour_pressure_dict():
    return {"hours": set(), "pressure_levels": set()}

def new_date_dict():
    return defaultdict(new_hour_pressure_dict)

def new_demand_dict():
    return defaultdict(new_date_dict)

# --- Grid ID Calculation ---
def get_grid_id(lat, lon, grid_size):
    """
    Calculate Grid ID based on lat/lon (e.g., N20E100).
    Aligns to the bottom-left corner of the grid.
    """
    # Floor to align with grid base
    lat_base = math.floor(lat / grid_size) * grid_size
    lon_base = math.floor(lon / grid_size) * grid_size
    
    # Special handling for Longitude 180:
    # Prevent 180 from becoming E180 (start of next grid), force it to belong to the previous grid (E160 for grid=20).
    if lon_base >= 180: 
        lon_base = 180 - grid_size
    
    lat_part = f"N{int(lat_base)}" if lat_base >= 0 else f"S{int(abs(lat_base))}"
    lon_part = f"E{int(lon_base)}" if lon_base >= 0 else f"W{int(abs(lon_base))}"
    return f"{lat_part}{lon_part}"

def get_grids_for_interval(min_lat, max_lat, min_lon, max_lon, grid_size):
    """
    Identify all Grid IDs covered by a rectangular bounding box.
    """
    grids = set()
    lat_start = np.floor(min_lat / grid_size) * grid_size
    lat_end = np.floor(max_lat / grid_size) * grid_size
    lon_start = np.floor(min_lon / grid_size) * grid_size
    lon_end = np.floor(max_lon / grid_size) * grid_size
    
    # Iterate through Latitude
    for lat in np.arange(lat_start, lat_end + grid_size, grid_size):
        # Iterate through Longitude
        for lon in np.arange(lon_start, lon_end + grid_size, grid_size):
            
            # [Filter Logic] Latitude
            # Skip if the grid top is below -90 or grid bottom is above 90
            if (lat + grid_size) <= -90 or lat >= 90: 
                continue
            
            # [Filter Logic] Longitude
            # Only process base points within [-180, 180)
            if lon < -180 or lon >= 180: 
                continue

            grids.add(get_grid_id(lat, lon, grid_size))
    return grids

def parse_grid_id_to_area(grid_id, grid_size):
    """
    Parse Grid ID (e.g., N20W120) into ERA5 area format [North, West, South, East].
    Includes clamping logic to prevent out-of-bound coordinates.
    """
    parts = re.match(r"([NS])(\d+)([WE])(\d+)", grid_id)
    if not parts:
        raise ValueError(f"Invalid grid_id: {grid_id}")
    
    lat_sign = 1 if parts.group(1) == 'N' else -1
    lon_sign = 1 if parts.group(3) == 'E' else -1
    
    lat_base = int(parts.group(2)) * lat_sign
    lon_base = int(parts.group(4)) * lon_sign
    
    # --- Clamping Logic ---
    # Latitude limit [-90, 90]
    north = min(lat_base + grid_size, 90.0)
    south = max(lat_base, -90.0)
    
    # Longitude limit [-180, 180]
    east = min(lon_base + grid_size, 180.0)
    west = max(lon_base, -180.0)
    
    # ERA5 format: [North, West, South, East]
    area = [north, west, south, east]
    return area

def get_bracketing_pressure_levels(p_min, p_max, all_levels_desc):
    """Select standard pressure levels that bracket the given pressure range."""
    if p_min > p_max: p_min, p_max = p_max, p_min
    try:
        # Find indices in descending array
        idx_low_bracket = np.where(all_levels_desc >= p_max)[0][-1]
        idx_high_bracket = np.where(all_levels_desc <= p_min)[0][0]
        
        if idx_high_bracket > idx_low_bracket: 
            idx_high_bracket, idx_low_bracket = idx_low_bracket, idx_high_bracket
            
        return set(all_levels_desc[idx_high_bracket : idx_low_bracket + 1])
    except IndexError:
        # Fallback default
        return {1000, 500, 300, 200}

# --- Worker Function ---
def process_file(file_path, grid_size, area_padding_deg, all_standard_levels_hpa):
    """
    Process a single interval_stats.parquet file to extract data requirements.
    """
    local_demand = new_demand_dict()
    
    # Columns required
    cols_to_read = [
        'start', 'end', 
        'min_latitude', 'max_latitude', 'min_longitude', 'max_longitude',
        'min_altitude', 'max_altitude',
        'is_cross_idl',                 # IDL crossing flag
        'lon_min_east', 'lon_max_east', # East hemisphere bounds
        'lon_min_west', 'lon_max_west'  # West hemisphere bounds
    ]
    
    try:
        try:
            df = pd.read_parquet(file_path, columns=cols_to_read)
        except Exception:
            # Fallback for older schema (missing IDL columns)
            base_cols = cols_to_read[:8]
            df = pd.read_parquet(file_path, columns=base_cols)
            df['is_cross_idl'] = False
            df['lon_min_east'] = np.nan

        df.dropna(subset=['start', 'end'], inplace=True)
        if df.empty: 
            return local_demand, (file_path.name, "success_empty")

        # Ensure datetime format
        if not np.issubdtype(df["start"].dtype, np.datetime64):
            df['start'] = pd.to_datetime(df['start'], utc=True)
            df['end'] = pd.to_datetime(df['end'], utc=True)

        for row in df.itertuples():
            # 1. Calculate Pressure Levels from Altitude
            min_alt_m = row.min_altitude * 0.3048 # ft to m
            max_alt_m = row.max_altitude * 0.3048
            try:
                atm_low = Atmosphere(min_alt_m); max_p = atm_low.pressure[0] / 100.0 # Pa to hPa
                atm_high = Atmosphere(max_alt_m); min_p = atm_high.pressure[0] / 100.0
                pressure_levels = get_bracketing_pressure_levels(min_p, max_p, all_standard_levels_hpa)
            except Exception:
                pressure_levels = {1000, 500, 300} # Fallback

            # 2. Determine Grid Coverage
            grid_cells = set()
            
            # Base latitude with padding and clamping
            lat_min_pad = max(-90.0, row.min_latitude - area_padding_deg)
            lat_max_pad = min(90.0, row.max_latitude + area_padding_deg)
            
            # === Branch A: IDL Crossing ===
            if getattr(row, 'is_cross_idl', False):
                # 2.1 East Hemisphere Part (Lon > 0)
                if pd.notna(row.lon_min_east) and pd.notna(row.lon_max_east):
                    lon_start = max(-180.0, row.lon_min_east - area_padding_deg)
                    lon_end = min(180.0, row.lon_max_east + area_padding_deg)
                    
                    grids_east = get_grids_for_interval(
                        lat_min_pad, lat_max_pad, lon_start, lon_end, grid_size
                    )
                    grid_cells.update(grids_east)
                
                # 2.2 West Hemisphere Part (Lon < 0)
                if pd.notna(row.lon_min_west) and pd.notna(row.lon_max_west):
                    lon_start = max(-180.0, row.lon_min_west - area_padding_deg)
                    lon_end = min(180.0, row.lon_max_west + area_padding_deg)
                    
                    grids_west = get_grids_for_interval(
                        lat_min_pad, lat_max_pad, lon_start, lon_end, grid_size
                    )
                    grid_cells.update(grids_west)
            
            # === Branch B: Standard Flight ===
            else:
                lon_start = max(-180.0, row.min_longitude - area_padding_deg)
                lon_end = min(180.0, row.max_longitude + area_padding_deg)
                
                grid_cells = get_grids_for_interval(
                    lat_min_pad, lat_max_pad, lon_start, lon_end, grid_size
                )

            # 3. Aggregate Time Requirements per Grid
            start_ts = row.start.floor('h')
            end_ts = row.end.ceil('h')
            
            all_hours_obj = pd.date_range(start_ts, end_ts, freq='h')
            
            for cell in grid_cells:
                for h_obj in all_hours_obj:
                    date_str = h_obj.strftime('%Y-%m-%d')
                    hour_int = h_obj.hour
                    
                    local_demand[cell][date_str]['hours'].add(hour_int)
                    local_demand[cell][date_str]['pressure_levels'].update(pressure_levels)
        
        return local_demand, (file_path.name, "success")

    except Exception as e:
        return dict(local_demand), (file_path.name, str(e))

def parse_args():
    ap = argparse.ArgumentParser(description="Generate ERA5 Download List (Train Data)")
    
    # Determine default workers: half of available CPUs (at least 1)
    cpu_count = os.cpu_count() or 2
    default_workers = max(1, cpu_count // 2)

    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT),
                    help=f"Root directory containing interval_stats.parquet. Default: {DEFAULT_ROOT}")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                    help=f"Output JSON file path. Default: {DEFAULT_OUT}")
    ap.add_argument("--grid_size", type=float, default=DEFAULT_GRID_SIZE, 
                    help=f"Grid size in degrees. Default: {DEFAULT_GRID_SIZE}")
    ap.add_argument("--area_padding_deg", type=float, default=DEFAULT_PADDING, 
                    help=f"Padding degrees for bounding box. Default: {DEFAULT_PADDING}")
    ap.add_argument("--max_workers", type=int, default=default_workers, 
                    help=f"Max parallel workers. Default: {default_workers}")
    return ap.parse_args()

def main():
    args = parse_args()
    ROOT = Path(args.root)
    OUT_FILE = Path(args.out)
    GRID_SIZE = args.grid_size
    
    # ERA5 Standard Pressure Levels (Descending)
    all_standard_levels_hpa = np.array([
        1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 
        450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1
    ])
    
    # 1. Scan Files
    if not ROOT.exists():
        print(f"Error: Source directory {ROOT} does not exist.")
        return

    all_stat_files = list(ROOT.rglob("interval_stats.parquet"))
    if not all_stat_files:
        print(f"Error: No interval_stats.parquet found in {ROOT}")
        return
    print(f"Found {len(all_stat_files)} stat files. Processing with grid size {GRID_SIZE}...")

    # 2. Parallel Aggregation
    final_demand = new_demand_dict()
    
    worker = partial(
        process_file, 
        grid_size=GRID_SIZE, 
        area_padding_deg=args.area_padding_deg,
        all_standard_levels_hpa=all_standard_levels_hpa
    )
    
    print(f"Starting parallel processing with {args.max_workers} workers...")
    with multiprocessing.Pool(processes=args.max_workers) as pool:
        results = pool.imap_unordered(worker, all_stat_files)
        
        for local_demand, (fname, status) in tqdm(results, total=len(all_stat_files)):
            if status == "success":
                # Merge local results into final demand
                for cell, dates in local_demand.items():
                    for date_str, req in dates.items():
                        final_demand[cell][date_str]['hours'].update(req['hours'])
                        final_demand[cell][date_str]['pressure_levels'].update(req['pressure_levels'])
            elif status != "success_empty":
                print(f"Failed {fname}: {status}")

    # 3. Generate Download Jobs
    print("Generating JSON jobs...")
    download_jobs = []
    
    # Iterate through aggregated demand
    for grid_id, dates_data in tqdm(final_demand.items()):
        try:
            # Parse area with clamping
            area = parse_grid_id_to_area(grid_id, GRID_SIZE)
            
            # Validate area dimensions
            if area[0] <= area[2] or area[1] >= area[3]:
                 continue
                 
        except ValueError:
            continue
            
        for date_str, req in dates_data.items():
            hours = sorted(list(req['hours']))
            levels = sorted(list(req['pressure_levels']), reverse=True) # High to Low
            
            if not hours or not levels: continue
            
            # Format requirements
            time_strs = [f"{h:02d}:00" for h in hours]
            level_strs = [str(int(p)) for p in levels]
            y, m, d = date_str.split('-')
            
            job = {
                "dataset": "reanalysis-era5-pressure-levels",
                "product_type": ["reanalysis"],
                "variable": ["u_component_of_wind", "v_component_of_wind", "geopotential"],
                "pressure_level": level_strs,
                "year": [y], "month": [m], "day": [d],
                "time": time_strs,
                "area": area, # [N, W, S, E]
                "data_format": "netcdf",
                "download_format": "unarchived",
                "metadata": {
                    "grid_id": grid_id,
                    "date": date_str
                }
            }
            download_jobs.append(job)

    # 4. Save Output
    output = {
        "metadata": {
            "source": str(ROOT),
            "total_jobs": len(download_jobs),
            "grid_size": GRID_SIZE,
            "logic": "Grid Aggregation with IDL Split"
        },
        "era5_download_jobs": download_jobs
    }
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(output, f, indent=4)
        
    print(f"Done! Generated {len(download_jobs)} jobs. Saved to {OUT_FILE}")

if __name__ == "__main__":
    main()