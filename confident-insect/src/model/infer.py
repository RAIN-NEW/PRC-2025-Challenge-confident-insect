#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
General Model Inference Script.

This script performs inference using a trained FuelPredictionTransformer model.
It reads processed test samples, predicts fuel consumption, and generates a submission file
based on a provided template.

Features:
- **Dynamic Path Resolution**: Adapts to project structure automatically.
- **Robust Mapping**: Aligns predictions with submission IDs using `interval_stats`.
- **Missing Data Handling**: Logs missing samples and fills NaNs with 0.0.
- **Optimization**: Uses efficient Parquet I/O and buffered processing.
"""

import argparse
import json
import math
import os
import shutil
import sys
import traceback
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, get_worker_info
from tqdm import tqdm

# Import local modules
try:
    from dataset import ERA5IterableDataset
    from model import FuelPredictionTransformer
except ImportError:
    # Handle case where script is run from root directory
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import ERA5IterableDataset
    from model import FuelPredictionTransformer

warnings.filterwarnings("ignore")

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/model/test.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[2]
except IndexError:
    project_root = Path.cwd()

# Default Paths
DEFAULT_INPUT_DIR = project_root / "data" / "intermediate" / "04_add_advanced_features" / "final"
DEFAULT_CHECKPOINT_DIR = project_root / "save_model"
DEFAULT_TEMPLATE_PATH = project_root / "data" / "intermediate" / "01_complement_data" / "fuel_final_submission.parquet"
DEFAULT_OUTPUT_FILE = project_root / "submissions" / "confident-insect_final.parquet"
DEFAULT_STATS_FILE = project_root / "data" / "intermediate" / "04_add_advanced_features" / "final" / "global_stats.json"
DEFAULT_LOG_DIR = project_root / "logs"

# Global Aircraft List
ALL_SUPPORTED_AIRCRAFT = [
    "A20N", "A21N", "A306", "A318", "A319", "A320", "A321",
    "A332", "A333", "A359", "A388",
    "B38M", "B39M", 
    "B737", "B738", "B739",
    "B744", "B748", "B752", "B763", "B772", "B77L", "B77W",
    "B788", "B789", 
    "MD11"
]

# =============================================

def parse_args():
    parser = argparse.ArgumentParser(description="General Model Inference")
    
    parser.add_argument("--ac_type", type=str, default="ALL", help="Target aircraft type or 'ALL'.")
    
    parser.add_argument("--input_dir", type=str, default=str(DEFAULT_INPUT_DIR),
                        help=f"Directory containing test samples. Default: {DEFAULT_INPUT_DIR}")
    
    parser.add_argument("--checkpoint_dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                        help=f"Directory containing model checkpoints. Default: {DEFAULT_CHECKPOINT_DIR}")
    
    parser.add_argument("--template_path", type=str, default=str(DEFAULT_TEMPLATE_PATH),
                        help=f"Path to submission template file. Default: {DEFAULT_TEMPLATE_PATH}")
    
    parser.add_argument("--output_file", type=str, default=str(DEFAULT_OUTPUT_FILE),
                        help=f"Path to save final submission file. Default: {DEFAULT_OUTPUT_FILE}")
    
    parser.add_argument("--stats_file", type=str, default=str(DEFAULT_STATS_FILE),
                        help=f"Path to global statistics json. Default: {DEFAULT_STATS_FILE}")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device.")
    
    return parser.parse_args()

# ==========================================
# 1. Optimized Inference Dataset
# ==========================================
class InferenceDataset(ERA5IterableDataset):
    """
    Optimized ERA5IterableDataset subclass for Inference.
    Features:
    - Disables shuffling.
    - Loads ID mappings to align predictions with the submission template.
    - Diagnostics for missing stats/mappings.
    """
    def __init__(self, *args, missing_log_path="missing_mapping_log.txt", **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle_buffer_size = 0 
        self._mapping_cache = {}
        self.missing_log_path = missing_log_path
        self.flight_id_col = 'flight_id'
        
        # Diagnostic counters
        self.stats = {
            "files_total": 0,
            "files_skipped_no_stats": 0,
            "samples_yielded": 0
        }
        
        if not hasattr(self, 'feat_mean'):
             raise RuntimeError(f"[CRITICAL] Stats not initialized. Check your stats_file path.")

    def _init_stats(self, cache_stats):
        # Override to prevent re-calculating stats during inference
        pass

    def calculate_total_samples(self):
        """
        Quickly scan all interval_stats.parquet files to get the exact total sample count.
        Crucial for accurate progress bars.
        """
        total = 0
        for f in self.files:
            stats_path = f.with_name("interval_stats.parquet")
            if stats_path.exists():
                try:
                    # Read metadata only for speed
                    meta = pq.read_metadata(stats_path)
                    total += meta.num_rows
                except:
                    pass
        return total

    def _get_interval_mapping(self, samples_path: Path):
        """Loads mapping from interval_id to (submission_idx, coverage_ratio)."""
        if samples_path in self._mapping_cache:
            return self._mapping_cache[samples_path]

        stats_path = samples_path.with_name("interval_stats.parquet")
        if not stats_path.exists():
            return None

        try:
            cols = [self.group_key, 'coverage_ratio', 'idx_orin', self.flight_id_col]
            pq_file = pq.ParquetFile(stats_path)
            available_cols = pq_file.schema.names
            
            use_flight_id = True
            if self.flight_id_col not in available_cols:
                cols.remove(self.flight_id_col)
                use_flight_id = False

            table = pq.read_table(stats_path, columns=cols)
            df = table.to_pandas()
            
            if 'idx_orin' not in df.columns:
                df['idx_orin'] = df[self.group_key] 

            interval_ids = df[self.group_key].astype(str).str.strip()
            
            if use_flight_id:
                flight_ids = df[self.flight_id_col].astype(str).str.strip()
                keys = zip(interval_ids, flight_ids)
            else:
                keys = zip(interval_ids, ["DUMMY"] * len(interval_ids))

            values = zip(df['idx_orin'], df['coverage_ratio'])
            mapping = dict(zip(keys, values))

            self._mapping_cache[samples_path] = mapping
            return mapping, use_flight_id
        except Exception as e:
            print(f"[ERR] Failed to load mapping from {stats_path}: {e}")
            raise e 

    def _process_dataframe_chunk(self, df, mapping=None, use_flight_id=True, min_len=10):
        if use_flight_id:
            if self.flight_id_col not in df.columns:
                raise RuntimeError(f"[CRITICAL ERROR] Mapping expects '{self.flight_id_col}' but dataframe misses it.")
            grouped = df.groupby([self.group_key, self.flight_id_col])
        else:
            grouped = df.groupby(self.group_key)
        
        chunk_samples = []

        for key, group in grouped:
            if mapping is None: continue
            
            if use_flight_id:
                if not isinstance(key, tuple): continue
                interval_id, flight_id = key
                k_interval = str(interval_id).strip()
                k_flight = str(flight_id).strip()
                if k_interval.endswith(".0"): k_interval = k_interval[:-2]
                lookup_key = (k_interval, k_flight)
            else:
                interval_id = key
                k_interval = str(interval_id).strip()
                if k_interval.endswith(".0"): k_interval = k_interval[:-2]
                lookup_key = (k_interval, "DUMMY")

            info = mapping.get(lookup_key)
            if info is None:
                # Log missing mappings for debugging
                try:
                    with open(self.missing_log_path, "a") as f:
                        f.write(f"Key:{lookup_key}\n")
                except: pass 
                continue
            
            submission_idx, coverage_ratio = info

            cols_check = self.feature_cols + [self.dt_col, self.cat_col]
            group = group.dropna(subset=cols_check)
            if len(group) < min_len:
                continue

            feats = group[self.feature_cols].values.astype(np.float32)
            raw_dt = group[self.dt_col].values.astype(np.float32)
            
            ac_str = group[self.cat_col].iloc[0]
            if ac_str not in self.ac_map:
                continue
            ac_idx = self.ac_map[ac_str]

            chunk_samples.append({
                'feats': feats,
                'raw_dt': raw_dt,
                'submission_idx': submission_idx, 
                'coverage_ratio': float(coverage_ratio),
                'ac_idx': ac_idx 
            })
            
        return chunk_samples

    def _data_generator(self, file_list, infinite=False):
        current_files = list(file_list)
        current_files.sort() 
        
        for f in current_files:
            try:
                self.stats["files_total"] += 1
                mapping_result = self._get_interval_mapping(f)
                
                if mapping_result is None: 
                    print(f"[WARN] Skipped file due to missing interval_stats: {f.name}")
                    self.stats["files_skipped_no_stats"] += 1
                    continue
                
                mapping, use_flight_id = mapping_result

                pf = pq.ParquetFile(f)
                buffer_df = None
                
                read_cols = self.feature_cols + [self.dt_col, self.group_key, self.cat_col]
                if use_flight_id:
                    read_cols.append(self.flight_id_col)
                
                available_cols = pf.schema.names
                if use_flight_id and self.flight_id_col not in available_cols:
                    print(f"[WARN] Data file {f} missing {self.flight_id_col}, skipping...")
                    continue

                for batch in pf.iter_batches(batch_size=50000, columns=read_cols):
                    df_chunk = batch.to_pandas()
                    if buffer_df is not None:
                        df_processing = pd.concat([buffer_df, df_chunk], ignore_index=True)
                    else:
                        df_processing = df_chunk
                    
                    if len(df_processing) == 0: continue

                    last_id = df_processing[self.group_key].iloc[-1]
                    mask_last = df_processing[self.group_key] == last_id
                    
                    if mask_last.all():
                        if len(df_processing) > 200000:
                            df_to_process = df_processing
                            buffer_df = None
                        else:
                            df_to_process = None
                            buffer_df = df_processing
                    else:
                        df_to_process = df_processing[~mask_last]
                        buffer_df = df_processing[mask_last]
                    
                    if df_to_process is not None:
                        samples = self._process_dataframe_chunk(df_to_process, mapping=mapping, use_flight_id=use_flight_id)
                        for s in samples: 
                            self.stats["samples_yielded"] += 1
                            yield s
                        del df_to_process, samples
                    
                    del df_chunk, df_processing

                if buffer_df is not None:
                    samples = self._process_dataframe_chunk(buffer_df, mapping=mapping, use_flight_id=use_flight_id)
                    for s in samples: 
                        self.stats["samples_yielded"] += 1
                        yield s
                    del buffer_df

            except Exception as e:
                print(f"[ERR] Error reading {f}: {e}")

            if not infinite: break

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_files = self.files
        else:
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            my_files = self.files[iter_start:iter_end]

        if len(my_files) == 0: return

        generator = self._data_generator(my_files, infinite=False)
        
        one_hot_matrix = np.eye(self.num_ac_classes, dtype=np.float32)
        
        for item in generator:
            feats = item['feats']
            raw_dt = item['raw_dt']
            submission_idx = item['submission_idx']
            cov = item['coverage_ratio']
            ac_idx = item['ac_idx']

            norm_feats = (feats - self.feat_mean) / self.feat_std
            norm_dt = (raw_dt - self.dt_mean) / self.dt_std
            norm_dt = norm_dt[:, np.newaxis]
            cov_vec = np.full((feats.shape[0], 1), cov, dtype=np.float32)
            ac_vec = one_hot_matrix[ac_idx:ac_idx+1, :]
            ac_vec_repeated = np.repeat(ac_vec, feats.shape[0], axis=0)

            input_x = np.concatenate([norm_feats, norm_dt, cov_vec, ac_vec_repeated], axis=1)

            yield {
                'x': torch.tensor(input_x, dtype=torch.float32),
                'raw_dt': torch.tensor(raw_dt, dtype=torch.float32),
                'submission_idx': submission_idx,
                'coverage_ratio': cov
            }

def inference_collate_fn(batch):
    xs = [item['x'] for item in batch]
    dts = [item['raw_dt'] for item in batch]
    idxs = [item['submission_idx'] for item in batch]
    covs = [item['coverage_ratio'] for item in batch]

    lengths = torch.tensor([len(x) for x in xs])
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    dt_padded = pad_sequence(dts, batch_first=True, padding_value=0)
    
    max_len = x_padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    
    covs = np.array(covs, dtype=np.float32)
    return x_padded, dt_padded, mask, idxs, covs

# ==========================================
# 3. Main Inference Loop
# ==========================================
def main():
    args = parse_args()
    print(f"[INFO] Device: {args.device}")
    
    # Setup Log directory
    log_dir = Path(DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    missing_log_file = log_dir / "inference_missing_mapping.log"
    if missing_log_file.exists():
        missing_log_file.unlink() 
    
    # Load Stats
    if not os.path.exists(args.stats_file):
        raise FileNotFoundError(f"Stats file not found: {args.stats_file}")
    
    print(f"[INFO] Loading GLOBAL stats from {args.stats_file}")
    with open(args.stats_file, 'r') as f:
        global_stats = json.load(f)
    
    for k in ['feat_mean', 'feat_std']:
        global_stats[k] = np.array(global_stats[k], dtype=np.float32)

    # Load Template
    print(f"[INFO] Loading submission template: {args.template_path}")
    df_sub = pd.read_parquet(args.template_path)
    
    # Identify Index Column
    id_col = 'idx'
    if id_col not in df_sub.columns:
        if 'interval_id' in df_sub.columns:
            id_col = 'interval_id'
        else:
            candidates = [c for c in df_sub.columns if 'id' in c.lower()]
            if candidates:
                id_col = candidates[0]
            print(f"[WARN] 'idx' column not found, using '{id_col}'")

    try:
        if df_sub[id_col].dtype == 'O':
            df_sub[id_col] = df_sub[id_col].astype(str)
        else:
            df_sub[id_col] = df_sub[id_col].astype(int)
    except:
        pass
    
    df_sub.set_index(id_col, inplace=True)
    
    TARGET_COL = 'fuel_kg'
    if TARGET_COL not in df_sub.columns:
        print(f"[WARN] '{TARGET_COL}' not in columns, creating it with NaNs.")
        df_sub[TARGET_COL] = np.nan
    
    print(f"[INFO] Template rows: {len(df_sub)}")

    # Check Data Root
    data_root = Path(args.input_dir)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Determine Aircraft Types to Process
    if args.ac_type == "ALL":
        found_ac_types = []
        for p in data_root.iterdir():
            if p.is_dir() and not p.name.startswith('.'): 
                found_ac_types.append(p.name)
        found_ac_types = sorted(found_ac_types)
    else:
        found_ac_types = [args.ac_type]

    print(f"[INFO] Found {len(found_ac_types)} aircraft types: {found_ac_types}")

    # Model Setup
    NUM_PHYS_FEATS = 13
    NUM_AC_CLASSES = len(ALL_SUPPORTED_AIRCRAFT)
    INPUT_DIM = NUM_PHYS_FEATS + 1 + 1 + NUM_AC_CLASSES
    
    print(f"[INFO] Building General Model (Input Dim: {INPUT_DIM})")
    model = FuelPredictionTransformer(
        input_dim=INPUT_DIM,
        d_model=128,
        nhead=4,
        num_encoder_layers=1, 
        dropout=0.0
    ).to(args.device)
    
    # Load Weights
    checkpoint_dir = Path(args.checkpoint_dir)
    model_path = checkpoint_dir / "model_best.pth"
    if not model_path.exists():
        print(f"[WARN] model_best.pth not found, trying model_latest.pth")
        model_path = checkpoint_dir / "model_latest.pth"
        
    print(f"[INFO] Loading weights: {model_path}")
    checkpoint = torch.load(model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    total_predictions = 0

    # === Inference Loop ===
    for ac in found_ac_types:
        print(f"\n>>> Processing Aircraft: {ac} ...")
        
        try:
            dataset = InferenceDataset(
                data_root=args.input_dir,
                ac_type=ac,
                mode='test',
                all_available_ac_types=ALL_SUPPORTED_AIRCRAFT,
                split='test',
                random_seed=42,
                external_stats=global_stats,
                missing_log_path=str(missing_log_file)
            )
            
            if len(dataset.files) == 0:
                print(f"[INFO] No files found for {ac}, skipping.")
                continue

            # Calculate accurate progress bar length
            print(f"[INFO] Scanning files to calculate precise total samples for {ac}...")
            total_valid_samples = dataset.calculate_total_samples()
            loader_len = math.ceil(total_valid_samples / args.batch_size)
            print(f"[INFO] Total valid samples: {total_valid_samples} -> {loader_len} batches")
            
            loader = DataLoader(
                dataset, 
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=inference_collate_fn,
                num_workers=4,
                pin_memory=True
            )
            
            ac_results_dict = {}
            
            with torch.no_grad():
                pbar = tqdm(loader, total=loader_len, desc=f"Infer {ac}", dynamic_ncols=True)
                
                for x, raw_dt, mask, idxs, covs in pbar:
                    x = x.to(args.device)
                    raw_dt = raw_dt.to(args.device)
                    mask = mask.to(args.device)
                    
                    preds = model(x, raw_dt, src_key_padding_mask=mask)
                    preds_np = preds.cpu().numpy()
                    
                    for i, idx in enumerate(idxs):
                        pred_val = float(preds_np[i])
                        cov_val = float(covs[i])
                        
                        # Adjust for coverage (Upscaling prediction to 100% coverage)
                        if cov_val > 1e-3:
                            final_pred = pred_val / cov_val
                        else:
                            final_pred = pred_val
                        
                        ac_results_dict[idx] = final_pred
            
            count_ac = len(ac_results_dict)
            total_predictions += count_ac
            
            if count_ac > 0:
                print(f"Updating template with {count_ac} predictions from {ac}...")
                print(f"[FILTER REPORT] Stats Samples: {total_valid_samples} -> Actual Yielded: {count_ac}")
                
                # Update DataFrame
                current_preds_series = pd.Series(ac_results_dict, name=TARGET_COL)
                if df_sub.index.dtype == 'O':
                    current_preds_series.index = current_preds_series.index.astype(str)
                elif pd.api.types.is_integer_dtype(df_sub.index):
                     current_preds_series.index = current_preds_series.index.astype(int)

                df_sub.update(current_preds_series)
                del ac_results_dict, current_preds_series
            else:
                print(f"[WARN] No valid predictions generated for {ac}. (All filtered out)")
                
        except Exception as e:
            print(f"[ERR] Failed to process {ac}: {e}")
            traceback.print_exc()
            continue

    print("\n>>> Finalizing submission file...")
    
    df_sub.reset_index(inplace=True)
    missing_mask = df_sub[TARGET_COL].isna()
    missing_count = missing_mask.sum()
    
    print("\n" + "="*40)
    print(f"Total Samples: {len(df_sub)}")
    print(f"Predicted:     {len(df_sub) - missing_count} (Accumulated: {total_predictions})")
    print(f"Missing:       {missing_count} ({missing_count/len(df_sub)*100:.2f}%)")
    print("="*40)

    if missing_log_file.exists():
        print(f"\n[DIAGNOSTIC] Missing mappings log saved to: {missing_log_file.absolute()}")

    # Save missing report
    if missing_count > 0:
        print(f"[WARN] Found {missing_count} missing samples. Saving report...")
        output_path = Path(args.output_file)
        missing_report_path = output_path.parent / "missing_report.csv"
        missing_report_path.parent.mkdir(parents=True, exist_ok=True)
        df_sub[missing_mask].to_csv(missing_report_path, index=False)
        print(f"[INFO] Missing report saved to {missing_report_path}")

    print(f"[FINAL CHECK] Ensuring no NaNs remain in '{TARGET_COL}'...")
    df_sub[TARGET_COL] = df_sub[TARGET_COL].fillna(0.0)
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_parquet(output_path)
    print(f"[SUCCESS] Saved submission to {output_path}")

if __name__ == "__main__":
    main()