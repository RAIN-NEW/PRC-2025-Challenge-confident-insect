import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
import json
import pyarrow.parquet as pq
import warnings
import math
import random

warnings.filterwarnings("ignore")

class ERA5IterableDataset(IterableDataset):
    """
    Iterable Dataset for loading flight trajectory data with ERA5 weather features.
    
    Expected Data Structure:
        data_root/
            AircraftTypeA/
                samples.parquet       (Data)
                interval_stats.parquet (Metadata/Splitting info)
            AircraftTypeB/
                ...
    
    Features:
        - Loads data in a streamed fashion (memory efficient).
        - Handles Train/Validation splitting based on 'interval_stats'.
        - Normalizes features using pre-computed statistics.
        - Adds One-Hot encoding for aircraft types.
    """
    def __init__(self, data_root, ac_type=None, mode='train',
                 cache_stats=True, shuffle_buffer_size=2000,
                 all_available_ac_types=None,
                 split='train',        
                 val_ratio=0.2,        
                 random_seed=42,      
                 external_stats=None  
                 ):
        
        self.root = Path(data_root)
        self.ac_type = ac_type
        self.shuffle_buffer_size = shuffle_buffer_size
        
        self.split = split
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        # 13 Physical Features (Aligned with 04_add_advanced_features output)
        self.feature_cols = [
            'gs_ve', 'gs_vn', 'as_u', 'as_v',
            'altitude', 'vertical_rate',
            'longitude', 'latitude',
            'gs_vn_dt',      
            'gs_ve_dt',      
            'vertical_rate_dt',
            'specific_energy',  # Specific Energy
            'energy_rate'       # Energy Rate
        ]
        self.dt_col = 'dt'
        self.fuel_col = 'fuel_rate_warm'
        self.group_key = 'interval_id'
        self.cat_col = 'aircraft_type'

        if all_available_ac_types is None:
            raise ValueError("Must provide all_available_ac_types list.")
        
        self.ac_map = {name: i for i, name in enumerate(sorted(all_available_ac_types))}
        self.num_ac_classes = len(all_available_ac_types)
        # print(f"[INFO] Dataset ({self.split}) configured with {self.num_ac_classes} aircraft types.")

        self.files = self._scan_files()
        
        self.estimated_len = 0
        if external_stats is not None:
            # Using external stats (usually for validation/test sets to use train stats)
            self.feat_mean = external_stats['feat_mean']
            self.feat_std = external_stats['feat_std']
            self.dt_mean = external_stats['dt_mean']
            self.dt_std = external_stats['dt_std']
            
            # Read total_samples from external_stats
            total_samples = external_stats.get('total_samples', 0)
            
            # Compatibility fallback for older stats files
            if total_samples == 0:
                 total_steps = external_stats.get('total_steps', 0)
                 total_samples = int(total_steps / 1000)

            if split == 'val':
                self.estimated_len = int(total_samples * val_ratio)
            else:
                self.estimated_len = int(total_samples * (1 - val_ratio))
        else:
            # Compute stats from data
            self._init_stats(cache_stats)
            # Adjust estimated length based on split ratio
            if self.split == 'train':
                self.estimated_len = int(self.estimated_len * (1 - self.val_ratio))
            else:
                self.estimated_len = int(self.estimated_len * self.val_ratio)

    def _scan_files(self):
        """Scans for parquet files in the root directory."""
        if self.ac_type:
            dir_path = self.root / self.ac_type
            # print(f"[INFO] Loading aircraft type: {self.ac_type}")
            files = sorted(list(dir_path.glob("samples.parquet")))
        else:
            # print("[INFO] Global mode: scanning all samples.parquet ...")
            files = sorted(list(self.root.rglob("samples.parquet")))

        if not files:
            raise FileNotFoundError(f"No samples.parquet found in {self.root}")
        return files

    def _init_stats(self, cache_stats):
            """Computes or loads feature statistics (mean/std)."""
            if self.ac_type:
                stats_file = self.root / self.ac_type / "stats.json"
            else:
                stats_file = self.root / "global_stats.json"

            if cache_stats and stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                feat_mean = np.array(stats['feat_mean'], dtype=np.float32)
                
                if feat_mean.shape[0] == len(self.feature_cols):
                    self.feat_mean = feat_mean
                    self.feat_std = np.array(stats['feat_std'], dtype=np.float32)
                    self.dt_mean = stats['dt_mean']
                    self.dt_std = stats['dt_std']
                    
                    # Prioritize reading total_samples. If missing, estimate from total_steps.
                    self.total_samples = stats.get('total_samples', int(stats.get('total_steps', 0) / 1000))
                    
                    # Set initial estimated_len (split logic applied later in __init__)
                    self.estimated_len = self.total_samples
                    
                    # print(f"[INFO] Loaded cached stats for {self.ac_type} (Total samples: {self.total_samples})")
                    return
                else:
                    print(f"[WARN] Stats dimension mismatch. Recomputing...")

            print(f"[INFO] Computing stats for {self.ac_type} (streamed)...")
            sum_feats = np.zeros(len(self.feature_cols), dtype=np.float64)
            sq_sum_feats = np.zeros(len(self.feature_cols), dtype=np.float64)
            sum_dt = 0.0
            sq_sum_dt = 0.0
            total_steps = 0
            total_samples_count = 0

            # stats_mode=True ensures iteration over all data without shuffling
            for item in self._data_generator(self.files, infinite=False, stats_mode=True):
                f = item['feats']          
                d = item['raw_dt']          
                sum_feats += f.sum(axis=0)
                sq_sum_feats += (f**2).sum(axis=0)
                sum_dt += d.sum()
                sq_sum_dt += (d**2).sum()
                total_steps += f.shape[0]
                total_samples_count += 1

            self.feat_mean = (sum_feats / total_steps).astype(np.float32)
            feat_var = sq_sum_feats / total_steps - self.feat_mean**2
            self.feat_std = np.sqrt(np.maximum(feat_var, 1e-6)).astype(np.float32)

            self.dt_mean = float(sum_dt / total_steps)
            dt_var = sq_sum_dt / total_steps - self.dt_mean**2
            self.dt_std = float(np.sqrt(max(dt_var, 1e-6)))

            self.total_samples = total_samples_count
            self.estimated_len = total_samples_count
            
            print(f"[INFO] Stats computed. Total VALID steps: {total_steps}, Total SAMPLES: {total_samples_count}")

            if cache_stats:
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_file, 'w') as f:
                    json.dump({
                        "feat_mean": self.feat_mean.tolist(),
                        "feat_std": self.feat_std.tolist(),
                        "dt_mean": self.dt_mean,
                        "dt_std": self.dt_std,
                        "total_steps": total_steps,
                        "total_samples": total_samples_count
                    }, f)

    def _get_valid_interval_ids(self, samples_path: Path, no_split=False):
        """
        Reads interval_stats.parquet to filter valid intervals based on coverage 
        and perform Train/Validation splitting.
        """
        if not hasattr(self, "_valid_interval_cache"):
            self._valid_interval_cache = {}

        # Cache key includes 'no_split' status to distinguish stats mode from training mode
        key = str(samples_path) + f"_{self.split}_{no_split}"
        if key in self._valid_interval_cache:
            return self._valid_interval_cache[key]

        stats_path = samples_path.with_name("interval_stats.parquet")
        if not stats_path.exists():
            return None

        try:
            stats_table = pq.read_table(stats_path, columns=[self.group_key, "coverage_ratio"])
            stats_df = stats_table.to_pandas()
            
            # Filter based on coverage ratio to exclude poor quality data
            stats_df = stats_df[stats_df["coverage_ratio"] >= 0.2]
            
            if stats_df.empty:
                self._valid_interval_cache[key] = {}
                return {}

            all_ids = sorted(stats_df[self.group_key].unique())
            
            # Logic: If no_split is True (stats computation), return all qualified IDs.
            # Otherwise, perform Train/Val split.
            if no_split:
                selected_ids = set(all_ids)
            else:
                rng = random.Random(self.random_seed)
                rng.shuffle(all_ids)
                
                split_idx = int(len(all_ids) * (1 - self.val_ratio))
                if self.split == 'train':
                    selected_ids = set(all_ids[:split_idx])
                else:
                    selected_ids = set(all_ids[split_idx:])
            
            stats_df = stats_df[stats_df[self.group_key].isin(selected_ids)]
            interval2cov = dict(zip(stats_df[self.group_key].tolist(), stats_df["coverage_ratio"].tolist()))
            self._valid_interval_cache[key] = interval2cov
            return interval2cov
            
        except Exception as e:
            print(f"[WARN] Failed to read interval_stats: {e}")
            self._valid_interval_cache[key] = None
            return None

    def _process_dataframe_chunk(self, df, interval2cov=None, min_len=10):
        """Processes a dataframe chunk into individual samples."""
        grouped = df.groupby(self.group_key)
        chunk_samples = []

        for interval_id, group in grouped:
            # Check validity via interval2cov
            if interval2cov is not None:
                cov = interval2cov.get(interval_id, None)
                if cov is None: continue # Filtered out ID
                cov = float(cov)
            else:
                cov = 1.0

            # Ensure all required columns are present
            group = group.dropna(subset=self.feature_cols + [self.dt_col, self.fuel_col, self.cat_col])
            if len(group) < min_len:
                continue

            feats = group[self.feature_cols].values.astype(np.float32)
            raw_dt = group[self.dt_col].values.astype(np.float32)
            fuel_rate = group[self.fuel_col].values.astype(np.float32)
            
            ac_str = group[self.cat_col].iloc[0]
            if ac_str not in self.ac_map:
                continue
            ac_idx = self.ac_map[ac_str]

            chunk_samples.append({
                'feats': feats,
                'raw_dt': raw_dt,
                'fuel_rate': fuel_rate,
                'coverage_ratio': cov,
                'ac_idx': ac_idx
            })
        return chunk_samples

    def _data_generator(self, file_list, infinite=False, stats_mode=False):
        """
        Main generator that yields samples from parquet files.
        """
        while True:
            current_files = list(file_list)
            if not stats_mode:
                random.shuffle(current_files)

            for f in current_files:
                try:
                    # Get valid IDs with splitting logic applied
                    interval2cov = self._get_valid_interval_ids(f, no_split=stats_mode)
                    
                    # Skip file if no valid IDs remain
                    if interval2cov is not None and len(interval2cov) == 0:
                        continue

                    pf = pq.ParquetFile(f)
                    buffer_df = None
                    read_cols = self.feature_cols + [self.dt_col, self.fuel_col, self.group_key, self.cat_col]

                    for batch in pf.iter_batches(batch_size=50000, columns=read_cols):
                        df_chunk = batch.to_pandas()
                        if buffer_df is not None:
                            df_processing = pd.concat([buffer_df, df_chunk], ignore_index=True)
                        else:
                            df_processing = df_chunk
                        
                        if len(df_processing) == 0: continue
                            
                        # Logic to prevent splitting interval_id across batches
                        last_id = df_processing[self.group_key].iloc[-1]
                        mask_last = df_processing[self.group_key] == last_id
                        
                        if mask_last.all():
                            # Buffer growth protection
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
                            samples = self._process_dataframe_chunk(df_to_process, interval2cov=interval2cov)
                            for s in samples: yield s
                            del df_to_process, samples
                        del df_chunk, df_processing

                    if buffer_df is not None:
                        samples = self._process_dataframe_chunk(buffer_df, interval2cov=interval2cov)
                        for s in samples: yield s
                        del buffer_df

                except Exception as e:
                    print(f"Error reading {f}: {e}")

            if not infinite:
                break

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
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
        shuffle_buffer = []
        one_hot_matrix = np.eye(self.num_ac_classes, dtype=np.float32)

        for item in generator:
            feats = item['feats']         # [L, 13]
            raw_dt = item['raw_dt']
            fuel_rate = item['fuel_rate']
            cov = item.get('coverage_ratio', 1.0)
            ac_idx = item['ac_idx']

            # Calculate Ground Truth Label (Total Fuel for the interval)
            total_fuel = np.sum(fuel_rate * raw_dt)
            
            # 1. Normalize continuous features
            norm_feats = (feats - self.feat_mean) / self.feat_std
            norm_dt = (raw_dt - self.dt_mean) / self.dt_std
            norm_dt = norm_dt[:, np.newaxis]

            # 2. Generate One-Hot Features
            ac_vec = one_hot_matrix[ac_idx:ac_idx+1, :]
            ac_vec_repeated = np.repeat(ac_vec, feats.shape[0], axis=0)

            # 3. Generate Coverage Feature
            cov_vec = np.full((feats.shape[0], 1), cov, dtype=np.float32)

            # 4. Concatenate All Features
            # Dim: 13(phys) + 1(dt) + 1(cov) + num_classes(ac_type)
            input_x = np.concatenate([norm_feats, norm_dt, cov_vec, ac_vec_repeated], axis=1)

            processed_sample = {
                'x': torch.tensor(input_x, dtype=torch.float32),
                'raw_dt': torch.tensor(raw_dt, dtype=torch.float32),
                'label': torch.tensor(total_fuel, dtype=torch.float32),
                'coverage_ratio': torch.tensor(cov, dtype=torch.float32)
            }

            if len(shuffle_buffer) < self.shuffle_buffer_size:
                shuffle_buffer.append(processed_sample)
            else:
                idx = random.randint(0, len(shuffle_buffer) - 1)
                yield shuffle_buffer[idx]
                shuffle_buffer[idx] = processed_sample

        random.shuffle(shuffle_buffer)
        for item in shuffle_buffer:
            yield item

    def __len__(self):
        return self.estimated_len
        
    def get_stats(self):
        return {
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'dt_mean': self.dt_mean,
            'dt_std': self.dt_std,
            'total_steps': 0, 
            'total_samples': getattr(self, 'total_samples', 0)
        }

def collate_fn(batch):
    """
    Collate function for DataLoader. Pads sequences to the longest in the batch.
    """
    xs = [item['x'] for item in batch]
    dts = [item['raw_dt'] for item in batch]
    labels = [item['label'] for item in batch]
    covs = [item['coverage_ratio'] for item in batch]

    lengths = torch.tensor([len(x) for x in xs])
    x_padded = pad_sequence(xs, batch_first=True, padding_value=0)
    dt_padded = pad_sequence(dts, batch_first=True, padding_value=0)
    max_len = x_padded.size(1)
    mask = torch.arange(max_len)[None, :] >= lengths[:, None]
    labels = torch.stack(labels)
    covs = torch.stack(covs)
    return x_padded, dt_padded, mask, labels, covs