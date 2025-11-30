# PRC 2025 Aviation Fuel Consumption Prediction Solution (Confident Insect)

This repository contains the complete solution for the **PRC 2025 Aviation Fuel Consumption Prediction Challenge**. The project implements a modular data processing pipeline and a Transformer-based deep learning model to automate the entire workflow—from raw trajectory cleaning and weather data integration to feature engineering and final fuel consumption prediction.

## 📁 Repository Structure

The repository is organized into data, source code, and configuration layers. Please note that due to file upload restrictions, **directories intended for specific data files contain only `.gitkeep` placeholders**.

```
confident-insect/
├── data/
│   ├── raw/                        # [Raw Data Storage]
│   │   └── prc-2025-datasets/      # Original datasets provided by organizers (Placeholders only)
│   │       ├── flights_train/      # Training set trajectory files
│   │       ├── flights_rank/       # Rank set trajectory files
│   │       ├── flights_final/      # Final set trajectory files
│   │       └── *.parquet           # Metadata files (flightlist, fuel, etc.)
│   ├── intermediate/               # [Intermediate Data Artifacts]
│   │   ├── 01_complement_data/     # Output of Step 1: Trajectories with restored Lat/Lon/Alt
│   │   ├── 02_build_script/        # Output of Step 2: Sample segments constructed from trajectories
│   │   ├── 03_add_airspeed/        # Output of Step 3: Samples merged with weather data (Airspeed added)
│   │   ├── 04_add_advanced_features/# Output of Step 4: Final feature engineering (Specific Energy, etc.)
│   │   └── climate_datasets/       # Weather data directory
│   │       ├── download_lists/     # Generated ERA5 download task JSONs
│   │       └── era5_*/             # Downloaded ERA5 NetCDF files
├── src/                            # [Core Source Code]
│   ├── model/                      # Model Definition, Training & Inference
│   │   ├── dataset.py              # Streaming DataLoader (IterableDataset)
│   │   ├── model.py                # FuelPredictionTransformer Architecture
│   │   ├── train.py                # Training Script
│   │   └── infer.py                # Inference & Submission Script
│   ├── preprocessing/              # Data Processing Pipeline
│   │   ├── train/                  # Scripts for Training Set (01-04)
│   │   ├── rank/                   # Scripts for Rank Set (01-04)
│   │   └── final/                  # Scripts for Final Set (01-04)
│   └── tools/                      # Auxiliary Tools
│       └── generate_grid_download_list_*.py # ERA5 Download List Generators
├── submissions/                    # [Submission Results]
│   └── confident-insect_final.parquet
├── logs/                           # TensorBoard Logs (Training output is printed to stdout)
├── save_model/                     # Model Checkpoints
├── requirements.txt                # Project Dependencies
└── README.md                       # Project Documentation
```

## 🚀 Quick Start

> **⚠️ Note**: All command-line operations below assume you are in the project root directory `confident-insect/`.

### 1. Prerequisites

Ensure your Python environment meets the requirements and install dependencies:

```
pip install -r requirements.txt
```

### 2. Data Preparation (Crucial)

#### 2.1 Raw Competition Data

Due to GitHub file size limits, the `data/raw/` directory does not contain actual data. **Before running any scripts, please place the original datasets provided by the competition organizers into the `data/raw/prc-2025-datasets/` directory.**

The structure must be strictly as follows:

- `data/raw/prc-2025-datasets/flights_train/*.parquet`
- `data/raw/prc-2025-datasets/flights_rank/*.parquet`
- `data/raw/prc-2025-datasets/flights_final/*.parquet`
- `data/raw/prc-2025-datasets/fuel_train.parquet`
- `data/raw/prc-2025-datasets/flightlist_train.parquet`
- (And corresponding metadata files for rank/final datasets)

#### 2.2 Weather Data Acquisition & Organization (ERA5)

This step is the most intricate but vital part of the preparation. The project utilizes the **ERA5 hourly data on pressure levels from 1940 to present** dataset to calculate wind speed and correct ground speed to true airspeed.

- **Official Dataset Link**: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview

> **Tip**: Generating the download list depends on statistics produced in **Pipeline Step 2**. If this is your first run, execute Steps 1 and 2 of the pipeline first, then return here to proceed with the download.

**1. Generate Download List** Generate a JSON task list based on the spatiotemporal distribution of flight trajectories:

```
# Generate download list for the train dataset
python src/tools/generate_era5_jobs_train.py
# (Repeat for rank and final datasets using their respective scripts)
```

Output file location: `data/intermediate/climate_datasets/download_lists/era5_download_train_GRID.json`

**2. Download & Organize Directory (Strict Structure)** Use the generated JSON file to request data from ECMWF (via CDS API). **Once downloaded, you MUST organize the NetCDF (`.nc`) files into the following hierarchy**, otherwise subsequent scripts will fail to locate them.

- **Root Directory**: `data/intermediate/climate_datasets/era5_train/` (Store rank/final data in their respective folders)
- **Level 1 Subdirectory (Grid ID)**: Use the `metadata.grid_id` value from the JSON directly, e.g., `N(-20)W(-20)S(-40)E(0)`.
- **Level 2 Subdirectory (Date)**: Format `YYYY-MM-DD`, e.g., `2025-04-16`.
- **Filename**: Must be `{GridID}-{Date}.nc`.

**📂 Expected File Structure Example:**

```
data/intermediate/climate_datasets/
├── era5_train/                          # Weather data root for Training Set
│   ├── N(-20)W(-20)S(-40)E(0)/          # <--- [L1] Grid ID
│   │   ├── 2025-04-13/                  # <--- [L2] Date
│   │   │   └── N(-20)W(-20)S(-40)E(0)-2025-04-13.nc  # <--- [File] Data
│   │   ├── 2025-04-16/
│   │   │   └── N(-20)W(-20)S(-40)E(0)-2025-04-16.nc
│   │   └── ...
│   └── ...
├── era5_rank/                           # Weather data for Rank Set
│   └── ... (Same structure)
└── era5_final/                          # Weather data for Final Set
    └── ... (Same structure)
```

### 3. Data Processing Pipeline

We have designed a staged processing pipeline. You need to run the following steps sequentially for the `train`, `rank`, and `final` datasets.

**(The commands below exemplify the `train` dataset. When processing `rank` or `final`, please execute the corresponding scripts in their respective directories.)**

#### Step 1: Data Complement & Cleaning

Restores missing latitude/longitude in raw trajectories, handles International Date Line (IDL) crossings, and performs temporal upsampling.

```
python src/preprocessing/train/01_complement_data.py
```

#### Step 2: Sample Construction & Segmentation

Slices trajectories into sample segments based on fuel labels and generates basic statistical metadata.

```
python src/preprocessing/train/02_build_script.py
```

> **Note**: After completing Step 2, ensure you have finished **"2.2 Weather Data Acquisition & Organization"** and prepared the ERA5 data before proceeding.

#### Step 3: Airspeed Calculation & Weather Fusion

Reads the downloaded weather data to calculate wind components, ground speed, and true airspeed.

```
python src/preprocessing/train/03_add_airspeed.py
```

#### Step 4: Advanced Feature Engineering

Calculates advanced kinematic features such as velocity components and their derivatives (acceleration) to generate the final training data.

```
python src/preprocessing/train/04_add_advanced_features.py
```

**⚠️ Prompt: Please ensure Steps 1-4 are also completed for the `rank` and `final` datasets before proceeding to inference.**

### 4. Model Training

Train the Transformer model using the processed `train` dataset. Supports automatic checkpoint resuming.

```
# Train a General Model (Mixed Aircraft Types)
python src/model/train.py \
    --data_root data/intermediate/04_add_advanced_features/train \
    --save_dir save_model \
    --batch_size 16 \
    --epochs 30 \
    --ac_type ALL
```

### 5. Inference & Submission

Use the trained model to perform inference on the `final` (or `rank`) dataset and generate a Parquet file compliant with the submission format. **Please pay attention to the specific paths and output filenames.**

#### 🟢 Generate Final Submission

```
python src/model/infer.py \
    --input_dir data/intermediate/04_add_advanced_features/final \
    --checkpoint_dir save_model \
    --template_path data/intermediate/01_complement_data/fuel_final_submission.parquet \
    --output_file submissions/confident-insect_final.parquet \
    --stats_file data/intermediate/04_add_advanced_features/final/global_stats.json
```

#### 🟡 Generate Rank (Leaderboard) Submission

```
python src/model/infer.py \
    --input_dir data/intermediate/04_add_advanced_features/rank \
    --checkpoint_dir save_model \
    --template_path data/intermediate/01_complement_data/fuel_rank_submission.parquet \
    --output_file submissions/confident-insect_rank.parquet \
    --stats_file data/intermediate/04_add_advanced_features/rank/global_stats.json
```

## 🔑 Key Technical Highlights

1. **Advanced Feature Engineering**: Implements sophisticated kinematic feature calculations, including ground/airspeed components and their accelerations, to assist the model in capturing flight state transitions.
2. **Robust Data Cleaning**: Developed vector-based restoration algorithms specifically to handle longitude jumps and trajectory interruptions caused by International Date Line (IDL) crossings.
3. **Scalable Data Loading**: Utilizes `pyarrow` and `PyTorch`'s `IterableDataset` to build a streaming data pipeline, minimizing memory usage when processing large-scale meteorological and trajectory data, and effectively preventing GPU OOM during training.
4. **Dynamic Weather Matching**: Implements a Grid-ID-based dynamic matching logic that loads ERA5 weather data only for the regions covered by the flight trajectory, optimizing data processing efficiency.

## 📜 License

All source code and additional datasets used in this project are openly available and published under the **GNU General Public License v3.0 (GPLv3)**, in accordance with the Challenge requirements.