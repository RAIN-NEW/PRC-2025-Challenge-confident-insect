#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train General Fuel Prediction Transformer.

This script trains a Transformer-based model to predict fuel consumption
based on flight trajectory data and ERA5 weather features.

Features:
- Supports single aircraft type or general model (ALL) training.
- Uses TensorBoard for logging losses and learning rates.
- Supports checkpoint resuming (continuing from previous interruptions).
- Dynamic path resolution relative to the project root.
"""

import argparse
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import local modules
# Handles imports regardless of whether script is run from root or src/model
try:
    from dataset import ERA5IterableDataset, collate_fn
    from model import FuelPredictionTransformer
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent))
    from dataset import ERA5IterableDataset, collate_fn
    from model import FuelPredictionTransformer

# ================= Configuration =================

# Dynamically resolve project root
# Assumes script is at: confident-insect/src/model/train.py
current_file = Path(__file__).resolve()
try:
    project_root = current_file.parents[2]
except IndexError:
    project_root = Path.cwd()

# Default Paths
DEFAULT_DATA_ROOT = project_root / "data" / "intermediate" / "04_add_advanced_features" / "train"
DEFAULT_SAVE_DIR = project_root / "save_model"
DEFAULT_LOG_DIR = project_root / "logs"

# Global Aircraft List (Vocabulary)
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

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train General Fuel Prediction Transformer")
    
    parser.add_argument(
        "--ac_type",
        type=str,
        default="ALL",
        help="Target aircraft type. Use 'ALL' to train a General Model on all data.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=f"Path to dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size") 
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None, 
        help="Max training steps per epoch. If None, iterates whole dataset.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR), 
        help=f"Directory to save model checkpoints. Default: {DEFAULT_SAVE_DIR}",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help=f"TensorBoard log directory. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")

    return parser.parse_args()

def evaluate(model, val_loader, criterion, device):
    """
    Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Use torch.no_grad() to prevent memory leaks during validation
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validating", leave=False)
        for x, raw_dt, mask, labels, cov in loop:
            x = x.to(device)
            raw_dt = raw_dt.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            cov = cov.to(device)
            
            # Target is the total fuel consumed in the interval (already summed in dataset)
            target = labels
            
            preds = model(
                x, 
                raw_dt, 
                coverage_ratio=cov, 
                src_key_padding_mask=mask
            )
            
            loss = criterion(preds, target)
            total_loss += loss.item()
            num_batches += 1
            
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss

def main():
    args = parse_args()
    set_seed(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ==========================================
    # 1. Configure Training Mode
    # ==========================================
    if args.ac_type.upper() == "ALL":
        print("[INFO] Mode: GENERAL MODEL TRAINING (All aircraft types mixed)")
        dataset_ac_type = None 
    else:
        print(f"[INFO] Mode: SINGLE TYPE TRAINING ({args.ac_type})")
        if args.ac_type not in ALL_SUPPORTED_AIRCRAFT:
            print(f"[WARN] {args.ac_type} is not in ALL_SUPPORTED_AIRCRAFT list!")
        dataset_ac_type = args.ac_type

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    run_name = time.strftime('%Y%m%d_%H%M%S') + "_General_submission" if dataset_ac_type is None else f"_{dataset_ac_type}_submission"
    run_dir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=run_dir)
    print(f"[TB] Logging to {run_dir}")

    # ==========================================
    # 2. Initialize Datasets
    # ==========================================
    print("\n[INFO] Initializing TRAINING Dataset...")
    train_dataset = ERA5IterableDataset(
        data_root=args.data_root,
        ac_type=dataset_ac_type,   
        mode='train',
        all_available_ac_types=ALL_SUPPORTED_AIRCRAFT,
        split='train',
        val_ratio=args.val_ratio,
        random_seed=42
    )

    if train_dataset.estimated_len == 0:
        raise ValueError(f"No data found in {args.data_root}")

    train_stats = train_dataset.get_stats()

    print("\n[INFO] Initializing VALIDATION Dataset...")
    val_dataset = ERA5IterableDataset(
        data_root=args.data_root,
        ac_type=dataset_ac_type,
        mode='train',
        all_available_ac_types=ALL_SUPPORTED_AIRCRAFT,
        split='val',
        val_ratio=args.val_ratio,
        random_seed=42,
        external_stats=train_stats
    )

    print(f"\n[INFO] Train Samples (Est): {train_dataset.estimated_len}")
    print(f"[INFO] Val Samples (Est):   {val_dataset.estimated_len}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=2, 
        pin_memory=True
    )

    # ==========================================
    # 3. Initialize Model
    # ==========================================
    # 13 Physical features (including specific_energy and energy_rate)
    NUM_PHYS_FEATS = 13 
    NUM_AC_CLASSES = len(ALL_SUPPORTED_AIRCRAFT)
    
    # Input Dim = 13 (Physical) + 1 (dt) + 1 (coverage) + OneHot Vector
    INPUT_DIM = NUM_PHYS_FEATS + 1 + 1 + NUM_AC_CLASSES
    
    print(f"[INFO] Model Input Dim = {INPUT_DIM}")

    model = FuelPredictionTransformer(
        input_dim=INPUT_DIM,
        d_model=128,
        nhead=4,
        num_encoder_layers=1,  
        dropout=0.1,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # ==========================================
    # 4. Resume Logic
    # ==========================================
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf') 
    latest_checkpoint_path = os.path.join(args.save_dir, "model_latest.pth")

    if args.resume:
        if os.path.exists(latest_checkpoint_path):
            print(f"\n[INFO] Resuming training from: {latest_checkpoint_path}")
            checkpoint = torch.load(latest_checkpoint_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'])
                except RuntimeError as e:
                    print(f"[WARN] Model mismatch. Loading matching keys only. Error: {e}")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

                if 'optimizer_state_dict' in checkpoint:
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except:
                        print("[WARN] Optimizer state mismatch. Resetting optimizer.")
                
                start_epoch = checkpoint.get('epoch', 0)
                global_step = checkpoint.get('global_step', 0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                print(f"[INFO] Resumed: Epoch {start_epoch}, Global Step {global_step}, Best Val Loss {best_val_loss:.4f}")
            else:
                print("[WARN] Legacy checkpoint format. Loading weights only (strict=False).")
                model.load_state_dict(checkpoint, strict=False)
        else:
            print(f"\n[WARN] --resume flag set but no checkpoint found at {latest_checkpoint_path}. Starting from scratch.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    for epoch in range(start_epoch, args.epochs):
        model.train() 
        train_loss = 0.0
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.epochs}] [Train]", leave=False)
        
        for step, (x, raw_dt, mask, labels, cov) in enumerate(loop):
            if args.steps_per_epoch is not None and step >= args.steps_per_epoch:
                break

            x = x.to(DEVICE)
            raw_dt = raw_dt.to(DEVICE)
            mask = mask.to(device=DEVICE)
            labels = labels.to(DEVICE)
            cov = cov.to(DEVICE)

            optimizer.zero_grad()

            target = labels 
            
            preds = model(
                x,
                raw_dt,
                coverage_ratio=cov,
                src_key_padding_mask=mask,
            )

            loss = criterion(preds, target)
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss = loss.item()
            train_loss += batch_loss
            num_batches += 1

            loop.set_postfix(loss=batch_loss)

            if global_step % 50 == 0:
                writer.add_scalar('Loss/Train_Step', batch_loss, global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
                writer.add_scalar('Grad/Norm', grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm, global_step)
            
            global_step += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else float('nan')
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        # --- Validation Phase ---
        print(f"Epoch {epoch+1}: Validating...")
        
        checkpoint = None
        try:
            avg_val_loss = evaluate(model, val_loader, criterion, DEVICE)
            
            writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
            writer.flush() 

            print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            checkpoint = {
                'epoch': epoch + 1,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss 
            }

            # 1. Save Best Model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(args.save_dir, "model_best.pth")
                checkpoint['best_val_loss'] = best_val_loss
                torch.save(checkpoint, best_path)
                print(f">>> New Best Model Saved (Val Loss: {best_val_loss:.4f})")
                
            # 2. Save Latest Model
            checkpoint['best_val_loss'] = best_val_loss 
            latest_path = os.path.join(args.save_dir, "model_latest.pth")
            torch.save(checkpoint, latest_path)
            print(f"    Saved latest model to: {latest_path}")
                
        except Exception as e:
            print(f"\n[WARN] Validation FAILED at Epoch {epoch+1}!")
            print(f"Error Details: {str(e)}")
            traceback.print_exc()
            
            # Emergency Save
            model.train()
            if checkpoint is None:
                checkpoint = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }
            
            latest_path = os.path.join(args.save_dir, "model_latest_crash_saved.pth")
            torch.save(checkpoint, latest_path)
            print(f"    [Safety] Saved emergency model to: {latest_path}")

    writer.close()
    print("\nGeneral Model Training Completed.")

if __name__ == "__main__":
    main()