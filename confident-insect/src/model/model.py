# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FuelPredictionTransformer(nn.Module):
    """
    Transformer-based model for aircraft fuel consumption prediction.
    
    Architecture:
    1. Input Projection: Projects features to high-dimensional space.
    2. Positional Encoding: Adds temporal information.
    3. Transformer Encoder: Captures sequential dependencies.
    4. Output Head (MLP): Predicts fuel mass per time step.
    """
    def __init__(
        self, 
        input_dim,   # Explicitly passed to prevent dimension mismatch
        d_model=128, 
        nhead=4, 
        num_encoder_layers=1, 
        dropout=0.1
    ): 
        """
        Args:
            input_dim: Dimension of input features.
                       Formula = 13 (Physical) + 1 (dt) + 1 (coverage) + num_ac_classes (OneHot)
                       Example: If 25 aircraft types, input_dim = 15 + 25 = 40.
            d_model: Hidden dimension size.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            dropout: Dropout rate.
        """
        super(FuelPredictionTransformer, self).__init__()
        self.d_model = d_model

        # 1. Input Projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=20000)

        # 3. Transformer Encoder
        # Using only Encoder to avoid memory overhead and complexity of Decoder during inference
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Output Head (MLP)
        # Directly predicts fuel consumption Mass (kg) for each time step
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, raw_dt=None, coverage_ratio=None, src_key_padding_mask=None):
        """
        Args:
            x: [B, T, input_dim] (Contains norm_feats, norm_dt, cov, one-hot vectors)
            raw_dt: [B, T] (Interface reserved, model learns primarily from norm_dt in x)
            coverage_ratio: [B] or [B, T] (Interface reserved)
            src_key_padding_mask: [B, T] (True indicates padding, to be masked)
        
        Returns:
            total_fuel: [B] Predicted total fuel consumption for the interval.
        """

        # Step 1: Embedding
        x_emb = self.input_projection(x) * math.sqrt(self.d_model)
        x_pe = self.pos_encoder(x_emb)

        # Step 2: Encoder
        # src_key_padding_mask ignores padding during Attention calculation
        encoder_out = self.transformer_encoder(
            x_pe, 
            src_key_padding_mask=src_key_padding_mask
        )

        # Step 3: MLP Prediction
        # raw_output: [B, T, 1] -> [B, T]
        raw_output = self.output_head(encoder_out).squeeze(-1)
        
        # Use Softplus to enforce Mass > 0
        # Output meaning: Fuel mass (kg) consumed in the current time step (corresponding to dt)
        instant_mass = F.softplus(raw_output)

        # Step 4: Mask Processing
        # Ensure predicted fuel at padding positions is 0 to avoid affecting sum
        if src_key_padding_mask is not None:
            valid_mask = (~src_key_padding_mask).float()
            instant_mass = instant_mass * valid_mask

        # Step 5: Interval Summation
        # Sum fuel consumption over all valid time steps to get total interval fuel
        # [B, T] -> [B]
        total_fuel = torch.sum(instant_mass, dim=1)

        return total_fuel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Dynamically generate PE if sequence length exceeds max_len (Handles extreme lengths)
            device = x.device
            new_pe = self._generate_pe(seq_len, self.d_model).to(device)
            x = x + new_pe[:, :seq_len, :]
        else:
            x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

    def _generate_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)