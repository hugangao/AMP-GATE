#!/usr/bin/env python3
"""Training entry point for the HeteroTri-Fusion AMP activity regressor.

Three heterogeneous encoders (ESM-2 semantics, physicochemical scales and
one-hot sequence motifs) are combined by sample-adaptive gated cross-attention.
The model predicts a continuous activity value with auxiliary supervision from
each individual branch.
"""

import argparse
import logging
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    def mean_squared_error(y_true, y_pred):
        return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1.0 - (ss_res / ss_tot))


logger = logging.getLogger(__name__)


# --- Standardised residue-level physicochemical scales (18 dimensions) ---
AA_LIST = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}  # Index 0 is reserved for padding.

RAW_PHYS_PROPERTIES = {
    'A': [-0.591, -1.302, -0.733,  1.570, -0.146,   0.24, -2.32,  0.60, -0.14,  1.30,  0.62,  1.8,  0.0, 6.00,  8.1,  67.0, 1.42, 0.83],
    'C': [-1.343,  0.465, -0.862, -1.020, -0.255,   0.84, -1.67,  3.71,  0.18, -2.65,  0.29,  2.5,  0.0, 5.07,  5.5,  86.0, 0.70, 1.19],
    'D': [ 1.050,  0.302, -3.656, -0.259, -3.242,   3.98,  0.93,  1.93, -2.46,  0.75, -0.90, -3.5, -1.0, 2.77, 13.0,  91.0, 1.01, 0.54],
    'E': [ 1.357, -1.453,  1.477,  0.113, -0.837,   3.11,  0.26, -0.11, -3.04, -0.25, -0.74, -3.5, -1.0, 3.22, 12.3, 109.0, 1.51, 0.37],
    'F': [-1.006, -0.590,  1.891, -0.397,  0.412,  -4.22,  1.94,  1.06,  0.54, -0.62,  1.19,  2.8,  0.0, 5.48,  5.2, 135.0, 1.13, 1.38],
    'G': [-0.384,  1.652,  1.330,  1.045,  2.064,   2.05, -4.06,  0.36, -0.82, -0.38,  0.48, -0.4,  0.0, 5.97,  9.0,  48.0, 0.57, 0.75],
    'H': [ 0.336, -0.417, -1.673, -1.474, -0.078,   2.47,  1.95,  0.26,  3.90,  0.09, -0.40, -3.2,  0.1, 7.59, 10.4, 118.0, 1.00, 0.87],
    'I': [-1.239, -0.547,  2.131,  0.393,  0.816,  -3.89, -1.73, -1.71, -0.84,  0.26,  1.38,  4.5,  0.0, 6.02,  5.2, 124.0, 1.08, 1.60],
    'K': [ 1.831, -0.561,  0.533, -0.277,  1.648,   2.29,  0.89, -2.49,  1.49,  0.31, -1.50, -3.9,  1.0, 9.74, 11.3, 135.0, 1.16, 0.74],
    'L': [-1.019, -0.987, -1.505,  1.266, -0.912,  -4.28, -1.30, -1.49, -0.72,  0.84,  1.06,  3.8,  0.0, 5.98,  4.9, 124.0, 1.21, 1.30],
    'M': [-0.663, -1.524,  2.219, -1.005,  1.212,  -2.85, -0.22,  0.47,  1.94, -0.98,  0.64,  1.9,  0.0, 5.74,  5.7, 124.0, 1.45, 1.05],
    'N': [ 0.945,  0.828,  1.299, -0.169,  0.933,   3.05,  1.62,  1.04, -1.15,  1.61, -0.78, -3.5,  0.0, 5.41, 11.6,  96.0, 0.67, 0.89],
    'P': [ 0.189,  2.081, -1.628, -0.421, -1.392,  -1.66,  0.27,  1.84,  0.70,  2.00,  0.12, -1.6,  0.0, 6.30,  8.0,  90.0, 0.57, 0.55],
    'Q': [ 0.931, -0.179, -3.005, -0.503, -1.853,   1.75,  0.50, -1.44, -1.34,  0.66, -0.85, -3.5,  0.0, 5.65, 10.5, 114.0, 1.11, 1.10],
    'R': [ 1.538, -0.055,  1.502,  0.440,  2.897,   3.52,  2.50, -3.50,  1.99, -0.17, -2.53, -4.5,  1.0, 10.76, 10.5, 148.0, 0.98, 0.93],
    'S': [-0.228,  1.399, -4.760,  0.670, -2.647,   2.39, -1.07,  1.15, -1.39,  0.67, -0.18, -0.8,  0.0, 5.68,  9.2,  73.0, 0.77, 0.75],
    'T': [-0.032,  0.326,  2.213,  0.908,  1.313,   0.75, -2.18, -1.12, -1.46, -0.40, -0.05, -0.7,  0.0, 5.60,  8.6,  93.0, 0.83, 1.19],
    'V': [-1.337, -0.279, -0.544,  1.242, -1.262,  -2.59, -2.64, -1.54, -0.85, -0.02,  1.08,  4.2,  0.0, 5.96,  5.9, 105.0, 1.06, 1.70],
    'W': [-0.595,  0.009,  0.672, -2.128, -0.184,  -4.36,  3.94,  0.59,  3.44, -1.59,  0.81, -0.9,  0.0, 5.89,  5.4, 163.0, 1.08, 1.37],
    'Y': [ 0.260,  0.830,  3.097, -0.838,  1.512,  -2.54,  2.44,  0.43,  0.04, -1.47,  0.26, -1.3,  0.0, 5.66,  6.2, 141.0, 0.69, 1.47],
}


def build_phys_embedding_table():
    mat = np.zeros((21, 18), dtype=np.float32)
    for aa, idx in AA_TO_IDX.items():
        mat[idx] = RAW_PHYS_PROPERTIES[aa]
    mean = mat[1:].mean(axis=0, keepdims=True)
    std = mat[1:].std(axis=0, keepdims=True) + 1e-6
    mat[1:] = (mat[1:] - mean) / std
    return torch.tensor(mat, dtype=torch.float32)


# --- Dataset and dynamic collation ---
class AMPRegressionDataset(Dataset):
    def __init__(self, csv_file, embedding_dir, id_col="ID", seq_col="Sequence", val_col="value", label_col="label", esm_layer=12):
        self.df = pd.read_csv(csv_file)
        self.embedding_dir = embedding_dir
        self.esm_layer = esm_layer
        self.id_col = id_col
        self.seq_col = seq_col
        self.val_col = val_col
        self.label_col = label_col

        self.valid_indices = []
        for i in range(len(self.df)):
            pid = str(self.df.iloc[i][self.id_col])
            emb_path = os.path.join(self.embedding_dir, f"{pid}.pt")
            if os.path.exists(emb_path):
                self.valid_indices.append(i)

        logger.info("Loaded %d/%d samples with ESM embeddings from %s",
                    len(self.valid_indices), len(self.df), csv_file)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.valid_indices[idx]]
        pid = str(row[self.id_col])
        seq = str(row[self.seq_col]).strip().upper()
        target_val = float(row[self.val_col])
        # Inactive samples are marked by the sentinel value 3.91339 when the label is absent.
        label = int(row[self.label_col]) if self.label_col in row and pd.notna(row[self.label_col]) else (0 if np.isclose(target_val, 3.91339, atol=1e-3) else 1)

        emb_path = os.path.join(self.embedding_dir, f"{pid}.pt")
        emb_data = torch.load(emb_path, map_location="cpu")

        esm_repr = emb_data["representations"][self.esm_layer]
        if esm_repr.size(0) == len(seq) + 2:
            esm_repr = esm_repr[1:-1]
        elif esm_repr.size(0) > len(seq):
            esm_repr = esm_repr[:len(seq)]

        token_ids = [AA_TO_IDX.get(aa, 0) for aa in seq]

        return {
            "id": pid,
            "seq": seq,
            "esm_feat": esm_repr,
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "target_val": torch.tensor(target_val, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long)
        }


def amp_reg_collate_fn(batch):
    ids = [item["id"] for item in batch]
    seqs = [item["seq"] for item in batch]
    targets = torch.stack([item["target_val"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)

    lengths = [len(s) for s in seqs]
    max_len = max(lengths)

    esm_list = [item["esm_feat"] for item in batch]
    tokens_list = [item["token_ids"] for item in batch]

    esm_padded = pad_sequence(esm_list, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(tokens_list, batch_first=True, padding_value=0)

    batch_size = len(batch)
    onehot = torch.zeros((batch_size, max_len, 20), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, t in enumerate(tokens_list):
        l = len(t)
        mask[i, :l] = True
        for j in range(l):
            idx = t[j].item()
            if 1 <= idx <= 20:
                onehot[i, j, idx - 1] = 1.0

    return {
        "ids": ids,
        "seqs": seqs,
        "esm_feats": esm_padded,
        "tokens": tokens_padded,
        "onehot": onehot,
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": targets,
        "labels": labels
    }


# --- Heterogeneous feature encoders ---
class MaskedMaxPooling(nn.Module):
    """Max-over-time pooling that ignores padded positions."""
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        # x: (B, L, C), mask: (B, L)
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        x_masked = x.masked_fill(~mask_expanded, -1e9)
        pooled, _ = torch.max(x_masked, dim=1)  # (B, C)
        return pooled


class PositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding for residue coordinates."""
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, L, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# Branch A: ESM-2 semantic refinement.
class TransformerSemanticRefiner(nn.Module):
    """Project ESM-2 features, refine with a Pre-LN Transformer, then max-pool."""
    def __init__(self, in_dim=480, out_dim=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(out_dim, max_len=256, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=out_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True  # Pre-LN improves optimisation stability.
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(out_dim)
        self.pool = MaskedMaxPooling()

    def forward(self, x, mask):
        h = self.proj(x)
        h = self.pos_encoder(h)  # Inject absolute positions.
        key_padding_mask = ~mask
        h_trans = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)
        h_out = self.norm(h_trans)
        pooled = self.pool(h_out, mask)
        return h_out, pooled, None


# Branch B: physicochemical feature extraction.
class PhysGatedConv1D(nn.Module):
    """Dual-scale gated convolutions for low-dimensional physicochemical scales."""
    def __init__(self, in_dim=18, proj_dim=64, out_dim=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        half_dim = proj_dim // 2
        # Local (k=3) and mid-range (k=5) value/gate convolutions.
        self.conv_val3 = nn.Conv1d(proj_dim, half_dim, kernel_size=3, padding=1)
        self.conv_gate3 = nn.Conv1d(proj_dim, half_dim, kernel_size=3, padding=1)
        self.conv_val5 = nn.Conv1d(proj_dim, half_dim, kernel_size=5, padding=2)
        self.conv_gate5 = nn.Conv1d(proj_dim, half_dim, kernel_size=5, padding=2)

        self.norm_conv = nn.LayerNorm(proj_dim)
        self.drop_conv = nn.Dropout(dropout)

        self.out_proj = nn.Sequential(
            nn.Linear(proj_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.pool = MaskedMaxPooling()

    def forward(self, x, mask):
        # x: (B, L, 18)
        h = self.proj(x)  # (B, L, 64)
        h_t = h.transpose(1, 2)  # (B, 64, L)

        # Gated linear units.
        gated3 = self.conv_val3(h_t) * torch.sigmoid(self.conv_gate3(h_t))  # (B, 32, L)
        gated5 = self.conv_val5(h_t) * torch.sigmoid(self.conv_gate5(h_t))  # (B, 32, L)

        gated_f = torch.cat([gated3, gated5], dim=1).transpose(1, 2)  # (B, L, 64)
        h_conv = self.drop_conv(self.norm_conv(gated_f))

        h_out = self.out_proj(h_conv)  # (B, L, 128)
        pooled = self.pool(h_out, mask)  # (B, 128)
        return h_out, pooled, None


# Branch C: local sequence motif extraction.
class SqueezeAndExcitation1D(nn.Module):
    """Channel attention with mask-aware global average pooling."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, mask):
        # x: (B, C, L)
        mask_f = mask.unsqueeze(1).float()  # (B, 1, L)
        denom = torch.clamp(mask_f.sum(dim=-1, keepdim=True), min=1.0)
        gap = (x * mask_f).sum(dim=-1, keepdim=True) / denom  # (B, C, 1)
        w = self.fc(gap.squeeze(-1)).unsqueeze(-1)  # (B, C, 1)
        return x * w


class MultiScaleInceptionSECNN(nn.Module):
    """Parallel multi-scale convolutions with SE attention for one-hot motifs."""
    def __init__(self, in_dim=20, out_dim=128, dropout=0.2):
        super().__init__()
        c_branch = out_dim // 4
        self.conv1 = nn.Conv1d(in_dim, c_branch, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_dim, c_branch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_dim, c_branch, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_dim, c_branch, kernel_size=7, padding=3)

        self.se = SqueezeAndExcitation1D(out_dim, reduction=4)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.pool = MaskedMaxPooling()

    def forward(self, x, mask):
        # x: (B, L, 20) -> (B, 20, L)
        x_t = x.transpose(1, 2)
        f1 = self.conv1(x_t)
        f3 = self.conv3(x_t)
        f5 = self.conv5(x_t)
        f7 = self.conv7(x_t)

        cat_f = torch.cat([f1, f3, f5, f7], dim=1)  # (B, 128, L)
        se_f = self.se(cat_f, mask)
        h = se_f.transpose(1, 2)
        h_out = self.drop(self.act(self.norm(h)))
        pooled = self.pool(h_out, mask)
        return h_out, pooled, None


# --- Sample-adaptive cross-modal gated fusion ---
class SampleAdaptiveCrossModalFusion(nn.Module):
    """Cross-attention from ESM queries to other modalities, followed by sample-wise gating.

    Gates are either temperature-scaled softmax or independent sigmoid units.
    """
    def __init__(self, feat_dim=128, nhead=4, dropout=0.2, gate_type="temp_softmax", gate_temp=1.2):
        super().__init__()
        self.feat_dim = feat_dim
        self.gate_type = gate_type
        self.gate_temp = max(float(gate_temp), 1e-4)

        # ESM queries physicochemical features.
        self.cross_phys = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_phys = nn.LayerNorm(feat_dim)
        self.drop_phys = nn.Dropout(dropout)

        # ESM queries sequence motifs.
        self.cross_motif = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm_motif = nn.LayerNorm(feat_dim)
        self.drop_motif = nn.Dropout(dropout)

        # Produces three modality logits per sample.
        self.gate_linear = nn.Sequential(
            nn.Linear(feat_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

        # Feed-forward aggregation.
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.Dropout(dropout)
        )
        self.norm_final = nn.LayerNorm(feat_dim)

    def forward(self, seq_esm, pool_esm, seq_phys, pool_phys, seq_motif, pool_motif, mask):
        key_padding_mask = ~mask

        # Cross-modal attention.
        attn_phys, w_phys = self.cross_phys(
            query=seq_esm, key=seq_phys, value=seq_phys, key_padding_mask=key_padding_mask
        )
        h_esm_phys = self.norm_phys(seq_esm + self.drop_phys(attn_phys))

        attn_motif, w_motif = self.cross_motif(
            query=h_esm_phys, key=seq_motif, value=seq_motif, key_padding_mask=key_padding_mask
        )
        h_cross = self.norm_motif(h_esm_phys + self.drop_motif(attn_motif))

        # Sample-wise modality gates.
        pools_cat = torch.cat([pool_esm, pool_phys, pool_motif], dim=-1)  # (B, feat_dim * 3)
        logits = self.gate_linear(pools_cat)  # (B, 3)

        if self.gate_type == "sigmoid":
            # Independent retention weights in (0, 1).
            gate_weights = torch.sigmoid(logits)
        else:
            # Smoothed softmax avoids modality collapse.
            gate_weights = F.softmax(logits / self.gate_temp, dim=-1)

        g_esm = gate_weights[:, 0].view(-1, 1, 1)
        g_phys = gate_weights[:, 1].view(-1, 1, 1)
        g_motif = gate_weights[:, 2].view(-1, 1, 1)

        # Weighted fusion.
        fused_seq = g_esm * h_cross + g_phys * seq_phys + g_motif * seq_motif
        fused_seq = self.norm_final(fused_seq + self.ffn(fused_seq))

        return fused_seq, gate_weights, w_phys, w_motif


# --- Main regression model ---
class HeteroTriFusionAMPRegressor(nn.Module):
    def __init__(self, esm_dim=480, feat_dim=256, nhead=4, dropout=0.2, gate_type="temp_softmax", gate_temp=1.2):
        super().__init__()
        phys_table = build_phys_embedding_table()
        self.phys_embed = nn.Embedding.from_pretrained(phys_table, freeze=True, padding_idx=0)

        # Heterogeneous encoders.
        self.branch_esm = TransformerSemanticRefiner(
            in_dim=esm_dim, out_dim=feat_dim, nhead=nhead, num_layers=2, dropout=dropout
        )
        self.branch_phys = PhysGatedConv1D(
            in_dim=18, proj_dim=64, out_dim=feat_dim, dropout=dropout
        )
        self.branch_motif = MultiScaleInceptionSECNN(
            in_dim=20, out_dim=feat_dim, dropout=dropout
        )

        # Sample-adaptive cross-modal fusion.
        self.fusion = SampleAdaptiveCrossModalFusion(
            feat_dim=feat_dim, nhead=nhead, dropout=dropout, gate_type=gate_type, gate_temp=gate_temp
        )
        # Masked max pooling over the fused sequence.
        self.global_pool = MaskedMaxPooling()

        # Main regression head.
        self.regressor = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1)  # Continuous activity value.
        )

        # Auxiliary single-modality heads.
        self.aux_esm_reg = nn.Linear(feat_dim, 1)
        self.aux_phys_reg = nn.Linear(feat_dim, 1)
        self.aux_motif_reg = nn.Linear(feat_dim, 1)

    def forward(self, esm_feats, tokens, onehot, mask):
        phys_feats = self.phys_embed(tokens)

        # Heterogeneous encoding.
        seq_esm, pool_esm, _ = self.branch_esm(esm_feats, mask)
        seq_phys, pool_phys, _ = self.branch_phys(phys_feats, mask)
        seq_motif, pool_motif, _ = self.branch_motif(onehot, mask)

        # Gated fusion.
        fused_seq, gate_w, w_phys, w_motif = self.fusion(
            seq_esm, pool_esm, seq_phys, pool_phys, seq_motif, pool_motif, mask
        )
        fused_pooled = self.global_pool(fused_seq, mask)
        main_pred = self.regressor(fused_pooled).squeeze(-1)

        # Auxiliary supervision.
        aux_esm_pred = self.aux_esm_reg(pool_esm).squeeze(-1)
        aux_phys_pred = self.aux_phys_reg(pool_phys).squeeze(-1)
        aux_motif_pred = self.aux_motif_reg(pool_motif).squeeze(-1)

        return main_pred, {
            "aux_preds": (aux_esm_pred, aux_phys_pred, aux_motif_pred),
            "gate_weights": gate_w,
            "cross_attn_phys": w_phys,
            "cross_attn_motif": w_motif
        }


# --- Regression metrics ---
def calculate_concordance_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n <= 1:
        return 0.5

    concordant = 0
    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total_pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1.0
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5

    if total_pairs == 0:
        return 0.5
    return float(concordant / total_pairs)


def calculate_regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pcc, p_val_pcc = pearsonr(y_true, y_pred)
        scc, p_val_scc = spearmanr(y_true, y_pred)
    else:
        pcc, p_val_pcc = 0.0, 1.0
        scc, p_val_scc = 0.0, 1.0

    ci = calculate_concordance_index(y_true, y_pred)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson_PCC": float(pcc),
        "PCC_p_value": float(p_val_pcc),
        "Spearman_SCC": float(scc),
        "SCC_p_value": float(p_val_scc),
        "CI": float(ci)
    }


# --- Training and evaluation ---
def train_one_epoch(model, loader, optimizer, criterion, aux_weight, device):
    model.train()
    total_loss, total_main_loss, total_aux_loss = 0.0, 0.0, 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        esm_feats = batch["esm_feats"].to(device)
        tokens = batch["tokens"].to(device)
        onehot = batch["onehot"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        main_pred, info = model(esm_feats, tokens, onehot, mask)

        loss_main = criterion(main_pred, targets)
        aux_esm, aux_phys, aux_motif = info["aux_preds"]
        loss_aux = criterion(aux_esm, targets) + criterion(aux_phys, targets) + criterion(aux_motif, targets)

        loss = loss_main + aux_weight * loss_aux
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        b_size = len(batch["ids"])
        total_loss += loss.item() * b_size
        total_main_loss += loss_main.item() * b_size
        total_aux_loss += loss_aux.item() * b_size

        all_preds.extend(main_pred.detach().cpu().numpy().tolist())
        all_targets.extend(targets.cpu().numpy().tolist())

    n_samples = len(all_targets)
    metrics = calculate_regression_metrics(all_targets, all_preds)
    metrics["Loss"] = total_loss / n_samples
    metrics["Main_Loss"] = total_main_loss / n_samples
    metrics["Aux_Loss"] = total_aux_loss / n_samples
    return metrics


def evaluate(model, loader, criterion, aux_weight, device):
    model.eval()
    total_loss, total_main_loss, total_aux_loss = 0.0, 0.0, 0.0
    all_preds, all_targets, all_labels = [], [], []
    all_aux_esm, all_aux_phys, all_aux_motif = [], [], []
    all_gate_weights = []

    with torch.no_grad():
        for batch in loader:
            esm_feats = batch["esm_feats"].to(device)
            tokens = batch["tokens"].to(device)
            onehot = batch["onehot"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["targets"].to(device)
            labels = batch["labels"]

            main_pred, info = model(esm_feats, tokens, onehot, mask)

            loss_main = criterion(main_pred, targets)
            aux_esm, aux_phys, aux_motif = info["aux_preds"]
            loss_aux = criterion(aux_esm, targets) + criterion(aux_phys, targets) + criterion(aux_motif, targets)
            loss = loss_main + aux_weight * loss_aux

            b_size = len(batch["ids"])
            total_loss += loss.item() * b_size
            total_main_loss += loss_main.item() * b_size
            total_aux_loss += loss_aux.item() * b_size

            all_preds.extend(main_pred.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

            all_aux_esm.extend(aux_esm.cpu().numpy().tolist())
            all_aux_phys.extend(aux_phys.cpu().numpy().tolist())
            all_aux_motif.extend(aux_motif.cpu().numpy().tolist())
            all_gate_weights.append(info["gate_weights"].cpu().numpy())

    n_samples = len(all_targets)
    metrics_all = calculate_regression_metrics(all_targets, all_preds)
    metrics_all["Loss"] = total_loss / n_samples
    metrics_all["Main_Loss"] = total_main_loss / n_samples
    metrics_all["Aux_Loss"] = total_aux_loss / n_samples

    # Active peptide subset.
    act_mask = np.array(all_labels) == 1
    if np.sum(act_mask) > 1:
        act_targets = np.array(all_targets)[act_mask]
        act_preds = np.array(all_preds)[act_mask]
        metrics_active = calculate_regression_metrics(act_targets, act_preds)
    else:
        metrics_active = metrics_all.copy()

    stacked_gates = np.concatenate(all_gate_weights, axis=0) if len(all_gate_weights) > 0 else np.zeros((n_samples, 3))
    return metrics_all, metrics_active, (all_preds, all_targets, all_labels, all_aux_esm, all_aux_phys, all_aux_motif, stacked_gates)


# --- Command-line entry point ---
def main():
    parser = argparse.ArgumentParser(
        description="HeteroTri-Fusion AMP Quantitative Activity Regressor (Specialized Architectures for Diverse Features)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset paths.
    parser.add_argument("-tr", "--train_csv", type=str, default="data/train.csv", help="Train dataset CSV path")
    parser.add_argument("-te", "--test_csv", type=str, default="data/test.csv", help="Test dataset CSV path")
    parser.add_argument("-ed", "--embedding_dir", type=str, default="data/esm_embeddings", help="Directory containing ESM .pt files")
    parser.add_argument("-o", "--output_dir", type=str, default="output", help="Output directory")

    # Column names and ESM configuration.
    parser.add_argument("--id_col", type=str, default="ID", help="Sequence ID column name")
    parser.add_argument("--seq_col", type=str, default="Sequence", help="Sequence column name")
    parser.add_argument("--val_col", type=str, default="value", help="Regression target value column name")
    parser.add_argument("--label_col", type=str, default="label", help="Activity label column name")
    parser.add_argument("--esm_layer", type=int, default=12, help="ESM representation layer index")
    parser.add_argument("--esm_dim", type=int, default=480, help="ESM embedding dimension")

    # Model hyperparameters.
    parser.add_argument("--feat_dim", type=int, default=256, help="Unified feature dimension for heterogeneous encoders")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads in Transformer and cross-attention")
    parser.add_argument("--aux_weight", type=float, default=0.2, help="Loss weight for auxiliary single-branch heads")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--gate_type", type=str, default="temp_softmax", choices=["temp_softmax", "sigmoid", "softmax"], help="Modality gating activation function")
    parser.add_argument("--gate_temp", type=float, default=1.2, help="Temperature for temp_softmax gating")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["huber", "smooth_l1", "mse", "l1"], help="Regression loss function")
    parser.add_argument("--huber_delta", type=float, default=2.0, help="Delta threshold for Huber loss")

    # Optimisation and early stopping.
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay penalty")
    parser.add_argument("--val_ratio", type=float, default=0.1765, help="Validation ratio within each target quantile bin")
    parser.add_argument("--monitor_metric", type=str, default="MSE", choices=["MSE", "RMSE", "MAE", "R2", "Pearson_PCC", "Spearman_SCC", "LOSS"], help="Early stopping monitor metric")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--lr_patience", type=int, default=8, help="Patience before decaying learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Fix random seeds.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Device=%s | train=%s | test=%s | embeddings=%s | output=%s",
                device, args.train_csv, args.test_csv, args.embedding_dir, args.output_dir)
    logger.info("Gate=%s (temp=%s) | loss=%s (delta=%s) | aux_weight=%s | monitor=%s (patience=%s)",
                args.gate_type, args.gate_temp, args.loss_type, args.huber_delta,
                args.aux_weight, args.monitor_metric, args.patience)

    # Quantile-stratified train/validation split.
    full_train_dataset = AMPRegressionDataset(
        csv_file=args.train_csv,
        embedding_dir=args.embedding_dir,
        id_col=args.id_col,
        seq_col=args.seq_col,
        val_col=args.val_col,
        label_col=args.label_col,
        esm_layer=args.esm_layer
    )

    test_dataset = AMPRegressionDataset(
        csv_file=args.test_csv,
        embedding_dir=args.embedding_dir,
        id_col=args.id_col,
        seq_col=args.seq_col,
        val_col=args.val_col,
        label_col=args.label_col,
        esm_layer=args.esm_layer
    )

    n_train_pool = len(full_train_dataset)
    indices = np.arange(n_train_pool)
    train_df = full_train_dataset.df.iloc[full_train_dataset.valid_indices].reset_index(drop=True)
    val_strat_bins = pd.qcut(train_df[args.val_col], q=10, labels=False, duplicates='drop').values

    rng = np.random.RandomState(args.seed)
    train_idx, val_idx = [], []
    for b_cls in np.unique(val_strat_bins):
        cls_idx = indices[val_strat_bins == b_cls]
        rng.shuffle(cls_idx)
        n_v = int(round(len(cls_idx) * args.val_ratio))
        val_idx.extend(cls_idx[:n_v])
        train_idx.extend(cls_idx[n_v:])

    train_subset = Subset(full_train_dataset, train_idx)
    val_subset = Subset(full_train_dataset, val_idx)

    logger.info("Train=%d | validation=%d | independent test=%d",
                len(train_subset), len(val_subset), len(test_dataset))

    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=amp_reg_collate_fn)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, collate_fn=amp_reg_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=amp_reg_collate_fn)

    # Build the heterogeneous model.
    model = HeteroTriFusionAMPRegressor(
        esm_dim=args.esm_dim,
        feat_dim=args.feat_dim,
        nhead=args.nhead,
        dropout=args.dropout,
        gate_type=args.gate_type,
        gate_temp=args.gate_temp
    ).to(device)

    # Loss function and optimiser.
    if args.loss_type in ["huber", "smooth_l1"]:
        criterion = nn.HuberLoss(delta=args.huber_delta) if hasattr(nn, "HuberLoss") else nn.SmoothL1Loss(beta=args.huber_delta)
    elif args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "l1":
        criterion = nn.L1Loss()
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Early stopping and learning rate scheduling.
    higher_is_better = args.monitor_metric in ["Pearson_PCC", "R2", "Spearman_SCC", "CI"]
    lr_mode = "max" if higher_is_better else "min"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=lr_mode, factor=0.5, patience=args.lr_patience
    )

    best_score = -float("inf") if higher_is_better else float("inf")
    patience_counter = 0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_hetero_regressor.pth")

    history_records = []
    logger.info("Training for up to %d epochs.", args.epochs)

    for epoch in range(1, args.epochs + 1):
        tr_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.aux_weight, device)
        val_all, val_act, (_, _, _, _, _, _, val_gates) = evaluate(model, val_loader, criterion, args.aux_weight, device)

        curr_score = val_all[args.monitor_metric] if args.monitor_metric != "LOSS" else val_all["Loss"]
        scheduler.step(curr_score)
        curr_lr = optimizer.param_groups[0]["lr"]

        is_better = (curr_score > best_score) if higher_is_better else (curr_score < best_score)

        epoch_log = (
            f"Epoch {epoch:03d}/{args.epochs:03d} | lr={curr_lr:.2e} | "
            f"train loss={tr_metrics['Loss']:.4f} (R2={tr_metrics['R2']:.3f}, PCC={tr_metrics['Pearson_PCC']:.3f}) | "
            f"val loss={val_all['Loss']:.4f}, MSE={val_all['MSE']:.4f}, "
            f"R2={val_all['R2']:.4f}, PCC={val_all['Pearson_PCC']:.4f}"
        )

        if is_better:
            best_score = curr_score
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "monitor_metric": args.monitor_metric,
                "args": vars(args)
            }, best_model_path)
            logger.info("%s | best %s=%.4f, model saved", epoch_log, args.monitor_metric, best_score)
        else:
            patience_counter += 1
            logger.info("%s | patience %d/%d", epoch_log, patience_counter, args.patience)

        mean_gates = np.mean(val_gates, axis=0)  # [esm_w, phys_w, motif_w]
        history_records.append({
            "Epoch": epoch,
            "Train_Loss": tr_metrics["Loss"],
            "Train_Main_Loss": tr_metrics["Main_Loss"],
            "Train_Aux_Loss": tr_metrics["Aux_Loss"],
            "Train_MSE": tr_metrics["MSE"],
            "Train_R2": tr_metrics["R2"],
            "Train_PCC": tr_metrics["Pearson_PCC"],
            "Val_Loss": val_all["Loss"],
            "Val_MSE": val_all["MSE"],
            "Val_R2": val_all["R2"],
            "Val_PCC": val_all["Pearson_PCC"],
            "Val_Spearman_SCC": val_all["Spearman_SCC"],
            "Val_MAE": val_all["MAE"],
            "Val_CI": val_all["CI"],
            "Gate_Weight_ESM": float(mean_gates[0]),
            "Gate_Weight_Phys": float(mean_gates[1]),
            "Gate_Weight_Motif": float(mean_gates[2]),
            "Learning_Rate": curr_lr
        })

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info("Early stopping after %d epochs without improvement.", args.patience)
            break

    # Save the training log.
    history_csv = os.path.join(args.output_dir, "training_history_log.csv")
    pd.DataFrame(history_records).to_csv(history_csv, index=False)
    logger.info("Training history written to %s", history_csv)

    # Evaluate the independent test set with the best checkpoint.
    logger.info("Evaluating the independent test set with the best model from epoch %d.", best_epoch)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_all, test_act, test_raw = evaluate(model, test_loader, criterion, args.aux_weight, device)
    all_preds, all_targets, all_labels, aux_esm, aux_phys, aux_motif, test_gates = test_raw

    test_df_orig = test_dataset.df.iloc[test_dataset.valid_indices].reset_index(drop=True)
    preds_df = pd.DataFrame({
        "ID": test_df_orig[args.id_col],
        "Sequence": test_df_orig[args.seq_col],
        "True_Value": all_targets,
        "Pred_Value": all_preds,
        "Residual": np.array(all_preds) - np.array(all_targets),
        "True_Label": all_labels,
        "Gate_ESM": test_gates[:, 0],
        "Gate_Phys": test_gates[:, 1],
        "Gate_Motif": test_gates[:, 2],
        "Pred_ESM": aux_esm,
        "Pred_Phys": aux_phys,
        "Pred_Motif": aux_motif
    })

    test_preds_csv = os.path.join(args.output_dir, "test_predictions.csv")
    preds_df.to_csv(test_preds_csv, index=False)
    logger.info("Test predictions written to %s", test_preds_csv)

    logger.info("Overall test set (n=%d): R2=%.4f | PCC=%.4f (p=%.2e) | SCC=%.4f (p=%.2e) | MSE=%.4f | MAE=%.4f | CI=%.4f",
                len(all_targets), test_all['R2'], test_all['Pearson_PCC'], test_all['PCC_p_value'],
                test_all['Spearman_SCC'], test_all['SCC_p_value'], test_all['MSE'], test_all['MAE'], test_all['CI'])
    logger.info("Mean gate weights [ESM / Phys / Motif]: [%.3f / %.3f / %.3f]",
                np.mean(test_gates[:, 0]), np.mean(test_gates[:, 1]), np.mean(test_gates[:, 2]))
    logger.info("Active AMP subset (n=%d): R2=%.4f | PCC=%.4f (p=%.2e) | SCC=%.4f (p=%.2e) | MSE=%.4f | MAE=%.4f | CI=%.4f",
                np.sum(np.array(all_labels) == 1), test_act['R2'], test_act['Pearson_PCC'], test_act['PCC_p_value'],
                test_act['Spearman_SCC'], test_act['SCC_p_value'], test_act['MSE'], test_act['MAE'], test_act['CI'])

    # Export the test metrics summary.
    summary_df = pd.DataFrame([
        {"Subset": "Overall Test Set", **test_all},
        {"Subset": "Active AMPs Subset", **test_act}
    ])
    summary_csv = os.path.join(args.output_dir, "test_metrics_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    logger.info("Test metrics summary written to %s", summary_csv)


if __name__ == "__main__":
    main()
