#!/usr/bin/env python3
"""Streaming inference for AMP-GATE.

ESM-2 embeddings are extracted on the fly from FASTA, CSV/TSV or a single
peptide string. Predictions are written batch by batch to keep memory usage
low; evaluation metrics are reported when ground-truth values are supplied.
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from Bio import SeqIO
    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False

try:
    import esm
    HAS_ESM = True
except ImportError:
    HAS_ESM = False

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
        return float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

from train import (  # noqa: E402
    AMPGATERegressor,
    AA_TO_IDX
)


class PeptideStreamDataset(Dataset):
    def __init__(self, input_source, id_col="ID", seq_col="Sequence", val_col="value"):
        self.records = []

        if isinstance(input_source, list):
            for i, item in enumerate(input_source):
                if isinstance(item, tuple):
                    self.records.append(item)
                else:
                    self.records.append((f"Seq_{i+1}", str(item).strip().upper(), None))
        elif os.path.isfile(input_source):
            ext = os.path.splitext(input_source)[1].lower()
            if ext in [".fasta", ".fa", ".fas", ".fna"]:
                if not HAS_BIOPYTHON:
                    with open(input_source, "r") as f:
                        curr_id, curr_seq = None, []
                        for line in f:
                            line = line.strip()
                            if line.startswith(">"):
                                if curr_id is not None:
                                    self.records.append((curr_id, "".join(curr_seq).upper(), None))
                                curr_id = line[1:].split()[0]
                                curr_seq = []
                            else:
                                curr_seq.append(line)
                        if curr_id is not None:
                            self.records.append((curr_id, "".join(curr_seq).upper(), None))
                else:
                    for record in SeqIO.parse(input_source, "fasta"):
                        self.records.append((str(record.id), str(record.seq).strip().upper(), None))
            elif ext in [".csv", ".tsv", ".txt"]:
                sep = "\t" if ext in [".tsv", ".txt"] else ","
                df = pd.read_csv(input_source, sep=sep)
                cols = {c.lower(): c for c in df.columns}
                matched_id = cols.get(id_col.lower(), df.columns[0])
                matched_seq = cols.get(seq_col.lower(), df.columns[1] if len(df.columns) > 1 else df.columns[0])
                matched_val = cols.get(val_col.lower(), None)

                for _, row in df.iterrows():
                    pid = str(row[matched_id])
                    seq = str(row[matched_seq]).strip().upper()
                    val = float(row[matched_val]) if (matched_val and pd.notna(row[matched_val])) else None
                    self.records.append((pid, seq, val))
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        else:
            self.records.append(("Custom_Peptide", str(input_source).strip().upper(), None))

        logger.info("Loaded %d peptide records.", len(self.records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def stream_collate_fn(batch):
    ids = [item[0] for item in batch]
    seqs = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    lengths = [len(s) for s in seqs]
    max_l = max(lengths)

    token_list, onehot_list = [], []
    for s in seqs:
        t_ids = [AA_TO_IDX.get(aa, 0) for aa in s]
        token_list.append(torch.tensor(t_ids, dtype=torch.long))

        oh = torch.zeros((len(s), 20), dtype=torch.float32)
        for j, aa in enumerate(s):
            if aa in AA_TO_IDX:
                oh[j, AA_TO_IDX[aa] - 1] = 1.0
        onehot_list.append(oh)

    tokens_padded = pad_sequence(token_list, batch_first=True, padding_value=0)
    onehot_padded = pad_sequence(onehot_list, batch_first=True, padding_value=0.0)

    mask = torch.zeros((len(seqs), max_l), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    return ids, seqs, targets, tokens_padded, onehot_padded, mask, lengths


def load_local_or_hub_esm(model_path=None, model_name="esm2_t12_35M_UR50D"):
    if not HAS_ESM:
        raise ImportError("fair-esm is not installed. Please install via: pip install fair-esm")

    if model_path and os.path.exists(model_path):
        logger.info("Loading local ESM-2 weights from %s", model_path)
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    else:
        logger.info("Loading ESM-2 %s from hub or local cache", model_name)
        if model_name == "esm2_t12_35M_UR50D":
            esm_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        elif model_name == "esm2_t6_8M_UR50D":
            esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif model_name == "esm2_t30_150M_UR50D":
            esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif model_name == "esm2_t33_650M_UR50D":
            esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        else:
            raise ValueError(f"Unknown ESM model name: {model_name}")

    esm_model = esm_model.to(DEVICE)
    esm_model.eval()
    return esm_model, alphabet


def run_streaming_prediction(
    input_source,
    model_path,
    output_csv,
    esm_model_path=None,
    esm_name="esm2_t12_35M_UR50D",
    esm_layer=12,
    batch_size=64,
    id_col="ID",
    seq_col="Sequence",
    val_col="value"
):
    t_start = time.time()
    logger.info("Device=%s | model=%s | ESM=%s (layer %d) | batch_size=%d",
                DEVICE, model_path, esm_model_path if esm_model_path else esm_name, esm_layer, batch_size)

    esm_model, alphabet = load_local_or_hub_esm(model_path=esm_model_path, model_name=esm_name)
    batch_converter = alphabet.get_batch_converter()

    logger.info("Loading regressor weights from %s", model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Restore the architecture hyperparameters saved with the checkpoint.
    saved_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(saved_args, dict):
        saved_args = vars(saved_args)
    nhead = saved_args.get("nhead", 4)
    dropout = saved_args.get("dropout", 0.2)
    gate_type = saved_args.get("gate_type", "temp_softmax")
    gate_temp = saved_args.get("gate_temp", 1.2)

    esm_in_dim = state_dict.get("branch_esm.proj.0.weight", torch.zeros((256, 480))).shape[1]
    feat_dim = state_dict.get("branch_esm.proj.0.weight", torch.zeros((256, 480))).shape[0]

    regressor = AMPGATERegressor(
        esm_dim=esm_in_dim,
        feat_dim=feat_dim,
        nhead=nhead,
        dropout=dropout,
        gate_type=gate_type,
        gate_temp=gate_temp
    ).to(DEVICE)
    regressor.load_state_dict(state_dict, strict=False)
    regressor.eval()

    dataset = PeptideStreamDataset(input_source, id_col=id_col, seq_col=seq_col, val_col=val_col)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=stream_collate_fn)

    results = {
        "ID": [], "Sequence": [], "Length": [], "Pred_Value": [],
        "Gate_ESM": [], "Gate_Phys": [], "Gate_Motif": [],
        "Pred_ESM": [], "Pred_Phys": [], "Pred_Motif": []
    }
    has_true_values, all_true_vals = False, []

    logger.info("Running inference on %d sequences in %d batches.", len(dataset), len(loader))

    with torch.no_grad():
        for b_ids, b_seqs, b_vals, b_tokens, b_onehot, b_mask, b_lens in loader:
            batch_data = list(zip(b_ids, b_seqs))
            _, _, esm_tokens = batch_converter(batch_data)
            esm_tokens = esm_tokens.to(DEVICE)

            esm_out = esm_model(esm_tokens, repr_layers=[esm_layer], return_contacts=False)
            tok_emb = esm_out["representations"][esm_layer]

            emb_list = []
            for i, seq_str in enumerate(b_seqs):
                L = len(seq_str)
                seq_emb = tok_emb[i, 1: L + 1, :]
                emb_list.append(seq_emb)

            esm_features = pad_sequence(emb_list, batch_first=True, padding_value=0.0).to(DEVICE)
            b_tokens = b_tokens.to(DEVICE)
            b_onehot = b_onehot.to(DEVICE)
            b_mask = b_mask.to(DEVICE)

            main_preds, info = regressor(esm_features, b_tokens, b_onehot, b_mask)
            main_preds_np = main_preds.cpu().numpy().flatten()
            gates_np = info["gate_weights"].cpu().numpy()
            aux_esm, aux_phys, aux_motif = info["aux_preds"]

            for i in range(len(b_ids)):
                results["ID"].append(b_ids[i])
                results["Sequence"].append(b_seqs[i])
                results["Length"].append(b_lens[i])
                results["Pred_Value"].append(round(float(main_preds_np[i]), 5))
                results["Gate_ESM"].append(round(float(gates_np[i, 0]), 4))
                results["Gate_Phys"].append(round(float(gates_np[i, 1]), 4))
                results["Gate_Motif"].append(round(float(gates_np[i, 2]), 4))
                results["Pred_ESM"].append(round(float(aux_esm[i].cpu().item()), 5))
                results["Pred_Phys"].append(round(float(aux_phys[i].cpu().item()), 5))
                results["Pred_Motif"].append(round(float(aux_motif[i].cpu().item()), 5))

                if b_vals[i] is not None:
                    has_true_values = True
                    all_true_vals.append(float(b_vals[i]))

            del esm_tokens, esm_out, tok_emb, emb_list, esm_features, b_tokens, b_onehot, b_mask, main_preds, info

    elapsed = time.time() - t_start
    logger.info("Inference finished in %.2f s (%.1f seq/s).", elapsed, len(dataset) / elapsed)

    df_out = pd.DataFrame(results)
    if has_true_values and len(all_true_vals) == len(df_out):
        df_out["True_Value"] = all_true_vals
        df_out["Residual"] = df_out["Pred_Value"] - df_out["True_Value"]
        cols = ["ID", "Sequence", "Length", "True_Value", "Pred_Value", "Residual", "Gate_ESM", "Gate_Phys", "Gate_Motif", "Pred_ESM", "Pred_Phys", "Pred_Motif"]
        df_out = df_out[cols]

        y_true = np.array(all_true_vals)
        y_pred = np.array(results["Pred_Value"])
        huber_crit = nn.HuberLoss(delta=2.0) if hasattr(nn, "HuberLoss") else nn.SmoothL1Loss(beta=2.0)
        huber_loss_val = float(huber_crit(torch.tensor(y_pred, dtype=torch.float32), torch.tensor(y_true, dtype=torch.float32)).item())
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pcc, p_val = pearsonr(y_true, y_pred) if np.std(y_true) > 1e-8 else (0.0, 1.0)
        scc, s_val = spearmanr(y_true, y_pred) if np.std(y_true) > 1e-8 else (0.0, 1.0)

        logger.info("Evaluation: Huber=%.4f | MSE=%.4f | MAE=%.4f | R2=%.4f | PCC=%.4f (p=%.2e) | SCC=%.4f (p=%.2e)",
                    huber_loss_val, mse, mae, r2, pcc, p_val, scc, s_val)
        logger.info("Mean gate weights [ESM / Phys / Motif]: [%.3f / %.3f / %.3f]",
                    df_out['Gate_ESM'].mean(), df_out['Gate_Phys'].mean(), df_out['Gate_Motif'].mean())

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Predictions written to %s", output_csv)
    return df_out


def main():
    parser = argparse.ArgumentParser(
        description="AMP-GATE: streaming AMP activity prediction with on-the-fly ESM-2 extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Input FASTA, CSV/TSV, or peptide sequence string")
    parser.add_argument("-m", "--model_path", type=str, default="model/amp_gate.pth", help="Trained model path (.pth)")
    parser.add_argument("-o", "--output", type=str, default="predictions.csv", help="Output CSV path")
    parser.add_argument("--esm_model_path", type=str, default=None, help="Local ESM-2 weights path")
    parser.add_argument("--esm_name", type=str, default="esm2_t12_35M_UR50D", help="ESM model architecture name")
    parser.add_argument("--esm_layer", type=int, default=12, help="ESM layer index")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--id_col", type=str, default="ID", help="Sequence ID column name")
    parser.add_argument("--seq_col", type=str, default="Sequence", help="Sequence column name")
    parser.add_argument("--val_col", type=str, default="value", help="Target value column name")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_streaming_prediction(
        input_source=args.input,
        model_path=args.model_path,
        output_csv=args.output,
        esm_model_path=args.esm_model_path,
        esm_name=args.esm_name,
        esm_layer=args.esm_layer,
        batch_size=args.batch_size,
        id_col=args.id_col,
        seq_col=args.seq_col,
        val_col=args.val_col
    )


if __name__ == "__main__":
    main()
