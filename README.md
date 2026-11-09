# AMP-GATE: HeteroTri-Fusion AMP Quantitative Activity Regressor

A PyTorch regressor that predicts the **quantitative antimicrobial activity** of a peptide sequence from three heterogeneous feature views, fused by sample-adaptive gated cross-attention.

## Background

Antimicrobial peptide (AMP) activity is not governed by a single type of signal. It depends simultaneously on global evolutionary semantics, residue-level physicochemical properties, and short local sequence motifs. A homogeneous encoder tends to blur these very different signals. AMP-GATE therefore assigns a dedicated encoder to each view:

| Branch | Input | Encoder | Rationale |
| --- | --- | --- | --- |
| A | ESM-2 residue embeddings (480-d, layer 12) | Sinusoidal positional encoding + 2-layer Pre-LN Transformer + masked max pooling | Global contextual and geometric semantics |
| B | 18 standardised physicochemical scales per residue | Dual-scale gated Conv1D (k=3, 5, GLU) + masked max pooling | Continuous charge / hydrophobicity priors without an over-parameterised RNN |
| C | 20-d one-hot sequence | Multi-scale Inception-1D (k=1, 3, 5, 7) + squeeze-and-excitation attention + masked max pooling | Discrete local motifs such as cationic clusters |

The ESM branch acts as the query in two cross-attention modules that retrieve physicochemical and motif information. A small gate network produces a per-peptide weight vector `[Gate_ESM, Gate_Phys, Gate_Motif]`, so each sequence decides how much to rely on each modality; the weights are exported with every prediction for interpretation. All branches use ReLU activations and masked max-over-time pooling, and each branch has an auxiliary regression head that adds a weighted auxiliary loss during training.

By default, training and evaluation use the datasets shipped in `data/`. The included checkpoint is ready for immediate inference and reports R² = 0.850, Pearson PCC = 0.922 and CI = 0.904 (see `model/test_metrics_summary.csv`).

## Key features

- **Heterogeneous tri-branch architecture** with modality-specific encoders rather than one shared backbone.
- **Sample-adaptive gated cross-modal fusion** with temperature-scaled softmax (`temp_softmax`) or independent `sigmoid` gates.
- **Auxiliary multi-task supervision** on the three single-branch predictions.
- **Quantile-stratified train/validation split**, AdamW, `ReduceLROnPlateau`, gradient clipping and early stopping.
- **Comprehensive metrics**: MSE, RMSE, MAE, R², Pearson PCC, Spearman SCC and concordance index (CI), reported overall and on the active-AMP subset.
- **Streaming low-memory inference** that extracts ESM-2 embeddings on the fly from FASTA, CSV/TSV, or a single sequence string, with automatic evaluation when ground-truth values are present.

## Repository layout

```
AMP-GATE/
├── Train_HeteroTriFusion_Regressor.py    # model, metrics, training entry point
├── Predict_HeteroTriFusion_Regressor.py # streaming inference entry point
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── train.csv / test.csv             # training and evaluation datasets
│   └── train.fasta / test.fasta         # matching sequences
├── model/
│   ├── best_hetero_regressor.pth        # example checkpoint (~26 MB)
│   └── test_metrics_summary.csv         # metrics recorded for that checkpoint
└── output/                              # created by training (git-ignored)
```

## Requirements

- Python 3.9+
- PyTorch >= 1.12 (`batch_first` attention, `HuberLoss`)
- numpy, pandas, scipy
- scikit-learn (optional; a built-in fallback is used)
- Biopython (optional; a plain-text FASTA parser is used as fallback)
- `fair-esm` (**prediction only**, for on-the-fly ESM-2 extraction)

```bash
pip install -r requirements.txt
```

The ESM-2 weights (`esm2_t12_35M_UR50D`, ~128 MB) are downloaded automatically on the first prediction run, or can be supplied locally with `--esm_model_path`.

## Data format

Training expects CSV files with the following columns (column names are configurable):

| Column | Meaning |
| --- | --- |
| `ID` | Unique identifier; must match the FASTA record ID and the embedding filename `<ID>.pt` |
| `Sequence` | Peptide sequence of the 20 canonical amino acids |
| `value` | Continuous regression target (log10(MIC)-like activity value) |
| `label` | `1` for active AMPs, `0` for inactive; used only for the active-subset metrics |
| `Length` | Sequence length (informational, not read by the code) |

If `label` is missing, inactive samples are inferred from the sentinel target value `3.91339`. Sequences are encoded over the 20 canonical amino acids and may contain up to 256 residues.

### Preparing ESM-2 embeddings for training

Training reads pre-computed per-token ESM-2 representations: one file `<ID>.pt` per sequence containing
`{"representations": {<layer>: tensor[L+2, 480]}}`. The following snippet generates exactly this format from the included FASTA files:

```python
import torch, esm
from Bio import SeqIO
from pathlib import Path

LAYER = 12
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model.eval()
converter = alphabet.get_batch_converter()
out_dir = Path("data/esm_embeddings")
out_dir.mkdir(parents=True, exist_ok=True)

for fasta in ["data/train.fasta", "data/test.fasta"]:
    for record in SeqIO.parse(fasta, "fasta"):
        _, _, tokens = converter([(record.id, str(record.seq))])
        with torch.no_grad():
            reps = model(tokens, repr_layers=[LAYER], return_contacts=False)["representations"][LAYER]
        torch.save({"representations": {LAYER: reps[0].cpu()}}, out_dir / f"{record.id}.pt")
```

For large datasets, batch the sequences before extraction. Embeddings are not stored in the repository; `data/esm_embeddings/` is git-ignored.

## Training

```bash
python Train_HeteroTriFusion_Regressor.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --embedding_dir data/esm_embeddings \
  --output_dir output
```

The defaults train on `data/train.csv` and evaluate on `data/test.csv`, so the same run is simply `python Train_HeteroTriFusion_Regressor.py`.

### Training parameters

| Argument | Default | Description |
| --- | --- | --- |
| `-tr`, `--train_csv` | `data/train.csv` | Training CSV path |
| `-te`, `--test_csv` | `data/test.csv` | Independent test CSV path |
| `-ed`, `--embedding_dir` | `data/esm_embeddings` | Directory of `<ID>.pt` ESM embeddings |
| `-o`, `--output_dir` | `output` | Directory for checkpoints and reports |
| `--id_col` | `ID` | ID column name |
| `--seq_col` | `Sequence` | Sequence column name |
| `--val_col` | `value` | Regression target column name |
| `--label_col` | `label` | Activity label column name |
| `--esm_layer` | `12` | ESM representation layer index |
| `--esm_dim` | `480` | ESM embedding dimension |
| `--feat_dim` | `256` | Shared feature dimension of the three encoders and fusion |
| `--nhead` | `4` | Attention heads in the Transformer and cross-attention |
| `--aux_weight` | `0.2` | Weight of the auxiliary single-branch losses |
| `--dropout` | `0.2` | Dropout rate |
| `--gate_type` | `temp_softmax` | Gate activation: `temp_softmax`, `sigmoid` or `softmax` (`softmax` is an alias of `temp_softmax`) |
| `--gate_temp` | `1.2` | Temperature for `temp_softmax` gating |
| `--loss_type` | `mse` | Main loss: `mse`, `huber`, `smooth_l1` or `l1` |
| `--huber_delta` | `2.0` | Delta for Huber / SmoothL1 loss |
| `-b`, `--batch_size` | `64` | Batch size |
| `-e`, `--epochs` | `100` | Maximum number of epochs |
| `--lr` | `5e-4` | Initial learning rate (AdamW) |
| `--weight_decay` | `0.001` | Weight decay |
| `--val_ratio` | `0.1765` | Validation fraction drawn from each target quantile bin |
| `--monitor_metric` | `MSE` | Early-stopping metric: `MSE`, `RMSE`, `MAE`, `R2`, `Pearson_PCC`, `Spearman_SCC`, `LOSS` |
| `--patience` | `15` | Early-stopping patience (`0` disables it) |
| `--lr_patience` | `8` | Patience before halving the learning rate |
| `--seed` | `42` | Random seed |

The device is selected automatically in the order CUDA → MPS → CPU.

### Training outputs (in `--output_dir`)

| File | Content |
| --- | --- |
| `best_hetero_regressor.pth` | Best checkpoint: model weights, optimiser state, epoch, best score and full arguments |
| `training_history_log.csv` | Per-epoch train/validation metrics, gate weights and learning rate |
| `test_predictions.csv` | Independent-test predictions, residuals, gate weights and auxiliary predictions |
| `test_metrics_summary.csv` | Final metrics for the overall test set and the active-AMP subset |

## Prediction

```bash
# FASTA input
python Predict_HeteroTriFusion_Regressor.py -i data/test.fasta -o predictions.csv

# CSV input containing a `value` column: metrics are computed automatically
python Predict_HeteroTriFusion_Regressor.py -i data/test.csv -o predictions_eval.csv

# A single sequence
python Predict_HeteroTriFusion_Regressor.py -i RKGLQKLVGRVHRLLRK -o predictions_single.csv
```

Supported inputs: `.fasta/.fa/.fas/.fna`, `.csv`, `.tsv/.txt` (tab-separated), a raw sequence string, or a Python list of `(id, sequence)` tuples when calling `run_streaming_hetero_regression()` programmatically.

### Prediction parameters

| Argument | Default | Description |
| --- | --- | --- |
| `-i`, `--input` | *required* | FASTA / CSV / TSV file, or a peptide sequence string |
| `-m`, `--model_path` | `model/best_hetero_regressor.pth` | Trained checkpoint (`.pth`) |
| `-o`, `--output` | `predictions.csv` | Output CSV path |
| `--esm_model_path` | `None` | Local ESM-2 weights; if omitted, hub/cache weights are used |
| `--esm_name` | `esm2_t12_35M_UR50D` | ESM architecture: `esm2_t6_8M_UR50D`, `esm2_t12_35M_UR50D`, `esm2_t30_150M_UR50D`, `esm2_t33_650M_UR50D` |
| `--esm_layer` | `12` | ESM representation layer index |
| `-b`, `--batch_size` | `64` | Inference batch size (lower it to reduce peak memory) |
| `--id_col` | `ID` | ID column name for tabular input |
| `--seq_col` | `Sequence` | Sequence column name for tabular input |
| `--val_col` | `value` | Ground-truth column name; if present, evaluation metrics are reported |

Model dimensions (`feat_dim`, `esm_dim`) and the gating configuration are restored from the checkpoint, so they do not need to be specified at prediction time. **The ESM model used for inference must produce the same embedding dimension as the training embeddings**: the shipped checkpoint uses `esm2_t12_35M_UR50D` (480-d, layer 12). Other ESM variants require retraining with the matching `--esm_dim`.

### Prediction outputs

The result CSV contains `ID`, `Sequence`, `Length`, `Pred_Value`, `Gate_ESM`, `Gate_Phys`, `Gate_Motif`, `Pred_ESM`, `Pred_Phys`, `Pred_Motif`. When a ground-truth column is available, `True_Value` and `Residual` are added and Huber loss, MSE, MAE, R², Pearson PCC and Spearman SCC are logged.

## License

Released under the [MIT License](LICENSE).
