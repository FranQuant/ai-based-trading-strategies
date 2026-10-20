# Runbook

## Purpose

This repository is a notebook-first research project for comparing TensorFlow/Keras deep learning architectures for S&P 500 directional trading strategies. The current notebooks build technical features from `data/GSPC_fixed.csv`, train binary classifiers on 28-day sequences, tune probability thresholds, and run transaction-cost-aware out-of-sample backtests.

The repository is intended for research and education. It is not a production trading system.

## Environment Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

There is also a Conda environment export at `envs/env_models_tf2.yml`:

```bash
conda env create -f envs/env_models_tf2.yml
conda activate ai_strategies_tf2
jupyter lab
```

Use one environment path per run. The `requirements.txt` file pins `tensorflow==2.12.0`, while the Conda export is a macOS-oriented environment containing `tensorflow-macos==2.13.0`. These are not identical environments, so results and compatibility can differ.

## Notebook Execution Order

Run notebooks from the `notebooks/` directory so relative paths such as `../data/GSPC_fixed.csv` and `../outputs/...` resolve correctly.

Recommended order:

1. `notebooks/1_stacked_lstm.ipynb`
2. `notebooks/2_gru.ipynb`
3. `notebooks/3_cnn_lstm.ipynb`
4. `notebooks/4_attention_lstm.ipynb`
5. `notebooks/5_transformer.ipynb`

Each notebook is largely self-contained and defines its own output directory. Running a notebook can retrain models and overwrite files under its corresponding `outputs/` subdirectory.

## Expected Artifacts

The notebooks can create model-specific directories under `outputs/`, including:

- `outputs/01_STACKED_LSTM/`
- `outputs/02_GRU/`
- `outputs/04_CNN_LSTM/`
- `outputs/04_ATT_LSTM_clean/`
- `outputs/05_TRANSFORMER/`

Expected generated artifacts include cross-validation predictions and labels, threshold sweep CSVs, tuning diagnostics, out-of-sample backtest summaries, simple strategy and benchmark return files, and equity/diagnostic plots.

At the time of this Phase 0 note, the tracked generated outputs are limited to:

- `outputs/04_ATT_LSTM_clean/oos_final/equity_curve_oos_enhanced.png`
- `outputs/05_TRANSFORMER/oos_final/equity_curve_oos_enhanced.png`

## Known Limitations

- Core logic lives inside notebooks, not importable modules.
- There are no automated tests for feature construction, sequence alignment, target shifting, scaler behavior, or backtest accounting.
- Notebook execution order inside a live kernel can affect reproducibility if cells are skipped or rerun out of order.
- Random seeds are set in notebooks, but deterministic backend settings and environment manifests are not centralized.
- Existing outputs are partial and should not be treated as a complete experiment registry.

## TensorFlow Caveats

- TensorFlow package compatibility depends on Python version, platform, and CPU/GPU backend.
- `tensorflow==2.12.0` from `requirements.txt` is different from the macOS `tensorflow-macos==2.13.0` environment export.
- Apple Silicon users may need the Conda/macOS path rather than plain `tensorflow`.
- Training results may vary across TensorFlow versions, hardware, BLAS libraries, and notebook execution state.
- Phase 0 does not migrate the project to PyTorch or change model logic.
