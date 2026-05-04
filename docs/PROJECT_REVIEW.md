# Project Review: AI-based-Trading-Strategies

## 1. Executive Summary

This repository is a notebook-first research project for comparing deep learning trading strategies on S&P 500 index data. The current implementation covers five TensorFlow/Keras architectures: Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, and Transformer. Each notebook follows a similar research workflow: load `data/GSPC_fixed.csv`, create technical indicators and lagged return features, build 28-day sequences, train a binary classifier, run cross-validation diagnostics, tune prediction thresholds, and perform an out-of-sample confirmatory backtest from 2022 onward.

The project already contains useful research assets: a clean single-asset dataset, model notebooks with broadly consistent stages, explicit transaction-cost-aware backtesting logic, QuantStats integration, and presentation materials. The main limitation is that nearly all logic lives inside notebooks. This makes the project hard to test, reproduce, compare, and migrate. A PyTorch migration should therefore not be a direct cell-by-cell translation. It should extract the shared data, feature, model, training, evaluation, and reporting logic into importable modules, then preserve the notebooks as thin experiment or presentation layers.

The recommended direction is to rebuild the core system as a small PyTorch research package with deterministic configuration, reusable datasets, model classes, walk-forward training, explicit artifact manifests, and regression tests around data alignment and backtesting. The existing notebooks should be preserved as reference implementations until parity checks confirm that the PyTorch pipeline produces comparable predictions, signal behavior, and backtest metrics.

## 2. Repository Inventory

- `README.md`: Project description, methodology summary, dependency installation instructions, structure overview, educational disclaimer, and embedded images for Transformer and Attention-LSTM out-of-sample equity curves.
- `requirements.txt`: Pip environment pinned around TensorFlow 2.12, NumPy 1.24.4, pandas 1.5.3, scikit-learn 1.3.2, Matplotlib, Seaborn, Optuna, QuantStats, Tabulate, and JupyterLab.
- `envs/env_models_tf2.yml`: Conda environment named `ai_strategies_tf2`, using Python 3.10.18 and Apple-oriented TensorFlow packages including `tensorflow-macos==2.13.0`, `keras==2.13.1`, `tensorboard==2.13.0`, pandas 2.2.0, scikit-learn 1.4.0, and Jupyter tooling.
- `data/GSPC_fixed.csv`: Daily S&P 500 index OHLCV data with columns `Date`, `Open`, `High`, `Low`, `Close`, `Adjusted_close`, and `Volume`. The inspected file has 2,809 data rows from 2014-01-06 through 2025-03-07.
- `notebooks/1_stacked_lstm.ipynb`: Stacked LSTM strategy notebook with 35 cells, including setup, feature engineering, sequence building, model training, stages A-G diagnostics, Stage H out-of-sample testing, and QuantStats analysis.
- `notebooks/2_gru.ipynb`: GRU strategy notebook with the same general workflow and a GRU-specific model definition.
- `notebooks/3_cnn_lstm.ipynb`: CNN-LSTM hybrid notebook using convolutional temporal feature extraction before LSTM sequence modeling.
- `notebooks/4_attention_lstm.ipynb`: Attention-LSTM notebook using stacked LSTM logic with a temporal attention mechanism.
- `notebooks/5_transformer.ipynb`: Transformer encoder notebook using dense feature projection, multi-head attention, residual normalization, feed-forward layers, global pooling, and sigmoid output.
- `outputs/04_ATT_LSTM_clean/oos_final/equity_curve_oos_enhanced.png`: Tracked presentation output for Attention-LSTM out-of-sample performance.
- `outputs/05_TRANSFORMER/oos_final/equity_curve_oos_enhanced.png`: Tracked presentation output for Transformer out-of-sample performance.
- `presentation_assets/1st_slide.png`: README and presentation image.
- `presentation_assets/Mkts_Dev.png`: README market-development image.
- `presentation_assets/AIIFC 2025.pdf`: Presentation deck artifact.
- `.gitignore`: Includes output and notebook-related ignore rules, while selected output images remain tracked.

## 3. Current Architecture

The current architecture is organized around self-contained notebooks rather than reusable Python modules. Each notebook embeds the full pipeline: imports, constants, data loading, plotting, feature engineering, sequence creation, model definition, training, cross-validation, threshold tuning, backtesting, visualization, and QuantStats reporting.

The common workflow is:

- Read `../data/GSPC_fixed.csv` with `Date` as the index.
- Normalize column names to lowercase.
- Convert OHLCV columns to numeric values and forward-fill missing values.
- Compute log returns and technical features.
- Create a one-step-ahead binary target from next-day log return direction.
- Drop rows with missing feature or target values.
- Convert tabular rows into rolling 28-day sequences.
- Train a TensorFlow/Keras classifier for each architecture.
- Save cross-validation predictions, labels, and indexes under model-specific `outputs` directories.
- Convert predicted probabilities into long, short, or neutral signals using tuned thresholds.
- Run a backtest with transaction costs, stops, holding constraints, and volatility targeting.
- Produce OOS charts and QuantStats output.

Architecturally, the five notebooks duplicate large blocks of code. The model-definition function is the primary difference across notebooks. This duplication increases maintenance risk: a bug fix in alignment, leakage prevention, backtest accounting, or threshold tuning must be copied consistently across all notebooks.

## 4. Data and Feature Pipeline

The data pipeline uses `data/GSPC_fixed.csv`, which contains daily S&P 500 OHLCV history from 2014-01-06 to 2025-03-07. The notebooks read this file locally, lowercase the column names, coerce `open`, `high`, `low`, `close`, `adjusted_close`, and `volume` to numeric values, sort by date, and forward-fill missing observations.

The shared feature pipeline computes:

- `log_returns` from adjusted close prices.
- `volatility_10`, a 10-day rolling standard deviation of log returns.
- `RSI_14`.
- MACD features: `MACD`, `MACD_line`, and `MACD_signal`.
- `OBV` using signed volume.
- Lagged log returns `lag_1` through `lag_7`.
- Multi-horizon returns `ret_3d`, `ret_5d`, and `ret_10d`.
- Annualized volatility features `vol_10d`, `vol_21d`, and `vol_63d`.
- `ATR_14` from high, low, and adjusted close.
- Rolling return shape features `ret_skew_21` and `ret_kurt_21`.
- `vol_zscore_20`.
- `vol_ratio`, defined from short and long volatility estimates.

The target is a binary label indicating whether the next log return is positive. Sequences are built with a 28-trading-day lookback, and each sequence label is aligned to the final timestamp of the input window. The OOS split used in the confirmatory stage is `2022-01-01`, meaning training and validation occur before 2022 and final OOS prediction occurs from 2022 onward.

The feature approach is reasonable for a research prototype, but the migration should make all feature definitions explicit, tested, and versioned. Alignment tests are especially important because a one-row shift error can materially inflate backtest results.

## 5. Model Inventory

- Stacked LSTM: Implemented in `notebooks/1_stacked_lstm.ipynb` through `get_stacked_lstm_model`. It uses two LSTM layers, batch normalization, dropout, and a sigmoid output for binary classification.
- GRU: Implemented in `notebooks/2_gru.ipynb` through `get_gru_model`. It uses a GRU sequence encoder with regularization and sigmoid classification output.
- CNN-LSTM: Implemented in `notebooks/3_cnn_lstm.ipynb` through `get_cnn_lstm`. It applies `Conv1D` to capture local temporal patterns before LSTM sequence modeling.
- Attention-LSTM: Implemented in `notebooks/4_attention_lstm.ipynb` through `get_stacked_att_lstm`. It uses stacked LSTM layers plus temporal attention for weighted sequence aggregation.
- Transformer: Implemented in `notebooks/5_transformer.ipynb` through `get_transformer_model`. It projects features into an embedding dimension, applies Keras `MultiHeadAttention`, residual layer normalization, a feed-forward block, global average pooling, dropout, and sigmoid output.

All models are currently binary classifiers that output a probability used by downstream thresholding. This design should be preserved in the PyTorch migration because it cleanly separates model inference from trading rule construction.

## 6. Backtesting and Evaluation

The notebooks include a shared `run_backtest_clean` function that aligns predictions to the prediction index, maps probabilities to long, short, or neutral signals, applies position management, and computes strategy returns. The backtest includes:

- Long and short thresholds, usually around a base threshold with a symmetric band.
- Optional neutral band between short and long thresholds.
- Minimum holding period logic.
- Daily stop logic.
- Rolling loss stop logic.
- Volatility targeting using a rolling volatility window.
- Maximum leverage cap.
- Transaction costs of 5 basis points per turnover event.
- Turnover and hit-rate reporting.

Evaluation includes cross-validation diagnostics with `TimeSeriesSplit`, classification metrics such as ROC AUC, threshold sweeps, Sharpe sensitivity, Stage B confirmatory backtests, Stage C risk/backtest configuration sweeps, Stage D prediction distribution review, Stage E threshold tuning diagnostics, and Stage H OOS confirmatory analysis from 2022 onward. QuantStats is used for richer performance reporting.

This is a strong foundation for a research notebook. For production-quality research, the evaluation layer should be rebuilt as deterministic scripts that emit structured metrics, figures, and metadata for each run. The current tracked `outputs` directory only contains final equity-curve images for Attention-LSTM and Transformer, so most intermediate experiment artifacts are either ignored, local-only, or absent from the repository state.

## 7. Methodological Risks

- Single-asset scope: The project evaluates only S&P 500 index data. A model can appear effective because it learned one index regime rather than a robust market pattern.
- Short OOS history: The OOS window starts at 2022-01-01. This is valuable because it includes difficult market conditions, but it is still a limited confirmatory sample.
- Threshold overfitting: Multiple threshold sweeps and configuration stages can overfit validation or CV results if final choices are not locked before OOS evaluation.
- Notebook state risk: Results may depend on cell execution order, hidden kernel state, or artifacts created in prior runs.
- Label simplicity: Next-day directional classification may not correspond directly to tradable edge after costs, turnover, and sizing constraints.
- Multiple-comparison risk: Five architectures, many thresholds, and multiple backtest configurations increase the chance of selecting an apparently strong result by luck.
- Feature leakage risk: The feature functions appear backward-looking, but rolling windows, target shifts, sequence alignment, scaling, and CV splits need automated tests to verify no future information leaks into training.
- Cost model simplification: A 5 bps turnover cost is useful, but real execution would also require spreads, slippage, market impact, borrow constraints for short exposure, and capacity assumptions.
- Benchmark ambiguity: README compares to SPY/buy-and-hold language, while the dataset is S&P 500 index data. A migration should make benchmark construction explicit.

## 8. Engineering Risks

- High duplication: Data loading, feature engineering, model training, backtesting, and reporting are repeated across notebooks.
- Low testability: Core logic is embedded in notebook cells instead of importable, unit-testable modules.
- Environment inconsistency: `requirements.txt` uses TensorFlow 2.12 and pandas 1.5.3, while `envs/env_models_tf2.yml` uses TensorFlow-MacOS 2.13, Keras 2.13.1, pandas 2.2.0, and scikit-learn 1.4.0.
- Platform coupling: The conda environment has a local prefix and Apple-specific TensorFlow packages, which limits portability.
- Artifact ambiguity: Notebooks write many outputs, but the repository currently tracks only two OOS equity-curve PNGs under `outputs`.
- No formal experiment registry: There is no manifest connecting a model run to code version, data version, hyperparameters, thresholds, metrics, and output files.
- No CI: There are no automated checks for formatting, tests, notebook execution, or reproducible metrics.
- No package boundary: There is no `src` package, no CLI, and no reusable Python API.
- Reproducibility gaps: Random seeds, deterministic backend settings, split manifests, and saved scalers are not centralized.
- Documentation drift: README references `AIFC 2025.pdf` in the tree, while the actual file is `presentation_assets/AIIFC 2025.pdf`.

## 9. What Should Be Preserved

- The overall research question: comparing deep learning architectures for systematic trading signals.
- The five-model inventory: Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, and Transformer.
- The single probability-output contract for all models.
- The feature set as a baseline feature version, including returns, volatility, RSI, MACD, OBV, ATR, rolling shape statistics, volume z-score, and lagged returns.
- The 28-day sequence formulation as an initial baseline.
- The 2022-plus OOS confirmatory concept.
- `TimeSeriesSplit` or an equivalent walk-forward validation pattern.
- Threshold-based conversion from probabilities to trading positions.
- Neutral-band support.
- Transaction-cost-aware backtesting.
- Volatility targeting and risk overlays, subject to clearer tests and documentation.
- QuantStats reporting, if it remains compatible with the new metrics pipeline.
- Existing presentation assets and tracked OOS equity-curve images as historical reference artifacts.
- The educational/research disclaimer.

## 10. What Should Be Rebuilt

- Rebuild feature engineering as a versioned, tested Python module.
- Rebuild dataset creation as a PyTorch-compatible `Dataset` and `DataLoader` pipeline.
- Rebuild model definitions as PyTorch `nn.Module` classes with a shared classifier interface.
- Rebuild training loops with explicit device handling, mixed precision where appropriate, early stopping, checkpointing, and deterministic seeding.
- Rebuild validation as walk-forward scripts that save fold-level predictions and metrics.
- Rebuild backtesting as a tested module that accepts predictions, dates, prices, returns, and a typed configuration object.
- Rebuild threshold tuning so that CV tuning, validation selection, and final OOS confirmation are separated.
- Rebuild experiment tracking with structured YAML or JSON configs and run manifests.
- Rebuild outputs so each run writes metrics, predictions, equity curves, config, code version, and data fingerprint.
- Rebuild notebooks as thin consumers of package APIs, not the source of truth.
- Rebuild environment management around a single authoritative PyTorch environment.

## 11. PyTorch Migration Roadmap

Phase 1: Freeze the TensorFlow baseline. Export a minimal baseline manifest for each current notebook: model name, input features, sequence length, split dates, thresholds, key hyperparameters, OOS metrics, prediction files if available, and final figures. This creates a parity target before changing frameworks.

Phase 2: Extract framework-independent code. Move data loading, feature engineering, sequence construction, split generation, thresholding, and backtesting into pure Python modules that do not depend on TensorFlow or PyTorch. Add tests for date sorting, numeric coercion, feature columns, target shift, sequence-label alignment, prediction-index alignment, cost accounting, and benchmark returns.

Phase 3: Define configuration files. Create model and experiment configs for Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, and Transformer. Include data path, feature version, sequence length, split dates, model hyperparameters, optimizer settings, early stopping settings, threshold rules, and backtest configuration.

Phase 4: Implement PyTorch datasets. Build a tabular-to-sequence dataset that returns `(features, label, timestamp)` and supports train, validation, CV, and OOS splits. Persist fitted scalers from training windows only and apply them to validation/OOS data without refitting.

Phase 5: Port models. Implement PyTorch versions of each architecture using a shared interface that returns logits. Use `BCEWithLogitsLoss` during training and apply sigmoid only for metrics, thresholding, and saved predictions.

Phase 6: Implement training and evaluation loops. Add deterministic seeds, device selection for CPU/CUDA/MPS, gradient clipping, early stopping, checkpoint saves, metric logging, and fold-level prediction export. Keep probability output parity with the current notebooks.

Phase 7: Rebuild backtest reporting. Convert model probabilities into signals using locked threshold configs, run backtests, save equity curves and CSV metrics, and optionally call QuantStats from a reporting layer.

Phase 8: Run parity checks. Compare TensorFlow and PyTorch pipelines on the same data split and feature version. Exact predictions are not expected, but shapes, indexes, labels, scaler behavior, signal counts, threshold semantics, and backtest accounting should match.

Phase 9: Convert notebooks. Replace long implementation notebooks with concise notebooks that import the package, run configured experiments, and render summaries. Keep the original notebooks archived until migration parity is accepted.

Phase 10: Add CI and release hygiene. Add unit tests, smoke tests for one tiny training run, linting, formatting, and documentation checks. CI should not require full model training, but it should verify that the core pipeline executes on a small deterministic sample.

## 12. Recommended New Repository Structure

```text
AI-based-Trading-Strategies/
├── configs/
│   ├── data/gspc.yml
│   ├── models/stacked_lstm.yml
│   ├── models/gru.yml
│   ├── models/cnn_lstm.yml
│   ├── models/attention_lstm.yml
│   └── models/transformer.yml
├── data/
│   └── GSPC_fixed.csv
├── docs/
│   └── PROJECT_REVIEW.md
├── envs/
│   ├── env_models_tf2.yml
│   └── env_models_torch.yml
├── notebooks/
│   ├── legacy_tf2/
│   └── pytorch_experiments/
├── outputs/
│   ├── historical/
│   └── runs/
├── presentation_assets/
├── src/
│   └── ai_trading_strategies/
│       ├── __init__.py
│       ├── backtesting.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── reporting.py
│       ├── splits.py
│       ├── training.py
│       └── models/
│           ├── __init__.py
│           ├── attention_lstm.py
│           ├── cnn_lstm.py
│           ├── gru.py
│           ├── stacked_lstm.py
│           └── transformer.py
├── tests/
│   ├── test_backtesting.py
│   ├── test_features.py
│   ├── test_sequences.py
│   └── test_splits.py
├── scripts/
│   ├── run_cv.py
│   ├── run_oos.py
│   └── make_report.py
├── README.md
├── requirements.txt
└── pyproject.toml
```

This structure separates reusable library code from experiments, keeps legacy TensorFlow notebooks available, and creates a path for reproducible PyTorch runs without losing the presentation-oriented material already in the repository.

## 13. Immediate Next Steps

1. Commit this audit as documentation only.
2. Create a migration branch for code changes after the documentation-only state is accepted.
3. Add a small `src/ai_trading_strategies` package with only data loading, feature engineering, sequence creation, and split utilities.
4. Add tests that prove target shifting, sequence-label alignment, scaler fitting, and OOS split behavior are correct.
5. Extract the current `run_backtest_clean` behavior into a tested module before changing model frameworks.
6. Define one YAML config for the existing Transformer notebook because it has tracked OOS output and is the best first parity target.
7. Implement a PyTorch Transformer baseline and compare its pipeline artifacts against the TensorFlow Transformer reference.
8. Port the remaining four models after the shared PyTorch training loop and backtest report are stable.
9. Replace duplicated notebook logic with calls into the package.
10. Resolve environment drift by adding a single PyTorch environment file and documenting whether CPU, CUDA, or Apple MPS is the primary supported runtime.
