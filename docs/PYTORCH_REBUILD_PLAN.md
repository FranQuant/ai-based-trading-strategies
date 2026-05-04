# PyTorch Rebuild Plan: AI Trading Strategies

## 1. Objective

The new PyTorch project will rebuild the AI trading strategy research pipeline as a clean, testable, reproducible codebase. It will preserve the methodological ideas from the legacy TensorFlow notebook repository while avoiding notebook state, duplicated implementation blocks, unclear artifact provenance, and unreproduced performance claims.

The rebuild should be separate from the TF2 repo because the current repository is best kept as a historical/reference artifact. Its notebooks are useful for understanding the original model lineup, features, target, validation stages, and backtest assumptions, but they are not a stable foundation for production-quality research code. A separate PyTorch repo allows the new project to start with explicit contracts, tests, configs, and run manifests without carrying TF2 environment drift or notebook execution risk.

## 2. Source References

- Legacy repo: `AI-based-Trading-Strategies`
- PyTorch style/reference repo: `pytorch_dl_apps`
- Legacy audit: `docs/PROJECT_REVIEW.md`
- Implementation-validity review: `docs/IMPLEMENTATION_VALIDATION.md`

The legacy repo should be treated as source material, not as code to copy. `PROJECT_REVIEW.md` defines the high-level architecture and migration direction. `IMPLEMENTATION_VALIDATION.md` defines what must be tested before the PyTorch pipeline can make public performance claims.

## 3. Design Principles

- Modularity: separate data loading, validation, feature engineering, target construction, sequence generation, splits, scaling, models, training, backtesting, metrics, and reporting.
- Reproducibility: every run should record data hash, config, code version, package versions, random seed, split dates, scaler policy, model parameters, thresholds, and output paths.
- Testability: core logic must be importable and covered by fast unit tests before any model training begins.
- No hidden notebook state: notebooks should call package APIs and render analysis; they should not define core implementation logic.
- Explicit data contracts: input files must have declared schemas, date behavior, numeric columns, missing-value policy, and content fingerprints.
- Explicit split contracts: train, validation, cross-validation, and OOS periods must be defined in config and emitted in manifests.
- No OOS leakage: OOS data must not influence feature selection, scaler fitting, threshold selection, architecture selection, or backtest parameter selection.
- Clear timing semantics: feature timestamp, prediction timestamp, signal timestamp, fill timestamp, and return timestamp must be explicit.
- Conservative claims: no performance claim is valid until regenerated under locked configs with reproducible artifacts.

## 4. What to Preserve from the TF2 Repo

- Dataset concept: use `GSPC_fixed.csv` as the initial parity dataset.
- Feature family: returns, rolling volatility, RSI, MACD, OBV, lagged returns, multi-horizon returns, ATR, skew/kurtosis, volume z-score, and volatility ratio.
- Target concept: next-day direction as the first parity target.
- Sequence design: 28-trading-day windows as the initial baseline.
- Model lineup: Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, and Transformer.
- Probability-output interface: models produce probabilities or logits convertible to probabilities.
- Threshold-to-signal logic: convert probability forecasts into long, short, or neutral positions.
- Shifted-position backtest principle: signals formed at date `t` must not earn return from date `t`.
- Cost-aware evaluation: transaction costs, turnover, benchmark alignment, and risk metrics remain part of the evaluation layer.

## 5. What Not to Reuse

- Notebook cells as implementation source.
- Duplicated feature, split, scaler, threshold, and backtest logic.
- Current threshold provenance as authoritative.
- Untested sequence/date/label alignment code.
- Untested backtest timing, turnover, and cost logic.
- Ambiguous SPY wording when the available dataset is S&P 500 index OHLCV.
- Mixed TF2 environment pins from `requirements.txt` and `envs/env_models_tf2.yml`.
- Unreproduced performance claims, charts, rankings, Sharpe ratios, CAGRs, drawdowns, and hit rates.
- Hidden generated artifacts that are not tied to configs and manifests.

## 6. Mandatory Validation Tests

- Data audit: verify required columns, row count, sorted dates, duplicate dates, blank cells, numeric coercion, date range, and file hash.
- Final-row target handling: ensure the final row with unavailable future return is removed or marked missing before label casting.
- Feature warmup and finite values: confirm expected warmup rows are dropped and all model features are finite after cleaning.
- Sequence/date/label alignment: prove with synthetic data that each 28-day feature window ending at `t` maps to the intended target return at `t+1`.
- Scaler fit window: prove scalers fit only on training rows and transform validation/OOS rows without refitting.
- Split/OOS integrity: verify fold date ranges, non-overlap, OOS start date, and no accidental OOS inclusion in tuning windows.
- Signal timing: prove predictions formed at `t` only affect returns after `t`.
- Transaction costs: test flat-to-long, long-to-flat, long-to-short, unchanged position, and volatility-targeted exposure changes.
- Benchmark alignment: prove strategy and benchmark use identical date windows and return definitions.

## 7. Proposed New Repository Structure

```text
ai-trading-strategies-pytorch/
├── configs/
│   ├── data/
│   │   └── gspc.yaml
│   ├── experiments/
│   │   ├── baseline_logistic.yaml
│   │   ├── stacked_lstm.yaml
│   │   ├── gru.yaml
│   │   ├── cnn_lstm.yaml
│   │   ├── attention_lstm.yaml
│   │   └── transformer.yaml
│   └── backtests/
│       └── default.yaml
├── data/
│   ├── raw/
│   │   └── GSPC_fixed.csv
│   └── processed/
├── docs/
│   ├── DESIGN.md
│   ├── VALIDATION.md
│   └── RUNBOOK.md
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_feature_review.ipynb
│   └── 03_results_review.ipynb
├── results/
│   ├── runs/
│   └── reports/
├── scripts/
│   ├── audit_data.py
│   ├── build_features.py
│   ├── run_cv.py
│   ├── run_oos.py
│   └── make_report.py
├── src/
│   └── ai_trading_strategies/
│       ├── __init__.py
│       ├── backtesting.py
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── metrics.py
│       ├── reporting.py
│       ├── scaling.py
│       ├── sequences.py
│       ├── splits.py
│       ├── targets.py
│       ├── training.py
│       └── models/
│           ├── __init__.py
│           ├── attention_lstm.py
│           ├── baseline.py
│           ├── cnn_lstm.py
│           ├── gru.py
│           ├── stacked_lstm.py
│           └── transformer.py
└── tests/
    ├── test_backtesting.py
    ├── test_data.py
    ├── test_features.py
    ├── test_scaling.py
    ├── test_sequences.py
    ├── test_splits.py
    └── test_targets.py
```

## 8. Environment Strategy

Start with a local CPU/Mac setup. The first milestone should not require CUDA, Colab, or long training runs. Use Python 3.11 or 3.12 if PyTorch and supporting packages are stable in the local environment.

Core dependencies:

- PyTorch for model definitions and later training.
- pandas for tabular data handling.
- numpy for numerical operations.
- scikit-learn for `StandardScaler`, baseline models, and simple metrics.
- matplotlib for plots.
- pytest for tests.
- pyarrow for optional Parquet feature caches.
- pydantic or dataclasses for typed configs and data/backtest contracts.
- jupyter for thin exploratory notebooks.

Optional later dependencies:

- CUDA-enabled PyTorch for GPU training.
- Colab-specific setup instructions.
- TensorBoard or MLflow only after the basic run manifest format is stable.

## 9. Implementation Roadmap

Milestone 1: Scaffold. Create the repository, package layout, `pyproject.toml`, basic config loading, pytest setup, and initial docs.

Milestone 2: Data contract. Implement CSV loading, schema validation, date checks, numeric coercion, data hash generation, and audit reports.

Milestone 3: Feature pipeline. Implement `feature_set_v1` from the TF2 reference as pure functions with warmup handling and finite-value checks.

Milestone 4: Target and sequence dataset. Implement next-day target construction with explicit final-row removal and 28-day sequence construction with timestamps.

Milestone 5: Split/scaler policy. Implement train/CV/OOS split contracts and scaler fitting that cannot include validation or OOS rows.

Milestone 6: Backtest engine. Implement signal conversion, shifted-position timing, transaction costs, turnover metrics, benchmark alignment, and synthetic tests.

Milestone 7: Baseline model. Add a non-deep baseline such as logistic regression or a tiny PyTorch MLP to validate the full pipeline without heavy training.

Milestone 8: LSTM/GRU. Implement PyTorch Stacked LSTM and GRU models using a common sequence-classifier interface.

Milestone 9: CNN-LSTM/Attention-LSTM. Implement hybrid and attention models after the basic recurrent path is stable.

Milestone 10: Transformer. Implement the Transformer encoder model with logits output and shared training/evaluation hooks.

Milestone 11: Colab/CUDA experiments. Add GPU run configs and Colab instructions after local tests and CPU smoke runs are reliable.

Milestone 12: Reporting and GitHub polish. Add result summaries, plots, run manifests, README, reproducibility notes, and publication caveats.

## 10. First Implementation Milestone

The first milestone should not train models. It should prove that the new repository can load, validate, transform, and align data correctly.

Scope:

- Data loading from `data/raw/GSPC_fixed.csv`.
- Data audit for schema, row count, date range, monotonic dates, duplicate dates, missing cells, numeric coercion, `Close` versus `Adjusted_close`, and file hash.
- Target construction for next-day direction with explicit handling of the unavailable final target.
- Feature engineering for the preserved TF2 feature family.
- Sequence construction for 28-day windows with explicit feature window start date, feature window end date, prediction date, and target date.
- Tests for data integrity, feature finite values, target final-row behavior, sequence/date/label alignment, scaler leakage, and OOS split integrity.

Acceptance:

- `pytest` passes locally.
- `python scripts/audit_data.py --config configs/data/gspc.yaml` emits a deterministic audit.
- No TensorFlow dependency exists.
- No model training code is required to pass the milestone.

## 11. Colab/CUDA Strategy

Colab and CUDA should be introduced only after local CPU development is reproducible and well tested. The local machine remains the source of truth for data contracts, feature generation, alignment tests, split definitions, scaler policy, and backtest logic.

Later Colab usage should be limited to heavier model training and larger experiment sweeps. Colab notebooks or scripts must consume the same configs as local runs and write the same run manifest fields. GPU runs should not define new preprocessing or evaluation logic. They should call the package APIs exactly as local scripts do.

Required Colab/CUDA practices:

- Pin install commands in a setup cell or setup script.
- Record PyTorch, CUDA, Python, pandas, numpy, and scikit-learn versions in each run manifest.
- Store results under `results/runs/<run_id>/`.
- Download or sync only generated artifacts, never manually edited metrics.
- Re-run local validation tests before accepting GPU results as publishable.

## 12. Deliverables

Expected files in the first new repo milestone:

- `pyproject.toml`
- `README.md`
- `.gitignore`
- `configs/data/gspc.yaml`
- `configs/backtests/default.yaml`
- `src/ai_trading_strategies/__init__.py`
- `src/ai_trading_strategies/config.py`
- `src/ai_trading_strategies/data.py`
- `src/ai_trading_strategies/features.py`
- `src/ai_trading_strategies/targets.py`
- `src/ai_trading_strategies/sequences.py`
- `src/ai_trading_strategies/splits.py`
- `src/ai_trading_strategies/scaling.py`
- `scripts/audit_data.py`
- `tests/test_data.py`
- `tests/test_features.py`
- `tests/test_targets.py`
- `tests/test_sequences.py`
- `tests/test_splits.py`
- `tests/test_scaling.py`
- `docs/DESIGN.md`
- `docs/VALIDATION.md`
- `docs/RUNBOOK.md`
- `data/raw/GSPC_fixed.csv` or documented instructions for placing it locally.

## 13. Go / No-Go Criteria

Go criteria before model training begins:

- Data audit passes and records a file hash.
- Target construction drops or marks the final unavailable future target explicitly.
- Feature pipeline produces the expected feature columns and finite values after warmup.
- Sequence tests prove date/window/label alignment on synthetic data and the real dataset.
- Split tests prove train, validation, CV, and OOS boundaries are correct.
- Scaler tests prove no validation or OOS data is used during fitting.
- Initial configs fully define data path, date column, price column, feature set, sequence length, split policy, and scaler policy.
- The first milestone has no TensorFlow dependency.
- `pytest` passes locally.

No-go conditions:

- Any off-by-one ambiguity remains in target, sequence, prediction, or backtest timing.
- Any scaler can be fit on validation or OOS rows.
- Any OOS rows can enter threshold selection or preprocessing fit state.
- The package requires notebooks to execute core logic.
- Performance claims are introduced before reproducible training and reporting exist.
