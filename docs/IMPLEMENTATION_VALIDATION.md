# Implementation Validation: AI-based-Trading-Strategies

## 1. Validation Objective

This is a methodological implementation-validity review of the existing TensorFlow/Scikit-learn notebook repository. It is not a full performance reproduction, not a TensorFlow training run, and not an end-to-end notebook execution review.

The purpose is to separate what can be validated from local static and lightweight evidence from what still requires full notebook execution. The review uses only safe local checks: repository listing, CSV inspection, and notebook JSON/code parsing with Python standard-library modules. No TensorFlow training was run, no notebooks were executed end to end, and no performance claim is treated as reproduced.

Validation categories used in this report:

- Code-inspection validated: supported by reading notebook source code or repository files.
- Lightweight-execution validated: supported by local scripts that parse CSV/notebook JSON without TensorFlow or `nbformat`.
- Requires full TensorFlow execution: cannot be confirmed without running model training/inference cells.
- Not validated from current repository: current files do not contain enough evidence.

## 2. Evidence Collected

Files inspected:

- `README.md`
- `requirements.txt`
- `envs/env_models_tf2.yml`
- `data/GSPC_fixed.csv`
- `notebooks/1_stacked_lstm.ipynb`
- `notebooks/2_gru.ipynb`
- `notebooks/3_cnn_lstm.ipynb`
- `notebooks/4_attention_lstm.ipynb`
- `notebooks/5_transformer.ipynb`
- `outputs/04_ATT_LSTM_clean/oos_final/equity_curve_oos_enhanced.png`
- `outputs/05_TRANSFORMER/oos_final/equity_curve_oos_enhanced.png`
- `presentation_assets/1st_slide.png`
- `presentation_assets/Mkts_Dev.png`
- `presentation_assets/AIIFC 2025.pdf`
- `docs/PROJECT_REVIEW.md`

Commands/checks used:

- `pwd`
- `git status --short`
- `find . -maxdepth 3 -type f | sort`
- Python standard-library CSV inspection using `csv` and `pathlib`
- Python standard-library notebook inspection using `json`, `re`, and `pathlib`

Summarized command evidence:

```text
pwd
/Users/air/Projects/AI-based-Trading-Strategies

git status --short
?? docs/IMPLEMENTATION_VALIDATION.md
```

The file inventory confirmed the expected repository shape: README, requirements, TF2 conda environment, five notebooks, one CSV dataset, selected output charts, presentation assets, `docs/PROJECT_REVIEW.md`, and this untracked validation document.

Notebook JSON evidence:

```text
notebooks/1_stacked_lstm.ipynb: total=35, code=19, markdown=16, factory=get_stacked_lstm_model
notebooks/2_gru.ipynb: total=35, code=19, markdown=16, factory=get_gru_model
notebooks/3_cnn_lstm.ipynb: total=35, code=19, markdown=16, factory=get_cnn_lstm
notebooks/4_attention_lstm.ipynb: total=34, code=18, markdown=16, factory=get_stacked_att_lstm
notebooks/5_transformer.ipynb: total=34, code=18, markdown=16, factory=get_transformer_model
```

## 3. Data Integrity Findings

Lightweight-execution validated:

```text
CSV_PATH data/GSPC_fixed.csv
COLUMNS ['Date', 'Open', 'High', 'Low', 'Close', 'Adjusted_close', 'Volume']
ROW_COUNT 2810
FIRST_DATE 2014-01-06
LAST_DATE 2025-03-07
DATE_MONOTONIC_STRICT True
DUPLICATE_DATE_COUNT 0
BLANK_CELL_COUNT 0
NON_NUMERIC_COUNTS {'Open': 0, 'High': 0, 'Low': 0, 'Close': 0, 'Adjusted_close': 0, 'Volume': 0}
CLOSE_ADJUSTED_CLOSE_DIFFERING_ROWS 0
```

The CSV is structurally clean under lightweight checks: expected columns, strictly increasing dates, no duplicate dates, no blank cells, and no non-numeric OHLCV cells. `Close` and `Adjusted_close` are identical for all inspected rows, so the data behaves like index OHLCV data rather than an adjusted ETF total-return series.

Missing-value risks are low at raw CSV level, but feature engineering intentionally creates warmup `NaN` values through rolling windows, lags, ATR, skew, kurtosis, and target shifting. The notebooks later drop rows with missing features and labels.

Validated by code inspection:

- Notebooks read the CSV with `parse_dates=["Date"]`, set `Date` as index, lowercase columns, coerce OHLCV columns to numeric, sort index, and forward-fill.
- Notebooks compute returns from `adjusted_close`.

Requires full TensorFlow execution:

- Whether each notebook successfully reruns from a fresh kernel after feature warmup and row dropping.

Not validated from current repository:

- Provenance of `GSPC_fixed.csv`.
- Whether any trading days are unexpectedly missing relative to an exchange calendar.
- Whether the dataset should be benchmarked as S&P 500 index, SPY ETF, or investable total-return proxy.

## 4. Target and Label Validation

Code-inspection validated:

All five notebooks contain the same target construction:

```python
df["label"] = (df["log_returns"].shift(-1) > 0).astype(int)
```

This is a next-trading-day directional label: the row at date `t` is labeled `1` if the log return at date `t+1` is positive, and `0` otherwise.

Validity assessment:

- The intended supervised learning target is clear and consistent across notebooks.
- Feature rows at date `t` are meant to predict the next return direction after date `t`.
- There is a subtle final-row issue: `NaN > 0` evaluates to `False`, so the last unavailable future return can become label `0` rather than missing. Because `df.dropna(subset=FEATURE_COLS + ["label"])` occurs after this cast, the final row may not be removed by target unavailability alone.

Requires lightweight follow-up execution:

- Recompute labels independently and confirm exact label/date alignment after feature `dropna`.
- Count how many labels fall in train, validation, and OOS windows.

Requires full TensorFlow execution:

- Whether target labels are used identically by every training, validation, prediction, and metric cell during actual notebook execution.

Not validated from current repository:

- Whether next-day direction is the best objective for tradable performance after transaction costs.

## 5. Feature Engineering Validation

Code-inspection validated:

All notebooks use the same feature expression:

```python
FEATURE_COLS = [
    "log_returns", "volatility_10", "RSI_14", "MACD", "OBV",
    "MACD_line", "MACD_signal",
    "ret_3d", "ret_5d", "ret_10d",
    "vol_10d", "vol_21d", "vol_63d", "vol_ratio",
    "ATR_14",
    "ret_skew_21", "ret_kurt_21",
    "vol_zscore_20",
] + [f"lag_{i}" for i in range(1, 8)]
```

Feature categories:

- Rolling return and volatility: `log_returns`, `volatility_10`, `vol_10d`, `vol_21d`, `vol_63d`, `vol_ratio`.
- Momentum/technical indicators: `RSI_14`, `MACD`, `MACD_line`, `MACD_signal`.
- Volume feature: `OBV`, `vol_zscore_20`.
- Lagged features: `lag_1` through `lag_7`.
- Multi-horizon return features: `ret_3d`, `ret_5d`, `ret_10d`.
- Range/volatility feature: `ATR_14`.
- Distribution shape features: `ret_skew_21`, `ret_kurt_21`.

Validity assessment:

- Most inspected features are backward-looking or same-day close-derived.
- No obvious feature uses future rows directly in the inspected code.
- Same-day close-derived features are valid only under a specific timing assumption: the signal for date `t` is generated after close `t` and traded from the next bar. The backtest attempts to enforce this with shifted positions.

Requires lightweight follow-up execution:

- Recompute feature matrix and check finite values after warmup.
- Confirm exact rows removed by `dropna`.
- Confirm no unintended division-by-zero artifacts survive in `vol_ratio`, RSI, ATR, skew, or kurtosis.

Requires full TensorFlow execution:

- Whether all notebooks execute these cells from a fresh kernel without relying on hidden state.

Not validated from current repository:

- Economic robustness of these indicators.
- Whether feature selection was made before or after seeing OOS performance.

## 6. Sequence Alignment Validation

Code-inspection validated:

All five notebooks define `SEQ_LEN = 28` and use the same sequence construction logic:

```python
X_seq = np.stack([Xv[i:i+seq_length] for i in range(len(Xv) - seq_length + 1)], axis=0)
y_seq = np.array([yv[i+seq_length-1] for i in range(len(Xv) - seq_length + 1)])
```

The notebooks also align prediction dates to the last timestamp in each sequence, for example:

```python
pred_index_oos = pd.DatetimeIndex(test_idx[SEQ_LEN-1:])
```

Validity assessment:

- A sequence ending on date `t` receives the label stored at date `t`.
- Since the label at date `t` is next-day direction, the model is intended to use data through date `t` to predict return at `t+1`.
- This is conceptually valid if the backtest only earns returns after date `t`.
- Off-by-one risk remains high enough to require a synthetic alignment test before reusing the logic.

Requires lightweight follow-up execution:

- Synthetic test with known dates, returns, labels, and expected sequence timestamps.
- Verify CV fold boundary behavior when sequences are created separately for train and validation rows.

Requires full TensorFlow execution:

- Whether predictions emitted by trained models preserve expected sequence order and length in every notebook.

Not validated from current repository:

- Actual saved prediction arrays and indexes for all models; most are generated artifacts not tracked in the repository.

## 7. Scaling and Leakage Validation

Code-inspection validated:

All notebooks import and use `StandardScaler`.

CV scaling pattern:

```python
sc = StandardScaler().fit(Xtr_2d)
Xtr = sc.transform(Xtr_2d)
Xva = sc.transform(Xva_2d)
```

OOS scaling pattern:

```python
sc = StandardScaler().fit(Xtrval2)
Xtrval = sc.transform(Xtrval2)
Xte = sc.transform(Xte2)
```

Validity assessment:

- The inspected code fits scalers on training data for CV folds and on pre-2022 train/validation data for OOS.
- That is the correct leakage-avoidance direction.
- The scaler is fit before sequence construction on tabular rows. This is acceptable if only training-period rows are used for fitting.
- No global `fit_transform` on all data was found in the parsed evidence.

Requires lightweight follow-up execution:

- Assert fit rows and transform rows by date for every fold.
- Save scaler date ranges and parameters in a manifest.

Requires full TensorFlow execution:

- Confirm executed notebooks do not skip or overwrite scaler cells through interactive state.

Not validated from current repository:

- Reproducibility of scaler artifacts, because scalers are not tracked outputs.

## 8. Cross-Validation and OOS Design

Code-inspection validated:

All notebooks use:

```python
tscv = TimeSeriesSplit(n_splits=6, test_size=252)
```

OOS logic uses:

```python
H_SPLIT = "2022-01-01"
trainval_idx = df.index[df.index < H_SPLIT]
test_idx = df.index[df.index >= H_SPLIT]
```

Validity assessment:

- The design separates rolling CV diagnostics from a final 2022+ OOS block.
- The 252-bar validation size is consistent with approximately one trading year.
- OOS from 2022 onward is a meaningful stress period, but it is still a single holdout period.
- Threshold tuning may contaminate final OOS conclusions if thresholds, model choices, or risk-overlay settings were selected after looking at OOS results.

Requires lightweight follow-up execution:

- Emit actual fold date ranges after feature `dropna`.
- Confirm validation folds are strictly after training folds.
- Confirm OOS sequence warmup does not accidentally remove or shift the expected first tradable OOS date differently across notebooks.

Requires full TensorFlow execution:

- Reproduce fold-level predictions and OOS predictions.
- Confirm that final OOS metrics come only from the locked final configuration.

Not validated from current repository:

- Whether final model/threshold selection was made before OOS inspection.

## 9. Model Comparability

Code-inspection validated:

All five notebooks share:

- Same data path: `../data/GSPC_fixed.csv`.
- Same feature list.
- Same target expression.
- Same sequence length: 28.
- Same CV split form: `TimeSeriesSplit(n_splits=6, test_size=252)`.
- Same OOS split date: `2022-01-01`.
- Same backtest function name: `run_backtest_clean`.
- Same probability-output trading interface.

Model factory functions:

```text
1_stacked_lstm.ipynb: get_stacked_lstm_model
2_gru.ipynb: get_gru_model
3_cnn_lstm.ipynb: get_cnn_lstm
4_attention_lstm.ipynb: get_stacked_att_lstm
5_transformer.ipynb: get_transformer_model
```

Comparability caveats:

- Threshold constants and bands differ in places. The Transformer starts with `BASE_TH = 0.40` and `BAND = 0.05`; several other notebooks start with `PRED_THRESHOLD = 0.4006` and `THRESH_BAND = 0.0866`.
- Attention-LSTM and Transformer use narrower Stage A sweeps than several other notebooks.
- CNN-LSTM writes to `outputs/04_CNN_LSTM`, while Attention-LSTM writes to `outputs/04_ATT_LSTM_clean`; numbering is inconsistent.
- Hyperparameters and callbacks may differ by notebook.

Requires full TensorFlow execution:

- Actual performance comparability across models.
- Whether random seeds, early stopping, callbacks, and training epochs produce fair model comparisons.

Not validated from current repository:

- Any definitive ranking of model architectures.

## 10. Backtest Validity

Code-inspection validated:

All notebooks define `run_backtest_clean`.

Signal construction:

```python
sig = np.select(
    [out["p"] >= cfg.long_threshold, out["p"] <= cfg.short_threshold],
    [1, -1],
    default=0
)
```

Strategy timing:

```python
out["strat_log_ret_gross"] = out["pos"].shift(1).fillna(0.0) * out[ret_col]
```

Transaction costs and turnover:

```python
pos_change = out["pos"].diff().fillna(0.0)
turnover_events = (pos_change != 0).astype(int)
cost = (cfg.cost_bps_per_turnover / 1e4) * np.abs(pos_change)
out["strat_log_cost"] = np.log1p(-out["cost"].clip(upper=0.999999)).fillna(0.0)
out["strat_log_ret"] = out["strat_log_ret_gross"] + out["strat_log_cost"]
```

Benchmark:

```python
out["bh_log_ret"] = out[ret_col]
out["bh_equity"] = np.exp(out["bh_log_ret"].cumsum())
```

Validity assessment:

- Long, short, and neutral signal logic is present.
- Shifted positions indicate the code tries to avoid same-bar return leakage.
- Transaction costs are applied to absolute exposure changes.
- Long-to-short flips should incur double one-sided cost because exposure changes by 2, which is reasonable if documented.
- Volatility targeting changes exposure continuously; costs therefore apply to volatility-driven exposure changes as well as signal flips.
- Benchmark returns are aligned to `out`, which is filtered to `pred_index`; this supports same-window benchmark alignment in the backtest function.

Requires lightweight follow-up execution:

- Synthetic backtest tests for signal timing, first tradable bar, long-to-short costs, neutral transitions, stop-loss behavior, and volatility-targeted turnover.

Requires full TensorFlow execution:

- Whether OOS prediction indexes exactly match the backtest window in every executed notebook.
- Whether generated QuantStats reports match notebook summaries.

Not validated from current repository:

- Realistic execution quality, spreads, slippage, borrow costs, market impact, or capacity.

## 11. Performance Claim Validation

Claims supported by code inspection:

- The repository contains implementations for Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, and Transformer notebooks.
- The notebooks use a common dataset, feature list, target expression, sequence length, CV pattern, OOS split, and backtest function.
- The backtest code includes long/short/neutral signals, shifted-position return timing, volatility targeting, stops, turnover tracking, transaction costs, and buy-and-hold benchmark returns.

Claims supported by lightweight execution without TensorFlow:

- The CSV is structurally clean under local checks.
- Notebook cell counts, model factory names, feature expressions, target expression, scaler lines, split lines, threshold lines, and backtest lines are present as described.
- `Close` and `Adjusted_close` are identical in the current CSV.

Claims requiring full TensorFlow notebook execution:

- Any reported classification accuracy, ROC AUC, Sharpe, CAGR, drawdown, hit rate, turnover, or equity curve value.
- Whether all five notebooks run from a clean kernel.
- Whether saved predictions, thresholds, and OOS plots are regenerated exactly.
- Whether callback behavior, early stopping, and model checkpoints are reproducible.
- Whether TensorFlow model comparisons are fair under controlled seeds and identical run protocol.

Claims requiring regenerated output comparison:

- README chart accuracy.
- Attention-LSTM and Transformer OOS equity-curve reproducibility.
- QuantStats report consistency.
- Any claim that one architecture outperforms another.

Claims that should be softened or caveated:

- "Realistic backtesting" should be caveated as transaction-cost-aware research backtesting, not execution-realistic trading simulation.
- "SPY benchmark" should be caveated because the local data is S&P 500 index OHLCV, not SPY ETF data.
- "Validated" should be limited to rolling CV/OOS workflow design unless full reproduction artifacts are generated.
- Any performance ranking should be described as exploratory until reproduced under a locked protocol.

## 12. Critical Issues

- Final target row may become label `0` because `NaN > 0` evaluates false after `shift(-1)`. This can pollute the last available row unless explicitly removed.
- Performance claims are not reproduced in this review. No Sharpe, CAGR, drawdown, or accuracy claim should be treated as validated.
- OOS conclusions may be contaminated if thresholds, risk settings, or model choices were selected after viewing OOS results.
- Sequence/backtest timing is plausible but not proven. A synthetic off-by-one test is required before reusing the logic.
- Benchmark naming is ambiguous: README/notebooks mention SPY, while the dataset is S&P 500 index data.

## 13. Moderate Issues

- Feature and backtest logic is duplicated across notebooks, increasing drift risk.
- Threshold constants and tuning ranges are not fully uniform across models.
- Turnover is reported as event count, while costs are based on absolute exposure change; both are useful but should be named separately.
- Volatility targeting can create cost from exposure changes even without signal changes; this should be documented and tested.
- Output directory numbering is inconsistent around CNN-LSTM and Attention-LSTM.
- Environment files disagree: `requirements.txt` and `envs/env_models_tf2.yml` pin different TensorFlow/Keras/pandas/scikit-learn versions.
- Current outputs are insufficient to audit all model runs.

## 14. Safe-to-Preserve Components

- Conceptual five-model comparison: Stacked LSTM, GRU, CNN-LSTM, Attention-LSTM, Transformer.
- Single probability-output interface for trading signals.
- Current CSV as a parity dataset, with a recorded hash.
- Feature set as `feature_set_v1`, after extracting into tested code.
- Next-day direction target as a parity objective, after fixing final-row handling.
- 28-day sequence windows, after synthetic alignment tests.
- Per-fold and pre-OOS `StandardScaler` fitting concept.
- `TimeSeriesSplit(n_splits=6, test_size=252)` as a baseline CV design.
- 2022+ OOS split as a first historical holdout.
- Shifted-position principle for trading after signal formation.
- Exposure-change transaction-cost accounting.
- Same-window buy-and-hold benchmark logic.

## 15. Do-Not-Reuse Components

- Notebook cells as production implementation.
- Hidden notebook state or manually generated local artifacts.
- Any threshold chosen from OOS inspection.
- Current performance claims without regenerated evidence.
- SPY wording unless actual SPY total-return data is introduced.
- Untracked scaler, prediction, threshold, and run metadata.
- Duplicated backtest and feature code.
- Ambiguous turnover naming.
- Output directory naming conventions.
- Mixed TF2 environment pins as the basis for the PyTorch environment.

## 16. Required Reproduction Checks Before Public Claims

Minimum lightweight checks before PyTorch migration:

- Data audit: columns, row count, date range, duplicate dates, blank cells, numeric coercion, content hash.
- Feature audit: finite values after warmup, expected dropped rows, no future-looking rolling features.
- Target audit: next-day label alignment and explicit removal of final unavailable target.
- Sequence audit: synthetic date/return test proving window, label, and prediction timestamp alignment.
- Scaling audit: fold and OOS scaler fit windows exclude validation/OOS rows.
- Backtest audit: synthetic tests for shifted signal timing, costs, turnover, neutral states, stops, and benchmark alignment.

Minimum full TensorFlow checks before public performance claims:

- Run each notebook from a clean kernel in a pinned environment.
- Save run manifests containing commit SHA, data hash, package versions, seed, split dates, thresholds, model hyperparameters, scaler policy, and backtest config.
- Regenerate CV predictions, OOS predictions, threshold files, metrics, QuantStats reports, and equity curves.
- Compare regenerated README charts against tracked output charts.
- Lock thresholds before OOS evaluation.
- Produce a model comparison table only from reruns under the same data, split, feature, threshold-selection, and backtest protocol.

## 17. Recommendation

Classification: publish as historical educational artifact, not as a fully validated research artifact.

The repository has a coherent methodological base that is worth preserving conceptually for a PyTorch rewrite: common feature set, next-day target, sequence modeling, fold-level scaling, time-series CV, OOS split, probability thresholds, shifted backtest timing, costs, and benchmark alignment. However, performance validity has not been reproduced, threshold provenance is not fully auditable from tracked files, and key alignment assumptions still require synthetic tests.

Recommendation: keep the current TF2 repository as a local historical reference and educational artifact. Do not present it as validated research until clean-kernel notebook reruns and regenerated artifacts support the claims. For the PyTorch rewrite, reuse the concepts, not the notebook implementation.
