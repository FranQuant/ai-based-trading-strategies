# Validation Notes

## Target Construction Risk

The notebooks construct a binary label from next-day log return direction:

```python
df["label"] = (df["log_returns"].shift(-1) > 0).astype(int)
```

This means features through date `t` are intended to predict whether the return at `t+1` is positive. The final row is a known risk because `NaN > 0` evaluates to `False` before casting, so the last unavailable target can become label `0` unless explicitly removed.

## Sequence Alignment Risk

The notebooks use `SEQ_LEN = 28` and align each label to the last timestamp in the input window. This is conceptually valid if a sequence ending on date `t` only earns returns after date `t`.

There are no automated tests proving that sequence timestamps, labels, prediction indexes, and backtest rows remain aligned across every notebook and split.

## Scaler Policy

The notebooks use `StandardScaler`. Cross-validation folds fit scalers on training rows, and the out-of-sample stage fits on pre-2022 train/validation rows before transforming the OOS rows.

This is the right policy direction, but scaler parameters, fit date ranges, and feature lists are not saved as first-class artifacts. Interactive notebook execution can also make it hard to prove which scaler was used for a specific result.

## Backtest Timing

The backtest applies shifted positions when computing strategy returns, which indicates the intended timing is: observe data and form a signal, then earn returns on a later bar.

This timing assumption still needs synthetic tests for first tradable date, OOS warmup behavior, signal-to-return alignment, and treatment of same-day close-derived features.

## Transaction Cost Assumptions

The backtest applies transaction costs based on absolute position changes, with a default of 5 basis points per turnover unit. Long-to-short flips can therefore incur double one-sided cost because exposure changes by 2.

The current cost model is useful for research, but it does not include spread, slippage, market impact, borrow fees, shorting constraints, taxes, or capacity limits. Volatility targeting can also create costs from exposure changes even without discrete signal flips.

## Reproducibility Gaps

- No centralized experiment configuration files.
- No saved manifest tying data hash, feature set, split dates, scaler state, thresholds, model parameters, and output artifacts together.
- No automated regression tests for target shifting, sequence alignment, scaler fitting, or backtest accounting.
- Notebook outputs can be overwritten by reruns.
- TensorFlow results can vary across package versions, hardware, and backend settings.
