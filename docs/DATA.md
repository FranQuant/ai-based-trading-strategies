# Data Notes

## Data File

The notebooks use:

```text
data/GSPC_fixed.csv
```

This file contains daily S&P 500 index OHLCV data. It is treated as the single source dataset for the current notebook experiments.

## Schema

Columns observed in the CSV:

| Column | Description |
| --- | --- |
| `Date` | Trading date |
| `Open` | Daily opening index level |
| `High` | Daily high index level |
| `Low` | Daily low index level |
| `Close` | Daily closing index level |
| `Adjusted_close` | Adjusted close used by notebooks for returns |
| `Volume` | Reported daily volume |

The file currently has 2,809 data rows plus a header row.

## Date Range

Observed range:

- Start: `2014-01-06`
- End: `2025-03-07`

The notebooks split the final out-of-sample window at `2022-01-01`, after feature creation and sequence construction.

## Notebook Feature Usage

The notebooks compute features from the raw OHLCV data, including log returns, volatility, RSI, MACD, OBV, ATR, volume z-score, lagged returns, multi-horizon returns, and rolling return shape statistics.

The target is a next-trading-day binary direction label based on shifted log returns.

## Benchmark Interpretation

The notebooks and README sometimes refer to a SPY or buy-and-hold benchmark. The local data file is S&P 500 index data, not SPY ETF total-return data.

Interpret benchmark results as an index-level buy-and-hold comparison over the same prediction/backtest window unless a future data update explicitly introduces an investable ETF or total-return benchmark.

## Known Limitations

- The data provenance and download process are not encoded in the repository.
- The dataset is single-asset and index-level, so it does not include ETF fees, dividends, borrow costs, spreads, or market impact.
- `Close` and `Adjusted_close` should be checked before relying on adjusted-return assumptions.
- Feature generation creates rolling-window and lag warmup rows that are later dropped by notebooks.
- There is no versioned data manifest recording file hash, vendor, timezone, market calendar, or missing-session policy.
