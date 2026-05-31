# Week 9

## Homework — Fashion MNIST CNN

| File | Purpose |
|------|---------|
| `Homework_09_Fashion_MNIST_CNN.ipynb` | CNN homework (baseline → experiments → Optuna → test) |

Kernel: **Python 3.11 (Week 9 TensorFlow)** (`week 7/.venv`). Run **Run All** from the top (Optuna ~25 trials may take a while on CPU).

---

## In-class — RNN with Keras (Housing Data)

## Files

| File | Purpose |
|------|---------|
| `Inclass_05_26.ipynb` | In-class RNN forecasting (SimpleRNN, MLP, 1D CNN) |
| `ma_lga_12345.csv` | Quarterly housing moving-average prices by subgroup |

## Setup

1. Open `Inclass_05_26.ipynb` in this folder.
2. Select kernel **Python 3.11 (Week 9 TensorFlow)** (same env as Week 7/8: `week 7/.venv`).
3. **Run All** from the top.

The imports cell sets the working directory to `Week 9` and checks that `ma_lga_12345.csv` exists.

## Activity outline

- **Early cells:** Load data, build lookback windows, initial SimpleRNN.
- **Middle:** Market-level MA (average across subgroups) — compare **RNN vs MLP vs 1D CNN**.
- **End (“Your Turn”):** Three SimpleRNN variants on **3-bedroom houses** (lookback, capacity, dropout). Fill in the **Model Exploration Summary** markdown from your `exp_comparison` table and plots.

## Re-download data

If the CSV is missing, copy it from the course site or from the repo root `ma_lga_12345.csv`.

---

## In-class — Bitcoin forecasting (RNN + XGBoost)

| File | Purpose |
|------|---------|
| `Inclass_05_28.ipynb` | Bitcoin close forecasting: SimpleRNN, LSTM, GRU, XGBoost |
| `dc.csv` | Daily Bitcoin OHLCV (2021–2024) |

1. Open `Inclass_05_28.ipynb` in this folder.
2. Kernel: **Python 3.11 (Week 9 TensorFlow)** (`week 7/.venv`). XGBoost cells call `ensure_libomp()` (loads `libomp` from Anaconda on macOS if needed).
3. **Run All** from the top (recurrent training may take several minutes on CPU).

The activity section improves each baseline RNN and compares against baseline and return-based XGBoost models.
