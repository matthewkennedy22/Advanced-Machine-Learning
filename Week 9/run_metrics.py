"""Compute notebook metrics (same logic as Inclass_05_26.ipynb). Writes metrics_cache.json."""
import os
import warnings

warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, SimpleRNN
from tensorflow.keras.models import Sequential

os.chdir(Path(__file__).resolve().parent)
EPOCHS = 100
tf.random.set_seed(42)


def create_sequences(dataset, look_back=2):
    x, y = [], []
    for i in range(len(dataset) - look_back):
        x.append(dataset[i : i + look_back, :])
        y.append(dataset[i + look_back, 0])
    return np.array(x), np.array(y)


def evaluate_metrics(model, test_x, test_y, scaler):
    pred = model.predict(test_x, verbose=0)
    pred_inv = scaler.inverse_transform(pred)
    y_inv = scaler.inverse_transform(test_y.reshape(-1, 1))
    rmse = float(np.sqrt(mean_squared_error(y_inv, pred_inv)))
    mae = float(mean_absolute_error(y_inv, pred_inv))
    return rmse, mae


def fit_model(model, train_x, train_y, test_x, test_y, scaler):
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_absolute_error"])
    hist = model.fit(
        train_x,
        train_y,
        epochs=EPOCHS,
        batch_size=8,
        validation_split=0.2,
        shuffle=False,
        verbose=0,
    )
    rmse, mae = evaluate_metrics(model, test_x, test_y, scaler)
    i = len(hist.history["loss"]) - 1
    return {
        "rmse": rmse,
        "mae": mae,
        "train_loss": hist.history["loss"][i],
        "val_loss": hist.history["val_loss"][i],
        "train_mae": hist.history["mean_absolute_error"][i],
        "val_mae": hist.history["val_mean_absolute_error"][i],
    }


def main():
    house = pd.read_csv("ma_lga_12345.csv")
    out = {}

    seq = house.copy().sort_values("saledate").reset_index(drop=True)
    sc = MinMaxScaler(feature_range=(0, 1))
    scaled = sc.fit_transform(seq[["MA"]])
    lb = 2
    ts = int(len(scaled) * 0.8)
    train_x, train_y = create_sequences(scaled[:ts], lb)
    test_x, test_y = create_sequences(scaled[ts:], lb)
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    m = Sequential([SimpleRNN(16, activation="tanh", input_shape=(lb, 1)), Dense(1)])
    out["Initial RNN"] = fit_model(m, train_x, train_y, test_x, test_y, sc)

    house["saledate"] = pd.to_datetime(house["saledate"], dayfirst=True)
    market = house.groupby("saledate")["MA"].mean().reset_index()
    sc_m = MinMaxScaler(feature_range=(0, 1))
    scaled_m = sc_m.fit_transform(market[["MA"]])
    ts = int(len(scaled_m) * 0.8)
    train_x, train_y = create_sequences(scaled_m[:ts], lb)
    test_x, test_y = create_sequences(scaled_m[ts:], lb)
    train_mlp = train_x.reshape(train_x.shape[0], -1)
    test_mlp = test_x.reshape(test_x.shape[0], -1)

    for name, model, tx, ex in [
        (
            "Market RNN",
            Sequential([SimpleRNN(16, activation="tanh", input_shape=(lb, 1)), Dense(1)]),
            train_x,
            test_x,
        ),
        (
            "Market MLP",
            Sequential(
                [
                    Dense(32, activation="relu", input_shape=(train_mlp.shape[1],)),
                    Dense(16, activation="relu"),
                    Dense(1),
                ]
            ),
            train_mlp,
            test_mlp,
        ),
        (
            "Market 1D CNN",
            Sequential(
                [Conv1D(16, 2, activation="relu", input_shape=(lb, 1)), Flatten(), Dense(1)]
            ),
            train_x,
            test_x,
        ),
    ]:
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        out[name] = fit_model(model, tx, train_y, ex, test_y, sc_m)

    sub = house[(house["type"] == "house") & (house["bedrooms"] == 3)].sort_values("saledate")

    def exp_rnn(lb, units, drop=0.0):
        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        sc = MinMaxScaler(feature_range=(0, 1))
        s = sc.fit_transform(sub[["MA"]])
        ts = int(len(s) * 0.8)
        tr_x, tr_y = create_sequences(s[:ts], lb)
        te_x, te_y = create_sequences(s[ts:], lb)
        layers = [SimpleRNN(units, activation="tanh", input_shape=(lb, 1))]
        if drop:
            layers.append(Dropout(drop))
        layers.append(Dense(1))
        return fit_model(Sequential(layers), tr_x, tr_y, te_x, te_y, sc)

    out["experiments"] = {
        "subgroup": "3-bedroom houses",
        "rows": int(len(sub)),
        "RNN_lb2_u16": {**exp_rnn(2, 16), "look_back": 2, "units": 16, "dropout": 0},
        "RNN_lb4_u32": {**exp_rnn(4, 32), "look_back": 4, "units": 32, "dropout": 0},
        "RNN_lb2_u32_drop": {**exp_rnn(2, 32, 0.2), "look_back": 2, "units": 32, "dropout": 0.2},
    }

    Path("metrics_cache.json").write_text(json.dumps(out, indent=2))
    print("Wrote metrics_cache.json")


if __name__ == "__main__":
    main()
