"""Generate Homework_09_Fashion_MNIST_CNN.ipynb"""
import json
from pathlib import Path
from textwrap import dedent


def md(s):
    lines = dedent(s).strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]}


def code(s):
    lines = dedent(s).strip("\n").split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [ln + "\n" for ln in lines],
    }


cells = []

cells.append(md("""
# Homework 9: CNN for Fashion MNIST

**Advanced Machine Learning**

Classify **Fashion MNIST** images (10 apparel categories, 28×28 grayscale) with Keras CNNs.

**Assignment checklist**

| Requirement | Section |
|-------------|---------|
| Load Fashion MNIST via Keras | §1 |
| Preprocessing for CNNs | §1 |
| Baseline CNN | §2 |
| ≥3 different configurations | §3a–3c |
| Change one factor at a time | §3 |
| Systematic tuning (Optuna) | §4 |
| Final model + held-out test | §5 |
| Metrics + train/val curves in conclusions | §2–§6 |
| Personal workflow | §7 |
"""))

cells.append(code("""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from sklearn.metrics import classification_report, confusion_matrix

tf.random.set_seed(42)
np.random.seed(42)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
NUM_CLASSES = 10
"""))

cells.append(md("## 1. Load and prepare data"))

cells.append(code("""
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

# Scale to [0, 1] and add channel dimension for Conv2D
X_train = (X_train.astype("float32") / 255.0)[..., np.newaxis]
X_test = (X_test.astype("float32") / 255.0)[..., np.newaxis]

print("Value range:", X_train.min(), "to", X_train.max())
print("Image shape:", X_train.shape[1:])
"""))

cells.append(code("""
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i].squeeze(), cmap="gray")
    ax.set_title(CLASS_NAMES[y_train[i]], fontsize=8)
    ax.axis("off")
plt.suptitle("Sample training images")
plt.tight_layout()
plt.show()
"""))

cells.append(md("""
## 2. Baseline CNN

Two convolution blocks (32 → 64 filters), max pooling, dense head — same style as Week 8 `Inclass_05_21` cats/dogs CNN.
"""))

cells.append(code("""
def build_baseline_cnn():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])


keras.backend.clear_session()
tf.random.set_seed(42)

baseline_model = build_baseline_cnn()
baseline_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

baseline_history = baseline_model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,
    verbose=1,
)
"""))

cells.append(code("""
def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title(f"{title} — loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


plot_history(baseline_history, "Baseline CNN")
"""))

cells.append(code("""
baseline_test_loss, baseline_test_acc = baseline_model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline test accuracy: {baseline_test_acc:.4f}")
print(f"Baseline test loss: {baseline_test_loss:.4f}")

y_pred_base = np.argmax(baseline_model.predict(X_test, verbose=0), axis=1)
print("\\nClassification report (baseline):")
print(classification_report(y_test, y_pred_base, target_names=CLASS_NAMES))
"""))

cells.append(md("""
## 3. Experiments — change one factor at a time

Fixed unless noted: Adam, `validation_split=0.2`, **15 epochs**, seed 42, same train/test split. Test accuracy is on the **held-out test set** (10,000 images).
"""))

cells.append(md("### 3a. Factor: architecture (filter widths)"))

cells.append(code("""
def build_cnn(filters=(32, 64), dense_units=64, dropout=0.0):
    reg_layers = []
    for f in filters:
        reg_layers += [
            layers.Conv2D(f, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
        ]
        if dropout:
            reg_layers.append(layers.Dropout(dropout))
    return keras.Sequential(
        [layers.Input(shape=(28, 28, 1))]
        + reg_layers
        + [
            layers.Flatten(),
            layers.Dense(dense_units, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )


def run_experiment(name, **build_kw):
    keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_cnn(**build_kw)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        X_train, y_train, epochs=15, batch_size=128,
        validation_split=0.2, verbose=0,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    last = len(hist.history["loss"]) - 1
    return {
        "name": name,
        "test_acc": acc,
        "test_loss": loss,
        "train_acc": hist.history["accuracy"][last],
        "val_acc": hist.history["val_accuracy"][last],
        "history": hist,
    }


arch_results = []
arch_results.append(run_experiment("baseline [32,64]", filters=(32, 64)))
arch_results.append(run_experiment("wider [64,128]", filters=(64, 128)))
arch_results.append(run_experiment("narrow [16,32]", filters=(16, 32)))

pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in arch_results])
"""))

cells.append(md("**Takeaway:** Wider filters increase capacity; compare test accuracy above — wider nets can help if not overfitting."))

cells.append(md("### 3b. Factor: dropout (architecture fixed at [32, 64])"))

cells.append(code("""
drop_results = []
for dr in [0.0, 0.25, 0.5]:
    drop_results.append(run_experiment(f"dropout {dr}", filters=(32, 64), dropout=dr))

pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in drop_results])
"""))

cells.append(md("**Takeaway:** Moderate dropout can reduce overfitting when train accuracy runs ahead of validation."))

cells.append(md("### 3c. Factor: batch size (baseline architecture, no dropout)"))

cells.append(code("""
batch_results = []

for bs in [32, 128, 256]:
    keras.backend.clear_session()
    tf.random.set_seed(42)
    model = build_cnn(filters=(32, 64))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    hist = model.fit(
        X_train, y_train, epochs=15, batch_size=bs,
        validation_split=0.2, verbose=0,
    )
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    last = len(hist.history["loss"]) - 1
    batch_results.append({
        "name": f"batch {bs}",
        "batch_size": bs,
        "test_acc": acc,
        "train_acc": hist.history["accuracy"][last],
        "val_acc": hist.history["val_accuracy"][last],
    })

pd.DataFrame(batch_results)
"""))

cells.append(md("**Takeaway:** Batch size affects noise in gradient updates; very large batches can generalize slightly differently on small images."))

cells.append(md("""
## 4. Tuning with Optuna

Search over conv filter sizes, dense units, learning rate, batch size, and dropout. Objective = best **validation loss** with early stopping (pattern from `Inclass_05_14`).
"""))

cells.append(code("""
def objective(trial):
    tf.random.set_seed(42)
    f1 = trial.suggest_categorical("filters_1", [32, 64, 128])
    f2 = trial.suggest_categorical("filters_2", [64, 128, 256])
    dense_u = trial.suggest_int("dense_units", 32, 128, step=32)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    model = build_cnn(filters=(f1, f2), dense_units=dense_u, dropout=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True,
    )
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop],
    )
    return min(history.history["val_loss"])


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=25)

print("Best validation loss:", study.best_value)
print("Best params:", study.best_params)
"""))

cells.append(code("""
plot_optimization_history(study)
plot_param_importances(study)
plot_slice(study)
"""))

cells.append(md("## 5. Final model and held-out test evaluation"))

cells.append(code("""
bp = study.best_params
keras.backend.clear_session()
tf.random.set_seed(42)

best_model = build_cnn(
    filters=(bp["filters_1"], bp["filters_2"]),
    dense_units=bp["dense_units"],
    dropout=bp["dropout"],
)
best_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=bp["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True,
)

history_final = best_model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=bp["batch_size"],
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop],
)

plot_history(history_final, "Optuna final CNN")
"""))

cells.append(code("""
final_test_loss, final_test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Final model test accuracy: {final_test_acc:.4f}")
print(f"Final model test loss: {final_test_loss:.4f}")

y_pred_final = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
print("\\nClassification report (final):")
print(classification_report(y_test, y_pred_final, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
plt.yticks(range(10), CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix — final model")
plt.tight_layout()
plt.show()
"""))

cells.append(code("""
# Summary table
summary = pd.DataFrame([
    {"Model": "Baseline CNN", "test_acc": baseline_test_acc},
    {"Model": "Best arch (manual)", "test_acc": max(r["test_acc"] for r in arch_results)},
    {"Model": "Best dropout (manual)", "test_acc": max(r["test_acc"] for r in drop_results)},
    {"Model": "Optuna final", "test_acc": final_test_acc},
])
summary
"""))

cells.append(md("""
## 6. Conclusions

*Update the numbers below from your run outputs if they differ slightly.*

**Data:** 60,000 training / 10,000 test images; pixel values scaled to [0, 1]; shape `(28, 28, 1)` for Conv2D.

**Baseline:** Two-block CNN; check train vs val curves for overfitting (train accuracy rising faster than val).

**One-factor experiments:**
- **Architecture:** Compare `arch_results` — wider vs narrow filter stacks.
- **Dropout:** Compare `drop_results` — whether val/test improve when train–val gap is large.
- **Batch size:** Compare `batch_results` — stability vs speed tradeoff.

**Optuna:** Best validation loss and params printed in §4; final model retrained with early stopping and evaluated once on **X_test**.

**Overfitting:** If training accuracy ≈ 0.95+ while validation stalls lower, use dropout, fewer filters, or early stopping (as in the final model).

**Class confusion:** Shirt vs T-shirt/top and Coat vs Pullover are commonly confused — see confusion matrix.

**Best model:** The Optuna-tuned CNN should match or beat the baseline on **test accuracy**; cite your `summary` table.
"""))

cells.append(md("""
## 7. Personal workflow (future reference)

1. **Load & inspect** — `fashion_mnist.load_data()`, plot samples, confirm shape and class balance.
2. **Preprocess for CNNs** — scale to [0, 1], add channel axis; keep official test set untouched.
3. **Baseline CNN** — small Conv2D stack + pool + dense; plot train/val loss and accuracy.
4. **Diagnose** — train–val gap → overfitting; both low → underfitting.
5. **One factor at a time** — architecture, then regularization, then batch size/learning rate; fix seed and epochs.
6. **Optuna** — tune on validation loss with early stopping; retrain best config once.
7. **Test once** — classification report + confusion matrix on held-out test.
8. **Pitfalls** — evaluating on test during tuning; forgetting `[..., np.newaxis]`; too many epochs without early stopping on small images.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.11 (Week 9 TensorFlow)",
            "language": "python",
            "name": "week9-tensorflow",
        },
        "language_info": {"name": "python", "version": "3.11.15"},
    },
    "cells": cells,
}

out = Path(__file__).parent / "Homework_09_Fashion_MNIST_CNN.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("Wrote", out, "cells:", len(cells))
