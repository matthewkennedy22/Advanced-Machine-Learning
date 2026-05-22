# Week 8 — CNNs (cats & dogs)

## Notebook

- `Inclass_05_21.ipynb` — MLP vs CNN on cat/dog images

## Data layout

The notebook expects:

```
Week 8/cats_dogs/
  cats_set/   # .jpg images
  dogs_set/   # .jpg images
```

You already have **500 images per class** in that folder. To rebuild or refresh:

```bash
cd "Week 8"
../week\ 7/.venv/bin/python setup_cats_dogs.py
```

## Python environment

Use the **Week 7 TensorFlow** venv (TensorFlow + `tensorflow-datasets`):

- Interpreter: `week 7/.venv/bin/python`
- Jupyter kernel: **Python 3.11 (Week 8 TensorFlow)** (same env as Week 7)

In VS Code/Cursor, open `Week 8/Inclass_05_21.ipynb` and select that kernel before **Run All**.
