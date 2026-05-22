"""Download cats vs dogs images into cats_dogs/cats_set and cats_dogs/dogs_set.

Uses TensorFlow Datasets (Microsoft ASIRRA subset). Run from Week 8:

    ../week\\ 7/.venv/bin/python setup_cats_dogs.py
"""
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds

DATA_ROOT = Path(__file__).resolve().parent / "cats_dogs"
CATS_DIR = DATA_ROOT / "cats_set"
DOGS_DIR = DATA_ROOT / "dogs_set"


def main() -> None:
    if CATS_DIR.exists() and DOGS_DIR.exists():
        n_cats = len(list(CATS_DIR.glob("*")))
        n_dogs = len(list(DOGS_DIR.glob("*")))
        if n_cats > 0 and n_dogs > 0:
            print(f"Already set up: {n_cats} cats, {n_dogs} dogs at {DATA_ROOT}")
            return

    print("Downloading cats_vs_dogs via TFDS (first run can take several minutes)...")
    builder = tfds.builder("cats_vs_dogs")
    builder.download_and_prepare()

    CATS_DIR.mkdir(parents=True, exist_ok=True)
    DOGS_DIR.mkdir(parents=True, exist_ok=True)
    label_dirs = {0: CATS_DIR, 1: DOGS_DIR}
    counts = {0: 0, 1: 0}

    ds = builder.as_dataset(split="train", shuffle_files=False)
    for i, ex in enumerate(ds):
        label = int(ex["label"].numpy())
        img = ex["image"].numpy()
        fname = ex["image/filename"].numpy().decode("utf-8")
        out = label_dirs[label] / Path(fname).name
        if not out.suffix:
            out = out.with_suffix(".jpg")
        if out.exists():
            out = label_dirs[label] / f"{i}_{out.name}"
        tf.keras.utils.save_img(str(out), img)
        counts[label] += 1
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1} images saved (cats={counts[0]}, dogs={counts[1]})")

    print(f"Done: {counts[0]} cats, {counts[1]} dogs -> {DATA_ROOT}")


if __name__ == "__main__":
    main()
