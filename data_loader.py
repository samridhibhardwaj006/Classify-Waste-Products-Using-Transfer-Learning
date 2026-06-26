"""
Data Loading & Preprocessing
Downloads the waste classification dataset and creates ImageDataGenerators
for train, validation, and test splits.
"""

import os
import requests
import zipfile
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Config ────────────────────────────────────────────────────────────────────
IMG_ROWS, IMG_COLS = 150, 150
BATCH_SIZE = 32
N_EPOCHS = 10
N_CLASSES = 2
VAL_SPLIT = 0.2
PATH_TRAIN = 'o-vs-r-split/train/'
PATH_TEST  = 'o-vs-r-split/test/'
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)
LABELS = ['O', 'R']
SEED = 42


def download_dataset():
    """Download and extract the waste classification dataset."""
    url = (
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
        "kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
    )
    file_name = "o-vs-r-split-reduced-1200.zip"

    if os.path.exists('o-vs-r-split'):
        print("✅ Dataset already present, skipping download.")
        return

    print("⬇️  Downloading dataset...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    print("📦 Extracting dataset...")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        members = zip_ref.infolist()
        with tqdm(total=len(members), unit='file') as pbar:
            for member in members:
                zip_ref.extract(member)
                pbar.update(1)

    os.remove(file_name)
    print("✅ Dataset ready.")


def create_generators():
    """
    Build and return train, validation, and test ImageDataGenerators.

    Returns
    -------
    train_generator, val_generator, test_generator
    """
    train_datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        rescale=1.0 / 255.0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(
        validation_split=VAL_SPLIT,
        rescale=1.0 / 255.0,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=PATH_TRAIN,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        target_size=(IMG_ROWS, IMG_COLS),
        subset='training',
    )

    val_generator = val_datagen.flow_from_directory(
        directory=PATH_TRAIN,
        seed=SEED,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        target_size=(IMG_ROWS, IMG_COLS),
        subset='validation',
    )

    # Task 2 — test generator
    test_generator = test_datagen.flow_from_directory(
        directory=PATH_TEST,
        class_mode='binary',
        seed=SEED,
        batch_size=BATCH_SIZE,
        shuffle=False,
        target_size=(IMG_ROWS, IMG_COLS),
    )

    # Task 3 — print train generator length
    print(f"Train generator length (steps per epoch): {len(train_generator)}")

    return train_generator, val_generator, test_generator


if __name__ == "__main__":
    download_dataset()
    train_gen, val_gen, test_gen = create_generators()
    print(test_gen)
