"""
Setup: Install and import all required libraries for the waste classification project.
Run this script first before anything else.
"""

import subprocess
import sys

def install_dependencies():
    packages = [
        "tensorflow==2.17.0",
        "scikit-learn==1.5.1",
        "matplotlib==3.9.2",
        "numpy<2.0",
        "tqdm",
        "requests",
    ]
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
    print("✅ All dependencies installed.")

if __name__ == "__main__":
    install_dependencies()

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import glob
import warnings
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from matplotlib.image import imread
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics

print(f"✅ Libraries imported successfully.")

# Task 1 — print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
