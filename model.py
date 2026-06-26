"""
Model Architecture
Builds the VGG16-based transfer learning model (feature extraction mode).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, LearningRateScheduler
)

CHECKPOINT_EXTRACT = 'O_R_tlearn_vgg16.keras'
CHECKPOINT_FINETUNE = 'O_R_tlearn_fine_tune_vgg16.keras'


# ── Learning rate schedule ────────────────────────────────────────────────────
def exp_decay(epoch):
    initial_lrate = 1e-4
    k = 0.1
    return initial_lrate * np.exp(-k * epoch)


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(epoch))
        print(f'lr: {exp_decay(len(self.losses)):.6f}')


def build_callbacks(checkpoint_path):
    loss_history = LossHistory()
    lrate = LearningRateScheduler(exp_decay)
    keras_callbacks = [
        EarlyStopping(monitor='val_loss', patience=4, mode='min', min_delta=0.01),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, mode='min'),
    ]
    return [loss_history, lrate] + keras_callbacks


# ── Feature Extraction Model ──────────────────────────────────────────────────
def build_extract_feat_model(input_shape=(150, 150, 3)):
    """
    Loads VGG16 with frozen weights and attaches a custom classifier head.
    All VGG16 layers are frozen — only the new Dense layers will be trained.
    """
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    output = vgg.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    basemodel = Model(vgg.input, output)

    for layer in basemodel.layers:
        layer.trainable = False

    model = Sequential([
        basemodel,
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])

    # Task 4 — print model summary
    model.summary()

    # Task 5 — compile
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=['accuracy'],
    )

    return model, basemodel


# ── Fine-Tuning Model ─────────────────────────────────────────────────────────
def build_fine_tune_model(input_shape=(150, 150, 3)):
    """
    Loads VGG16 and unfreezes the last conv block (block5_conv3 onward)
    to allow fine-tuning on the waste dataset.
    """
    vgg = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    output = vgg.layers[-1].output
    output = tf.keras.layers.Flatten()(output)
    basemodel = Model(vgg.input, output)

    # Freeze all layers first
    for layer in basemodel.layers:
        layer.trainable = False

    # Unfreeze from block5_conv3 onward
    set_trainable = False
    for layer in basemodel.layers:
        if layer.name == 'block5_conv3':
            set_trainable = True
        if set_trainable:
            layer.trainable = True

    for layer in basemodel.layers:
        print(f"  {layer.name}: trainable={layer.trainable}")

    model = Sequential([
        basemodel,
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        metrics=['accuracy'],
    )

    return model


if __name__ == "__main__":
    model, _ = build_extract_feat_model()
    print("✅ Extract-features model built successfully.")
