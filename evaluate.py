"""
Evaluation & Visualization
Loads both saved models, runs predictions on test images,
prints classification reports, and generates all required plots.

Saved plots go into ../results/
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
from sklearn import metrics

from model import CHECKPOINT_EXTRACT, CHECKPOINT_FINETUNE

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
class2num = lambda lst: [0 if x == 'O' else 1 for x in lst]
num2class = lambda lst: ['O' if x < 0.5 else 'R' for x in lst]


def plot_curves(history, metric, title, ylabel, save_name):
    """Plot training and validation curves and save to results/."""
    plt.figure(figsize=(5, 5))
    plt.plot(history.history[metric],        label=f'Training {ylabel}')
    plt.plot(history.history[f'val_{metric}'], label=f'Validation {ylabel}')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    out = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  💾 Saved → {out}")


def plot_image_with_title(image, model_name, actual_label, predicted_label, save_name=None):
    plt.figure()
    plt.imshow(image)
    plt.title(f"Model: {model_name}\nActual: {actual_label} | Predicted: {predicted_label}")
    plt.axis('off')
    if save_name:
        out = os.path.join(RESULTS_DIR, save_name)
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  💾 Saved → {out}")
    plt.show()


# ── Load test images ──────────────────────────────────────────────────────────
def load_test_images(img_dim=(150, 150)):
    test_files_O = glob.glob('./o-vs-r-split/test/O/*')
    test_files_R = glob.glob('./o-vs-r-split/test/R/*')
    test_files = test_files_O[:50] + test_files_R[:50]

    test_imgs = [
        tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(img, target_size=img_dim)
        )
        for img in test_files
    ]
    test_imgs = np.array(test_imgs)
    test_labels = [Path(fn).parent.name for fn in test_files]

    test_imgs_scaled = test_imgs.astype('float32') / 255.0
    return test_imgs, test_imgs_scaled, test_labels


# ── Main evaluation ───────────────────────────────────────────────────────────
def evaluate(extract_feat_history=None, fine_tune_history=None):
    """
    Parameters
    ----------
    extract_feat_history : keras History object (from train.py), optional.
        Pass when running evaluate directly after training in the same session.
    fine_tune_history    : keras History object, optional.
    """

    # ── Plot curves (Tasks 6, 7, 8) ──────────────────────────────────────────
    if extract_feat_history:
        # Task 6 — accuracy curves for extract-feat model
        plot_curves(
            extract_feat_history, 'accuracy',
            title='Accuracy Curve',
            ylabel='Accuracy',
            save_name='plot_accuracy_curve.png',
        )

    if fine_tune_history:
        # Task 7 — loss curves for fine-tune model
        plot_curves(
            fine_tune_history, 'loss',
            title='Loss Curve',
            ylabel='Loss',
            save_name='plot_loss_curve.png',
        )
        # Task 8 — accuracy curves for fine-tune model
        plot_curves(
            fine_tune_history, 'accuracy',
            title='Accuracy Curve',
            ylabel='Accuracy',
            save_name='plot_finetune_model.png',
        )

    # ── Load saved models ─────────────────────────────────────────────────────
    print("\n📦 Loading saved models...")
    extract_feat_model = tf.keras.models.load_model(CHECKPOINT_EXTRACT)
    fine_tune_model    = tf.keras.models.load_model(CHECKPOINT_FINETUNE)

    # ── Load & scale test images ──────────────────────────────────────────────
    test_imgs, test_imgs_scaled, test_labels = load_test_images()

    # ── Predictions ───────────────────────────────────────────────────────────
    preds_efm = num2class(extract_feat_model.predict(test_imgs_scaled, verbose=0))
    preds_ftm = num2class(fine_tune_model.predict(test_imgs_scaled, verbose=0))

    # ── Classification reports ────────────────────────────────────────────────
    print("\n── Extract Features Model ──────────────────────────────────────")
    print(metrics.classification_report(test_labels, preds_efm))
    print("── Fine-Tuned Model ────────────────────────────────────────────")
    print(metrics.classification_report(test_labels, preds_ftm))

    # ── Task 9 — test image with Extract Features Model (index = 1) ──────────
    index_to_plot = 1
    plot_image_with_title(
        image=test_imgs[index_to_plot].astype('uint8'),
        model_name='Extract Features Model',
        actual_label=test_labels[index_to_plot],
        predicted_label=preds_efm[index_to_plot],
        save_name='extract_features_model.png',
    )

    # ── Task 10 — test image with Fine-Tuned Model (index = 1) ───────────────
    plot_image_with_title(
        image=test_imgs[index_to_plot].astype('uint8'),
        model_name='Fine-Tuned Model',
        actual_label=test_labels[index_to_plot],
        predicted_label=preds_ftm[index_to_plot],
        save_name='finetuned_model.png',
    )

    print("\n✅ Evaluation complete. Check the results/ folder for all plots.")


if __name__ == "__main__":
    evaluate()
