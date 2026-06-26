"""
Training
Trains the feature-extraction model and the fine-tuned model,
then saves both to disk.
"""

from data_loader import download_dataset, create_generators, BATCH_SIZE
from model import (
    build_extract_feat_model,
    build_fine_tune_model,
    build_callbacks,
    CHECKPOINT_EXTRACT,
    CHECKPOINT_FINETUNE,
)


def train_extract_feat_model(train_gen, val_gen):
    print("\n" + "=" * 60)
    print("PHASE 1: Feature Extraction (all VGG16 layers frozen)")
    print("=" * 60)

    model, _ = build_extract_feat_model()
    callbacks = build_callbacks(CHECKPOINT_EXTRACT)

    history = model.fit(
        train_gen,
        steps_per_epoch=5,
        epochs=10,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        verbose=1,
    )
    print(f"✅ Best model saved to {CHECKPOINT_EXTRACT}")
    return history


def train_fine_tune_model(train_gen, val_gen):
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-Tuning (block5_conv3 onward unfrozen)")
    print("=" * 60)

    model = build_fine_tune_model()
    callbacks = build_callbacks(CHECKPOINT_FINETUNE)

    history = model.fit(
        train_gen,
        steps_per_epoch=5,
        epochs=10,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        verbose=1,
    )
    print(f"✅ Best model saved to {CHECKPOINT_FINETUNE}")
    return history


if __name__ == "__main__":
    download_dataset()
    train_gen, val_gen, test_gen = create_generators()

    extract_feat_history = train_extract_feat_model(train_gen, val_gen)
    fine_tune_history    = train_fine_tune_model(train_gen, val_gen)

    print("\n✅ Training complete. Run evaluate.py to see test results.")
