"""
train_cnn.py
============
MobileNetV2 transfer learning for Indian currency fake/real classification.

Pipeline
--------
1. Load images from currency_dataset_split/{train,val,test}/{real,fake}/
2. Freeze MobileNetV2 base (ImageNet weights) -> train custom head
3. Phase 2: Unfreeze top 30 layers, fine-tune at lower LR
4. Report per-epoch train/val accuracy + loss
5. Evaluate on test set -> classification report + confusion matrix
6. Save model as currency_cnn_model.h5
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime

# ── Keras / TF imports ────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]
from tensorflow.keras import layers  # type: ignore[attr-defined]
from tensorflow.keras.applications import MobileNetV2  # type: ignore[attr-defined]
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[attr-defined]
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,  # type: ignore[attr-defined]
                                        ModelCheckpoint)

from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(r"c:\Users\Asus\Downloads\ETA Hackathon")
SPLIT_DIR    = BASE_DIR / "currency_dataset_split"
OUTPUT_DIR   = BASE_DIR
MODEL_PATH   = OUTPUT_DIR / "currency_cnn_model.h5"
PLOT_PATH    = OUTPUT_DIR / "training_curves.png"
CM_PATH      = OUTPUT_DIR / "confusion_matrix.png"

IMG_SIZE     = (224, 224)
BATCH_SIZE   = 32
SEED         = 42
EPOCHS_HEAD  = 15      # Phase 1: frozen base
EPOCHS_FINE  = 10      # Phase 2: fine-tune top layers
FINE_TUNE_AT = 100     # Unfreeze from this layer onwards (MobileNetV2 has 154)

tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── 1. Data generators ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Currency CNN Training — MobileNetV2")
print("  TF version:", tf.__version__)
print("="*60)

# MobileNetV2 expects [-1, 1] pixel range via preprocess_input
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False,  # don't flip — note orientation matters
    brightness_range=[0.85, 1.15],
)
val_test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
)

def make_gen(datagen, split, shuffle=True):
    return datagen.flow_from_directory(
        SPLIT_DIR / split,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",   # real=1, fake=0  (alphabetical: fake<real)
        classes=["fake", "real"],
        shuffle=shuffle,
        seed=SEED,
    )

train_gen = make_gen(train_datagen,   "train", shuffle=True)
val_gen   = make_gen(val_test_datagen,"val",   shuffle=False)
test_gen  = make_gen(val_test_datagen,"test",  shuffle=False)

print(f"\n  Class indices : {train_gen.class_indices}")
print(f"  Train samples : {train_gen.samples}")
print(f"  Val   samples : {val_gen.samples}")
print(f"  Test  samples : {test_gen.samples}")
print(f"  Batch size    : {BATCH_SIZE}")
print(f"  Steps/epoch   : {len(train_gen)}\n")

# ── 2. Build model ────────────────────────────────────────────────────────────
print("  Building model (MobileNetV2 + custom head)...")
base = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False   # freeze entire base for Phase 1

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

total_params   = model.count_params()
trainable      = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"  Total params     : {total_params:,}")
print(f"  Trainable (head) : {trainable:,}\n")

# ── 3. Phase 1 — train head ───────────────────────────────────────────────────
print("  PHASE 1: Training classification head (base frozen)")
print("-"*60)

callbacks_p1 = [
    EarlyStopping(monitor="val_accuracy", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-6, verbose=1),
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_p1,
    verbose=1,
)

# ── 4. Phase 2 — fine-tune ────────────────────────────────────────────────────
print(f"\n  PHASE 2: Fine-tuning (unfreezing from layer {FINE_TUNE_AT})")
print("-"*60)

base.trainable = True
for layer in base.layers[:FINE_TUNE_AT]:
    layer.trainable = False

fine_tune_params = sum(np.prod(v.shape) for v in model.trainable_weights)
print(f"  Trainable params now: {fine_tune_params:,}")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_p2 = [
    EarlyStopping(monitor="val_accuracy", patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                      min_lr=1e-7, verbose=1),
    ModelCheckpoint(str(MODEL_PATH), monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_p2,
    verbose=1,
)

# ── 5. Merge histories & print epoch table ────────────────────────────────────
def merge(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

all_train_acc  = merge(history1, history2, "accuracy")
all_val_acc    = merge(history1, history2, "val_accuracy")
all_train_loss = merge(history1, history2, "loss")
all_val_loss   = merge(history1, history2, "val_loss")

print("\n" + "="*60)
print("  EPOCH-BY-EPOCH RESULTS")
print("="*60)
print(f"  {'Epoch':>6}  {'Train Acc':>10}  {'Val Acc':>10}  {'Train Loss':>11}  {'Val Loss':>9}")
print("  " + "-"*54)
for i, (ta, va, tl, vl) in enumerate(zip(
        all_train_acc, all_val_acc, all_train_loss, all_val_loss), 1):
    phase = "P1" if i <= len(history1.history["accuracy"]) else "P2"
    print(f"  {i:>4} {phase}  {ta:>10.4f}  {va:>10.4f}  {tl:>11.4f}  {vl:>9.4f}")

# ── 6. Plot training curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("MobileNetV2 — Currency Fake/Real Detection", fontsize=14)

ep = range(1, len(all_train_acc) + 1)
p1_end = len(history1.history["accuracy"])

ax1.plot(ep, all_train_acc, "b-o", markersize=4, label="Train Acc")
ax1.plot(ep, all_val_acc,   "r-o", markersize=4, label="Val Acc")
ax1.axvline(p1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)

ax2.plot(ep, all_train_loss, "b-o", markersize=4, label="Train Loss")
ax2.plot(ep, all_val_loss,   "r-o", markersize=4, label="Val Loss")
ax2.axvline(p1_end + 0.5, color="gray", linestyle="--", alpha=0.7, label="Fine-tune start")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(PLOT_PATH), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Training curves saved -> {PLOT_PATH}")

# ── 7. Test evaluation ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  TEST SET EVALUATION")
print("="*60)

# Load best checkpoint
model = keras.models.load_model(str(MODEL_PATH))

test_gen.reset()
y_pred_prob = model.predict(test_gen, verbose=1)
y_pred = (y_pred_prob.squeeze() >= 0.5).astype(int)
y_true = test_gen.classes   # 0=fake, 1=real

class_names = ["fake", "real"]

print("\n  Classification Report:")
print("  " + "-"*44)
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
for line in report.split("\n"):
    print("  " + line)

# ── 8. Confusion matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)

print("\n  Confusion Matrix:")
print("                 Predicted")
print("                 fake   real")
print(f"  Actual fake  :  {cm[0,0]:>4}   {cm[0,1]:>4}")
print(f"  Actual real  :  {cm[1,0]:>4}   {cm[1,1]:>4}")

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set", fontsize=13)
plt.tight_layout()
plt.savefig(str(CM_PATH), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Confusion matrix saved -> {CM_PATH}")

# ── 9. Leakage / sanity check ────────────────────────────────────────────────
print("\n" + "="*60)
print("  SANITY CHECK")
print("="*60)

test_acc = np.mean(y_pred == y_true)
val_acc_final  = max(all_val_acc)
train_acc_final = max(all_train_acc)
overfit_gap = train_acc_final - val_acc_final

print(f"  Test  accuracy   : {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"  Best val accuracy: {val_acc_final:.4f} ({val_acc_final*100:.2f}%)")
print(f"  Train-val gap    : {overfit_gap:.4f}")

if test_acc >= 0.999:
    print("\n  [!] SUSPICIOUSLY PERFECT (>=99.9%) — Investigating...")
    print("      Possible causes:")
    print("      1. Data leakage between train and test splits")
    print("      2. Test set too small / not representative")
    print("      3. Dataset too easy (e.g. simple color/format differences)")
    print("      4. Label encoding mismatch (all predictions one class)")
    fake_pred_pct = np.mean(y_pred == 0) * 100
    real_pred_pct = np.mean(y_pred == 1) * 100
    print(f"      Predicted fake: {fake_pred_pct:.1f}%  |  real: {real_pred_pct:.1f}%")
    if fake_pred_pct < 5 or real_pred_pct < 5:
        print("      >> Model is collapsing to single class — likely trivial separability")
elif test_acc >= 0.97:
    print("\n  [NOTE] Very high accuracy (>=97%). Verify with fresh unseen images.")
elif overfit_gap > 0.10:
    print("\n  [!] Overfitting detected (train-val gap > 10%). Consider more dropout or data.")
else:
    print("\n  [OK] Accuracy and gap look reasonable. Model generalizes well.")

print(f"\n  Model saved -> {MODEL_PATH}")
print("="*60 + "\n")
