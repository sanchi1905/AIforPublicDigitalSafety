"""
finetune_phase3.py
==================
Phase 3 deep fine-tuning of the MobileNetV2 currency classifier.

Flow
----
1. Load currency_cnn_model.h5 if it exists; otherwise run full
   Phase-1 + Phase-2 training first.
2. Evaluate on the test set  ->  BEFORE metrics.
3. Unfreeze the last 30 layers of the MobileNetV2 base.
4. Fine-tune at LR = 1e-5 for up to 10 epochs (EarlyStopping patience=4).
5. Evaluate on the test set  ->  AFTER metrics.
6. Print a side-by-side comparison + recommendation for the demo.
7. Save the best Phase-3 model as currency_cnn_model_p3.h5
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import io
from pathlib import Path
from datetime import datetime

# Force UTF-8 output so Unicode chars don't crash on Windows cp1252 terminals
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── TF / Keras ────────────────────────────────────────────────────────────────
import tensorflow as tf  # type: ignore[attr-defined]
from tensorflow import keras  # type: ignore[attr-defined]
from tensorflow.keras import layers  # type: ignore[attr-defined]
from tensorflow.keras.applications import MobileNetV2  # type: ignore[attr-defined]
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore[attr-defined]
from tensorflow.keras.callbacks import (  # type: ignore[attr-defined]
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, roc_auc_score
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parents[1]
SPLIT_DIR     = BASE_DIR / "currency_dataset_split"
MODEL_P2_PATH = BASE_DIR / "currency_cnn_model.h5"
MODEL_P3_PATH = BASE_DIR / "currency_cnn_model_p3.h5"
PLOT_P3_PATH  = BASE_DIR / "results" / "finetune_p3_curves.png"
CM_P3_PATH    = BASE_DIR / "results" / "confusion_matrix_p3.png"

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42

# Phase-3 hyper-params (per user request)
P3_UNFREEZE_LAST = 30    # unfreeze last 30 layers of base
P3_LR            = 1e-5
P3_EPOCHS        = 10
P3_PATIENCE      = 4

# Phase-1 / Phase-2 fallback defaults
EPOCHS_P1       = 15
EPOCHS_P2       = 10
FINE_TUNE_AT_P2 = 100

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("\n" + "="*65)
print("  Currency CNN — Phase 3 Deep Fine-Tuning")
print(f"  TF {tf.__version__}   |   {datetime.now():%Y-%m-%d %H:%M}")
print("="*65)

# ── Data generators ───────────────────────────────────────────────────────────
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input  # type: ignore[attr-defined]

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=False,
    brightness_range=[0.85, 1.15],
)
val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess)


def make_gen(datagen, split, shuffle=True):
    return datagen.flow_from_directory(
        SPLIT_DIR / split,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=["fake", "real"],
        shuffle=shuffle,
        seed=SEED,
    )


train_gen = make_gen(train_datagen,    "train", shuffle=True)
val_gen   = make_gen(val_test_datagen, "val",   shuffle=False)
test_gen  = make_gen(val_test_datagen, "test",  shuffle=False)

print(f"\n  Class indices : {train_gen.class_indices}")
print(f"  Train / Val / Test : {train_gen.samples} / {val_gen.samples} / {test_gen.samples}")


# ── Helper: full evaluation on test set ───────────────────────────────────────
def evaluate_model(model, label="Model"):
    test_gen.reset()
    probs  = model.predict(test_gen, verbose=0).squeeze()
    preds  = (probs >= 0.5).astype(int)
    labels = test_gen.classes

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="weighted")
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    fake_recall = tn / (tn + fp) if (tn + fp) else 0.0
    real_recall = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"\n  -- {label} --")
    print(f"     Accuracy        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"     Weighted F1     : {f1:.4f}")
    print(f"     ROC-AUC         : {auc:.4f}")
    print(f"     Fake recall     : {fake_recall*100:.2f}%  (catches {int(tn)}/{int(tn+fp)} fake notes)")
    print(f"     Real recall     : {real_recall*100:.2f}%  (passes  {int(tp)}/{int(tp+fn)} real notes)")
    print(f"     Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    report = classification_report(labels, preds,
                                   target_names=["fake", "real"], digits=4)
    for line in report.split("\n"):
        print("     " + line)

    return dict(acc=acc, f1=f1, auc=auc,
                fake_recall=fake_recall, real_recall=real_recall,
                cm=cm, probs=probs, preds=preds)


# ── Load or build the Phase-2 model ───────────────────────────────────────────
if MODEL_P2_PATH.exists():
    print(f"\n  Loading existing Phase-2 model: {MODEL_P2_PATH}")
    model_p2 = keras.models.load_model(str(MODEL_P2_PATH))  # type: ignore[attr-defined]
    print("  Model loaded successfully.")
else:
    print(f"\n  {MODEL_P2_PATH.name} not found — training from scratch (Phase 1 + 2)...")

    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False,
                       weights="imagenet")
    base.trainable = False

    inputs  = keras.Input(shape=(*IMG_SIZE, 3))  # type: ignore[attr-defined]
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(256, activation="relu")(x)
    x       = layers.Dropout(0.4)(x)
    x       = layers.Dense(64, activation="relu")(x)
    x       = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model_p2 = keras.Model(inputs, outputs)  # type: ignore[attr-defined]

    model_p2.compile(optimizer=keras.optimizers.Adam(1e-3),  # type: ignore[attr-defined]
                     loss="binary_crossentropy", metrics=["accuracy"])

    print("\n  PHASE 1 — head only (base frozen)")
    model_p2.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_P1, verbose=1,
        callbacks=[
            EarlyStopping("val_accuracy", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        ],
    )

    print(f"\n  PHASE 2 — partial unfreeze (layer {FINE_TUNE_AT_P2}+)")
    base.trainable = True
    for lyr in base.layers[:FINE_TUNE_AT_P2]:
        lyr.trainable = False
    model_p2.compile(optimizer=keras.optimizers.Adam(1e-4),  # type: ignore[attr-defined]
                     loss="binary_crossentropy", metrics=["accuracy"])
    model_p2.fit(
        train_gen, validation_data=val_gen, epochs=EPOCHS_P2, verbose=1,
        callbacks=[
            EarlyStopping("val_accuracy", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
            ModelCheckpoint(str(MODEL_P2_PATH), monitor="val_accuracy",
                            save_best_only=True, verbose=1),
        ],
    )
    model_p2 = keras.models.load_model(str(MODEL_P2_PATH))  # type: ignore[attr-defined]
    print(f"  Phase-2 model saved -> {MODEL_P2_PATH}")

# ── BEFORE baseline ───────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  BEFORE PHASE-3 — TEST SET (Phase-2 baseline)")
print("="*65)
before = evaluate_model(model_p2, label="Phase-2 (before fine-tune)")

# ── Locate MobileNetV2 sub-model inside the functional model ──────────────────
base_model = None
for lyr in model_p2.layers:
    if (isinstance(lyr, tf.keras.Model)  # type: ignore[attr-defined]
            and "mobilenet" in lyr.name.lower()):
        base_model = lyr
        break

# Fallback: largest sub-model (>50 layers)
if base_model is None:
    for lyr in model_p2.layers:
        sub_layers = getattr(lyr, "layers", [])
        if len(sub_layers) > 50:
            base_model = lyr
            break

if base_model is None:
    print("\n  [ERROR] Cannot find MobileNetV2 inside saved model. Exiting.")
    sys.exit(1)

total_base   = len(base_model.layers)
unfreeze_n   = min(P3_UNFREEZE_LAST, total_base)
freeze_until = total_base - unfreeze_n

print(f"\n  Base model : '{base_model.name}' ({total_base} layers)")
print(f"  Freezing   : layers 0 – {freeze_until-1}")
print(f"  Unfreezing : layers {freeze_until} – {total_base-1}  ({unfreeze_n} layers)")

base_model.trainable = True
for lyr in base_model.layers[:freeze_until]:
    lyr.trainable = False

trainable_p3 = sum(np.prod(v.shape) for v in model_p2.trainable_weights)
print(f"  Trainable parameters in Phase-3 : {trainable_p3:,}")

# ── Recompile at very low LR ──────────────────────────────────────────────────
model_p2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=P3_LR),  # type: ignore[attr-defined]
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# ── Phase-3 training ──────────────────────────────────────────────────────────
print(f"\n  PHASE 3 — deep fine-tune | LR={P3_LR} | max epochs={P3_EPOCHS}")
print("-"*65)

callbacks_p3 = [
    EarlyStopping(monitor="val_accuracy", patience=P3_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                      min_lr=1e-8, verbose=1),
    ModelCheckpoint(str(MODEL_P3_PATH), monitor="val_accuracy",
                    save_best_only=True, verbose=1),
]

history_p3 = model_p2.fit(
    train_gen,
    validation_data=val_gen,
    epochs=P3_EPOCHS,
    callbacks=callbacks_p3,
    verbose=1,
)

# Load the best checkpoint from Phase 3
model_p3 = keras.models.load_model(str(MODEL_P3_PATH))  # type: ignore[attr-defined]

# ── AFTER evaluation ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  AFTER PHASE-3 — TEST SET")
print("="*65)
after = evaluate_model(model_p3, label="Phase-3 (after fine-tune)")

# ── Side-by-side table ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  COMPARISON: Phase-2  vs  Phase-3")
print("="*65)


def fmt_delta(b, a, pct=False):
    d    = a - b
    sign = "+" if d >= 0 else ""
    if pct:
        return f"{a*100:.2f}%  ({sign}{d*100:.2f}pp)"
    return f"{a:.4f}  ({sign}{d:.4f})"


rows = [
    ("Accuracy",    before["acc"],         after["acc"],         True),
    ("Weighted F1", before["f1"],          after["f1"],          False),
    ("ROC-AUC",     before["auc"],         after["auc"],         False),
    ("Fake recall", before["fake_recall"], after["fake_recall"], True),
    ("Real recall", before["real_recall"], after["real_recall"], True),
]

print(f"  {'Metric':<20} {'Phase-2':>14}   {'Phase-3 (delta)':>22}")
print("  " + "-"*60)
for name, b_val, a_val, is_pct in rows:
    b_str = f"{b_val*100:.2f}%" if is_pct else f"{b_val:.4f}"
    a_str = fmt_delta(b_val, a_val, pct=is_pct)
    print(f"  {name:<20} {b_str:>14}   {a_str:>22}")

# ── Demo recommendation logic ────────────────────────────────────────────────
print("\n" + "="*65)
print("  DEMO RECOMMENDATION")
print("="*65)

acc_gain         = after["acc"]         - before["acc"]
f1_gain          = after["f1"]          - before["f1"]
auc_gain         = after["auc"]         - before["auc"]
fake_recall_drop = before["fake_recall"] - after["fake_recall"]

# Decision rules (fraud detection: missing a fake is expensive)
if fake_recall_drop > 0.02:
    # Phase-3 made us miss more fakes → bad for fraud detection
    winner      = "Phase-2"
    winner_path = MODEL_P2_PATH
    reason = (
        f"Phase-3 reduced fake-recall by {fake_recall_drop*100:.1f}pp. "
        "In a fraud-detection demo, missing fake notes is far more costly "
        "than false alarms. Use the Phase-2 model."
    )
elif acc_gain >= 0.005 and f1_gain >= 0:
    winner      = "Phase-3"
    winner_path = MODEL_P3_PATH
    reason = (
        f"Phase-3 improved accuracy by {acc_gain*100:+.2f}pp and F1 by "
        f"{f1_gain:+.4f} without hurting fake-recall. Use the deeper model."
    )
elif acc_gain >= 0 and auc_gain > 0.005:
    winner      = "Phase-3"
    winner_path = MODEL_P3_PATH
    reason = (
        f"AUC improved by {auc_gain:+.4f}, meaning Phase-3 ranks positives "
        "better — valuable if the demo threshold is adjustable."
    )
else:
    # No meaningful improvement — keep Phase-2 (simpler, less risk of overfit)
    winner      = "Phase-2"
    winner_path = MODEL_P2_PATH
    reason = (
        "Phase-3 fine-tuning produced no meaningful gain (<0.5pp accuracy, "
        "<0.005 AUC). On this small dataset Phase-2 is already near-optimal. "
        "Further fine-tuning risks overfitting with no demo benefit."
    )

print(f"  RECOMMENDED MODEL : {winner}")
print(f"  FILE TO USE       : {winner_path.name}")
print(f"\n  Reason: {reason}\n")
print("="*65)

# -- Plot Phase-3 training curves ----------------------------------------------
ep = range(1, len(history_p3.history["accuracy"]) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Phase-3 Fine-Tuning Curves  (LR = 1e-5, last-30 layers)", fontsize=12)

ax1.plot(ep, history_p3.history["accuracy"],     "b-o", ms=5, label="Train Acc")
ax1.plot(ep, history_p3.history["val_accuracy"], "r-o", ms=5, label="Val Acc")
ax1.axhline(before["acc"], color="gray", ls="--", alpha=0.7,
            label=f"P2 baseline ({before['acc']:.3f})")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)

ax2.plot(ep, history_p3.history["loss"],     "b-o", ms=5, label="Train Loss")
ax2.plot(ep, history_p3.history["val_loss"], "r-o", ms=5, label="Val Loss")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch")
ax2.legend(); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(str(PLOT_P3_PATH), dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Training curves  -> {PLOT_P3_PATH}")

# -- Confusion matrix side-by-side -----------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Test-Set Confusion Matrices", fontsize=12)

for ax, cm, title in [
    (ax1, before["cm"], f"Phase-2  (acc={before['acc']:.3f})"),
    (ax2, after["cm"],  f"Phase-3  (acc={after['acc']:.3f})"),
]:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["fake", "real"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=11)

plt.tight_layout()
plt.savefig(str(CM_P3_PATH), dpi=150, bbox_inches="tight")
plt.close()
print(f"  Confusion matrices -> {CM_P3_PATH}\n")
