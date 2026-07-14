"""
currency_predictor.py
=====================
Importable inference module for the Indian currency fake/real classifier.

Public API
----------
    from currency_predictor import predict_image, load_model_once

    result = predict_image("path/to/note.jpg")
    # -> {"verdict": "fake" | "real", "confidence": 0.9732}

FastAPI usage example
---------------------
    from fastapi import FastAPI, UploadFile, File
    import shutil, tempfile
    from currency_predictor import predict_image

    app = FastAPI()

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            shutil.copyfileobj(file.file, tmp)
            result = predict_image(tmp.name)
        return result
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

# ── lazy TF import so the module loads fast if TF is slow to initialise ───────
_tf   = None
_keras = None
_model = None   # singleton – loaded once per process


# Default model path – override via env var or the load_model_once() argument
_DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "currency_cnn_model.h5"

IMG_SIZE = (224, 224)   # must match training


# ── Internal helpers ──────────────────────────────────────────────────────────

def _import_tf():
    """Import TensorFlow/Keras once and cache the reference."""
    global _tf, _keras
    if _tf is None:
        import tensorflow as tf          # type: ignore[attr-defined]
        from tensorflow import keras     # type: ignore[attr-defined]
        _tf   = tf
        _keras = keras
    return _tf, _keras


def _preprocess(image_path: str | Path) -> np.ndarray:
    """
    Load an image from *image_path*, resize to 224×224, apply MobileNetV2
    preprocessing (scales pixels to [-1, 1]), and return a (1, 224, 224, 3)
    float32 numpy array ready for model.predict().

    Raises
    ------
    FileNotFoundError  – if the path does not exist.
    ValueError         – if the file cannot be decoded as an image.
    """
    tf, keras = _import_tf()

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # Read & decode
    raw = tf.io.read_file(str(path))
    try:
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    except Exception as exc:
        raise ValueError(f"Cannot decode image '{path}': {exc}") from exc

    # Resize
    img = tf.image.resize(img, IMG_SIZE)           # (224, 224, 3)  float32

    # MobileNetV2 expects [-1, 1] — use the official preprocessing fn
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # type: ignore[attr-defined]

    # Add batch dimension
    img = tf.expand_dims(img, axis=0)              # (1, 224, 224, 3)
    return img.numpy()


# ── Public API ────────────────────────────────────────────────────────────────

def load_model_once(model_path: str | Path | None = None) -> object:
    """
    Load the Keras model exactly once (singleton pattern).  Subsequent calls
    return the cached model without re-reading disk.

    Parameters
    ----------
    model_path : str | Path | None
        Path to the .h5 model file.  Defaults to *currency_cnn_model.h5*
        next to this module, or the ``CURRENCY_MODEL_PATH`` environment
        variable if set.

    Returns
    -------
    A compiled Keras model.

    Raises
    ------
    FileNotFoundError  – if the model file cannot be found.
    """
    global _model

    if _model is not None:
        return _model                               # already loaded

    tf, keras = _import_tf()

    # Resolve path: argument > env var > default next to this file > default hardcoded
    if model_path is None:
        env_path = os.environ.get("CURRENCY_MODEL_PATH")
        if env_path:
            model_path = Path(env_path)
        else:
            # Try same directory as this file first, then the hardcoded default
            local = Path(__file__).parent / "currency_cnn_model.h5"
            model_path = local if local.exists() else _DEFAULT_MODEL

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "Train it first with:  python train_cnn.py\n"
            "Or set the CURRENCY_MODEL_PATH environment variable."
        )

    _model = keras.models.load_model(str(model_path))  # type: ignore[attr-defined]
    return _model


def predict_image(
    image_path: str | Path,
    model_path: str | Path | None = None,
    threshold: float = 0.5,
) -> dict:
    """
    Predict whether a currency note image is real or fake.

    Parameters
    ----------
    image_path : str | Path
        Path to the image file (JPG, PNG, BMP, etc.).
    model_path : str | Path | None
        Optional override for the model file location.
        If *None*, uses the default / env-var path (see load_model_once).
    threshold : float
        Decision threshold on the sigmoid output.
        Scores >= threshold -> "real"; below -> "fake". Default 0.5.

    Returns
    -------
    dict with keys:
        "verdict"    : "real" or "fake"
        "confidence" : float in [0.0, 1.0] – how certain the model is
                       about *its* verdict (always ≥ 0.5 when unambiguous).
        "raw_score"  : float in [0.0, 1.0] – raw sigmoid output
                       (closer to 1.0 = more likely real).

    Raises
    ------
    FileNotFoundError  – if image or model file cannot be found.
    ValueError         – if the image file is corrupt / unreadable.

    Examples
    --------
    >>> result = predict_image("test/real/orig_103.jpg")
    >>> result
    {'verdict': 'real', 'confidence': 0.9821, 'raw_score': 0.9821}

    >>> result = predict_image("test/fake/orig_103.jpg")
    >>> result
    {'verdict': 'fake', 'confidence': 0.9143, 'raw_score': 0.0857}
    """
    model = load_model_once(model_path)

    img        = _preprocess(image_path)
    raw_score  = float(model.predict(img, verbose=0).squeeze())

    is_real   = raw_score >= threshold
    verdict   = "real" if is_real else "fake"
    # Confidence = how far from the decision boundary, normalised to [0,1]
    confidence = raw_score if is_real else (1.0 - raw_score)

    return {
        "verdict":    verdict,
        "confidence": round(confidence, 6),
        "raw_score":  round(raw_score,  6),
    }
