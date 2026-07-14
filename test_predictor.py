"""
test_predictor.py
=================
Quick smoke-test for currency_predictor.py.

Picks 2 fake + 2 real images from the test split and prints predictions.
Run after training completes:
    python test_predictor.py
"""

import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

# ── Locate sample images ──────────────────────────────────────────────────────
BASE     = Path(r"c:\Users\Asus\Downloads\ETA Hackathon")
TEST_DIR = BASE / "currency_dataset_split" / "test"


def pick_images(cls: str, n: int = 2) -> list[Path]:
    folder = TEST_DIR / cls
    imgs   = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not imgs:
        print(f"  [warn] no images found in {folder}")
        return []
    return imgs[:n]


samples = (
    [(p, "fake") for p in pick_images("fake", 2)]
    + [(p, "real") for p in pick_images("real", 2)]
)

if not samples:
    print("ERROR: No test images found. Check currency_dataset_split/test/.")
    sys.exit(1)

# ── Import the module ─────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE))
from currency_predictor import predict_image  # noqa: E402

# ── Check model exists ────────────────────────────────────────────────────────
model_path = BASE / "currency_cnn_model.h5"
if not model_path.exists():
    print(
        "\nERROR: currency_cnn_model.h5 not found.\n"
        "The model is still training. Re-run this script once train_cnn.py finishes.\n"
    )
    sys.exit(1)

# ── Run predictions ───────────────────────────────────────────────────────────
GREEN = ""
RED   = ""
RESET = ""

print("\n" + "="*65)
print("  currency_predictor  —  smoke test")
print("="*65)
print(f"  {'Image':<40} {'True':>6}  {'Verdict':>6}  {'Conf':>7}  {'OK?':>4}")
print("  " + "-"*62)

correct = 0
for img_path, true_label in samples:
    result     = predict_image(img_path)
    verdict    = result["verdict"]
    confidence = result["confidence"]
    raw        = result["raw_score"]
    ok         = verdict == true_label

    if ok:
        correct += 1
        tag = "[OK]"
    else:
        tag = "[!!]"

    name = img_path.parent.name + "/" + img_path.name   # e.g. fake/orig_103.jpg
    print(f"  {name:<40} {true_label:>6}  {verdict:>6}  {confidence:>6.1%}  {tag}")

print("  " + "-"*62)
print(f"  Correct: {correct}/{len(samples)}\n")

# ── Show full dict output for the first image ─────────────────────────────────
print("  Full dict output for first sample:")
first_result = predict_image(samples[0][0])
print(f"    predict_image('{samples[0][0].name}')")
print(f"    → {first_result}")
print()

# ── FastAPI snippet ───────────────────────────────────────────────────────────
print("="*65)
print("  FastAPI integration snippet")
print("="*65)
print("""
  from fastapi import FastAPI, UploadFile, File, HTTPException
  import shutil, tempfile, os
  from currency_predictor import predict_image

  app = FastAPI(title="Currency Fraud Detector")

  @app.post("/predict", summary="Classify a currency note image")
  async def predict(file: UploadFile = File(...)):
      suffix = os.path.splitext(file.filename)[-1] or ".jpg"
      with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
          shutil.copyfileobj(file.file, tmp)
          tmp_path = tmp.name
      try:
          return predict_image(tmp_path)
      except Exception as e:
          raise HTTPException(status_code=422, detail=str(e))
      finally:
          os.unlink(tmp_path)
""")
print("="*65 + "\n")
