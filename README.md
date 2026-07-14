# AI for Public Digital Safety — Indian Currency Fraud Detection

A deep-learning pipeline that classifies Indian currency notes as **real** or **fake** using **MobileNetV2** transfer learning. Built for the ETA Hackathon.

---

## Project Structure

```
├── train_cnn.py          # Phase 1 + 2 training (MobileNetV2, ImageNet weights)
├── finetune_phase3.py    # Phase 3 deep fine-tuning (last-30-layer unfreeze, LR=1e-5)
├── currency_predictor.py # Importable inference module → predict_image()
├── test_predictor.py     # Smoke test: runs predict_image on 4 sample images
├── augment_dataset.py    # Offline augmentation script
├── organize_dataset.py   # Dataset organisation helper
├── split_dataset.py      # train/val/test split (70/15/15)
├── training_curves.png   # Accuracy & loss curves (Phase 1 + 2)
├── confusion_matrix.png  # Test-set confusion matrix (Phase 2)
├── finetune_p3_curves.png# Phase 3 training curves
├── confusion_matrix_p3.png # Side-by-side confusion matrices
└── augmentation_preview.jpg
```

---

## Model Performance

| Phase | Description | Test Accuracy |
|---|---|---|
| Phase 1 | Frozen MobileNetV2 base, train head only | ~94% val |
| Phase 2 | Unfreeze from layer 100+, LR=1e-4 | **96.06%** test |
| Phase 3 | Unfreeze last 30 layers, LR=1e-5 | 95.70% test |

**Best model: Phase 2** (`currency_cnn_model.h5`)

Confusion matrix on 279 test images:
```
              Predicted
              fake   real
Actual fake :  137      6
Actual real :    5    131
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install tensorflow scikit-learn matplotlib seaborn
```

### 2. Train the model
```bash
python train_cnn.py
```

### 3. Run inference
```python
from currency_predictor import predict_image

result = predict_image("path/to/note.jpg")
print(result)
# → {'verdict': 'real', 'confidence': 0.9821, 'raw_score': 0.9821}
```

### 4. FastAPI endpoint (for your teammate)
```python
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil, tempfile, os
from currency_predictor import predict_image

app = FastAPI(title="Currency Fraud Detector")

@app.post("/predict")
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
```

---

## Dataset Layout (expected)

```
currency_dataset_split/
├── train/
│   ├── fake/   (664 images)
│   └── real/   (634 images)
├── val/
│   ├── fake/   (142 images)
│   └── real/   (136 images)
└── test/
    ├── fake/   (143 images)
    └── real/   (136 images)
```

> Dataset images are excluded from this repo.
> Pre-trained model weights (`currency_cnn_model.h5`) can be downloaded here: **[Insert Google Drive Link Here]**
> Alternatively, you can run `train_cnn.py` to reproduce and retrain the model locally.

---

## Tech Stack

- **Python 3.11**
- **TensorFlow 2.19 / Keras**
- **MobileNetV2** (ImageNet pretrained)
- **scikit-learn** — metrics & evaluation
- **FastAPI** — REST inference endpoint
