# AI for Public Digital Safety — ETA Hackathon

A two-module AI pipeline for public digital safety, built for the ETA Hackathon:

1. **Currency Fraud Detector** — classifies Indian currency notes as real or fake using MobileNetV2
2. **Scam Detection Module** — classifies SMS/WhatsApp messages as `scam`, `suspicious`, or `safe` using a heuristic + LLM hybrid pipeline

---

## Project Structure

```
├── api/
│   ├── main.py               # Scam detection FastAPI app (port 8001)
│   └── schemas.py            # Pydantic schemas
├── data_processing/
│   ├── organize_dataset.py   # Dataset organisation helper
│   ├── split_dataset.py      # train/val/test split (70/15/15)
│   └── augment_dataset.py    # Offline augmentation script
├── training/
│   ├── train_cnn.py          # Phase 1 + 2 training (MobileNetV2, ImageNet weights)
│   └── finetune_phase3.py    # Phase 3 deep fine-tuning (last-30-layer unfreeze, LR=1e-5)
├── inference/
│   ├── currency_predictor.py # Importable inference module -> predict_image()
│   └── test_predictor.py     # Smoke test: runs predict_image on 4 sample images
├── heuristics/
│   ├── extract.py            # Fast heuristic checks (URL, urgency, OTP, etc.)
│   └── merge.py              # Merge heuristic + LLM outputs
├── llm/
│   └── classify.py           # Groq (Llama 3.3 70B) LLM classifier
├── frontend/
│   └── index.html            # Redesigned light-theme frontend
├── tests/
│   ├── test_samples.py       # Heuristics-only tests (no API key needed)
│   └── test_llm.py           # LLM integration tests
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── finetune_p3_curves.png
│   ├── confusion_matrix_p3.png
│   └── augmentation_preview.jpg
├── main.py                   # Currency detector FastAPI app (port 8000)
├── requirements.txt
└── README.md
```

---

## Module 1 — Currency Fraud Detector

### Model Performance

| Phase | Description | Test Accuracy |
|---|---|---|
| Phase 1 | Frozen MobileNetV2 base, train head only | ~94% val |
| Phase 2 | Unfreeze from layer 100+, LR=1e-4 | **96.06%** test |
| Phase 3 | Unfreeze last 30 layers, LR=1e-5 | 95.70% test |

**Best model: Phase 2** (`currency_cnn_model.h5`)

### Quick Start

```bash
pip install tensorflow scikit-learn matplotlib seaborn fastapi uvicorn python-multipart
python -m uvicorn main:app --port 8000
```

**POST** `/predict` — upload a currency note image, get `real`/`fake` verdict  
**GET** `/health` — health check

### Run inference directly

```python
import sys; sys.path.insert(0, "./inference")
from currency_predictor import predict_image

result = predict_image("path/to/note.jpg")
# -> {'verdict': 'real', 'confidence': 0.9821, 'raw_score': 0.9821}
```

---

## Module 2 — Scam Detection

Hybrid heuristic + LLM pipeline that classifies SMS/WhatsApp-style messages
(and any URLs they contain) as `scam`, `suspicious`, or `safe`.

### How it works

1. **Heuristic pass** (`heuristics/extract.py`) — fast, free, explainable checks:
   URL shorteners, IP-based URLs, suspicious TLDs, lookalike domains against
   known Indian brands, urgency language, OTP/PIN/UPI requests, prize/lottery bait.
2. **LLM pass** (`llm/classify.py`) — sends the message + heuristic flags to
   Groq (Llama 3.3 70B) for contextual judgment and a human-readable explanation.
3. **Merge** (`heuristics/merge.py`) — heuristic-only fallback if LLM call fails.

### Quick Start

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-here"   # https://console.groq.com
python -m uvicorn api.main:app --reload --port 8001
```

### Example request

```
POST /predict/scam
{"text": "Your SBI account will be BLOCKED. Verify now: http://sbi-kyc.xyz/verify"}
```

```json
{
  "module": "scam",
  "verdict": "scam",
  "confidence": 0.93,
  "explanation": "This message uses urgency and a suspicious lookalike link to try to steal your bank login."
}
```

---

## Tech Stack

- **Python 3.11**
- **TensorFlow 2.19 / Keras** + **MobileNetV2**
- **FastAPI** + **Uvicorn** — REST endpoints
- **Groq API (Llama 3.3 70B)** — LLM classification
- **scikit-learn** — metrics & evaluation
