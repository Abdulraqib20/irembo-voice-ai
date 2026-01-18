# Irembo Voice AI Intent Classification System

This repository implements a multilingual intent classification system for Irembo Voice AI, targeting Kinyarwanda, English, and code-switched utterances for Rwanda’s e-government services.

## Results Summary (Test Split)

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|--------------|
| Rule-Based Baseline | 88.41% | 89.37% | 90.34% | 88.57% |
| AfroXLMR-base (Tuned) | 95.65% | 95.63% | 96.54% | 95.26% |

### Per-Language Performance (AfroXLMR-base)

| Language | Accuracy | Samples |
|----------|----------|---------|
| English | 90.5% | 21 |
| Code-Switched | 100.0% | 9 |
| Kinyarwanda | 97.4% | 39 |

## System Overview

- Language detection uses lightweight Kinyarwanda/English markers for routing.
- Primary inference runs on a fine-tuned AfroXLMR transformer.
- Confidence-based fallback routes to a rule-based classifier, then to Mixtral-8x7B (LLM) when enabled.
- Each request is logged with confidence, fallback reason, and latency for monitoring.

## Supported Intents (13)

1. `check_application_status`
2. `start_new_application`
3. `requirements_information`
4. `fees_information`
5. `appointment_booking`
6. `cancel_or_reschedule_appointment`
7. `payment_help`
8. `reset_password_login_help`
9. `document_upload_help`
10. `update_application_details`
11. `service_eligibility`
12. `speak_to_agent`
13. `complaint_or_support_ticket`

## Repository Structure

```
├── data/
│   ├── voiceai_intent_train.csv
│   ├── voiceai_intent_val.csv
│   ├── voiceai_intent_test.csv
│   └── processed/
├── models/
│   └── transformer_intent/
├── notebooks/
│   └── eda_analysis.ipynb
├── reports/
│   └── figures/
├── scripts/
│   ├── preprocess_irembo.py
│   ├── train_transformer.py
│   ├── eval_transformer.py
│   ├── eval_baseline.py
│   └── eval_hybrid.py
└── src/
    ├── api/
    ├── config/
    ├── domain/
    └── services/
```

## Setup

### Requirements

- Python 3.10+
- Install inference dependencies:

```bash
pip install -r requirements.txt
```

- Training/evaluation dependencies:

```bash
pip install datasets scikit-learn pandas
```

### Preprocess Data

```bash
python scripts/preprocess_irembo.py
```

### Train the Transformer

```bash
python scripts/train_transformer.py
```

### Evaluate

```bash
python scripts/eval_transformer.py \
  --model-dir models/transformer_intent \
  --test-file data/processed/test_hf.json

python scripts/eval_baseline.py
python scripts/eval_hybrid.py
```

## Run the API

### Local

```bash
uvicorn src.api.inference_api:app --reload --port 8000
```

### Docker

```bash
docker compose up --build
```

### Example Request

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"utterance_text": "Ndashaka kureba status ya application yanjye"}'
```

## Optional LLM Fallback

Set `GROQ_API_KEY` to enable Mixtral-8x7B fallback. If not provided, the system runs with the transformer and rule-based tiers only.

## License

This project was developed as part of the Irembo Machine Learning Engineer assessment.

## Acknowledgments

- Davlan/afro-xlmr-base
- HuggingFace Transformers
- Irembo Rwanda (dataset and problem specification)
