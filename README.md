# Irembo Voice AI - Intent Classification System

A machine learning solution for classifying user intents in a multilingual (Kinyarwanda/English) Voice AI system for Rwanda's e-government platform.

## 📊 Results Summary

| Model | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|-------|----------|----------|-----------------|--------------|
| Rule-Based Baseline | 88.41% | 0.8937 | 0.9034 | 0.8857 |
| **AfroXLMR-base (Tuned)** | **95.65%** | **0.9563** | **0.9654** | **0.9526** |

### Per-Language Performance (AfroXLMR-base)

| Language | Accuracy | Samples |
|----------|----------|---------|
| English | 90.5% | 21 |
| Code-Switched | 100.0% | 9 |
| **Kinyarwanda** | **97.4%** | 39 |

## 🎯 Supported Intents (13)

1. `check_application_status` - Track application progress
2. `start_new_application` - Begin new service application
3. `requirements_information` - Document/eligibility requirements
4. `fees_information` - Service costs and payments
5. `appointment_booking` - Schedule appointments
6. `cancel_or_reschedule_appointment` - Modify appointments
7. `payment_help` - Payment issues and methods
8. `reset_password_login_help` - Account access issues
9. `document_upload_help` - File upload assistance
10. `update_application_details` - Modify submitted information
11. `service_eligibility` - Check qualification criteria
12. `speak_to_agent` - Human agent escalation
13. `complaint_or_support_ticket` - File complaints

## 🗂️ Project Structure

```
├── data/
│   ├── voiceai_intent_train.csv    # Training data (561 samples)
│   ├── voiceai_intent_val.csv      # Validation data (70 samples)
│   ├── voiceai_intent_test.csv     # Test data (69 samples)
│   └── processed/                   # Preprocessed HuggingFace format
├── models/
│   └── transformer_intent/          # Fine-tuned AfroXLMR model
├── notebooks/
│   └── eda_analysis.ipynb          # Exploratory data analysis
├── reports/
│   └── figures/                     # Visualization outputs
├── scripts/
│   ├── preprocess_irembo.py        # Data preprocessing
│   ├── train_transformer.py        # Model training
│   ├── eval_transformer.py         # Model evaluation
│   ├── eval_baseline.py            # Rule-based baseline
│   └── eval_hybrid.py              # Hybrid classifier evaluation
└── src/
    ├── domain/
    │   ├── entities/
    │   │   └── intent_schema.py    # Intent definitions
    │   └── services/
    │       ├── rule_based_classifier.py
    │       ├── groq_classifier.py
    │       └── hybrid_classifier.py
    └── services/
        └── enhanced_language_detector.py
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n voice python=3.12 -y
conda activate voice

# Install dependencies
pip install transformers datasets torch scikit-learn pandas
pip install sentencepiece accelerate
```

### 2. Preprocess Data

```bash
python scripts/preprocess_irembo.py
```

### 3. Train Transformer Model

```bash
python scripts/train_transformer.py
```

### 4. Evaluate Model

```bash
# Evaluate transformer
python scripts/eval_transformer.py \
    --model-dir models/transformer_intent \
    --test-file data/processed/test_hf.json

# Evaluate baseline
python scripts/eval_baseline.py
```

## 🔧 Model Architecture

- **Base Model**: [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base)
- **Fine-tuning**: 10 epochs, warmup ratio 0.1, weight decay 0.01
- **Optimizer**: AdamW
- **Learning Rate**: 2e-5

### Why AfroXLMR?

AfroXLMR is specifically pre-trained on African languages including Kinyarwanda, making it ideal for:
- Low-resource language handling
- Code-switching between Kinyarwanda and English
- Cultural and linguistic nuances

## 🌍 Language Support

The system handles three language modes:
- **English (`en`)**: Pure English utterances
- **Kinyarwanda (`rw`)**: Pure Kinyarwanda utterances
- **Mixed (`mixed`)**: Code-switched utterances combining both languages

Example code-switched utterance:
> "Ndashaka kubona **status** y'application yanjye" (I want to see the status of my application)

## 📈 Training Hyperparameters

```python
TrainingConfig(
    model_name="Davlan/afro-xlmr-base",
    num_epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    max_length=128
)
```

## 🧪 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Macro F1**: Average F1 across all intents (treats all intents equally)
- **Per-Intent F1**: Individual intent performance
- **Per-Language Accuracy**: Performance breakdown by language

## 📊 Dataset Statistics

| Split | Samples | Languages |
|-------|---------|-----------|
| Train | 561 | rw: 43.7%, en: 28.7%, mixed: 27.6% |
| Validation | 70 | Stratified |
| Test | 69 | Stratified |

## 🔄 Hybrid Classification Pipeline

```
User Utterance
      ↓
[Language Detection] → Detect rw/en/mixed
      ↓
[Rule-Based Classifier] → Keyword matching
      ↓ (if confidence < 0.7)
[Transformer Model] → AfroXLMR inference
      ↓ (if confidence < 0.5)
[LLM Fallback] → Groq API
      ↓ (if still uncertain)
[Human Agent] → Escalation
```

## 📁 Key Files

| File | Description |
|------|-------------|
| `scripts/train_transformer.py` | Fine-tuning pipeline |
| `scripts/eval_transformer.py` | Evaluation with per-language metrics |
| `src/domain/entities/intent_schema.py` | Intent definitions and examples |
| `src/domain/services/rule_based_classifier.py` | Keyword-based classifier |
| `src/services/enhanced_language_detector.py` | Kinyarwanda/English detection |

## 🛠️ Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
# Format with black
black src/ scripts/

# Lint with flake8
flake8 src/ scripts/
```

## 📝 License

This project was developed as part of the Irembo Machine Learning Engineer assessment.

## 🙏 Acknowledgments

- [Davlan/afro-xlmr-base](https://huggingface.co/Davlan/afro-xlmr-base) - Multilingual African language model
- [HuggingFace Transformers](https://huggingface.co/transformers/) - Model training framework
- Irembo Rwanda - Dataset and problem specification
