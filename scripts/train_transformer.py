"""
Transformer Training Script for Irembo Voice AI Intent Classification.

Fine-tunes a multilingual transformer (e.g., AfroXLMR or XLM-R) on the
Kinyarwanda/English intent dataset prepared by preprocess_irembo.py.
"""

# Standard library imports
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party imports
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class TrainingConfig:
    """Configuration for transformer fine-tuning."""

    model_name: str
    train_file: str
    val_file: str
    output_dir: str
    max_length: int
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    gradient_accumulation_steps: int
    seed: int


def load_hf_json(file_path: str) -> Dict[str, Any]:
    """Load HuggingFace-format JSON produced by preprocess_irembo.py.

    Args:
        file_path (str): Path to JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(data: Dict[str, Any]) -> Dataset:
    """Create a HuggingFace Dataset from preprocessed data.

    Args:
        data (Dict[str, Any]): Parsed JSON with "data" list.

    Returns:
        Dataset: HuggingFace Dataset with text/label/language fields.
    """
    samples = data.get("data", [])
    if not samples:
        raise ValueError("No samples found in provided JSON file.")
    return Dataset.from_list(samples)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """Tokenize dataset using the provided tokenizer.

    Args:
        dataset (Dataset): Input dataset.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        max_length (int): Max sequence length.

    Returns:
        Dataset: Tokenized dataset.
    """
    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize_batch, batched=True)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for classification.

    Args:
        eval_pred: Predictions object from Trainer.

    Returns:
        Dict[str, float]: Metrics dictionary.
    """
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def save_training_config(config: TrainingConfig, label2id: Dict[str, int]) -> None:
    """Save training configuration to output directory.

    Args:
        config (TrainingConfig): Training configuration.
        label2id (Dict[str, int]): Label mapping.
    """
    output_path = Path(config.output_dir) / "training_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": config.model_name,
                "train_file": config.train_file,
                "val_file": config.val_file,
                "output_dir": config.output_dir,
                "max_length": config.max_length,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "seed": config.seed,
                "label2id": label2id,
            },
            f,
            indent=2,
        )


def parse_args() -> TrainingConfig:
    """Parse CLI arguments.

    Returns:
        TrainingConfig: Parsed configuration.
    """
    parser = argparse.ArgumentParser(description="Train transformer for Irembo intents")
    parser.add_argument(
        "--model-name",
        type=str,
        default="Davlan/afro-xlmr-base",
        help="Pretrained transformer model name",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/processed/train_hf.json",
        help="Training data JSON file",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="data/processed/val_hf.json",
        help="Validation data JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/transformer_intent",
        help="Output directory for model artifacts",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return TrainingConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )


def main() -> None:
    """Main entry point for training."""
    config = parse_args()

    # Load data
    train_data = load_hf_json(config.train_file)
    val_data = load_hf_json(config.val_file)

    label2id = train_data.get("label2id", {})
    id2label = train_data.get("id2label", {})

    if not label2id:
        raise ValueError("label2id missing from training data.")

    train_dataset = build_dataset(train_data)
    val_dataset = build_dataset(val_data)

    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={int(k): v for k, v in id2label.items()} if id2label else None,
    )

    # Tokenize datasets
    train_dataset = tokenize_dataset(train_dataset, tokenizer, config.max_length)
    val_dataset = tokenize_dataset(val_dataset, tokenizer, config.max_length)

    # Set format for PyTorch
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=20,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        seed=config.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and save
    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    save_training_config(config, label2id)

    print(f"✅ Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
