"""
Transformer Evaluation Script for Irembo Voice AI Intent Classification.

Evaluates a fine-tuned transformer model on the test split and reports:
- Overall accuracy and macro F1
- Per-language accuracy (en, rw, mixed)
"""

# Standard library imports
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

# Third-party imports
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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


def compute_metrics(predictions, labels) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        predictions: Predicted label ids.
        labels: Ground-truth label ids.

    Returns:
        Dict[str, float]: Metrics dictionary.
    """
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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate transformer for Irembo intents")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/transformer_intent",
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/processed/test_hf.json",
        help="Test data JSON file",
    )
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Main entry point for evaluation."""
    args = parse_args()

    if not Path(args.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")

    # Load test data
    test_data = load_hf_json(args.test_file)
    test_dataset = build_dataset(test_data)

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    # Tokenize
    test_dataset = tokenize_dataset(test_dataset, tokenizer, args.max_length)
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label", "language"])

    # Evaluate
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)

    logits = predictions.predictions
    preds = logits.argmax(axis=-1)
    labels = predictions.label_ids

    metrics = compute_metrics(preds, labels)

    print("=" * 70)
    print("IREMBO VOICE AI - TRANSFORMER EVALUATION")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")

    # Per-language accuracy
    print("\nPer-Language Accuracy:")
    languages = test_dataset["language"]
    lang_stats = {}
    for i, lang in enumerate(languages):
        if lang not in lang_stats:
            lang_stats[lang] = {"correct": 0, "total": 0}
        lang_stats[lang]["total"] += 1
        if preds[i] == labels[i]:
            lang_stats[lang]["correct"] += 1

    for lang, stats in sorted(lang_stats.items()):
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        lang_name = {"en": "English", "rw": "Kinyarwanda", "mixed": "Code-switched"}.get(lang, lang)
        print(f"  {lang_name:20} {stats['correct']:3}/{stats['total']:3} ({acc:5.1f}%)")

    print("\n✅ Transformer evaluation complete.")


if __name__ == "__main__":
    main()
