import json
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.rule_based_classifier import RuleBasedClassifier
from src.domain.entities.intent_schema import IntentType

def loadDataset(filePath: str):
    samples = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

def evaluate(classifier, samples):
    correct = 0
    total = len(samples)
    intentStats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for sample in samples:
        result = classifier.classify(sample['query'], sample.get('language', 'en'))
        predicted = result.intent.value
        actual = sample['intent']

        intentStats[actual]['total'] += 1

        if predicted == actual:
            correct += 1
            intentStats[actual]['correct'] += 1

    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"RULE-BASED CLASSIFIER EVALUATION")
    print(f"{'='*60}\n")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%\n")
    print(f"{'='*60}")
    print(f"PER-INTENT BREAKDOWN")
    print(f"{'='*60}\n")

    for intent in sorted(intentStats.keys()):
        stats = intentStats[intent]
        intentAccuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        print(f"{intent:25} {stats['correct']:3}/{stats['total']:3} ({intentAccuracy:5.1f}%)")

    print(f"\n{'='*60}\n")

    return accuracy, intentStats

if __name__ == "__main__":
    testFile = "data/processed/test.jsonl"

    if not Path(testFile).exists():
        print(f"Error: {testFile} not found")
        sys.exit(1)

    print("Loading test dataset...")
    testSamples = loadDataset(testFile)
    print(f"Loaded {len(testSamples)} test samples\n")

    print("Initializing classifier...")
    classifier = RuleBasedClassifier()

    print("Running evaluation...")
    accuracy, stats = evaluate(classifier, testSamples)

    print(f"✅ Evaluation complete! Baseline accuracy: {accuracy:.2f}%")
