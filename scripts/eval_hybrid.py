import json
import sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.hybrid_classifier import HybridClassifier

def loadDataset(filePath: str):
    samples = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples

def evaluate(classifier, samples, maxSamples=None):
    if maxSamples:
        samples = samples[:maxSamples]

    correct = 0
    total = len(samples)
    intentStats = defaultdict(lambda: {'correct': 0, 'total': 0})
    totalLatency = 0
    methodCounts = defaultdict(int)

    for i, sample in enumerate(samples, 1):
        print(f"Processing {i}/{total}...", end='\r')
        result = classifier.classify(sample['query'], sample.get('language', 'en'))
        predicted = result.intent.value
        actual = sample['intent']
        totalLatency += result.latency_ms
        methodCounts[result.method] += 1

        intentStats[actual]['total'] += 1

        if predicted == actual:
            correct += 1
            intentStats[actual]['correct'] += 1

    avgLatency = totalLatency / total if total > 0 else 0
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"HYBRID CLASSIFIER EVALUATION")
    print(f"{'='*60}\n")
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Avg Latency: {avgLatency:.2f}ms\n")
    print(f"Method Distribution:")
    for method, count in methodCounts.items():
        pct = (count / total) * 100
        print(f"  {method}: {count} ({pct:.1f}%)")
    print(f"\n{'='*60}")
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

    print("Initializing Hybrid classifier...")
    classifier = HybridClassifier(useGroq=True)

    print("Running evaluation...")
    accuracy, stats = evaluate(classifier, testSamples)

    print(f"✅ Evaluation complete! Hybrid accuracy: {accuracy:.2f}%")
