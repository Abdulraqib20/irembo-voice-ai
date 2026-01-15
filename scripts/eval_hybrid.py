"""
Hybrid Classifier Evaluation Script for Irembo Voice AI

Evaluates the hybrid (rule-based + LLM) classifier on the Irembo intent dataset.
Reports accuracy, F1 scores, per-intent metrics, and language-specific performance.
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.hybrid_classifier import HybridClassifier


def loadDatasetJsonl(filePath: str) -> List[Dict[str, Any]]:
    """Load JSONL format dataset"""
    samples = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def loadDatasetCsv(filePath: str) -> List[Dict[str, Any]]:
    """Load CSV format dataset"""
    samples = []
    with open(filePath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "id": row.get("utterance_id", ""),
                "text": row.get("utterance_text", ""),
                "intent": row.get("intent", ""),
                "language": row.get("language", "en"),
                "asr_confidence": float(row.get("asr_confidence", 1.0)),
            })
    return samples


def loadDataset(filePath: str) -> List[Dict[str, Any]]:
    """Load dataset from file (auto-detect format)"""
    if filePath.endswith('.jsonl'):
        return loadDatasetJsonl(filePath)
    elif filePath.endswith('.csv'):
        return loadDatasetCsv(filePath)
    else:
        raise ValueError(f"Unsupported file format: {filePath}")


def computeF1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def evaluate(classifier, samples: List[Dict[str, Any]], maxSamples: Optional[int] = None):
    """
    Evaluate classifier on samples.
    
    Returns detailed metrics including:
    - Overall accuracy
    - Per-intent precision/recall/F1
    - Per-language accuracy
    - Latency statistics
    """
    if maxSamples:
        samples = samples[:maxSamples]

    correct = 0
    total = len(samples)
    
    # Per-intent stats
    intentStats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0})
    
    # Per-language stats
    languageStats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Confusion matrix data
    confusionMatrix = defaultdict(lambda: defaultdict(int))
    
    totalLatency = 0
    methodCounts = defaultdict(int)
    
    # Track errors for analysis
    errors = []

    for i, sample in enumerate(samples, 1):
        print(f"Processing {i}/{total}...", end='\r')
        
        text = sample.get('text', sample.get('query', ''))
        language = sample.get('language', 'en')
        
        result = classifier.classify(text, language)
        predicted = result.intent.value
        actual = sample['intent']
        
        totalLatency += result.latency_ms
        methodCounts[result.method] += 1
        
        # Update confusion matrix
        confusionMatrix[actual][predicted] += 1
        
        # Update per-intent stats
        intentStats[actual]['total'] += 1
        if predicted == actual:
            correct += 1
            intentStats[actual]['tp'] += 1
        else:
            intentStats[actual]['fn'] += 1
            intentStats[predicted]['fp'] += 1
            errors.append({
                'text': text,
                'actual': actual,
                'predicted': predicted,
                'confidence': result.confidence,
                'language': language,
            })
        
        # Update per-language stats
        languageStats[language]['total'] += 1
        if predicted == actual:
            languageStats[language]['correct'] += 1

    avgLatency = totalLatency / total if total > 0 else 0
    accuracy = (correct / total) * 100 if total > 0 else 0

    # Print results
    print(f"\n{'='*70}")
    print(f"IREMBO VOICE AI - HYBRID CLASSIFIER EVALUATION")
    print(f"{'='*70}\n")
    
    print(f"📊 OVERALL METRICS")
    print(f"   Total Samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Avg Latency: {avgLatency:.2f}ms\n")
    
    print(f"🔀 METHOD DISTRIBUTION")
    for method, count in sorted(methodCounts.items()):
        pct = (count / total) * 100
        print(f"   {method}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"🌍 PER-LANGUAGE ACCURACY")
    print(f"{'='*70}\n")
    
    for lang in sorted(languageStats.keys()):
        stats = languageStats[lang]
        langAcc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        langName = {'en': 'English', 'rw': 'Kinyarwanda', 'mixed': 'Code-switched'}.get(lang, lang)
        print(f"   {langName:20} {stats['correct']:3}/{stats['total']:3} ({langAcc:5.1f}%)")
    
    print(f"\n{'='*70}")
    print(f"📈 PER-INTENT METRICS")
    print(f"{'='*70}\n")
    print(f"{'Intent':40} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
    print("-" * 70)
    
    macroF1 = 0
    weightedF1 = 0
    totalSupport = 0
    
    for intent in sorted(intentStats.keys()):
        stats = intentStats[intent]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        support = stats['total']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = computeF1(precision, recall)
        
        macroF1 += f1
        weightedF1 += f1 * support
        totalSupport += support
        
        print(f"{intent:40} {precision:6.2f} {recall:6.2f} {f1:6.2f} {support:8}")
    
    numIntents = len(intentStats)
    macroF1 = macroF1 / numIntents if numIntents > 0 else 0
    weightedF1 = weightedF1 / totalSupport if totalSupport > 0 else 0
    
    print("-" * 70)
    print(f"{'Macro Avg':40} {'':>6} {'':>6} {macroF1:6.2f} {total:8}")
    print(f"{'Weighted Avg':40} {'':>6} {'':>6} {weightedF1:6.2f} {total:8}")
    
    print(f"\n{'='*70}\n")
    
    # Show sample errors
    if errors:
        print(f"❌ SAMPLE ERRORS (showing first 5)")
        print("-" * 70)
        for error in errors[:5]:
            print(f"   Text: {error['text'][:60]}...")
            print(f"   Actual: {error['actual']}")
            print(f"   Predicted: {error['predicted']} (conf: {error['confidence']:.2f})")
            print(f"   Language: {error['language']}")
            print()

    return {
        'accuracy': accuracy,
        'macro_f1': macroF1,
        'weighted_f1': weightedF1,
        'avg_latency_ms': avgLatency,
        'per_intent': dict(intentStats),
        'per_language': dict(languageStats),
        'method_distribution': dict(methodCounts),
        'errors': errors,
    }


if __name__ == "__main__":
    # Try processed JSONL first, then fall back to CSV
    testFiles = [
        "data/processed/test.jsonl",
        "data/voiceai_intent_test.csv",
    ]
    
    testFile = None
    for f in testFiles:
        if Path(f).exists():
            testFile = f
            break
    
    if not testFile:
        print(f"Error: No test file found. Tried: {testFiles}")
        print("Run `python scripts/preprocess_irembo.py` first to generate processed files.")
        sys.exit(1)

    print(f"Loading test dataset from {testFile}...")
    testSamples = loadDataset(testFile)
    print(f"Loaded {len(testSamples)} test samples\n")

    print("Initializing Hybrid classifier...")
    try:
        classifier = HybridClassifier(useGroq=True)
        print("  ✓ Groq LLM enabled for fallback")
    except Exception as e:
        print(f"  ⚠ Groq unavailable, using rule-based only: {e}")
        classifier = HybridClassifier(useGroq=False)

    print("\nRunning evaluation...")
    results = evaluate(classifier, testSamples)

    print(f"✅ Evaluation complete!")
    print(f"   Accuracy: {results['accuracy']:.2f}%")
    print(f"   Macro F1: {results['macro_f1']:.4f}")
    print(f"   Weighted F1: {results['weighted_f1']:.4f}")
