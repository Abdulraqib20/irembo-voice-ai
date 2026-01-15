"""
Baseline (Rule-Based) Classifier Evaluation Script for Irembo Voice AI

Evaluates the keyword-based rule classifier on the Irembo intent dataset.
This serves as a fast baseline for comparison with LLM and transformer models.
"""

import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.rule_based_classifier import RuleBasedClassifier
from src.domain.entities.intent_schema import IntentType


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
            })
    return samples


def loadDatasetJsonl(filePath: str) -> List[Dict[str, Any]]:
    """Load JSONL format dataset"""
    samples = []
    with open(filePath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
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


def evaluate(classifier, samples: List[Dict[str, Any]]):
    """Evaluate rule-based classifier on samples"""
    correct = 0
    total = len(samples)
    
    intentStats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total': 0})
    languageStats = defaultdict(lambda: {'correct': 0, 'total': 0})
    totalLatency = 0

    for sample in samples:
        text = sample.get('text', sample.get('query', ''))
        language = sample.get('language', 'en')
        
        result = classifier.classify(text, language)
        predicted = result.intent.value
        actual = sample['intent']
        totalLatency += result.latency_ms

        intentStats[actual]['total'] += 1
        languageStats[language]['total'] += 1
        
        if predicted == actual:
            correct += 1
            intentStats[actual]['tp'] += 1
            languageStats[language]['correct'] += 1
        else:
            intentStats[actual]['fn'] += 1
            intentStats[predicted]['fp'] += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    avgLatency = totalLatency / total if total > 0 else 0

    print(f"\n{'='*70}")
    print(f"IREMBO VOICE AI - RULE-BASED BASELINE EVALUATION")
    print(f"{'='*70}\n")
    print(f"📊 OVERALL METRICS")
    print(f"   Total Samples: {total}")
    print(f"   Correct: {correct}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Avg Latency: {avgLatency:.4f}ms (very fast!)\n")
    
    print(f"🌍 PER-LANGUAGE ACCURACY")
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
    numIntents = 0
    
    for intent in sorted(intentStats.keys()):
        stats = intentStats[intent]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        support = stats['total']
        
        if support == 0:
            continue
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = computeF1(precision, recall)
        
        macroF1 += f1
        numIntents += 1
        
        print(f"{intent:40} {precision:6.2f} {recall:6.2f} {f1:6.2f} {support:8}")
    
    macroF1 = macroF1 / numIntents if numIntents > 0 else 0
    
    print("-" * 70)
    print(f"{'Macro F1':40} {'':>6} {'':>6} {macroF1:6.2f}")
    print(f"\n{'='*70}\n")

    return {
        'accuracy': accuracy,
        'macro_f1': macroF1,
        'avg_latency_ms': avgLatency,
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
        sys.exit(1)

    print(f"Loading test dataset from {testFile}...")
    testSamples = loadDataset(testFile)
    print(f"Loaded {len(testSamples)} test samples\n")

    print("Initializing Rule-Based classifier...")
    classifier = RuleBasedClassifier()

    print("Running evaluation...")
    results = evaluate(classifier, testSamples)

    print(f"✅ Evaluation complete!")
    print(f"   Baseline Accuracy: {results['accuracy']:.2f}%")
    print(f"   Macro F1: {results['macro_f1']:.4f}")
