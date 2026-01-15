"""
Dataset Splitter
Stratified train/val/test splitting with balance verification
"""

import json
import random
from typing import Dict, List, Any, Tuple
from pathlib import Path
from collections import defaultdict

class DatasetSplitter:
    """Split dataset into train/val/test with stratification"""

    def __init__(self, randomSeed: int = 42):
        self.randomSeed = randomSeed
        random.seed(randomSeed)

    def stratifiedSplit(
        self,
        samples: List[Dict[str, Any]],
        trainRatio: float = 0.7,
        valRatio: float = 0.15,
        testRatio: float = 0.15
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if abs(trainRatio + valRatio + testRatio - 1.0) > 0.01:
            raise ValueError("Split ratios must sum to 1.0")

        grouped = self._groupByIntentAndLanguage(samples)

        trainData = []
        valData = []
        testData = []

        for group in grouped.values():
            random.shuffle(group)

            n = len(group)
            trainEnd = int(n * trainRatio)
            valEnd = trainEnd + int(n * valRatio)

            trainData.extend(group[:trainEnd])
            valData.extend(group[trainEnd:valEnd])
            testData.extend(group[valEnd:])

        random.shuffle(trainData)
        random.shuffle(valData)
        random.shuffle(testData)

        return trainData, valData, testData

    def _groupByIntentAndLanguage(
        self,
        samples: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        grouped = defaultdict(list)

        for sample in samples:
            intent = sample.get("intent", "unknown")
            language = sample.get("language", "en")
            key = f"{intent}_{language}"
            grouped[key].append(sample)

        return grouped

    def verifyBalance(
        self,
        trainData: List[Dict[str, Any]],
        valData: List[Dict[str, Any]],
        testData: List[Dict[str, Any]],
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        totalSamples = len(trainData) + len(valData) + len(testData)

        checks = {
            "total_samples": totalSamples,
            "train_samples": len(trainData),
            "val_samples": len(valData),
            "test_samples": len(testData),
            "split_ratios": {
                "train": len(trainData) / totalSamples,
                "val": len(valData) / totalSamples,
                "test": len(testData) / totalSamples
            },
            "intent_distribution": self._checkIntentBalance(
                trainData, valData, testData, tolerance
            ),
            "language_distribution": self._checkLanguageBalance(
                trainData, valData, testData, tolerance
            ),
            "no_data_leakage": self._checkNoOverlap(trainData, valData, testData),
            "all_checks_passed": True
        }

        if not checks["intent_distribution"]["balanced"]:
            checks["all_checks_passed"] = False
            print("⚠️  Warning: Intent distribution imbalanced")

        if not checks["language_distribution"]["balanced"]:
            checks["all_checks_passed"] = False
            print("⚠️  Warning: Language distribution imbalanced")

        if not checks["no_data_leakage"]:
            checks["all_checks_passed"] = False
            print("⚠️  Warning: Data leakage detected between splits")

        return checks

    def _checkIntentBalance(
        self,
        trainData: List[Dict[str, Any]],
        valData: List[Dict[str, Any]],
        testData: List[Dict[str, Any]],
        tolerance: float
    ) -> Dict[str, Any]:
        trainIntents = self._countIntents(trainData)
        valIntents = self._countIntents(valData)
        testIntents = self._countIntents(testData)

        allIntents = set(trainIntents.keys()) | set(valIntents.keys()) | set(testIntents.keys())

        distribution = {}
        balanced = True

        for intent in allIntents:
            trainCount = trainIntents.get(intent, 0)
            valCount = valIntents.get(intent, 0)
            testCount = testIntents.get(intent, 0)
            total = trainCount + valCount + testCount

            if total > 0:
                trainPct = trainCount / total
                valPct = valCount / total
                testPct = testCount / total

                intentBalanced = (
                    abs(trainPct - 0.7) <= tolerance and
                    abs(valPct - 0.15) <= tolerance and
                    abs(testPct - 0.15) <= tolerance
                )

                distribution[intent] = {
                    "train": trainCount,
                    "val": valCount,
                    "test": testCount,
                    "train_pct": trainPct,
                    "val_pct": valPct,
                    "test_pct": testPct,
                    "balanced": intentBalanced
                }

                if not intentBalanced:
                    balanced = False

        return {
            "balanced": balanced,
            "per_intent": distribution
        }

    def _checkLanguageBalance(
        self,
        trainData: List[Dict[str, Any]],
        valData: List[Dict[str, Any]],
        testData: List[Dict[str, Any]],
        tolerance: float
    ) -> Dict[str, Any]:
        trainLangs = self._countLanguages(trainData)
        valLangs = self._countLanguages(valData)
        testLangs = self._countLanguages(testData)

        allLangs = set(trainLangs.keys()) | set(valLangs.keys()) | set(testLangs.keys())

        distribution = {}
        balanced = True

        for lang in allLangs:
            trainCount = trainLangs.get(lang, 0)
            valCount = valLangs.get(lang, 0)
            testCount = testLangs.get(lang, 0)
            total = trainCount + valCount + testCount

            if total > 0:
                trainPct = trainCount / total
                valPct = valCount / total
                testPct = testCount / total

                langBalanced = (
                    abs(trainPct - 0.7) <= tolerance and
                    abs(valPct - 0.15) <= tolerance and
                    abs(testPct - 0.15) <= tolerance
                )

                distribution[lang] = {
                    "train": trainCount,
                    "val": valCount,
                    "test": testCount,
                    "train_pct": trainPct,
                    "val_pct": valPct,
                    "test_pct": testPct,
                    "balanced": langBalanced
                }

                if not langBalanced:
                    balanced = False

        return {
            "balanced": balanced,
            "per_language": distribution
        }

    def _checkNoOverlap(
        self,
        trainData: List[Dict[str, Any]],
        valData: List[Dict[str, Any]],
        testData: List[Dict[str, Any]]
    ) -> bool:
        trainQueries = set(s["query"] for s in trainData)
        valQueries = set(s["query"] for s in valData)
        testQueries = set(s["query"] for s in testData)

        trainValOverlap = trainQueries & valQueries
        trainTestOverlap = trainQueries & testQueries
        valTestOverlap = valQueries & testQueries

        if trainValOverlap or trainTestOverlap or valTestOverlap:
            print(f"Data leakage detected:")
            print(f"  Train-Val overlap: {len(trainValOverlap)} samples")
            print(f"  Train-Test overlap: {len(trainTestOverlap)} samples")
            print(f"  Val-Test overlap: {len(valTestOverlap)} samples")
            return False

        return True

    def _countIntents(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = defaultdict(int)
        for sample in samples:
            counts[sample.get("intent", "unknown")] += 1
        return dict(counts)

    def _countLanguages(self, samples: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = defaultdict(int)
        for sample in samples:
            counts[sample.get("language", "en")] += 1
        return dict(counts)

    def loadSamplesFromJsonl(self, filePath: str) -> List[Dict[str, Any]]:
        samples = []
        with open(filePath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    def saveToJsonl(self, samples: List[Dict[str, Any]], outputPath: str):
        outputFile = Path(outputPath)
        outputFile.parent.mkdir(parents=True, exist_ok=True)

        with open(outputFile, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    def saveSplits(
        self,
        trainData: List[Dict[str, Any]],
        valData: List[Dict[str, Any]],
        testData: List[Dict[str, Any]],
        outputDir: str
    ):
        outputPath = Path(outputDir)
        outputPath.mkdir(parents=True, exist_ok=True)

        self.saveToJsonl(trainData, str(outputPath / "train.jsonl"))
        self.saveToJsonl(valData, str(outputPath / "val.jsonl"))
        self.saveToJsonl(testData, str(outputPath / "test.jsonl"))

        print(f"\n✓ Saved splits to {outputDir}/")
        print(f"  - train.jsonl: {len(trainData)} samples")
        print(f"  - val.jsonl: {len(valData)} samples")
        print(f"  - test.jsonl: {len(testData)} samples")

def main():
    inputFile = "data/raw/generated/all_intents.jsonl"
    outputDir = "data/processed"

    print("=" * 60)
    print("Dataset Splitter")
    print("=" * 60)

    import os
    if not os.path.exists(inputFile):
        print(f"\nError: Input file not found: {inputFile}")
        print("Please run template_generator.py first.")
        return

    splitter = DatasetSplitter(randomSeed=42)

    print(f"\nLoading samples from {inputFile}...")
    samples = splitter.loadSamplesFromJsonl(inputFile)
    print(f"  ✓ Loaded {len(samples)} samples")

    print("\nPerforming stratified split (70/15/15)...")
    trainData, valData, testData = splitter.stratifiedSplit(
        samples,
        trainRatio=0.7,
        valRatio=0.15,
        testRatio=0.15
    )

    print("\nVerifying balance...")
    checks = splitter.verifyBalance(trainData, valData, testData)

    print("\n" + "=" * 60)
    print("Split Statistics")
    print("=" * 60)

    print(f"\nTotal Samples: {checks['total_samples']}")
    print(f"Train Samples: {checks['train_samples']} ({checks['split_ratios']['train']:.1%})")
    print(f"Val Samples: {checks['val_samples']} ({checks['split_ratios']['val']:.1%})")
    print(f"Test Samples: {checks['test_samples']} ({checks['split_ratios']['test']:.1%})")

    print("\nIntent Distribution Check:")
    intentDist = checks['intent_distribution']
    print(f"  Balanced: {'✓' if intentDist['balanced'] else '✗'}")

    print("\nLanguage Distribution Check:")
    langDist = checks['language_distribution']
    print(f"  Balanced: {'✓' if langDist['balanced'] else '✗'}")

    print("\nData Leakage Check:")
    print(f"  No overlap: {'✓' if checks['no_data_leakage'] else '✗'}")

    print(f"\nOverall: {'✓ All checks passed' if checks['all_checks_passed'] else '✗ Some checks failed'}")

    print("\nSaving splits...")
    splitter.saveSplits(trainData, valData, testData, outputDir)

    metadataPath = Path(outputDir) / "metadata.json"
    with open(metadataPath, 'w', encoding='utf-8') as f:
        json.dump(checks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to {metadataPath}")

    print("\n✓ Dataset splitting complete!")

if __name__ == "__main__":
    main()
