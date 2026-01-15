"""
Multilingual Dataset Generator
Translates English samples to Yoruba, Hausa, and Pidgin
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import time

try:
    from googletrans import Translator
except ImportError:
    print("Warning: googletrans not installed. Install with: pip install googletrans==4.0.0-rc1")
    Translator = None

class MultilingualGenerator:
    """Generate multilingual variations using translation"""

    SUPPORTED_LANGUAGES = {
        "en": "English",
        "yo": "Yoruba",
        "ha": "Hausa",
        "pcm": "Nigerian Pidgin"
    }

    def __init__(self):
        if Translator is None:
            raise ImportError("googletrans required. Install with: pip install googletrans==4.0.0-rc1")

        self.translator = Translator()
        self.translatedSamples = []
        self.translationCache = {}

    def translate(
        self,
        query: str,
        sourceLang: str = "en",
        targetLang: str = "yo"
    ) -> Optional[str]:
        cacheKey = f"{sourceLang}:{targetLang}:{query}"

        if cacheKey in self.translationCache:
            return self.translationCache[cacheKey]

        try:
            result = self.translator.translate(
                query,
                src=sourceLang,
                dest=targetLang
            )
            # Handle async result if needed
            if hasattr(result, '__await__'):
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't use await in sync context
                        return None
                    else:
                        result = loop.run_until_complete(result)
                except RuntimeError:
                    # No event loop
                    result = asyncio.run(result)

            translated = result.text if hasattr(result, 'text') else str(result)
            self.translationCache[cacheKey] = translated
            time.sleep(0.1)
            return translated

        except Exception as e:
            # Silently skip translation errors to avoid cluttering output
            return None

    def translateSample(
        self,
        sample: Dict[str, Any],
        targetLang: str
    ) -> Optional[Dict[str, Any]]:
        sourceLang = sample.get("language", "en")

        if sourceLang == targetLang:
            return sample

        translated = self.translate(sample["query"], sourceLang, targetLang)

        if translated:
            return {
                "query": translated,
                "intent": sample["intent"],
                "entities": sample.get("entities", {}),
                "language": targetLang,
                "source": "translated",
                "original_query": sample["query"],
                "original_language": sourceLang,
                "generated_at": datetime.utcnow().isoformat()
            }

        return None

    def translateBatch(
        self,
        samples: List[Dict[str, Any]],
        targetLang: str,
        maxSamples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        translated = []

        samplesToTranslate = samples[:maxSamples] if maxSamples else samples

        print(f"Translating {len(samplesToTranslate)} samples to {self.SUPPORTED_LANGUAGES[targetLang]}...")

        for i, sample in enumerate(samplesToTranslate):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(samplesToTranslate)}")

            result = self.translateSample(sample, targetLang)
            if result:
                translated.append(result)

        self.translatedSamples.extend(translated)
        print(f"  ✓ Translated {len(translated)} samples")

        return translated

    def generateMultilingualDataset(
        self,
        samples: List[Dict[str, Any]],
        targetLanguages: List[str],
        maxSamplesPerLang: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        allTranslations = {}

        for lang in targetLanguages:
            if lang == "en":
                continue

            if lang not in self.SUPPORTED_LANGUAGES:
                print(f"Warning: Language '{lang}' not supported. Skipping.")
                continue

            translations = self.translateBatch(samples, lang, maxSamplesPerLang)
            allTranslations[lang] = translations

        return allTranslations

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

        print(f"✓ Saved {len(samples)} samples to {outputFile}")

    def saveAllTranslations(
        self,
        translations: Dict[str, List[Dict[str, Any]]],
        outputDir: str
    ):
        outputPath = Path(outputDir)
        outputPath.mkdir(parents=True, exist_ok=True)

        for lang, samples in translations.items():
            filename = outputPath / f"all_intents_{lang}.jsonl"
            self.saveToJsonl(samples, str(filename))

        allSamples = []
        for samples in translations.values():
            allSamples.extend(samples)

        if allSamples:
            combinedFile = outputPath / "all_multilingual.jsonl"
            self.saveToJsonl(allSamples, str(combinedFile))

    def getStatistics(self) -> Dict[str, Any]:
        if not self.translatedSamples:
            return {"total_samples": 0}

        langCounts = {}
        intentCounts = {}

        for sample in self.translatedSamples:
            lang = sample["language"]
            intent = sample["intent"]

            langCounts[lang] = langCounts.get(lang, 0) + 1
            intentCounts[intent] = intentCounts.get(intent, 0) + 1

        return {
            "total_samples": len(self.translatedSamples),
            "language_counts": langCounts,
            "intent_counts": intentCounts,
            "cache_size": len(self.translationCache)
        }

def main():
    inputFile = "data/raw/generated/all_intents.jsonl"
    outputDir = "data/augmented"

    print("=" * 60)
    print("Multilingual Dataset Generator")
    print("=" * 60)

    import os
    if not os.path.exists(inputFile):
        print(f"\nError: Input file not found: {inputFile}")
        print("Please run template_generator.py first.")
        return

    generator = MultilingualGenerator()

    print(f"\nLoading English samples from {inputFile}...")
    samples = generator.loadSamplesFromJsonl(inputFile)
    print(f"  ✓ Loaded {len(samples)} samples")

    sampleSubset = samples[:100]
    print(f"\nUsing subset of {len(sampleSubset)} samples for translation")
    print("(Remove limit for full dataset)")

    targetLanguages = ["yo", "ha"]
    print(f"\nTarget languages: {', '.join(generator.SUPPORTED_LANGUAGES[lang] for lang in targetLanguages)}")

    translations = generator.generateMultilingualDataset(
        sampleSubset,
        targetLanguages,
        maxSamplesPerLang=100
    )

    print(f"\nSaving translations...")
    generator.saveAllTranslations(translations, outputDir)

    print("\n" + "=" * 60)
    print("Translation Statistics")
    print("=" * 60)

    stats = generator.getStatistics()
    print(f"\nTotal Translated Samples: {stats['total_samples']}")
    print(f"Translation Cache Size: {stats['cache_size']}")

    print("\nLanguage Distribution:")
    for lang, count in sorted(stats['language_counts'].items()):
        langName = generator.SUPPORTED_LANGUAGES.get(lang, lang)
        print(f"  {langName:20s}: {count:3d} samples")

    print("\nIntent Distribution:")
    for intent, count in sorted(stats['intent_counts'].items()):
        print(f"  {intent:25s}: {count:3d} samples")

    print("\n✓ Translation complete!")
    print("\nNote: Pidgin (pcm) translation requires manual creation or")
    print("specialized models. Consider using LLM-based generation instead.")

if __name__ == "__main__":
    main()
