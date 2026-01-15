#!/usr/bin/env python3
"""
Dataset Generation Pipeline Runner
Executes the full pipeline: Template → LLM → Multilingual → Split
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
projectRoot = Path(__file__).parent.parent
sys.path.insert(0, str(projectRoot))

from src.data.generators.template_generator import TemplateGenerator
from src.data.generators.llm_generator import LLMParaphraser
from src.data.generators.multilingual_generator import MultilingualGenerator
from src.data.processors.splitter import DatasetSplitter


class PipelineRunner:
    """Orchestrates the full dataset generation pipeline"""

    def __init__(self, outputDir: Path):
        self.outputDir = outputDir
        self.outputDir.mkdir(parents=True, exist_ok=True)

        # Pipeline stages
        self.templateGen = TemplateGenerator(
            templatesPath=str(projectRoot / "data" / "raw" / "templates" / "intent_templates.json")
        )

        # Optional stages (only initialize if needed)
        self.llmGen = None
        self.multilingualGen = None
        self.splitter = DatasetSplitter(randomSeed=42)

        # Track pipeline metrics
        self.metrics = {
            "startTime": datetime.now().isoformat(),
            "stages": {},
            "totalSamples": 0,
            "endTime": None,
            "durationSeconds": None
        }

    def runTemplateGeneration(self, samplesPerIntent: int = 100) -> Path:
        """Step 1: Generate base samples from templates"""
        print(f"\n{'='*60}")
        print(f"STAGE 1: Template Generation")
        print(f"{'='*60}")

        stageStart = datetime.now()
        outputPath = self.outputDir / "base_samples.jsonl"

        # Generate all intents
        self.templateGen.generateAllIntents(
            samplesPerIntent=samplesPerIntent
        )

        # Save to JSONL
        self.templateGen.saveToJsonl(self.templateGen.generatedSamples, str(outputPath))

        # Get statistics
        stats = self.templateGen.getStatistics()

        stageDuration = (datetime.now() - stageStart).total_seconds()

        self.metrics["stages"]["template_generation"] = {
            "samplesPerIntent": samplesPerIntent,
            "totalSamples": stats["total_samples"],
            "intentCounts": stats["intent_counts"],
            "uniqueQueries": stats["unique_queries"],
            "uniqueRatio": stats["unique_queries"] / stats["total_samples"] if stats["total_samples"] > 0 else 0,
            "durationSeconds": stageDuration,
            "outputFile": str(outputPath)
        }

        print(f"\n✅ Generated {stats['total_samples']} base samples")
        print(f"⏱️  Duration: {stageDuration:.2f}s")

        return outputPath

    async def runLLMParaphrasing(self, inputPath: Path, paraphrasesPerSample: int = 3) -> Path:
        """Step 2: Generate natural variations using LLM (OPTIONAL)"""
        print(f"\n{'='*60}")
        print(f"STAGE 2: LLM Paraphrasing (Optional)")
        print(f"{'='*60}")

        response = input("\nGenerate LLM paraphrases? (y/n) [costs ~$0.10-0.15]: ").lower()
        if response != 'y':
            print("⏭️  Skipping LLM paraphrasing")
            self.metrics["stages"]["llm_paraphrasing"] = {"skipped": True}
            return inputPath

        # Initialize LLM generator
        if self.llmGen is None:
            try:
                self.llmGen = LLMParaphraser()
            except Exception as e:
                print(f"\n❌ Failed to initialize LLM generator: {e}")
                print("⏭️  Skipping LLM paraphrasing")
                self.metrics["stages"]["llm_paraphrasing"] = {"skipped": True, "error": str(e)}
                return inputPath

        stageStart = datetime.now()
        outputPath = self.outputDir / "paraphrased_samples.jsonl"

        # Load base samples
        samples = []
        with open(inputPath, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        print(f"\nParaphrasing {len(samples)} samples...")
        allSamples = []

        for sample in samples:
            paraphrases = self.llmGen.paraphrase(
                query=sample["query"],
                intent=sample["intent"],
                entities=sample.get("entities", {}),
                numVariations=paraphrasesPerSample,
                language=sample.get("language", "en")
            )
            allSamples.extend(paraphrases)
            await asyncio.sleep(0.1)

        # Save to JSONL
        with open(outputPath, 'w', encoding='utf-8') as f:
            for sample in allSamples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        stageDuration = (datetime.now() - stageStart).total_seconds()

        self.metrics["stages"]["llm_paraphrasing"] = {
            "inputSamples": len(samples),
            "outputSamples": len(allSamples),
            "paraphrasesPerSample": paraphrasesPerSample,
            "durationSeconds": stageDuration,
            "outputFile": str(outputPath)
        }

        print(f"\n✅ Generated {len(allSamples)} paraphrased samples")
        print(f"⏱️  Duration: {stageDuration:.2f}s")

        return outputPath

    def runMultilingualTranslation(self, inputPath: Path) -> Path:
        """Step 3: Translate samples to multiple languages (OPTIONAL)"""
        print(f"\n{'='*60}")
        print(f"STAGE 3: Multilingual Translation (Optional)")
        print(f"{'='*60}")

        response = input("\nTranslate to Yoruba and Hausa? (y/n) [requires googletrans]: ").lower()
        if response != 'y':
            print("⏭️  Skipping multilingual translation")
            self.metrics["stages"]["multilingual_translation"] = {"skipped": True}
            return inputPath

        # Initialize multilingual generator
        if self.multilingualGen is None:
            try:
                self.multilingualGen = MultilingualGenerator()
            except Exception as e:
                print(f"\n❌ Failed to initialize multilingual generator: {e}")
                print("⏭️  Skipping multilingual translation")
                self.metrics["stages"]["multilingual_translation"] = {"skipped": True, "error": str(e)}
                return inputPath

        stageStart = datetime.now()
        outputPath = self.outputDir / "multilingual_samples.jsonl"

        # Load English samples
        samples = []
        with open(inputPath, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        print(f"\nTranslating {len(samples)} samples to Yoruba and Hausa...")

        # Translate to Yoruba
        print("\n📝 Note: Translation may fail due to googletrans library issues.")
        print("   If translation fails, English-only dataset will be used.")

        yorubaSamples = self.multilingualGen.translateBatch(
            samples=samples,
            targetLang="yo"
        )

        # Translate to Hausa
        hausaSamples = self.multilingualGen.translateBatch(
            samples=samples,
            targetLang="ha"
        )

        # Combine all samples
        allSamples = samples + yorubaSamples + hausaSamples

        # If no translations succeeded, just use English
        if len(yorubaSamples) == 0 and len(hausaSamples) == 0:
            print("\n⚠️  Translation failed. Using English-only dataset.")
            print("   This is fine for initial development!")
            allSamples = samples

        # Save multilingual dataset
        self.multilingualGen.saveToJsonl(allSamples, str(outputPath))

        stageDuration = (datetime.now() - stageStart).total_seconds()

        self.metrics["stages"]["multilingual_translation"] = {
            "inputSamples": len(samples),
            "yorubaSamples": len(yorubaSamples),
            "hausaSamples": len(hausaSamples),
            "totalSamples": len(allSamples),
            "durationSeconds": stageDuration,
            "outputFile": str(outputPath)
        }

        print(f"\n✅ Generated {len(allSamples)} multilingual samples")
        print(f"   - English: {len(samples)}")
        print(f"   - Yoruba: {len(yorubaSamples)}")
        print(f"   - Hausa: {len(hausaSamples)}")
        print(f"⏱️  Duration: {stageDuration:.2f}s")

        return outputPath

    def runDatasetSplitting(self, inputPath: Path) -> dict:
        """Step 4: Split into train/val/test with stratification"""
        print(f"\n{'='*60}")
        print(f"STAGE 4: Dataset Splitting")
        print(f"{'='*60}")

        stageStart = datetime.now()

        # Load all samples
        samples = []
        with open(inputPath, 'r', encoding='utf-8') as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        print(f"\nSplitting {len(samples)} samples with stratification...")

        # Perform stratified split
        trainData, valData, testData = self.splitter.stratifiedSplit(
            samples=samples,
            trainRatio=0.7,
            valRatio=0.15,
            testRatio=0.15
        )

        # Save splits to files
        self.splitter.saveSplits(
            trainData=trainData,
            valData=valData,
            testData=testData,
            outputDir=str(self.outputDir)
        )

        # Verify balance
        balanceReport = self.splitter.verifyBalance(trainData, valData, testData)

        stageDuration = (datetime.now() - stageStart).total_seconds()

        self.metrics["stages"]["dataset_splitting"] = {
            "totalSamples": len(samples),
            "trainSamples": len(trainData),
            "valSamples": len(valData),
            "testSamples": len(testData),
            "balanceReport": balanceReport,
            "durationSeconds": stageDuration
        }

        print(f"\n✅ Split complete:")
        print(f"   - Train: {len(trainData)} samples")
        print(f"   - Val: {len(valData)} samples")
        print(f"   - Test: {len(testData)} samples")
        print(f"⏱️  Duration: {stageDuration:.2f}s")

        if not balanceReport["all_checks_passed"]:
            print("\n⚠️  WARNING: Some quality checks failed!")
            if not balanceReport["intent_distribution"]["balanced"]:
                print("   - Intent distribution imbalanced")
            if not balanceReport["language_distribution"]["balanced"]:
                print("   - Language distribution imbalanced")
            if not balanceReport["no_data_leakage"]:
                print("   - Data leakage detected")

        return {
            "train": trainData,
            "val": valData,
            "test": testData,
            "balanceReport": balanceReport
        }

    def savePipelineMetrics(self):
        """Save pipeline execution metrics"""
        self.metrics["endTime"] = datetime.now().isoformat()
        startTime = datetime.fromisoformat(self.metrics["startTime"])
        endTime = datetime.fromisoformat(self.metrics["endTime"])
        self.metrics["durationSeconds"] = (endTime - startTime).total_seconds()

        metricsPath = self.outputDir / "pipeline_metrics.json"
        with open(metricsPath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        print(f"\n📊 Pipeline metrics saved to: {metricsPath}")

    async def runFullPipeline(self, samplesPerIntent: int = 100):
        """Execute the complete pipeline"""
        print(f"\n{'#'*60}")
        print(f"# INTENT CLASSIFICATION DATASET GENERATION PIPELINE")
        print(f"{'#'*60}")
        print(f"\nOutput Directory: {self.outputDir}")
        print(f"Samples Per Intent: {samplesPerIntent}")
        print(f"Start Time: {self.metrics['startTime']}")

        try:
            # Stage 1: Template generation
            baseSamplesPath = self.runTemplateGeneration(samplesPerIntent)

            # Stage 2: LLM paraphrasing (optional)
            paraphrasedPath = await self.runLLMParaphrasing(baseSamplesPath)

            # Stage 3: Multilingual translation (optional)
            multilingualPath = self.runMultilingualTranslation(paraphrasedPath)

            # Stage 4: Dataset splitting
            splitData = self.runDatasetSplitting(multilingualPath)

            # Save metrics
            self.savePipelineMetrics()

            # Print final summary
            self.printSummary()

            print(f"\n{'='*60}")
            print(f"✅ PIPELINE COMPLETE")
            print(f"{'='*60}")

        except Exception as e:
            print(f"\n❌ PIPELINE FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.savePipelineMetrics()
            sys.exit(1)

    def printSummary(self):
        """Print pipeline execution summary"""
        print(f"\n{'='*60}")
        print(f"PIPELINE SUMMARY")
        print(f"{'='*60}")

        # Template generation
        if "template_generation" in self.metrics["stages"]:
            tg = self.metrics["stages"]["template_generation"]
            print(f"\n1. Template Generation:")
            print(f"   - Samples: {tg['totalSamples']}")
            print(f"   - Unique ratio: {tg['uniqueRatio']:.2%}")
            print(f"   - Duration: {tg['durationSeconds']:.2f}s")

        # LLM paraphrasing
        if "llm_paraphrasing" in self.metrics["stages"]:
            lp = self.metrics["stages"]["llm_paraphrasing"]
            if not lp.get("skipped"):
                print(f"\n2. LLM Paraphrasing:")
                print(f"   - Input: {lp['inputSamples']}")
                print(f"   - Output: {lp['outputSamples']}")
                print(f"   - Duration: {lp['durationSeconds']:.2f}s")
            else:
                print(f"\n2. LLM Paraphrasing: SKIPPED")

        # Multilingual translation
        if "multilingual_translation" in self.metrics["stages"]:
            mt = self.metrics["stages"]["multilingual_translation"]
            if not mt.get("skipped"):
                print(f"\n3. Multilingual Translation:")
                print(f"   - English: {mt['inputSamples']}")
                print(f"   - Yoruba: {mt['yorubaSamples']}")
                print(f"   - Hausa: {mt['hausaSamples']}")
                print(f"   - Total: {mt['totalSamples']}")
                print(f"   - Duration: {mt['durationSeconds']:.2f}s")
            else:
                print(f"\n3. Multilingual Translation: SKIPPED")

        # Dataset splitting
        if "dataset_splitting" in self.metrics["stages"]:
            ds = self.metrics["stages"]["dataset_splitting"]
            print(f"\n4. Dataset Splitting:")
            print(f"   - Train: {ds['trainSamples']}")
            print(f"   - Val: {ds['valSamples']}")
            print(f"   - Test: {ds['testSamples']}")
            print(f"   - Balanced: {'✅' if ds['balanceReport'].get('all_checks_passed', False) else '❌'}")
            print(f"   - Duration: {ds['durationSeconds']:.2f}s")

        # Total
        if self.metrics["durationSeconds"]:
            print(f"\n📊 Total Duration: {self.metrics['durationSeconds']:.2f}s")


async def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Generate intent classification dataset")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to generate per intent (default: 100)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for generated data (default: data/processed)"
    )

    args = parser.parse_args()

    # Create pipeline runner
    runner = PipelineRunner(outputDir=args.output)

    # Run pipeline
    await runner.runFullPipeline(samplesPerIntent=args.samples)


if __name__ == "__main__":
    asyncio.run(main())
