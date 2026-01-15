"""
LLM-Based Paraphrasing Generator
Uses Groq to generate natural variations of template-generated queries
"""

import json
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
import os

try:
    from groq import Groq
except ImportError:
    print("Warning: Groq library not installed. Install with: pip install groq")
    Groq = None

class LLMParaphraser:
    """Generate natural paraphrases using LLM"""

    def __init__(self, apiKey: Optional[str] = None):
        if Groq is None:
            raise ImportError("Groq library required. Install with: pip install groq")

        self.apiKey = apiKey or os.getenv("GROQ_API_KEY")
        if not self.apiKey:
            raise ValueError("GROQ_API_KEY not found in environment or constructor")

        self.client = Groq(api_key=self.apiKey)
        self.model = "openai/gpt-oss-120b"
        self.paraphrasedSamples = []

    def paraphrase(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        numVariations: int = 5,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        prompt = self._buildParaphrasePrompt(query, intent, entities, numVariations, language)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data augmentation expert for Nigerian financial services."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,
                max_tokens=500,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            variations = result.get("variations", [])

            paraphrased = []
            for var in variations:
                paraphrased.append({
                    "query": var.get("query", ""),
                    "intent": intent,
                    "entities": entities,
                    "language": language,
                    "source": "paraphrased",
                    "original_query": query,
                    "generated_at": datetime.utcnow().isoformat()
                })

            self.paraphrasedSamples.extend(paraphrased)
            return paraphrased

        except Exception as e:
            print(f"Error paraphrasing query '{query}': {e}")
            return []

    def _buildParaphrasePrompt(
        self,
        query: str,
        intent: str,
        entities: Dict[str, Any],
        numVariations: int,
        language: str
    ) -> str:
        entityStr = json.dumps(entities, indent=2) if entities else "None"

        prompt = f"""Generate {numVariations} natural variations of this financial query.

Original Query: "{query}"
Intent: {intent}
Entities: {entityStr}
Language: {language}

Requirements:
1. Keep the same intent and ALL entity values EXACTLY as they are
2. Use natural Nigerian English or Pidgin expressions
3. Include both formal and informal variations
4. Vary sentence structure and word order
5. Make queries sound like real user questions
6. DO NOT change any numbers, names, or entity values
7. Keep queries between 5-20 words

Example variations should include:
- Question format: "Can you...", "How do I..."
- Statement format: "I want to...", "I need to..."
- Informal: "Abeg help me...", "Make I..."

Return ONLY valid JSON in this exact format:
{{
  "variations": [
    {{"query": "first variation here"}},
    {{"query": "second variation here"}},
    {{"query": "third variation here"}},
    {{"query": "fourth variation here"}},
    {{"query": "fifth variation here"}}
  ]
}}"""

        return prompt

    async def paraphraseBatch(
        self,
        samples: List[Dict[str, Any]],
        variationsPerSample: int = 3,
        batchSize: int = 10
    ) -> List[Dict[str, Any]]:
        allParaphrased = []

        for i in range(0, len(samples), batchSize):
            batch = samples[i:i + batchSize]
            print(f"Processing batch {i // batchSize + 1} ({len(batch)} samples)...")

            batchResults = []
            for sample in batch:
                paraphrased = self.paraphrase(
                    query=sample["query"],
                    intent=sample["intent"],
                    entities=sample.get("entities", {}),
                    numVariations=variationsPerSample,
                    language=sample.get("language", "en")
                )
                batchResults.extend(paraphrased)

                await asyncio.sleep(0.1)

            allParaphrased.extend(batchResults)
            print(f"  ✓ Generated {len(batchResults)} paraphrases")

        return allParaphrased

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

        print(f"✓ Saved {len(samples)} paraphrased samples to {outputFile}")

    def getStatistics(self) -> Dict[str, Any]:
        if not self.paraphrasedSamples:
            return {"total_samples": 0}

        intentCounts = {}
        for sample in self.paraphrasedSamples:
            intent = sample["intent"]
            intentCounts[intent] = intentCounts.get(intent, 0) + 1

        avgQueryLength = sum(
            len(s["query"].split()) for s in self.paraphrasedSamples
        ) / len(self.paraphrasedSamples)

        uniqueQueries = len(set(s["query"] for s in self.paraphrasedSamples))

        return {
            "total_samples": len(self.paraphrasedSamples),
            "intent_counts": intentCounts,
            "avg_query_length_words": avgQueryLength,
            "unique_queries": uniqueQueries,
            "uniqueness_ratio": uniqueQueries / len(self.paraphrasedSamples)
        }

async def main():
    inputFile = "data/raw/generated/all_intents.jsonl"
    outputFile = "data/raw/paraphrased/all_intents_paraphrased.jsonl"

    print("=" * 60)
    print("LLM-Based Paraphrasing Generator")
    print("=" * 60)

    if not os.path.exists(inputFile):
        print(f"\nError: Input file not found: {inputFile}")
        print("Please run template_generator.py first to generate base samples.")
        return

    paraphraser = LLMParaphraser()

    print(f"\nLoading samples from {inputFile}...")
    samples = paraphraser.loadSamplesFromJsonl(inputFile)
    print(f"  ✓ Loaded {len(samples)} samples")

    sampleSubset = samples[:50]
    print(f"\nGenerating paraphrases for {len(sampleSubset)} samples...")
    print("(Using subset for demo - remove limit for full dataset)")

    paraphrased = await paraphraser.paraphraseBatch(
        sampleSubset,
        variationsPerSample=3,
        batchSize=10
    )

    print(f"\nSaving paraphrased samples...")
    paraphraser.saveToJsonl(paraphrased, outputFile)

    print("\n" + "=" * 60)
    print("Paraphrasing Statistics")
    print("=" * 60)

    stats = paraphraser.getStatistics()
    print(f"\nTotal Paraphrased Samples: {stats['total_samples']}")
    print(f"Unique Queries: {stats['unique_queries']}")
    print(f"Uniqueness Ratio: {stats['uniqueness_ratio']:.2%}")
    print(f"Avg Query Length: {stats['avg_query_length_words']:.1f} words")

    print("\nIntent Distribution:")
    for intent, count in sorted(stats['intent_counts'].items()):
        print(f"  {intent:25s}: {count:3d} samples")

    print("\n✓ Paraphrasing complete!")

if __name__ == "__main__":
    asyncio.run(main())
