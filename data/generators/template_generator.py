"""
Template-Based Dataset Generator
Generates synthetic training data using predefined templates and slot filling
"""

import json
import random
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime

class TemplateGenerator:
    """Generate synthetic data samples using template-based slot filling"""

    def __init__(self, templatesPath: str):
        self.templatesPath = Path(templatesPath)
        self.templates = self._loadTemplates()
        self.generatedSamples = []

    def _loadTemplates(self) -> Dict[str, Any]:
        with open(self.templatesPath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generateSamples(
        self,
        intent: str,
        numSamples: int,
        language: str = "en"
    ) -> List[Dict[str, Any]]:
        if intent not in self.templates:
            raise ValueError(f"Intent '{intent}' not found in templates")

        intentData = self.templates[intent]
        templates = intentData["templates"]
        entities = intentData.get("entities", {})

        samples = []

        for i in range(numSamples):
            template = random.choice(templates)
            sample = self._fillTemplate(template, entities, intent, language)
            if sample:
                samples.append(sample)

        self.generatedSamples.extend(samples)
        return samples

    def _fillTemplate(
        self,
        template: str,
        entities: Dict[str, List],
        intent: str,
        language: str
    ) -> Optional[Dict[str, Any]]:
        query = template
        extractedEntities = {}

        if "{amount}" in template and "amount" in entities:
            amount = random.choice(entities["amount"])
            query = query.replace("{amount}", f"₦{amount:,}")
            extractedEntities["amount"] = amount
            extractedEntities["currency"] = "NGN"

        if "{recipient}" in template:
            if "recipient_names" in entities and random.random() > 0.5:
                recipient = random.choice(entities["recipient_names"])
                query = query.replace("{recipient}", recipient)
                extractedEntities["recipient_name"] = recipient
            elif "phone_numbers" in entities:
                phone = random.choice(entities["phone_numbers"])
                query = query.replace("{recipient}", phone)
                extractedEntities["recipient_phone"] = self._normalizePhone(phone)

        if "{name}" in template and "recipient_names" in entities:
            name = random.choice(entities["recipient_names"])
            query = query.replace("{name}", name)
            extractedEntities["recipient_name"] = name

        if "{phone}" in template and "phone_numbers" in entities:
            phone = random.choice(entities["phone_numbers"])
            query = query.replace("{phone}", phone)
            extractedEntities["recipient_phone"] = self._normalizePhone(phone)

        if "{bill_type}" in template and "bill_types" in entities:
            billType = random.choice(entities["bill_types"])
            query = query.replace("{bill_type}", billType)
            extractedEntities["bill_type"] = billType

        if "{provider}" in template and "providers" in entities:
            provider = random.choice(entities["providers"])
            query = query.replace("{provider}", provider)
            extractedEntities["provider"] = provider

        if "{goal}" in template and "goals" in entities:
            goal = random.choice(entities["goals"])
            query = query.replace("{goal}", goal)
            extractedEntities["goal_name"] = goal

        if "{frequency}" in template and "frequency" in entities:
            frequency = random.choice(entities["frequency"])
            query = query.replace("{frequency}", frequency)
            extractedEntities["frequency"] = frequency

        if "{time_period}" in template and "time_periods" in entities:
            timePeriod = random.choice(entities["time_periods"])
            query = query.replace("{time_period}", timePeriod)
            extractedEntities["time_period"] = timePeriod

        if "{purpose}" in template and "purposes" in entities:
            purpose = random.choice(entities["purposes"])
            query = query.replace("{purpose}", purpose)
            extractedEntities["loan_purpose"] = purpose

        if "{topic}" in template and "topics" in entities:
            topic = random.choice(entities["topics"])
            query = query.replace("{topic}", topic)
            extractedEntities["topic"] = topic

        if "{issue}" in template and "issues" in entities:
            issue = random.choice(entities["issues"])
            query = query.replace("{issue}", issue)
            extractedEntities["issue_type"] = issue

        return {
            "query": query,
            "intent": intent,
            "entities": extractedEntities,
            "language": language,
            "source": "template",
            "template": template,
            "generated_at": datetime.utcnow().isoformat()
        }

    def _normalizePhone(self, phone: str) -> str:
        cleaned = phone.replace(" ", "").replace("-", "")
        if cleaned.startswith("0"):
            return f"+234{cleaned[1:]}"
        elif cleaned.startswith("234"):
            return f"+{cleaned}"
        else:
            return f"+234{cleaned}"

    def generateAllIntents(
        self,
        samplesPerIntent: int = 100,
        language: str = "en"
    ) -> Dict[str, List[Dict[str, Any]]]:
        allSamples = {}

        for intent in self.templates.keys():
            if intent == "unknown":
                continue

            print(f"Generating {samplesPerIntent} samples for intent: {intent}")
            samples = self.generateSamples(intent, samplesPerIntent, language)
            allSamples[intent] = samples
            print(f"  ✓ Generated {len(samples)} samples")

        return allSamples

    def saveToJsonl(self, samples: List[Dict[str, Any]], outputPath: str):
        outputFile = Path(outputPath)
        outputFile.parent.mkdir(parents=True, exist_ok=True)

        with open(outputFile, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(samples)} samples to {outputFile}")

    def saveAllIntents(self, allSamples: Dict[str, List], outputDir: str):
        outputPath = Path(outputDir)
        outputPath.mkdir(parents=True, exist_ok=True)

        for intent, samples in allSamples.items():
            filename = outputPath / f"{intent}.jsonl"
            self.saveToJsonl(samples, str(filename))

        combinedSamples = []
        for samples in allSamples.values():
            combinedSamples.extend(samples)

        allFilename = outputPath / "all_intents.jsonl"
        self.saveToJsonl(combinedSamples, str(allFilename))

        print(f"\n✓ Total samples generated: {len(combinedSamples)}")

    def getStatistics(self) -> Dict[str, Any]:
        if not self.generatedSamples:
            return {"total_samples": 0}

        intentCounts = {}
        entityCounts = {}
        totalEntities = 0

        for sample in self.generatedSamples:
            intent = sample["intent"]
            intentCounts[intent] = intentCounts.get(intent, 0) + 1

            for entityType in sample["entities"].keys():
                entityCounts[entityType] = entityCounts.get(entityType, 0) + 1
                totalEntities += 1

        avgQueryLength = sum(
            len(s["query"].split()) for s in self.generatedSamples
        ) / len(self.generatedSamples)

        return {
            "total_samples": len(self.generatedSamples),
            "intent_counts": intentCounts,
            "entity_counts": entityCounts,
            "total_entities": totalEntities,
            "avg_entities_per_sample": totalEntities / len(self.generatedSamples),
            "avg_query_length_words": avgQueryLength,
            "unique_queries": len(set(s["query"] for s in self.generatedSamples))
        }

def main():
    templatesPath = "data/raw/templates/intent_templates.json"
    outputDir = "data/raw/generated"

    print("=" * 60)
    print("Template-Based Dataset Generator")
    print("=" * 60)

    generator = TemplateGenerator(templatesPath)

    print("\nGenerating samples for all intents...")
    allSamples = generator.generateAllIntents(samplesPerIntent=100, language="en")

    print("\nSaving to JSONL files...")
    generator.saveAllIntents(allSamples, outputDir)

    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)

    stats = generator.getStatistics()
    print(f"\nTotal Samples: {stats['total_samples']}")
    print(f"Unique Queries: {stats['unique_queries']}")
    print(f"Avg Query Length: {stats['avg_query_length_words']:.1f} words")
    print(f"Avg Entities per Sample: {stats['avg_entities_per_sample']:.2f}")

    print("\nIntent Distribution:")
    for intent, count in sorted(stats['intent_counts'].items()):
        print(f"  {intent:25s}: {count:3d} samples")

    print("\nEntity Distribution:")
    for entity, count in sorted(stats['entity_counts'].items()):
        print(f"  {entity:25s}: {count:3d} occurrences")

    print("\n✓ Dataset generation complete!")

if __name__ == "__main__":
    main()
