import os
import time
import json
from typing import List
from groq import Groq
from dotenv import load_dotenv
from src.domain.entities.intent_schema import IntentType, IntentResult

class GroqClassifier:
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        # Load environment variables from .env file
        load_dotenv(override=True)

        apiKey = os.getenv("GROQ_API_KEY")
        if not apiKey:
            raise ValueError("GROQ_API_KEY not set")

        self.client = Groq(api_key=apiKey)
        self.model = model
        self.intents = [
            "check_balance", "send_money", "pay_bill", "savings_goal",
            "transaction_history", "get_loan", "financial_education",
            "customer_support", "greeting", "unknown"
        ]
        self.lastCallTime = 0
        self.minDelay = 0.5

    def classify(self, query: str, language: str = "en") -> IntentResult:
        elapsed = time.time() - self.lastCallTime
        if elapsed < self.minDelay:
            time.sleep(self.minDelay - elapsed)
        startTime = time.time()

        languageNames = {
            'en': 'English',
            'yo': 'Yoruba',
            'ha': 'Hausa',
            'ig': 'Igbo',
            'pcm': 'Nigerian Pidgin'
        }
        detectedLanguage = languageNames.get(language, 'English')

        prompt = f"""You are a multilingual financial assistant for Nigerian users.

Classify this query into ONE intent: {', '.join(self.intents)}

Query: "{query}"
Language: {detectedLanguage}

Context:
- Understand Nigerian languages: English, Yoruba, Hausa, Igbo, Nigerian Pidgin
- Handle code-switching (queries mixing multiple languages)
- Recognize cultural context and local payment methods (MTN, Airtel, NEPA, etc.)

Examples:
- "Wetin be my balance?" → check_balance
- "I wan send money" → send_money
- "Bawo ni balance mi?" → check_balance
- "Ina son in aika kuɗi" → send_money
- "Kedu ego m?" → check_balance

IMPORTANT: Respond with ONLY valid JSON:
{{"intent": "intent_name", "confidence": 0.85}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response")

            content = content.strip()

            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))

            intentType = IntentType(intent)

        except Exception as e:
            print(f"Error: {e}")
            intentType = IntentType.UNKNOWN
            confidence = 0.0

        self.lastCallTime = time.time()
        latency = (time.time() - startTime) * 1000

        return IntentResult(
            intent=intentType,
            confidence=confidence,
            entities={},
            language=language,
            method="groq_llm",
            latency_ms=round(latency, 2)
        )

    def batchClassify(self, queries: List[str], language: str = "en") -> List[IntentResult]:
        return [self.classify(query, language) for query in queries]
