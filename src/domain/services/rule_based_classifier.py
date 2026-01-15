from typing import Dict, List, Tuple
import re
import time
import logging
from src.domain.entities.intent_schema import IntentType, IntentResult

logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    def __init__(self):
        self.intentKeywords = {

        }

        self.phonePattern = re.compile(r'\b0[7-9]\d{9}\b|\+234[7-9]\d{9}\b')
        self.amountPattern = re.compile(r'₦?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?')

    def classify(self, query: str, language: str = "en") -> IntentResult:
        startTime = time.time()
        query = query.lower().strip()

        scores = {}
        for intent, keywords in self.intentKeywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            scores[intent] = score

        maxScore = max(scores.values())
        if maxScore == 0:
            topIntent = IntentType.UNKNOWN
            confidence = 0.0
        else:
            topIntent = max(scores.items(), key=lambda x: x[1])[0]

            # Improved confidence calculation
            # Base confidence scales better with keyword matches
            if maxScore == 1:
                confidence = 0.45  # Single keyword match - low confidence
            elif maxScore == 2:
                confidence = 0.72  # Two keywords - medium confidence
            elif maxScore >= 3:
                confidence = 0.88  # Three+ keywords - high confidence
            else:
                confidence = 0.0

        hasPhone = bool(self.phonePattern.search(query))
        hasAmount = bool(self.amountPattern.search(query))

        sendPattern = re.compile(r'\b(send|transfer|give)\b.*\b(to|give)\b', re.IGNORECASE)
        hasSendPattern = bool(sendPattern.search(query))

        latency = (time.time() - startTime) * 1000

        return IntentResult(
            intent=topIntent,
            confidence=confidence,
            entities={},
            language=language,
            method="rule_based",
            latency_ms=round(latency, 2)
        )

    def batchClassify(self, queries: List[str], language: str = "en") -> List[IntentResult]:
        return [self.classify(query, language) for query in queries]
