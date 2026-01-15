import time
from typing import Optional
from src.domain.services.rule_based_classifier import RuleBasedClassifier
from src.domain.services.groq_classifier import GroqClassifier
from src.domain.entities.intent_schema import IntentResult, IntentType, ConfidenceThreshold

class HybridClassifier:
    def __init__(self, useGroq: bool = True):
        self.ruleClassifier = RuleBasedClassifier()
        self.groqClassifier = GroqClassifier() if useGroq else None
        self.useGroq = useGroq

    def classify(self, query: str, language: str = "en") -> IntentResult:
        ruleResult = self.ruleClassifier.classify(query, language)

        # If rule-based has high confidence, trust it immediately
        if ruleResult.confidence >= ConfidenceThreshold.HIGH:
            return ruleResult

        # If Groq is disabled, return rule-based result
        if not self.useGroq or not self.groqClassifier:
            return ruleResult

        # For low confidence (<70%), always use LLM as fallback
        if ruleResult.confidence < ConfidenceThreshold.MEDIUM:
            groqResult = self.groqClassifier.classify(query, language)

            # If LLM has higher confidence, use it
            if groqResult.confidence > ruleResult.confidence:
                return groqResult

            # Otherwise return rule-based result
            return ruleResult

        # For medium confidence (70-85%), trust rule-based
        return ruleResult
