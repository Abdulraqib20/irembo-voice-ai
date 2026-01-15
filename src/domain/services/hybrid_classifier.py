"""
Hybrid Intent Classifier for Irembo Voice AI
Combines rule-based (fast) and LLM-based (accurate) classification.
Uses confidence-based routing for optimal accuracy/latency tradeoff.
"""

import time
from typing import Optional
from src.domain.services.rule_based_classifier import RuleBasedClassifier
from src.domain.services.groq_classifier import GroqClassifier
from src.domain.entities.intent_schema import IntentResult, IntentType, ConfidenceThreshold


class HybridClassifier:
    """
    Hybrid classifier that routes between rule-based and LLM classifiers.
    
    Strategy:
    - High confidence (>= 0.85): Trust rule-based immediately
    - Medium confidence (0.70-0.85): Trust rule-based (fast path)
    - Low confidence (< 0.70): Use LLM fallback
    - Very low confidence (< 0.40): Recommend escalation to agent
    """
    
    def __init__(self, useGroq: bool = True):
        self.ruleClassifier = RuleBasedClassifier()
        self.groqClassifier = None
        self.useGroq = useGroq
        
        if useGroq:
            try:
                self.groqClassifier = GroqClassifier()
            except ValueError as e:
                print(f"Warning: Groq disabled - {e}")
                self.useGroq = False

    def classify(self, query: str, language: str = "en") -> IntentResult:
        """
        Classify user intent using hybrid approach.
        
        Args:
            query: User utterance text
            language: Language code (en, rw, mixed)
            
        Returns:
            IntentResult with predicted intent, confidence, and method used
        """
        startTime = time.time()
        
        # First try rule-based classification
        ruleResult = self.ruleClassifier.classify(query, language)

        # High confidence - trust rule-based immediately
        if ruleResult.confidence >= ConfidenceThreshold.HIGH:
            ruleResult.method = "hybrid_rule"
            return ruleResult

        # Groq disabled - return rule-based result
        if not self.useGroq or not self.groqClassifier:
            return ruleResult

        # Low confidence - use LLM fallback
        if ruleResult.confidence < ConfidenceThreshold.MEDIUM:
            try:
                groqResult = self.groqClassifier.classify(query, language)
                
                # If LLM has higher confidence, use it
                if groqResult.confidence > ruleResult.confidence:
                    groqResult.method = "hybrid_llm"
                    return groqResult
                    
            except Exception as e:
                print(f"Groq fallback failed: {e}")
            
            # LLM failed or had lower confidence - return rule-based
            return ruleResult

        # Medium confidence - trust rule-based (fast path)
        ruleResult.method = "hybrid_rule"
        return ruleResult

    def classifyWithFallback(self, query: str, language: str = "en") -> tuple[IntentResult, bool]:
        """
        Classify with explicit escalation recommendation.
        
        Returns:
            Tuple of (IntentResult, should_escalate_to_agent)
        """
        result = self.classify(query, language)
        shouldEscalate = result.should_escalate()
        return result, shouldEscalate
