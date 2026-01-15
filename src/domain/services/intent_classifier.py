"""
Intent Classification Service for financial intents.

Provides basic keyword-based intent classification.
For advanced classification, use HybridClassifier instead.
"""

from src.domain.core_entities import Language, Intent


class IntentClassificationService:
    """Business logic for intent classification"""

    INTENT_KEYWORDS = {
        "balance_inquiry": ["balance", "account", "how much", "money", "owo mi da"],
        "transfer": ["send", "transfer", "pay", "fi ranṣẹ", "aika"],
        "bill_payment": ["bill", "electricity", "water", "nepa", "ina", "dstv", "gotv"],
        "vtu_airtime": ["buy airtime", "purchase airtime", "recharge", "top up", "load airtime", "airtime for", "buy credit"],
        "vtu_data": ["buy data", "purchase data", "data bundle", "data plan", "gb data", "mb data", "need data"],
        "loan": ["loan", "borrow", "awin"],
        "savings": ["save", "saving", "ifowopamọ"],
        "advice": ["help", "advice", "guide", "iranlọwọ"]
    }

    @staticmethod
    def classifyIntent(message: str, language: Language = Language.ENGLISH) -> Intent:
        """Classify user intent from message"""
        messageLower = message.lower()
        detectedIntent = "general_inquiry"
        confidence = 0.5
        entities = {}

        for intent, keywords in IntentClassificationService.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in messageLower:
                    detectedIntent = intent
                    confidence = 0.8
                    break
            if confidence > 0.7:
                break

        # Extract amounts
        amounts = []
        for word in message.split():
            cleanWord = word.replace(",", "").replace("₦", "")
            try:
                amount = float(cleanWord)
                amounts.append(amount)
            except ValueError:
                continue

        if amounts:
            entities["amount"] = amounts[0]

        return Intent(
            intent_type=detectedIntent,
            confidence=confidence,
            entities=entities,
            raw_message=message,
            language=language
        )
