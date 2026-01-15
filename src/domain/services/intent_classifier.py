"""
Intent Classification Service for Irembo Voice AI
Provides basic keyword-based intent classification.
For advanced classification, use HybridClassifier instead.

NOTE: This is a legacy service. Prefer using RuleBasedClassifier 
or HybridClassifier for new implementations.
"""

from src.domain.core_entities import Language, Intent


class IntentClassificationService:
    """Legacy business logic for intent classification"""

    # Kinyarwanda/English keyword patterns for all 13 Irembo intents
    INTENT_KEYWORDS = {
        "check_application_status": [
            "status", "progress", "where is my", "igeze", "aho igeze", 
            "yarangiye", "kureba status", "check niba"
        ],
        "start_new_application": [
            "start", "new application", "apply", "gusaba", "ntangira he",
            "gukora application", "gutangira"
        ],
        "requirements_information": [
            "requirements", "documents", "what do i need", "amabwiriza",
            "ibisabwa", "ngomba kuzana", "nkeneye"
        ],
        "fees_information": [
            "how much", "fee", "cost", "price", "amafaranga", "ikiguzi", "angahe"
        ],
        "appointment_booking": [
            "book", "appointment", "schedule", "gufata appointment",
            "gufata igihe", "rendez-vous", "booking"
        ],
        "cancel_or_reschedule_appointment": [
            "cancel", "reschedule", "change appointment", "guhindura igihe",
            "guhagarika"
        ],
        "payment_help": [
            "payment", "paid", "not confirmed", "nishyuye", "kwishyura",
            "ntibyemejwe", "payment failed"
        ],
        "reset_password_login_help": [
            "password", "login", "otp", "cannot log in", "kwinjira",
            "password yibagiwe", "reset password"
        ],
        "document_upload_help": [
            "upload", "attachment", "kongera ifoto", "gushyiraho"
        ],
        "update_application_details": [
            "update", "correct", "made a mistake", "guhindura amakuru",
            "modify"
        ],
        "service_eligibility": [
            "eligible", "eligibility", "who can apply", "nemerewe",
            "wemerewe", "can i apply"
        ],
        "speak_to_agent": [
            "speak to agent", "talk to agent", "human", "customer support",
            "kuvugana n'umukozi", "mumpuze"
        ],
        "complaint_or_support_ticket": [
            "complaint", "problem", "issue", "support ticket", "ikibazo",
            "kugaragaza ikibazo", "ubufasha"
        ],
    }

    @staticmethod
    def classifyIntent(message: str, language: Language = Language.ENGLISH) -> Intent:
        """Classify user intent from message"""
        messageLower = message.lower()
        detectedIntent = "unknown"
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

        # Extract amounts (RWF currency context)
        amounts = []
        for word in message.split():
            cleanWord = word.replace(",", "").replace("RWF", "").replace("Frw", "")
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
