"""
Rule-Based Intent Classifier for Irembo Voice AI
Keyword-based classification for Kinyarwanda/English multilingual utterances.
Designed as fast baseline with explicit patterns for government service intents.
"""

from typing import Dict, List, Tuple, Optional, Any
import re
import time
import logging
from src.domain.entities.intent_schema import IntentType, IntentResult, ConfidenceThreshold

logger = logging.getLogger(__name__)


class RuleBasedClassifier:
    """
    Rule-based intent classifier using keyword matching.
    Handles Kinyarwanda, English, and code-switched utterances.
    """
    
    def __init__(self):
        # Intent keywords for all 13 intents
        # Includes English, Kinyarwanda, and common code-switched patterns
        self.intentKeywords: Dict[IntentType, List[str]] = {
            IntentType.CHECK_APPLICATION_STATUS: [
                # English
                "status", "progress", "where is my", "check status",
                "submitted", "application status", "track",
                # Kinyarwanda
                "igeze", "aho igeze", "yarangiye", "kureba status",
                "kumenya aho", "sinzi aho", "case yanjye",
                # Mixed patterns
                "status ya application", "check niba",
            ],
            
            IntentType.START_NEW_APPLICATION: [
                # English
                "start new", "new application", "begin application",
                "apply for", "how do i apply", "help me apply",
                "i want to apply", "start application",
                # Kinyarwanda
                "gusaba", "ntangira he", "gukora application",
                "gutangira gusaba", "nshaka gusaba", "ndashaka gukora",
                "nkeneye gukora", "application nshya",
                # Mixed
                "help me gutangira", "new application ya",
            ],
            
            IntentType.REQUIREMENTS_INFORMATION: [
                # English
                "requirements", "what documents", "what do i need",
                "documents needed", "what are the requirements",
                # Kinyarwanda
                "amabwiriza", "ibisabwa", "nkeneye iki", "ngomba kuzana",
                "ni iki ngomba", "documents nkeneye", "ni izihe",
                # Mixed
                "ni izihe requirements", "what documents nkeneye",
            ],
            
            IntentType.FEES_INFORMATION: [
                # English
                "how much", "fee", "cost", "price", "payment amount",
                "what is the fee", "fees",
                # Kinyarwanda
                "amafaranga", "ikiguzi", "angahe", "bisaba amafaranga",
                "kumenya ikiguzi",
                # Mixed
                "how much ni",
            ],
            
            IntentType.APPOINTMENT_BOOKING: [
                # English
                "book appointment", "schedule", "booking",
                "make appointment", "reserve",
                # Kinyarwanda
                "gufata appointment", "gufata igihe", "rendez-vous",
                "booking yigihe", "nashaka gufata",
                # Mixed
                "help me book", "book rendez-vous",
            ],
            
            IntentType.CANCEL_OR_RESCHEDULE_APPOINTMENT: [
                # English
                "cancel appointment", "reschedule", "change appointment",
                "cancel my appointment", "change time",
                # Kinyarwanda
                "guhindura igihe", "guhagarika", "guhindura appointment",
                "gusiba rendez-vous",
                # Mixed
                "cancel appointment yanjye",
            ],
            
            IntentType.PAYMENT_HELP: [
                # English
                "payment failed", "paid but", "not confirmed",
                "how to pay", "payment issue", "payment problem",
                "my payment", "i paid",
                # Kinyarwanda
                "nishyuye", "kwishyura", "ntibyemejwe", "payment yaranze",
                "uburyo bwo kwishyura", "ntabasha kwishyura",
                # Mixed
                "nishyuye but", "payment ntabwo",
            ],
            
            IntentType.RESET_PASSWORD_LOGIN_HELP: [
                # English
                "password", "login", "cannot log in", "otp",
                "forgot password", "reset password", "sign in",
                "log in", "account access",
                # Kinyarwanda
                "kwinjira", "password yibagiwe", "ntabasha kwinjira",
                "guhindura password", "otp ntabwo",
                # Mixed
                "reset password yanjye", "help me reset",
            ],
            
            IntentType.DOCUMENT_UPLOAD_HELP: [
                # English
                "upload", "attachment", "upload failing", "attach document",
                "cannot upload", "file upload",
                # Kinyarwanda
                "kongera ifoto", "gushyiraho", "attachment ntabwo",
                "scan", "nashaka kongera",
                # Mixed
                "ntabasha gushyiraho attachment",
            ],
            
            IntentType.UPDATE_APPLICATION_DETAILS: [
                # English
                "update", "correct", "made a mistake", "change details",
                "modify application", "edit application",
                # Kinyarwanda
                "guhindura amakuru", "gukosorora",
                # Mixed
                "update details za", "nkeneye update",
            ],
            
            IntentType.SERVICE_ELIGIBILITY: [
                # English
                "eligible", "eligibility", "who can apply", "am i eligible",
                "can i apply", "eligibility rules", "qualified",
                # Kinyarwanda
                "nemerewe", "wemerewe", "ni nde wemerewe",
                "iba isabwa", "ni iki cyiciro",
                # Mixed
                "eligibility ya", "eligibility ni iyihe",
            ],
            
            IntentType.SPEAK_TO_AGENT: [
                # English
                "speak to agent", "talk to agent", "human agent",
                "customer support", "talk to someone", "real person",
                "connect me", "speak to someone",
                # Kinyarwanda
                "kuvugana n'umukozi", "mumpuze", "umuntu umbwire",
                "ndashaka kuvugana",
                # Mixed
                "mumpuze n'agent",
            ],
            
            IntentType.COMPLAINT_OR_SUPPORT_TICKET: [
                # English
                "complaint", "problem", "issue", "support ticket",
                "raise complaint", "file complaint", "help with problem",
                # Kinyarwanda
                "ikibazo", "kugaragaza ikibazo", "gutanga ikibazo",
                "ubufasha", "yarandemereye", "mwandikire ticket",
                # Mixed
                "open ticket", "please open ticket",
            ],
        }

        # Rwandan phone pattern (07X or +250)
        self.phonePattern = re.compile(r'\b07[2389]\d{7}\b|\+250[7][2389]\d{7}\b')
        
        # Amount pattern (RWF currency)
        self.amountPattern = re.compile(r'(?:RWF|Frw|FRW)?\s?\d{1,3}(?:[,.\s]\d{3})*(?:\.\d{2})?')
        
        # Application number pattern
        self.appNumberPattern = re.compile(r'\b[A-Z]{2,4}[-/]?\d{6,10}\b', re.IGNORECASE)
        
        # Service name extraction patterns
        self.servicePatterns = {
            "passport": re.compile(r'\b(passport|pasiporo)\b', re.IGNORECASE),
            "driving_license": re.compile(r'\b(driving\s*license|permis|driver\'?s?\s*license)\b', re.IGNORECASE),
            "birth_certificate": re.compile(r'\b(birth\s*certificate|icyemezo\s*cy\'?amavuko)\b', re.IGNORECASE),
            "marriage_certificate": re.compile(r'\b(marriage\s*certificate|icyemezo\s*cy\'?ubukwe)\b', re.IGNORECASE),
            "id_replacement": re.compile(r'\b(id\s*replacement|indangamuntu|gusimbuza\s*indangamuntu|id\s*card)\b', re.IGNORECASE),
            "tax_clearance": re.compile(r'\b(tax\s*clearance|attestation\s*y\'?imisoro)\b', re.IGNORECASE),
            "land_title": re.compile(r'\b(land\s*title|icyangombwa\s*cy\'?ubutaka)\b', re.IGNORECASE),
            "business_registration": re.compile(r'\b(business\s*registration|kwandikisha\s*ubucuruzi)\b', re.IGNORECASE),
            "visa_on_arrival": re.compile(r'\b(visa(\s*on\s*arrival)?|visa\s*yo\s*kuza)\b', re.IGNORECASE),
            "esia": re.compile(r'\besia\b', re.IGNORECASE),
        }

    def extractEntities(self, query: str) -> Dict[str, Any]:
        """Extract entities from utterance"""
        entities = {}
        
        # Extract phone number
        phoneMatch = self.phonePattern.search(query)
        if phoneMatch:
            entities["phone_number"] = phoneMatch.group()
        
        # Extract amount
        amountMatch = self.amountPattern.search(query)
        if amountMatch:
            entities["amount"] = amountMatch.group()
        
        # Extract application number
        appMatch = self.appNumberPattern.search(query)
        if appMatch:
            entities["application_number"] = appMatch.group()
        
        # Extract service name
        for serviceName, pattern in self.servicePatterns.items():
            if pattern.search(query):
                entities["service_name"] = serviceName
                break
        
        return entities

    def classify(self, query: str, language: str = "en") -> IntentResult:
        """
        Classify user intent based on keyword matching.
        
        Args:
            query: User utterance text
            language: Language code (en, rw, mixed)
            
        Returns:
            IntentResult with predicted intent and confidence
        """
        startTime = time.time()
        queryLower = query.lower().strip()

        # Score each intent based on keyword matches
        scores: Dict[IntentType, int] = {}
        for intent, keywords in self.intentKeywords.items():
            score = 0
            for keyword in keywords:
                if keyword in queryLower:
                    score += 1
            scores[intent] = score

        # Find best match
        if not scores or max(scores.values()) == 0:
            topIntent = IntentType.UNKNOWN
            confidence = 0.0
        else:
            topIntent = max(scores.items(), key=lambda x: x[1])[0]
            maxScore = scores[topIntent]

            # Confidence scaling based on keyword matches
            if maxScore == 1:
                confidence = 0.50  # Single keyword - low confidence
            elif maxScore == 2:
                confidence = 0.72  # Two keywords - medium confidence
            elif maxScore == 3:
                confidence = 0.85  # Three keywords - high confidence
            elif maxScore >= 4:
                confidence = 0.92  # Four+ keywords - very high confidence
            else:
                confidence = 0.0

        # Extract entities
        entities = self.extractEntities(query)
        
        # Boost confidence if relevant entities found
        if entities.get("service_name") and topIntent in [
            IntentType.REQUIREMENTS_INFORMATION,
            IntentType.FEES_INFORMATION,
            IntentType.START_NEW_APPLICATION,
            IntentType.APPOINTMENT_BOOKING,
        ]:
            confidence = min(0.95, confidence + 0.05)

        latency = (time.time() - startTime) * 1000

        return IntentResult(
            intent=topIntent,
            confidence=confidence,
            entities=entities,
            language=language,
            method="rule_based",
            latency_ms=round(latency, 2)
        )

    def batchClassify(self, queries: List[str], language: str = "en") -> List[IntentResult]:
        """Classify multiple queries"""
        return [self.classify(query, language) for query in queries]
