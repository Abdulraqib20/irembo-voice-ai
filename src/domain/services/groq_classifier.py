"""
Groq LLM-Based Intent Classifier for Irembo Voice AI
Uses Groq's fast inference for Kinyarwanda/English intent classification.
Designed as fallback for ambiguous cases that rule-based cannot handle.
"""

import os
import time
import json
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from src.domain.entities.intent_schema import IntentType, IntentResult


class GroqClassifier:
    """
    LLM-based intent classifier using Groq API.
    Handles complex Kinyarwanda/English code-switched utterances.
    """
    
    def __init__(self, model: str = "mixtral-8x7b-32768"):
        # Load environment variables from .env file
        load_dotenv(override=True)

        apiKey = os.getenv("GROQ_API_KEY")
        if not apiKey:
            raise ValueError("GROQ_API_KEY not set in environment")

        self.client = Groq(api_key=apiKey)
        self.model = model
        
        # All 13 Irembo intents
        self.intents = [
            "check_application_status",
            "start_new_application",
            "requirements_information",
            "fees_information",
            "appointment_booking",
            "cancel_or_reschedule_appointment",
            "payment_help",
            "reset_password_login_help",
            "document_upload_help",
            "update_application_details",
            "service_eligibility",
            "speak_to_agent",
            "complaint_or_support_ticket",
        ]
        
        self.lastCallTime = 0
        self.minDelay = 0.3  # Rate limiting

    def _buildPrompt(self, query: str, language: str) -> str:
        """Build the classification prompt with Kinyarwanda context"""
        languageNames = {
            'en': 'English',
            'rw': 'Kinyarwanda',
            'mixed': 'Kinyarwanda-English (code-switched)',
        }
        detectedLanguage = languageNames.get(language, 'English')

        prompt = f"""You are an intent classifier for Irembo, Rwanda's e-government platform Voice AI assistant.
Citizens call to access government services like passport applications, driving licenses, birth certificates, etc.

Your task: Classify the user's query into exactly ONE intent from this list:
{', '.join(self.intents)}

CONTEXT:
- Language detected: {detectedLanguage}
- Users speak Kinyarwanda, English, or mix both (code-switching is common)
- Common services: passport (pasiporo), driving license (permis), birth certificate (icyemezo cy'amavuko), ID replacement (indangamuntu), tax clearance (attestation y'imisoro), visa, ESIA, land title, business registration, marriage certificate

INTENT DEFINITIONS:
- check_application_status: User wants to know the progress/status of an existing application
- start_new_application: User wants to begin a new application for a service
- requirements_information: User asks what documents/requirements are needed
- fees_information: User asks about costs/fees for a service
- appointment_booking: User wants to schedule/book an appointment
- cancel_or_reschedule_appointment: User wants to cancel or change appointment time
- payment_help: User has payment issues (failed payment, not confirmed, how to pay)
- reset_password_login_help: User cannot log in, forgot password, OTP issues
- document_upload_help: User has trouble uploading documents/attachments
- update_application_details: User wants to correct/modify an existing application
- service_eligibility: User asks if they qualify/are eligible for a service
- speak_to_agent: User explicitly wants to talk to a human agent
- complaint_or_support_ticket: User wants to file a complaint or report a problem

KINYARWANDA PATTERNS:
- "Ndashaka" / "Nashaka" = "I want"
- "igeze he?" = "where has it reached?" (status inquiry)
- "aho igeze" = "where it is" (status)
- "amafaranga angahe" = "how much money" (fees)
- "gufata appointment" = "to take/book appointment"
- "ntibyemejwe" = "not confirmed" (payment issues)
- "kwinjira" = "to enter/log in"

EXAMPLES:
- "Ndashaka kureba status ya application yanije." → check_application_status
- "Ni iki ngomba kuzana kugira ngo nsabe passport?" → requirements_information
- "Nishyuye ariko ntibyemejwe." → payment_help
- "Nashaka gufata appointment kuri pasiporo." → appointment_booking
- "OTP ntabwo igeze, what can I do?" → reset_password_login_help
- "Am I eligible to apply for driving license?" → service_eligibility

USER QUERY: "{query}"

Respond with ONLY valid JSON (no markdown, no explanation):
{{"intent": "intent_name", "confidence": 0.85}}"""

        return prompt

    def classify(self, query: str, language: str = "en") -> IntentResult:
        """
        Classify user intent using Groq LLM.
        
        Args:
            query: User utterance text
            language: Language code (en, rw, mixed)
            
        Returns:
            IntentResult with predicted intent and confidence
        """
        # Rate limiting
        elapsed = time.time() - self.lastCallTime
        if elapsed < self.minDelay:
            time.sleep(self.minDelay - elapsed)
        
        startTime = time.time()
        
        prompt = self._buildPrompt(query, language)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from Groq")

            content = content.strip()

            # Clean markdown formatting if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            intentStr = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))

            # Validate intent is in our list
            if intentStr not in self.intents and intentStr != "unknown":
                # Try to match partial intent name
                for validIntent in self.intents:
                    if intentStr in validIntent or validIntent in intentStr:
                        intentStr = validIntent
                        break
                else:
                    intentStr = "unknown"
                    confidence = 0.0

            intentType = IntentType(intentStr)

        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}, content: {content}")
            intentType = IntentType.UNKNOWN
            confidence = 0.0
        except ValueError as e:
            # Intent not in enum
            print(f"Intent value error: {e}")
            intentType = IntentType.UNKNOWN
            confidence = 0.0
        except Exception as e:
            print(f"Groq classification error: {e}")
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
        """Classify multiple queries (sequential due to rate limiting)"""
        return [self.classify(query, language) for query in queries]
