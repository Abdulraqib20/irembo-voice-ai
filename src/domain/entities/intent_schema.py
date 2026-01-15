"""
Intent Classification Schema for Irembo Voice AI
Defines all supported government service intents, entities, and confidence thresholds
for Kinyarwanda/English multilingual Voice AI system.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class IntentType(Enum):
    """
    13 intent types for Irembo government services Voice AI.
    Covers citizen interactions for applications, payments, appointments, and support.
    """
    # Application & Status
    CHECK_APPLICATION_STATUS = "check_application_status"
    START_NEW_APPLICATION = "start_new_application"
    UPDATE_APPLICATION_DETAILS = "update_application_details"
    
    # Information Requests
    REQUIREMENTS_INFORMATION = "requirements_information"
    FEES_INFORMATION = "fees_information"
    SERVICE_ELIGIBILITY = "service_eligibility"
    
    # Appointments
    APPOINTMENT_BOOKING = "appointment_booking"
    CANCEL_OR_RESCHEDULE_APPOINTMENT = "cancel_or_reschedule_appointment"
    
    # Payment & Technical Help
    PAYMENT_HELP = "payment_help"
    RESET_PASSWORD_LOGIN_HELP = "reset_password_login_help"
    DOCUMENT_UPLOAD_HELP = "document_upload_help"
    
    # Support & Escalation
    SPEAK_TO_AGENT = "speak_to_agent"
    COMPLAINT_OR_SUPPORT_TICKET = "complaint_or_support_ticket"
    
    # Fallback
    UNKNOWN = "unknown"


class EntityType(Enum):
    """Entity types that can be extracted from user utterances"""
    # Service Types
    SERVICE_NAME = "service_name"  # passport, visa, birth_certificate, etc.
    
    # Application Details
    APPLICATION_NUMBER = "application_number"
    APPLICATION_ID = "application_id"
    
    # Location & Time
    LOCATION = "location"  # Kigali, Huye, Rusizi, etc.
    DATE = "date"
    TIME = "time"
    
    # Document Types
    DOCUMENT_TYPE = "document_type"
    
    # Payment
    AMOUNT = "amount"
    PAYMENT_METHOD = "payment_method"  # mobile_money, bank, etc.
    
    # User Info
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"


class ConfidenceThreshold:
    """Confidence thresholds for classification decisions"""
    HIGH = 0.85      # Rule-based trusted, no fallback needed
    MEDIUM = 0.70    # Acceptable confidence
    LOW = 0.50       # Requires LLM fallback
    ESCALATE = 0.40  # Escalate to human agent


@dataclass
class IntentResult:
    """Result from intent classification"""
    intent: IntentType
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    method: str = "unknown"  # rule_based, groq_llm, transformer, hybrid
    latency_ms: float = 0.0
    
    def should_escalate(self) -> bool:
        """Check if confidence is too low and should escalate to agent"""
        return self.confidence < ConfidenceThreshold.ESCALATE
    
    def needs_fallback(self) -> bool:
        """Check if LLM fallback is needed"""
        return self.confidence < ConfidenceThreshold.MEDIUM


@dataclass
class IntentDefinition:
    """Definition of an intent with examples and requirements"""
    intent_type: IntentType
    description: str
    examples_en: List[str]
    examples_rw: List[str]  # Kinyarwanda examples
    examples_mixed: List[str]  # Code-switched examples
    required_entities: List[EntityType]
    optional_entities: List[EntityType]
    requires_auth: bool
    requires_confirmation: bool
    handler_function: str


# Intent definitions with Kinyarwanda examples
INTENT_DEFINITIONS: Dict[IntentType, IntentDefinition] = {
    IntentType.CHECK_APPLICATION_STATUS: IntentDefinition(
        intent_type=IntentType.CHECK_APPLICATION_STATUS,
        description="User wants to check the status of an existing application",
        examples_en=[
            "What is the status of my application?",
            "Where is my passport application now?",
            "I want to check the status of my application.",
            "Can you tell me the progress of my ESIA application?",
        ],
        examples_rw=[
            "Ndashaka kumenya aho application igeze.",
            "Ese passport yanije igeze he?",
            "Natanze application ariko sinzi aho igeze.",
            "Checka niba application yanjye yarangiye.",
        ],
        examples_mixed=[
            "Ndashaka kureba status ya application yanije.",
            "I submitted application number ariko status sinayibona.",
            "Ndashaka kureba status ya application yanjye ya ESIA.",
        ],
        required_entities=[],
        optional_entities=[EntityType.APPLICATION_NUMBER, EntityType.SERVICE_NAME],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="check_application_status",
    ),
    IntentType.START_NEW_APPLICATION: IntentDefinition(
        intent_type=IntentType.START_NEW_APPLICATION,
        description="User wants to start a new application for a service",
        examples_en=[
            "I want to start a new application for passport.",
            "How do I begin an application for driving license?",
            "Help me apply for tax clearance.",
        ],
        examples_rw=[
            "Nshaka gusaba passport, ntangira he?",
            "Nkeneye gukora application nshya ya birth certificate.",
            "Ndashaka gutangira gusaba attestation y'imisoro.",
        ],
        examples_mixed=[
            "Help me gutangira application ya passport.",
            "Ndashaka gukora new application ya marriage certificate.",
        ],
        required_entities=[EntityType.SERVICE_NAME],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="start_new_application",
    ),
    IntentType.REQUIREMENTS_INFORMATION: IntentDefinition(
        intent_type=IntentType.REQUIREMENTS_INFORMATION,
        description="User wants to know requirements for a service",
        examples_en=[
            "What are the requirements for passport?",
            "What documents do I need for visa on arrival?",
        ],
        examples_rw=[
            "Ni iki ngomba kuzana kugira ngo nsabe passport?",
            "Amabwiriza yo gusaba driving license ni ayahe?",
            "Ni izihe documents nkeneye kuri visa yo kuza?",
        ],
        examples_mixed=[
            "Ni izihe requirements za passport?",
            "What documents nkeneye kuri ESIA?",
        ],
        required_entities=[EntityType.SERVICE_NAME],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="get_requirements",
    ),
    IntentType.FEES_INFORMATION: IntentDefinition(
        intent_type=IntentType.FEES_INFORMATION,
        description="User wants to know fees/costs for a service",
        examples_en=[
            "How much does passport cost?",
            "What is the fee for driving license?",
            "Please tell me the payment amount for ESIA.",
        ],
        examples_rw=[
            "Nshaka kumenya ikiguzi cya passport.",
            "Ese birth certificate bisaba amafaranga angahe?",
            "Amafaranga ya kwandikisha ubucuruzi ni angahe?",
        ],
        examples_mixed=[
            "How much ni passport?",
        ],
        required_entities=[EntityType.SERVICE_NAME],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="get_fees",
    ),
    IntentType.APPOINTMENT_BOOKING: IntentDefinition(
        intent_type=IntentType.APPOINTMENT_BOOKING,
        description="User wants to book an appointment",
        examples_en=[
            "I want to book an appointment for passport.",
            "Help me schedule an appointment.",
        ],
        examples_rw=[
            "Nashaka gufata appointment kuri passport.",
            "Mumfashishe gufata igihe cyo marriage certificate.",
            "Ndashaka booking yigihe kuri gusimbuza indangamuntu.",
        ],
        examples_mixed=[
            "Help me book rendez-vous for ID replacement.",
            "Nashaka gufata appointment ya ESIA i Muhanga.",
        ],
        required_entities=[EntityType.SERVICE_NAME],
        optional_entities=[EntityType.LOCATION, EntityType.DATE, EntityType.TIME],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="book_appointment",
    ),
    IntentType.CANCEL_OR_RESCHEDULE_APPOINTMENT: IntentDefinition(
        intent_type=IntentType.CANCEL_OR_RESCHEDULE_APPOINTMENT,
        description="User wants to cancel or reschedule an appointment",
        examples_en=[
            "Please cancel my appointment for driving license.",
            "Can I change the appointment time?",
        ],
        examples_rw=[
            "Nashaka guhindura igihe cya rendez-vous.",
            "Ndashaka guhagarika appointment yanjye.",
        ],
        examples_mixed=[
            "Cancel appointment yanjye please.",
        ],
        required_entities=[],
        optional_entities=[EntityType.SERVICE_NAME, EntityType.DATE],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="cancel_reschedule_appointment",
    ),
    IntentType.PAYMENT_HELP: IntentDefinition(
        intent_type=IntentType.PAYMENT_HELP,
        description="User needs help with payment issues",
        examples_en=[
            "I paid but it is not confirmed.",
            "My payment failed for birth certificate.",
            "How do I pay for passport on mobile money?",
        ],
        examples_rw=[
            "Nishyuye ariko ntibyemejwe.",
            "Payment yaranze kuri passport.",
            "Uburyo bwo kwishyura permis ni ubuhe?",
        ],
        examples_mixed=[
            "Nishyuye but ntabwo byemejwe.",
            "How do I pay for passport on mobile money?",
        ],
        required_entities=[],
        optional_entities=[EntityType.SERVICE_NAME, EntityType.AMOUNT, EntityType.PAYMENT_METHOD],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="payment_help",
    ),
    IntentType.RESET_PASSWORD_LOGIN_HELP: IntentDefinition(
        intent_type=IntentType.RESET_PASSWORD_LOGIN_HELP,
        description="User needs help with password reset or login issues",
        examples_en=[
            "I cannot log in to my account.",
            "The OTP is not coming through.",
            "I forgot my password.",
        ],
        examples_rw=[
            "Ntabasha kwinjira muri account yanjye.",
            "OTP ntabwo igeze.",
            "Nabonye ikibazo cyo kwinjira, password yibagiwe.",
            "Nshaka guhindura password yanjye.",
        ],
        examples_mixed=[
            "Help me reset password yanjye.",
            "OTP ntabwo igeze, what can I do?",
        ],
        required_entities=[],
        optional_entities=[EntityType.EMAIL, EntityType.PHONE_NUMBER],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="reset_password_help",
    ),
    IntentType.DOCUMENT_UPLOAD_HELP: IntentDefinition(
        intent_type=IntentType.DOCUMENT_UPLOAD_HELP,
        description="User needs help uploading documents",
        examples_en=[
            "The attachment upload is failing.",
            "I need to update my application details.",
        ],
        examples_rw=[
            "Nashaka kongera ifoto/scan kuri request.",
            "Ntabasha gushyiraho attachment.",
        ],
        examples_mixed=[
            "Ntabasha gushyiraho attachment kuri case.",
        ],
        required_entities=[],
        optional_entities=[EntityType.DOCUMENT_TYPE, EntityType.APPLICATION_NUMBER],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="document_upload_help",
    ),
    IntentType.UPDATE_APPLICATION_DETAILS: IntentDefinition(
        intent_type=IntentType.UPDATE_APPLICATION_DETAILS,
        description="User wants to update details on an existing application",
        examples_en=[
            "I made a mistake in my application. How do I correct it?",
            "I need to update my application details.",
        ],
        examples_rw=[
            "Nashaka guhindura amakuru muri application.",
        ],
        examples_mixed=[
            "Nkeneye update details za application yanjye.",
        ],
        required_entities=[],
        optional_entities=[EntityType.APPLICATION_NUMBER],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="update_application",
    ),
    IntentType.SERVICE_ELIGIBILITY: IntentDefinition(
        intent_type=IntentType.SERVICE_ELIGIBILITY,
        description="User wants to know eligibility criteria for a service",
        examples_en=[
            "Am I eligible to apply for driving license?",
            "Who can apply for passport?",
            "What are the eligibility rules for marriage certificate?",
        ],
        examples_rw=[
            "Ese nemerewe gusaba passport?",
            "Ni nde wemerewe attestation y'imisoro?",
        ],
        examples_mixed=[
            "Eligibility ya birth certificate ni iyihe?",
        ],
        required_entities=[EntityType.SERVICE_NAME],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="check_eligibility",
    ),
    IntentType.SPEAK_TO_AGENT: IntentDefinition(
        intent_type=IntentType.SPEAK_TO_AGENT,
        description="User wants to speak to a human agent",
        examples_en=[
            "I want to talk to an agent.",
            "Can I speak to customer support?",
            "Connect me to a human.",
        ],
        examples_rw=[
            "Ndashaka kuvugana n'umukozi.",
            "Mumpuze n'umuntu umbwire neza.",
        ],
        examples_mixed=[
            "Mumpuze n'agent please.",
        ],
        required_entities=[],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="escalate_to_agent",
    ),
    IntentType.COMPLAINT_OR_SUPPORT_TICKET: IntentDefinition(
        intent_type=IntentType.COMPLAINT_OR_SUPPORT_TICKET,
        description="User wants to file a complaint or support ticket",
        examples_en=[
            "I need help with a problem on passport.",
            "I want to raise a complaint about visa on arrival.",
        ],
        examples_rw=[
            "Ndashaka kugaragaza ikibazo cyabaye kuri passport.",
            "Serivisi ya ESIA yarandemereye, ndasaba ubufasha.",
            "Mwandikire ticket y'ubufasha kuri ESIA.",
            "Ndashaka gutanga ikibazo/complaint kuri permis.",
        ],
        examples_mixed=[
            "Please open ticket, ikibazo cya ID replacement.",
        ],
        required_entities=[],
        optional_entities=[EntityType.SERVICE_NAME, EntityType.APPLICATION_NUMBER],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="create_support_ticket",
    ),
}


# Service name mappings for entity extraction
SERVICE_NAME_MAPPINGS = {
    # English
    "passport": "passport",
    "visa": "visa_on_arrival",
    "visa on arrival": "visa_on_arrival",
    "driving license": "driving_license",
    "driver's license": "driving_license",
    "birth certificate": "birth_certificate",
    "marriage certificate": "marriage_certificate",
    "id replacement": "id_replacement",
    "id card": "id_replacement",
    "tax clearance": "tax_clearance",
    "land title": "land_title",
    "business registration": "business_registration",
    "esia": "esia",
    
    # Kinyarwanda / French borrowings
    "pasiporo": "passport",
    "permis": "driving_license",
    "icyemezo cy'amavuko": "birth_certificate",
    "icyemezo cy'ubukwe": "marriage_certificate",
    "indangamuntu": "id_replacement",
    "gusimbuza indangamuntu": "id_replacement",
    "attestation y'imisoro": "tax_clearance",
    "icyangombwa cy'ubutaka": "land_title",
    "kwandikisha ubucuruzi": "business_registration",
}
