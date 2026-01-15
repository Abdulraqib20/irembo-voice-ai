"""
Domain Entities Package
Contains intent schema, entity definitions, and extraction utilities
"""

from .intent_schema import (
    IntentType,
    EntityType,
    BillType,
    BillProvider,
    IntentDefinition,
    IntentResult,
    ConfidenceThreshold,
    INTENT_SCHEMA,
    getIntentExamples,
    getAllIntents,
    getIntentDefinition,
    requiresAuth,
    requiresConfirmation
)

from .entity_extraction import (
    EntityExtractor,
    EntityValidator,
    normalizePhone,
    formatAmount,
    parseAmount
)

# Import from core_entities.py (renamed to avoid naming conflict with entities/ folder)
from ..core_entities import (
    Language,
    TransactionType,
    TransactionCategory,
    TransactionStatus,
    IDType,
    User,
    Wallet,
    Transaction,
    Beneficiary,
    AuditLog,
    Referral,
    Bill,
    ConversationMessage,
    Intent,
    FinancialAdvice,
    VoiceSession
)

__all__ = [
    "IntentType",
    "EntityType",
    "BillType",
    "BillProvider",
    "IntentDefinition",
    "IntentResult",
    "ConfidenceThreshold",
    "INTENT_SCHEMA",
    "getIntentExamples",
    "getAllIntents",
    "getIntentDefinition",
    "requiresAuth",
    "requiresConfirmation",
    "EntityExtractor",
    "EntityValidator",
    "normalizePhone",
    "formatAmount",
    "parseAmount",
    "Language",
    "TransactionType",
    "TransactionCategory",
    "TransactionStatus",
    "IDType",
    "User",
    "Wallet",
    "Transaction",
    "Beneficiary",
    "AuditLog",
    "Referral",
    "Bill",
    "ConversationMessage",
    "Intent",
    "FinancialAdvice",
    "VoiceSession"
]
