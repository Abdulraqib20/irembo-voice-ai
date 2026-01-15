"""
Domain entities for Agentic Finclusion
Core business objects and value objects

Updated to match banking schema integration (Oct 2025)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    YORUBA = "yo"
    HAUSA = "ha"
    IGBO = "ig"
    PIDGIN = "pcm"

class TransactionType(Enum):
    """Types of financial transactions (banking schema)"""
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"

class TransactionCategory(Enum):
    """Transaction categories"""
    TRANSFER = "Transfer"
    BILLS = "Bills"
    AIRTIME = "Airtime"
    DATA = "Data"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class IDType(Enum):
    """ID document types"""
    NIN = "NIN"
    BVN = "BVN"
    PASSPORT = "Passport"
    DRIVERS_LICENSE = "Driver's License"

@dataclass
class User:
    """
    User entity with KYC information (aligned with banking schema).

    Attributes:
        user_id: Unique user identifier (Integer in DB, can be String in memory)
        first_name: User's first name
        last_name: User's last name
        phone_number: WhatsApp/primary contact number (unique)
        date_of_birth: Date of birth for age verification
        id_type: Type of ID document (NIN, Passport, etc.)
        id_number: ID document number
        referral_code: User's unique referral code for viral growth
        address: Full address details
        pin_hash: Hashed PIN for authentication
        is_active: Account active status
        preferred_language: User's preferred language
    """
    user_id: str
    first_name: str
    last_name: str
    phone_number: str
    pin_hash: str
    date_of_birth: Optional[date] = None
    id_type: Optional[str] = None
    id_number: Optional[str] = None
    referral_code: Optional[str] = None
    street: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    is_active: bool = True
    preferred_language: Language = Language.ENGLISH
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    @property
    def full_name(self) -> str:
        """Get full name"""
        return f"{self.first_name} {self.last_name}"

    @property
    def address(self) -> str:
        """Get formatted address"""
        parts = [self.street, self.city, self.state]
        return ", ".join(p for p in parts if p)

@dataclass
class Wallet:
    """
    Wallet/Account entity (aligned with banking schema).

    Represents a user's financial account with balance tracking.
    """
    wallet_id: str
    user_id: str
    account_name: str
    account_number: str
    bank_name: str = "Providus Bank"
    balance: Decimal = Decimal("0.00")
    currency: str = "NGN"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()

    @property
    def formatted_balance(self) -> str:
        """Get formatted balance with currency"""
        return f"₦{self.balance:,.2f}"


@dataclass
class Transaction:
    """
    Financial transaction entity (aligned with banking schema).

    Supports DEBIT/CREDIT operations with atomic balance tracking.
    """
    transaction_id: str
    wallet_id: str
    amount: Decimal
    transaction_type: TransactionType
    category: TransactionCategory
    reference: str
    status: TransactionStatus = TransactionStatus.PENDING
    balance_before: Optional[Decimal] = None
    balance_after: Optional[Decimal] = None
    transaction_metadata: Optional[Dict[str, Any]] = None
    currency: str = "NGN"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.transaction_metadata is None:
            self.transaction_metadata = {}

    @property
    def is_debit(self) -> bool:
        """Check if transaction is debit"""
        return self.transaction_type == TransactionType.DEBIT

    @property
    def is_credit(self) -> bool:
        """Check if transaction is credit"""
        return self.transaction_type == TransactionType.CREDIT

    @property
    def formatted_amount(self) -> str:
        """Get formatted amount with currency"""
        return f"₦{self.amount:,.2f}"


@dataclass
class Beneficiary:
    """
    Saved payment beneficiary for quick transfers.
    """
    beneficiary_id: str
    user_id: str
    name: str
    bank_name: str
    account_number: str
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class AuditLog:
    """
    Security and compliance audit trail.
    """
    log_id: str
    user_id: str
    action: str
    module: str
    log_metadata: Optional[str] = None
    ip_address: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class Referral:
    """
    User referral tracking for viral growth.
    """
    referral_id: str
    referrer_id: str
    referred_id: str
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class Bill:
    """Bill payment entity"""
    bill_id: str
    user_id: str
    bill_type: str  # electricity, water, internet, etc.
    provider: str
    account_number: str
    amount: float
    currency: str = "NGN"
    due_date: Optional[datetime] = None
    status: str = "unpaid"
    created_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

@dataclass
class ConversationMessage:
    """Chat conversation message"""
    message_id: str
    user_id: str
    message_type: str  # user or assistant
    content: str
    language: Language = Language.ENGLISH
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Intent:
    """User intent classification"""
    intent_type: str
    confidence: float
    entities: Dict[str, Any]
    raw_message: str
    language: Language = Language.ENGLISH

@dataclass
class FinancialAdvice:
    """Financial advice response"""
    advice_type: str
    content: str
    confidence: float
    language: Language = Language.ENGLISH
    sources: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class VoiceSession:
    """Voice interaction session"""
    session_id: str
    user_id: str
    language: Language = Language.ENGLISH
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    transcript: Optional[List[str]] = None
    audio_duration: float = 0.0

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.utcnow()
        if self.transcript is None:
            self.transcript = []
