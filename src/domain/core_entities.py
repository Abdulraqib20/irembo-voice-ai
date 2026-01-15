"""
Domain entities for Irembo Voice AI
Core business objects and value objects for Kinyarwanda/English multilingual system
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum


class Language(Enum):
    """Supported languages for Irembo Voice AI"""
    ENGLISH = "en"
    KINYARWANDA = "rw"
    MIXED = "mixed"  # Code-switched Kinyarwanda/English
    
    @classmethod
    def from_string(cls, lang_str: str) -> "Language":
        """Convert string to Language enum"""
        mapping = {
            "en": cls.ENGLISH,
            "rw": cls.KINYARWANDA,
            "mixed": cls.MIXED,
        }
        return mapping.get(lang_str.lower(), cls.ENGLISH)


@dataclass
class Intent:
    """Classified intent result"""
    intent_type: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    raw_message: str = ""
    language: Language = Language.ENGLISH


@dataclass
class Utterance:
    """Voice AI utterance from user"""
    utterance_id: str
    utterance_text: str
    language: Language
    timestamp: Optional[datetime] = None
    channel: Optional[str] = None  # voice_call, whatsapp_voice_note, ivr, mobile_app_voice
    device_type: Optional[str] = None  # android, ios, feature_phone, web
    region: Optional[str] = None  # Kigali, Northern, Western, Eastern, Southern
    asr_confidence: float = 1.0
    duration_seconds: float = 0.0


@dataclass
class ClassificationResult:
    """Full classification result with metadata"""
    utterance: Utterance
    intent: Intent
    processing_time_ms: float = 0.0
    model_version: str = "v1.0"
    fallback_used: bool = False
