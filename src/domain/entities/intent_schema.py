"""
Intent Classification Schema
Defines all supported financial intents, entities, and confidence thresholds
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class IntentType(Enum):

class EntityType(Enum):


@dataclass
class IntentDefinition:
    intent_type: IntentType
    description: str
    examples_en: List[str]
    required_entities: List[EntityType]
    optional_entities: List[EntityType]
    requires_auth: bool
    requires_confirmation: bool
    handler_function: str
