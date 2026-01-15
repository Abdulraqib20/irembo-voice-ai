"""
Domain entities
Core business objects and value objects

"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, date
from decimal import Decimal
from enum import Enum

class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"


@dataclass
