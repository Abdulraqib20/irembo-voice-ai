"""
Entity Extraction Utilities
Handles extraction and validation of entities from user queries
"""

import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dateutil import parser as date_parser

class EntityExtractor:
    """Extract and validate entities from user queries"""

    def __init__(self):
        