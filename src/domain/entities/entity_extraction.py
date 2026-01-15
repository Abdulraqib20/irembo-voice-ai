"""
Entity Extraction Utilities
Handles extraction and validation of entities from user queries
"""

import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from src.domain.entities.intent_schema import EntityType, BillType, BillProvider

class EntityExtractor:
    """Extract and validate entities from user queries"""

    def __init__(self):
        self.amountPattern = re.compile(r'₦?\s?(\d{1,3}(?:,?\d{3})*(?:\.\d{2})?)')
        self.phonePattern = re.compile(r'(?:\+?234|0)([789]\d{9})')
        self.namePattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')

        self.billKeywords = {
            BillType.ELECTRICITY.value: [
                "electricity", "light", "EKEDC", "IKEDC", "AEDC", "power",
                "ina", "wutar lantarki"
            ],
            BillType.WATER.value: ["water", "ruwa", "omi"],
            BillType.INTERNET.value: ["internet", "data", "wifi"],
            BillType.AIRTIME.value: ["airtime", "recharge", "credit"],
            BillType.DATA.value: ["data", "internet bundle"],
            BillType.CABLE_TV.value: ["DSTV", "GOtv", "Startimes", "cable", "tv"]
        }

        self.providerKeywords = {
            BillProvider.EKEDC.value: ["EKEDC", "Eko Electric"],
            BillProvider.IKEDC.value: ["IKEDC", "Ikeja Electric"],
            BillProvider.AEDC.value: ["AEDC", "Abuja Electric"],
            BillProvider.MTN.value: ["MTN"],
            BillProvider.AIRTEL.value: ["Airtel"],
            BillProvider.GLO.value: ["Glo", "Globacom"],
            BillProvider.NINE_MOBILE.value: ["9mobile", "Etisalat"],
            BillProvider.DSTV.value: ["DSTV", "DStv"],
            BillProvider.GOTV.value: ["GOtv", "Gotv"],
            BillProvider.STARTIMES.value: ["Startimes", "StarTimes"]
        }

    def extract(self, query: str, intent: str) -> Dict[str, Any]:
        entities = {}

        queryLower = query.lower()

        if "amount" in query or "₦" in query or re.search(r'\d{3,}', query):
            amount = self.extractAmount(query)
            if amount:
                entities[EntityType.AMOUNT.value] = amount
                entities[EntityType.CURRENCY.value] = "NGN"

        if "phone" in intent.lower() or "send" in intent.lower() or "transfer" in intent.lower():
            phone = self.extractPhone(query)
            if phone:
                entities[EntityType.RECIPIENT_PHONE.value] = phone

            name = self.extractName(query)
            if name:
                entities[EntityType.RECIPIENT_NAME.value] = name

        if "bill" in intent.lower() or "pay" in intent.lower():
            billType = self.extractBillType(query)
            if billType:
                entities[EntityType.BILL_TYPE.value] = billType

            provider = self.extractProvider(query)
            if provider:
                entities[EntityType.PROVIDER.value] = provider

        if "history" in intent.lower() or "transactions" in intent.lower():
            timePeriod = self.extractTimePeriod(query)
            if timePeriod:
                entities[EntityType.TIME_PERIOD.value] = timePeriod

        return entities

    def extractAmount(self, query: str) -> Optional[float]:
        match = self.amountPattern.search(query)
        if match:
            amountStr = match.group(1).replace(",", "")
            try:
                return float(amountStr)
            except ValueError:
                return None

        words = query.lower().split()
        for i, word in enumerate(words):
            if word in ["naira", "₦", "ngn"]:
                if i > 0:
                    try:
                        return float(words[i - 1].replace(",", ""))
                    except ValueError:
                        pass

        return None

    def extractPhone(self, query: str) -> Optional[str]:
        match = self.phonePattern.search(query)
        if match:
            return f"+234{match.group(1)}"

        cleanQuery = re.sub(r'[^\d]', '', query)
        if len(cleanQuery) == 11 and cleanQuery[0] == '0':
            return f"+234{cleanQuery[1:]}"
        elif len(cleanQuery) == 10:
            return f"+234{cleanQuery}"

        return None

    def extractName(self, query: str) -> Optional[str]:
        patterns = [
            r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bfor\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bgive\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bsi\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
            r'\bga\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                name = match.group(1).strip()
                if len(name) >= 3 and name not in ["You", "Me", "Him", "Her", "The"]:
                    return name

        return None

    def extractBillType(self, query: str) -> Optional[str]:
        queryLower = query.lower()

        for billType, keywords in self.billKeywords.items():
            for keyword in keywords:
                if keyword.lower() in queryLower:
                    return billType

        return None

    def extractProvider(self, query: str) -> Optional[str]:
        queryUpper = query.upper()

        for provider, keywords in self.providerKeywords.items():
            for keyword in keywords:
                if keyword.upper() in queryUpper:
                    return provider

        return None

    def extractTimePeriod(self, query: str) -> Optional[Dict[str, str]]:
        queryLower = query.lower()
        now = datetime.now()

        if "today" in queryLower or "loni" in queryLower:
            return {
                "start_date": now.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
                "description": "today"
            }

        if "yesterday" in queryLower:
            yesterday = now - timedelta(days=1)
            return {
                "start_date": yesterday.strftime("%Y-%m-%d"),
                "end_date": yesterday.strftime("%Y-%m-%d"),
                "description": "yesterday"
            }

        if "last week" in queryLower or "ọsẹ to kọja" in queryLower:
            startDate = now - timedelta(days=7)
            return {
                "start_date": startDate.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
                "description": "last week"
            }

        if "last month" in queryLower or "oṣu to kọja" in queryLower:
            startDate = now - timedelta(days=30)
            return {
                "start_date": startDate.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
                "description": "last month"
            }

        if "this month" in queryLower:
            startDate = now.replace(day=1)
            return {
                "start_date": startDate.strftime("%Y-%m-%d"),
                "end_date": now.strftime("%Y-%m-%d"),
                "description": "this month"
            }

        return None

class EntityValidator:
    """Validate extracted entities"""

    @staticmethod
    def validateAmount(amount: float) -> tuple[bool, Optional[str]]:
        if amount <= 0:
            return False, "Amount must be greater than zero"
        if amount > 1_000_000:
            return False, "Amount exceeds maximum limit of ₦1,000,000"
        return True, None

    @staticmethod
    def validatePhone(phone: str) -> tuple[bool, Optional[str]]:
        if not phone.startswith("+234"):
            return False, "Phone number must be Nigerian (+234)"
        if len(phone) != 14:
            return False, "Invalid phone number format"
        if phone[4] not in ['7', '8', '9']:
            return False, "Phone number must start with 07, 08, or 09"
        return True, None

    @staticmethod
    def validateBillType(billType: str) -> tuple[bool, Optional[str]]:
        validTypes = [bt.value for bt in BillType]
        if billType not in validTypes:
            return False, f"Invalid bill type. Must be one of: {', '.join(validTypes)}"
        return True, None

    @staticmethod
    def validateProvider(provider: str) -> tuple[bool, Optional[str]]:
        validProviders = [bp.value for bp in BillProvider]
        if provider not in validProviders:
            return False, f"Invalid provider. Must be one of: {', '.join(validProviders)}"
        return True, None

def normalizePhone(phone: str) -> str:
    cleaned = re.sub(r'[^\d+]', '', phone)

    if cleaned.startswith('0'):
        return f"+234{cleaned[1:]}"
    elif cleaned.startswith('234'):
        return f"+{cleaned}"
    elif cleaned.startswith('+234'):
        return cleaned
    else:
        return f"+234{cleaned}"

def formatAmount(amount: float, currency: str = "NGN") -> str:
    if currency == "NGN":
        return f"₦{amount:,.2f}"
    else:
        return f"{currency} {amount:,.2f}"

def parseAmount(amountStr: str) -> Optional[float]:
    cleaned = re.sub(r'[₦,\s]', '', amountStr)
    try:
        return float(cleaned)
    except ValueError:
        return None
