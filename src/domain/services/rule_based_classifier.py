from typing import Dict, List, Tuple
import re
import time
import logging
from src.domain.entities.intent_schema import IntentType, IntentResult

logger = logging.getLogger(__name__)

class RuleBasedClassifier:
    def __init__(self):
        self.intentKeywords = {
            IntentType.DEPOSIT_FUNDS: [
                "deposit", "fund", "add money", "top up", "load", "credit my account",
                "fund wallet", "deposit money", "add funds", "put money",
                "fi owó sí", "fi sí", "kó owó", "kun wallet", "mo fẹ fi owo sii",
                "saka", "ƙara kuɗi", "cika wallet", "zuba kuɗi",
                "tinye ego", "kwụnye", "mejupụta wallet", "mu ego",
                "fund my account", "add money", "deposit", "top up"
            ],
            IntentType.CHECK_BALANCE: [
                "balance", "how much", "money do i have", "check balance",
                "funds", "available", "account balance", "my balance", "balance inquiry",
                "owo mi", "owó", "àkójọpọ̀", "bawo ni balance",
                "kuɗi", "yawan", "adadin", "kudin da nake",
                "ego m", "nwere", "akaụntụ", "kedu ego",
                "my money", "how much money", "wetin be my balance"
            ],
            IntentType.SEND_MONEY: [
                "send", "transfer", "give", "remit", "wire",
                "send money", "transfer money", "send to", "pay to",
                "ránṣẹ́", "fi owó", "dá owó", "ranse owo", "mo fẹ́ ránsẹ́",
                "aika", "tura", "aiko", "son in aika",
                "ziga", "zipu", "nye ego", "biko ziga",
                "send money", "transfer", "give money", "i wan send"
            ],
            IntentType.PAY_BILL: [
                "bill", "electricity", "water", "cable", "nepa", "ekedc", "dstv", "gotv",
                "pay bill", "pay light", "subscription", "san", "ìgbéléwó", "biya", "kwụọ",
                "pay nepa", "pay electricity", "pay water", "pay dstv"
            ],
            IntentType.VTU_AIRTIME: [
                "buy airtime", "purchase airtime", "recharge", "top up airtime", "load airtime",
                "airtime for", "buy credit", "airtime", "buy mtn airtime", "buy glo airtime",
                "buy airtel airtime", "buy 9mobile airtime", "recharge phone", "load credit",
                "ra airtime", "sayi airtime", "zụta airtime", "i need airtime",
                "get me airtime", "airtime recharge", "mobile recharge"
            ],
            IntentType.VTU_DATA: [
                "buy data", "purchase data", "data bundle", "data plan", "gb data", "mb data",
                "need data", "get data", "buy mtn data", "buy glo data", "buy airtel data",
                "buy 9mobile data", "data subscription", "ra data", "sayi data", "zụta data",
                "i need data", "get me data", "internet data", "mobile data",
                "data plans", "what data", "available data", "show me data", "list data plans",
                "data package", "data offer", "data bundle",
                "show me the plans", "show the plans", "show plans", "show me plans",
                "what plans", "available plans", "list plans", "see the plans", "see plans", "view plans"
            ],
            IntentType.SAVINGS_GOAL: [
                "save", "savings", "goal", "target", "plan",
                "save money", "savings goal", "set aside", "put away",
                "pamọ́", "kó owó", "ìfowópamọ́",
                "ajiye", "tanadi kuɗi",
                "chekwa", "ego echekwara",
                "save money", "keep money"
            ],
            IntentType.TRANSACTION_HISTORY: [
                "history", "transactions", "spent", "spending", "statement",
                "transaction history", "show transactions", "recent transactions",
                "what did i", "what have i",
                "ìtàn", "itan owo", "owó tí mo ná", "iṣẹ́ owó", "ti mo na",
                "tarihi", "tarihin", "abin da na kashe", "jerin kashe", "kuɗi na",
                "akụkọ", "ego m jiri", "ihe m jiri ego", "jiri",
                "transaction history", "wetin i spend"
            ],
            IntentType.GET_LOAN: [
                "loan", "borrow", "credit", "advance",
                "need loan", "apply loan", "get loan", "loan application",
                "yá owó", "àwín", "gbé owó", "mo fẹ gbe owo ya", "gbe owo ya",
                "bashin", "rancen", "samun bashi",
                "mbinye", "ego mbinye", "ịgbazinye",
                "borrow money", "collect loan", "need loan"
            ],
            IntentType.FINANCIAL_EDUCATION: [
                "how do i", "what is", "explain", "teach", "learn", "advice",
                "interest", "budget", "invest", "financial", "education", "how to",
                "báwo", "kọ́ mi", "ṣàlàyé",
                "yaya", "koya", "bayyana",
                "kedu", "kụziere", "kọwaa",
                "how", "teach me", "explain", "how do i save", "save money?"
            ],
            IntentType.CUSTOMER_SUPPORT: [
                "help", "problem", "issue", "blocked", "failed", "error",
                "support", "pin", "password", "forgot", "reset", "stuck",
                "block my", "freeze", "account number", "card", "atm",
                "ìrànwọ́", "ìṣòro", "àtúnṣe",
                "taimako", "taimaka", "matsala", "sake saita", "allah",
                "enyemaka", "nsogbu", "tọgharịa", "achọrọ m enyemaka",
                "help", "problem", "reset pin", "support", "help me"
            ],
            IntentType.GREETING: [
                "hello", "hi", "hey", "good morning", "good afternoon",
                "good evening", "greetings", "bawo", "sannu", "how far",
                "káàárọ̀", "káàsán", "pẹlẹ́",
                "ina kwana", "yaya kake",
                "ndeewọ", "ụtụtụ ọma", "kedu",
                "wetin dey", "good morning"
            ]
        }

        self.phonePattern = re.compile(r'\b0[7-9]\d{9}\b|\+234[7-9]\d{9}\b')
        self.amountPattern = re.compile(r'₦?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?')

    def classify(self, query: str, language: str = "en") -> IntentResult:
        startTime = time.time()
        query = query.lower().strip()

        scores = {}
        for intent, keywords in self.intentKeywords.items():
            score = 0
            for keyword in keywords:
                if keyword in query:
                    score += 1
            scores[intent] = score

        maxScore = max(scores.values())
        if maxScore == 0:
            topIntent = IntentType.UNKNOWN
            confidence = 0.0
        else:
            topIntent = max(scores.items(), key=lambda x: x[1])[0]

            # Improved confidence calculation
            # Base confidence scales better with keyword matches
            if maxScore == 1:
                confidence = 0.45  # Single keyword match - low confidence
            elif maxScore == 2:
                confidence = 0.72  # Two keywords - medium confidence
            elif maxScore >= 3:
                confidence = 0.88  # Three+ keywords - high confidence
            else:
                confidence = 0.0

        hasPhone = bool(self.phonePattern.search(query))
        hasAmount = bool(self.amountPattern.search(query))

        sendPattern = re.compile(r'\b(send|transfer|give)\b.*\b(to|give)\b', re.IGNORECASE)
        hasSendPattern = bool(sendPattern.search(query))

        # VTU keywords detection (airtime, data, recharge, buy)
        vtuPattern = re.compile(r'\b(buy|purchase|airtime|recharge|data|credit|load|top\s*up)\b', re.IGNORECASE)
        hasVTUKeywords = bool(vtuPattern.search(query))

        # 🔥 NEW: VTU SHORTHAND PATTERN DETECTION
        # Pattern: "{amount} {network} for {phone}" (e.g., "100 mtn for 08012345678")
        vtuShorthandPattern = re.compile(
            r'\b\d+\s*(mtn|glo|airtel|9mobile|etisalat)\s+(?:for|to)\s+0[7-9]\d{9}\b',
            re.IGNORECASE
        )
        hasVTUShorthand = bool(vtuShorthandPattern.search(query))

        # Network keyword detection (strong VTU indicator)
        networkKeywords = ['mtn', 'glo', 'airtel', '9mobile', 'etisalat']
        hasNetworkKeyword = any(network in query for network in networkKeywords)

        # Data plan inquiry detection
        dataPlanPattern = re.compile(r'\b(data\s+plan|data\s+bundle|available\s+data|data\s+package)\b', re.IGNORECASE)
        isDataPlanInquiry = bool(dataPlanPattern.search(query))

        # 🔥 CRITICAL: VTU SHORTHAND PATTERN OVERRIDES EVERYTHING
        if hasVTUShorthand:
            # Pattern like "100 mtn for 08012345678" is DEFINITELY airtime
            topIntent = IntentType.VTU_AIRTIME
            confidence = 0.95
            logger.info(f"🎯 VTU shorthand detected: '{query}' → vtu_airtime (0.95)")

        # 🔥 DATA PLAN INQUIRY OVERRIDE
        elif isDataPlanInquiry:
            topIntent = IntentType.VTU_DATA
            confidence = 0.90
            logger.info(f"📦 Data plan inquiry detected: '{query}' → vtu_data (0.90)")

        # Network keyword + amount/phone is likely VTU
        elif hasNetworkKeyword and (hasAmount or hasPhone):
            if hasAmount and hasPhone:
                # Has network, amount, and phone → Very likely airtime
                if topIntent not in [IntentType.VTU_AIRTIME, IntentType.VTU_DATA]:
                    topIntent = IntentType.VTU_AIRTIME
                confidence = min(confidence + 0.25, 0.92)
            elif hasAmount:
                # Has network and amount → Likely airtime
                if topIntent not in [IntentType.VTU_AIRTIME, IntentType.VTU_DATA]:
                    topIntent = IntentType.VTU_AIRTIME
                confidence = min(confidence + 0.20, 0.88)

        # If VTU keywords detected, NEVER convert to SEND_MONEY
        elif hasVTUKeywords and topIntent in [IntentType.VTU_AIRTIME, IntentType.VTU_DATA]:
            # Boost confidence for VTU with phone + amount
            if hasPhone and hasAmount:
                confidence = min(confidence + 0.20, 0.95)
            elif hasAmount:
                confidence = min(confidence + 0.15, 0.92)
        elif (hasPhone and hasAmount) or (hasSendPattern and hasAmount):
            # Only convert to SEND_MONEY if NO VTU keywords and NO network keywords
            if topIntent == IntentType.SEND_MONEY:
                confidence = min(confidence + 0.15, 0.98)
            elif topIntent in [IntentType.PAY_BILL, IntentType.CHECK_BALANCE, IntentType.DEPOSIT_FUNDS] and not hasVTUKeywords and not hasNetworkKeyword:
                topIntent = IntentType.SEND_MONEY
                confidence = 0.85

        if hasAmount and topIntent == IntentType.PAY_BILL and not hasSendPattern and not hasVTUKeywords:
            confidence = min(confidence + 0.1, 0.95)

        if len(query.split()) <= 3 and topIntent == IntentType.GREETING:
            confidence = min(confidence + 0.2, 0.98)

        latency = (time.time() - startTime) * 1000

        return IntentResult(
            intent=topIntent,
            confidence=confidence,
            entities={},
            language=language,
            method="rule_based",
            latency_ms=round(latency, 2)
        )

    def batchClassify(self, queries: List[str], language: str = "en") -> List[IntentResult]:
        return [self.classify(query, language) for query in queries]
