"""
Intent Classification Schema
Defines all supported financial intents, entities, and confidence thresholds
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class IntentType(Enum):
    DEPOSIT_FUNDS = "deposit_funds"
    CHECK_BALANCE = "check_balance"
    SEND_MONEY = "send_money"
    PAY_BILL = "pay_bill"
    VTU_AIRTIME = "vtu_airtime"  # Buy airtime (VTU)
    VTU_DATA = "vtu_data"  # Buy data bundle (VTU)
    SAVINGS_GOAL = "savings_goal"
    TRANSACTION_HISTORY = "transaction_history"
    GET_LOAN = "get_loan"
    FINANCIAL_EDUCATION = "financial_education"
    CUSTOMER_SUPPORT = "customer_support"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class EntityType(Enum):
    AMOUNT = "amount"
    CURRENCY = "currency"
    RECIPIENT_PHONE = "recipient_phone"
    RECIPIENT_NAME = "recipient_name"
    BILL_TYPE = "bill_type"
    PROVIDER = "provider"
    ACCOUNT_NUMBER = "account_number"
    TIME_PERIOD = "time_period"
    TRANSACTION_TYPE = "transaction_type"
    GOAL_NAME = "goal_name"
    TARGET_DATE = "target_date"
    FREQUENCY = "frequency"
    LOAN_AMOUNT = "loan_amount"
    LOAN_PURPOSE = "loan_purpose"
    LOAN_DURATION = "loan_duration"
    TOPIC = "topic"
    ISSUE_TYPE = "issue_type"

class BillType(Enum):
    ELECTRICITY = "electricity"
    WATER = "water"
    INTERNET = "internet"
    AIRTIME = "airtime"
    DATA = "data"
    CABLE_TV = "cable_tv"

class BillProvider(Enum):
    EKEDC = "EKEDC"
    IKEDC = "IKEDC"
    AEDC = "AEDC"
    MTN = "MTN"
    AIRTEL = "Airtel"
    GLO = "Glo"
    NINE_MOBILE = "9mobile"
    DSTV = "DSTV"
    GOTV = "GOtv"
    STARTIMES = "Startimes"

@dataclass
class IntentDefinition:
    intent_type: IntentType
    description: str
    examples_en: List[str]
    examples_yo: List[str]
    examples_ha: List[str]
    examples_pcm: List[str]
    required_entities: List[EntityType]
    optional_entities: List[EntityType]
    requires_auth: bool
    requires_confirmation: bool
    handler_function: str

INTENT_SCHEMA: Dict[IntentType, IntentDefinition] = {
    IntentType.DEPOSIT_FUNDS: IntentDefinition(
        intent_type=IntentType.DEPOSIT_FUNDS,
        description="User wants to deposit/fund their wallet",
        examples_en=[
            "Deposit ₦50000",
            "I want to fund my wallet",
            "Add money to my account",
            "Top up my wallet with ₦10000",
            "Deposit money",
            "Fund my account with ₦25000",
            "Add ₦5000 to my wallet"
        ],
        examples_yo=[
            "Fi owo sii",
            "Mo fẹ fi owo kun wallet mi",
            "Fi ₦50000 sii",
            "Mo fẹ fi owo si account mi",
            "Kun wallet mi pelu ₦10000"
        ],
        examples_ha=[
            "Saka kudi",
            "Ina son cika wallet na",
            "Saka ₦50000",
            "Ƙara kudi zuwa asusun na",
            "Cika wallet na da ₦10000"
        ],
        examples_pcm=[
            "Put money for my wallet",
            "I wan fund my account",
            "Add ₦50000 for my wallet",
            "Top up my wallet",
            "Put money inside"
        ],
        required_entities=[EntityType.AMOUNT],
        optional_entities=[],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="handle_deposit"
    ),

    IntentType.CHECK_BALANCE: IntentDefinition(
        intent_type=IntentType.CHECK_BALANCE,
        description="User wants to check account balance",
        examples_en=[
            "What is my balance?",
            "How much money do I have?",
            "Check my account balance",
            "Balance inquiry",
            "Show me my balance",
            "What's my account balance?",
            "How much do I have in my account?"
        ],
        examples_yo=[
            "Elo ni mo ni lowo?",
            "Kini iwọn owó mi?",
            "Wo owó mi",
            "Elo ni owo mi?",
            "Wo account balance mi"
        ],
        examples_ha=[
            "Menene ma'aunin asusun ku?",
            "Nawa kudin da nake da shi?",
            "Nuna mini kudina",
            "Balance dinata menene?",
            "Ku nuna mini balance na"
        ],
        examples_pcm=[
            "How much money I get?",
            "Wetin be my balance?",
            "Show me my balance",
            "How much dey my account?",
            "I wan check my balance"
        ],
        required_entities=[],
        optional_entities=[],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="handle_balance_check"
    ),

    IntentType.SEND_MONEY: IntentDefinition(
        intent_type=IntentType.SEND_MONEY,
        description="User wants to transfer money to another person",
        examples_en=[
            "Send ₦5000 to Musa",
            "Transfer money to 08012345678",
            "I want to send ₦10000 to my friend",
            "Can you transfer ₦2500 to Amina?",
            "Send money to my brother",
            "Transfer ₦15000 to 08087654321"
        ],
        examples_yo=[
            "Fi owo ranṣẹ si Ade",
            "Fi ₦5000 ranṣẹ si Bola",
            "Mo fẹ fi owo ranṣẹ",
            "Ranṣẹ owo si ọrẹ mi",
            "Fi owo ranṣẹ si arakunrin mi"
        ],
        examples_ha=[
            "Aika kudi ga Musa",
            "Ka aika ₦5000 ga abokina",
            "Ina son aika kudi",
            "Aika kudi zuwa ga 08012345678",
            "Ka aika kudi ga kawuna"
        ],
        examples_pcm=[
            "Send money give Musa",
            "Transfer ₦5000 give my guy",
            "I wan send money",
            "Send money give my brother",
            "Transfer money for 08012345678"
        ],
        required_entities=[EntityType.AMOUNT],
        optional_entities=[
            EntityType.RECIPIENT_PHONE,
            EntityType.RECIPIENT_NAME,
            EntityType.CURRENCY
        ],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="handle_money_transfer"
    ),

    IntentType.PAY_BILL: IntentDefinition(
        intent_type=IntentType.PAY_BILL,
        description="User wants to pay utility bills (electricity, water, cable TV)",
        examples_en=[
            "Pay my electricity bill",
            "I need to pay EKEDC",
            "Pay ₦3000 for light",
            "Pay my DSTV subscription",
            "I want to pay water bill",
            "Pay NEPA bill"
        ],
        examples_yo=[
            "San iná mi",
            "Mo fẹ san EKEDC",
            "San ₦3000 fun ina",
            "San DSTV mi"
        ],
        examples_ha=[
            "Biya kudin wutar lantarki",
            "Ina son biya EKEDC",
            "Biya ₦3000 don wuta",
            "Biya kudin DSTV na"
        ],
        examples_pcm=[
            "Pay my light bill",
            "I wan pay EKEDC",
            "Pay ₦3000 for light",
            "Pay my DSTV"
        ],
        required_entities=[EntityType.BILL_TYPE],
        optional_entities=[
            EntityType.AMOUNT,
            EntityType.PROVIDER,
            EntityType.ACCOUNT_NUMBER
        ],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="handle_bill_payment"
    ),

    IntentType.VTU_AIRTIME: IntentDefinition(
        intent_type=IntentType.VTU_AIRTIME,
        description="User wants to buy airtime (VTU service)",
        examples_en=[
            "Buy ₦100 MTN airtime",
            "Buy airtime for ₦500",
            "Recharge ₦200 for 08012345678",
            "Buy ₦1000 MTN airtime for 08012345678",
            "I need airtime",
            "Recharge my phone",
            "Buy credit",
            "Top up ₦500 airtime",
            "Load ₦300 airtime for me",
            "Purchase MTN airtime"
        ],
        examples_yo=[
            "Ra airtime ₦100",
            "Mo fẹ ra airtime MTN",
            "Recharge ₦500 fun mi",
            "Ra airtime ₦200 fun 08012345678",
            "Mo nilo airtime"
        ],
        examples_ha=[
            "Sayi airtime ₦100",
            "Ina son sayin airtime MTN",
            "Recharge ₦500",
            "Sayi airtime ₦200 don 08012345678",
            "Ina bukatar airtime"
        ],
        examples_pcm=[
            "Buy airtime ₦100",
            "I wan buy MTN airtime",
            "Recharge ₦500 for me",
            "Buy airtime ₦200 for 08012345678",
            "I need airtime"
        ],
        required_entities=[EntityType.AMOUNT],
        optional_entities=[
            EntityType.PROVIDER,
            EntityType.RECIPIENT_PHONE
        ],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="handle_vtu_airtime"
    ),

    IntentType.VTU_DATA: IntentDefinition(
        intent_type=IntentType.VTU_DATA,
        description="User wants to buy data bundle (VTU service)",
        examples_en=[
            "Buy 1GB MTN data",
            "Buy data bundle",
            "I need data",
            "Purchase 2GB data for 08012345678",
            "Buy MTN data plan",
            "Get me 500MB data",
            "Buy data for ₦500",
            "I want to buy data",
            "Purchase Glo data bundle",
            "Buy 5GB data"
        ],
        examples_yo=[
            "Ra data 1GB",
            "Mo fẹ ra data MTN",
            "Mo nilo data",
            "Ra data 2GB fun 08012345678",
            "Ra data bundle"
        ],
        examples_ha=[
            "Sayi data 1GB",
            "Ina son sayin data MTN",
            "Ina bukatar data",
            "Sayi data 2GB don 08012345678",
            "Sayi data bundle"
        ],
        examples_pcm=[
            "Buy data 1GB",
            "I wan buy MTN data",
            "I need data",
            "Buy 2GB data for 08012345678",
            "Buy data bundle"
        ],
        required_entities=[],
        optional_entities=[
            EntityType.AMOUNT,
            EntityType.PROVIDER,
            EntityType.RECIPIENT_PHONE
        ],
        requires_auth=True,
        requires_confirmation=True,
        handler_function="handle_vtu_data"
    ),

    IntentType.SAVINGS_GOAL: IntentDefinition(
        intent_type=IntentType.SAVINGS_GOAL,
        description="User wants to save money or set savings goals",
        examples_en=[
            "I want to save ₦50000",
            "Help me save for rent",
            "Create a savings plan",
            "I want to save ₦10000 every month",
            "Set up savings goal for ₦100000",
            "How can I save money?"
        ],
        examples_yo=[
            "Mo fẹ tọju owo",
            "Ṣe iranl̩ọw̩ọ fun mi lati tọju owo",
            "Mo fẹ pa owo mi mọ",
            "Ṣe atilẹyin fun ipin owo",
            "Mo fẹ tọju ₦50000"
        ],
        examples_ha=[
            "Ina son ajiye kudi",
            "Taimaka mini in ajiye kudi",
            "Ina son tsara ajiya",
            "In ajiye ₦50000",
            "Ina son ajiye kudi kowane wata"
        ],
        examples_pcm=[
            "I wan save money",
            "Help me save money",
            "I wan save ₦50000",
            "Make I save money for rent",
            "I wan save money every month"
        ],
        required_entities=[],
        optional_entities=[
            EntityType.AMOUNT,
            EntityType.GOAL_NAME,
            EntityType.TARGET_DATE,
            EntityType.FREQUENCY
        ],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="handle_savings"
    ),

    IntentType.TRANSACTION_HISTORY: IntentDefinition(
        intent_type=IntentType.TRANSACTION_HISTORY,
        description="User wants to view past transactions",
        examples_en=[
            "Show my transactions",
            "What did I spend last month?",
            "Transaction history",
            "Show me my spending",
            "What are my recent transactions?",
            "Show transactions from last week"
        ],
        examples_yo=[
            "Fi awọn iṣowo mi han mi",
            "Kini mo na ni oṣu to kọja?",
            "Itan iṣowo mi",
            "Fi ina mi han mi",
            "Awọn iṣowo ti mo ṣe laipe"
        ],
        examples_ha=[
            "Nuna mini kasuwancin da na yi",
            "Me na kashe a watan jiya?",
            "Nuna mini tarihin kasuwanci",
            "Kasuwanci na na baya-bayan nan",
            "Nuna mini yadda na kashe kudi"
        ],
        examples_pcm=[
            "Show me my transactions",
            "Wetin I spend last month?",
            "Show my transaction history",
            "Make I see how I don spend money",
            "Show me my recent transactions"
        ],
        required_entities=[],
        optional_entities=[EntityType.TIME_PERIOD, EntityType.TRANSACTION_TYPE],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="handle_transaction_history"
    ),

    IntentType.GET_LOAN: IntentDefinition(
        intent_type=IntentType.GET_LOAN,
        description="User inquiring about or applying for a loan",
        examples_en=[
            "I need a loan",
            "Can I borrow money?",
            "Loan application",
            "I want to apply for a loan of ₦100000",
            "How can I get a loan?",
            "What are your loan requirements?"
        ],
        examples_yo=[
            "Mo nilo awin",
            "Ṣe mo le ya owo?",
            "Mo fẹ gba awin",
            "Mo fẹ ya owo ₦100000",
            "Bawo ni mo ṣe le gba awin?"
        ],
        examples_ha=[
            "Ina bukatar bashi",
            "Zan iya karbar bashi?",
            "Ina son neman bashi",
            "Ina son karbar bashi ₦100000",
            "Ta yaya zan iya samun bashi?"
        ],
        examples_pcm=[
            "I need loan",
            "I fit borrow money?",
            "I wan apply for loan",
            "I wan collect loan of ₦100000",
            "How I go fit get loan?"
        ],
        required_entities=[],
        optional_entities=[
            EntityType.LOAN_AMOUNT,
            EntityType.LOAN_PURPOSE,
            EntityType.LOAN_DURATION
        ],
        requires_auth=True,
        requires_confirmation=False,
        handler_function="handle_loan_inquiry"
    ),

    IntentType.FINANCIAL_EDUCATION: IntentDefinition(
        intent_type=IntentType.FINANCIAL_EDUCATION,
        description="User asking for financial advice or education",
        examples_en=[
            "How do I save money?",
            "What is interest?",
            "Teach me about budgeting",
            "How can I manage my money better?",
            "What is compound interest?",
            "Tips for saving money"
        ],
        examples_yo=[
            "Bawo ni mo ṣe le pa owo mi mọ?",
            "Kini iwọn?",
            "Kọ mi nipa iṣakoso owo",
            "Bawo ni mo ṣe le ṣakoso owo mi daradara?",
            "Bawo ni mo ṣe le tọju owo?"
        ],
        examples_ha=[
            "Yaya zan iya ajiye kudi?",
            "Menene riba?",
            "Koya mini game da tsarin kudi",
            "Yaya zan iya sarrafa kudina?",
            "Koya mini yadda ake ajiye kudi"
        ],
        examples_pcm=[
            "How I go fit save money?",
            "Wetin be interest?",
            "Teach me how to manage money",
            "How I fit take care of my money well?",
            "Wetin I suppose know about money?"
        ],
        required_entities=[],
        optional_entities=[EntityType.TOPIC],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="handle_financial_education"
    ),

    IntentType.CUSTOMER_SUPPORT: IntentDefinition(
        intent_type=IntentType.CUSTOMER_SUPPORT,
        description="User needs help with account issues or general support",
        examples_en=[
            "I forgot my PIN",
            "My card is blocked",
            "Help me",
            "I have a problem",
            "My transaction failed",
            "I need customer support"
        ],
        examples_yo=[
            "Mo gbagbe PIN mi",
            "Kaadi mi ti di",
            "Ṣe iranl̩ọw̩ọ fun mi",
            "Mo ni iṣoro kan",
            "Iṣowo mi kuna"
        ],
        examples_ha=[
            "Na manta PIN dina",
            "An toshe katin na",
            "Ka taimake ni",
            "Ina da matsala",
            "Kasuwanci na ya kasa"
        ],
        examples_pcm=[
            "I forget my PIN",
            "My card don block",
            "Help me",
            "I get problem",
            "My transaction no work"
        ],
        required_entities=[],
        optional_entities=[EntityType.ISSUE_TYPE],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="handle_support"
    ),

    IntentType.GREETING: IntentDefinition(
        intent_type=IntentType.GREETING,
        description="User greeting or starting conversation",
        examples_en=[
            "Hello",
            "Good morning",
            "Hi there",
            "Hey",
            "Good afternoon",
            "Good evening",
            "Greetings"
        ],
        examples_yo=[
            "Bawo ni",
            "Ẹ káàárọ̀",
            "Ẹ kú ọ̀sán",
            "Ẹ kú irọlẹ",
            "Ẹ káàbọ̀"
        ],
        examples_ha=[
            "Sannu",
            "Barka da safe",
            "Barka da yamma",
            "Ina kwana",
            "Yaya dai?"
        ],
        examples_pcm=[
            "How far",
            "Good morning",
            "How you dey",
            "Wetin dey happen",
            "Hello"
        ],
        required_entities=[],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="handle_greeting"
    ),

    IntentType.UNKNOWN: IntentDefinition(
        intent_type=IntentType.UNKNOWN,
        description="Intent cannot be determined with confidence",
        examples_en=[],
        examples_yo=[],
        examples_ha=[],
        examples_pcm=[],
        required_entities=[],
        optional_entities=[],
        requires_auth=False,
        requires_confirmation=False,
        handler_function="handle_unknown_intent"
    )
}

class ConfidenceThreshold:
    HIGH = 0.85
    MEDIUM = 0.70
    LOW = 0.50
    AMBIGUOUS = 0.50

@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    entities: Dict[str, Any]
    language: str
    method: str
    latency_ms: float
    reasoning: Optional[str] = None

    def requiresConfirmation(self) -> bool:
        intent_def = INTENT_SCHEMA[self.intent]
        return (
            intent_def.requires_confirmation and
            self.confidence >= ConfidenceThreshold.MEDIUM
        )

    def isHighConfidence(self) -> bool:
        return self.confidence >= ConfidenceThreshold.HIGH

    def isMediumConfidence(self) -> bool:
        return (
            self.confidence >= ConfidenceThreshold.MEDIUM and
            self.confidence < ConfidenceThreshold.HIGH
        )

    def isLowConfidence(self) -> bool:
        return (
            self.confidence >= ConfidenceThreshold.LOW and
            self.confidence < ConfidenceThreshold.MEDIUM
        )

    def isAmbiguous(self) -> bool:
        return self.confidence < ConfidenceThreshold.AMBIGUOUS

def getIntentExamples(intent_type: IntentType, language: str = "en") -> List[str]:
    intent_def = INTENT_SCHEMA[intent_type]

    if language == "yo":
        return intent_def.examples_yo
    elif language == "ha":
        return intent_def.examples_ha
    elif language == "pcm":
        return intent_def.examples_pcm
    else:
        return intent_def.examples_en

def getAllIntents() -> List[str]:
    return [intent.value for intent in IntentType]

def getIntentDefinition(intent_type: IntentType) -> IntentDefinition:
    return INTENT_SCHEMA[intent_type]

def requiresAuth(intent_type: IntentType) -> bool:
    return INTENT_SCHEMA[intent_type].requires_auth

def requiresConfirmation(intent_type: IntentType) -> bool:
    return INTENT_SCHEMA[intent_type].requires_confirmation
