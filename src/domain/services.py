"""
Domain services for business logic
"""

from typing import Optional, Dict, Any
from src.domain.entities import (
    User, Transaction, Bill, Intent, Language,
    TransactionType, TransactionStatus
)
import uuid
from datetime import datetime

class UserService:
    """Business logic for user operations"""

    @staticmethod
    def createUser(userId: str, phoneNumber: Optional[str] = None,
                   preferredLanguage: Language = Language.ENGLISH) -> User:
        """Create a new user"""
        return User(
            user_id=userId,
            phone_number=phoneNumber,
            preferred_language=preferredLanguage,
            account_balance=0.0,
            is_verified=False
        )

    @staticmethod
    def verifyUser(user: User) -> User:
        """Verify user identity"""
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        return user

    @staticmethod
    def updateBalance(user: User, amount: float) -> User:
        """Update user account balance"""
        user.account_balance += amount
        user.updated_at = datetime.utcnow()
        return user

class TransactionService:
    """Business logic for financial transactions"""

    @staticmethod
    def createTransaction(userId: str, transactionType: TransactionType,
                         amount: float, recipient: Optional[str] = None,
                         description: Optional[str] = None) -> Transaction:
        """Create a new transaction"""
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=userId,
            transaction_type=transactionType,
            amount=amount,
            recipient=recipient,
            description=description,
            status=TransactionStatus.PENDING,
            reference=str(uuid.uuid4())[:8].upper()
        )

    @staticmethod
    def completeTransaction(transaction: Transaction) -> Transaction:
        """Mark transaction as completed"""
        transaction.status = TransactionStatus.COMPLETED
        transaction.completed_at = datetime.utcnow()
        return transaction

    @staticmethod
    def failTransaction(transaction: Transaction) -> Transaction:
        """Mark transaction as failed"""
        transaction.status = TransactionStatus.FAILED
        transaction.completed_at = datetime.utcnow()
        return transaction

    @staticmethod
    def validateTransaction(transaction: Transaction, userBalance: float) -> bool:
        """Validate if transaction can be processed"""
        if transaction.amount <= 0:
            return False
        if transaction.transaction_type in [TransactionType.WITHDRAWAL, TransactionType.TRANSFER]:
            return userBalance >= transaction.amount
        return True

class BillService:
    """Business logic for bill payments"""

    @staticmethod
    def createBill(userId: str, billType: str, provider: str,
                   accountNumber: str, amount: float) -> Bill:
        """Create a new bill"""
        from src.domain.entities import Bill
        return Bill(
            bill_id=str(uuid.uuid4()),
            user_id=userId,
            bill_type=billType,
            provider=provider,
            account_number=accountNumber,
            amount=amount
        )

    @staticmethod
    def markBillPaid(bill) -> None:
        """Mark bill as paid"""
        bill.status = "paid"

class IntentClassificationService:
    """Business logic for intent classification"""

    INTENT_KEYWORDS = {
        "balance_inquiry": ["balance", "account", "how much", "money", "owo mi da"],
        "transfer": ["send", "transfer", "pay", "fi ranṣẹ", "aika"],
        "bill_payment": ["bill", "electricity", "water", "nepa", "ina"],
        "airtime": ["airtime", "recharge", "credit", "kaadi"],
        "loan": ["loan", "borrow", "awin"],
        "savings": ["save", "saving", "ifowopamọ"],
        "advice": ["help", "advice", "guide", "iranlọwọ"]
    }

    @staticmethod
    def classifyIntent(message: str, language: Language = Language.ENGLISH) -> Intent:
        """Classify user intent from message"""
        messageLower = message.lower()
        detectedIntent = "general_inquiry"
        confidence = 0.5
        entities = {}

        for intent, keywords in IntentClassificationService.INTENT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in messageLower:
                    detectedIntent = intent
                    confidence = 0.8
                    break
            if confidence > 0.7:
                break

        amounts = []
        for word in message.split():
            cleanWord = word.replace(",", "").replace("₦", "")
            try:
                amount = float(cleanWord)
                amounts.append(amount)
            except ValueError:
                continue

        if amounts:
            entities["amount"] = amounts[0]

        return Intent(
            intent_type=detectedIntent,
            confidence=confidence,
            entities=entities,
            raw_message=message,
            language=language
        )
