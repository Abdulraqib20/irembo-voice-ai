"""
Dialogue Manager with Multi-Turn Slot-Filling State Machine.

Handles:
- Intent-based slot requirements (transfer, check_balance, pay_bill, loan_inquiry)
- Intelligent slot prompting in Nigerian languages
- Context-aware slot extraction from user responses
- Slot validation and confirmation
- Multi-turn conversation tracking
- Integration with slot-filling repository for persistence
"""

import logging
import re
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SlotStatus(Enum):
    """Status of slot-filling process"""
    ACTIVE = "active"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    FAILED = "failed"


@dataclass
class SlotDefinition:
    """Definition of a required slot"""
    name: str
    prompt: Dict[str, str]  # Language-specific prompts
    validation_pattern: Optional[str] = None
    validation_fn: Optional[Callable] = None
    examples: Optional[Dict[str, List[str]]] = None


@dataclass
class SlotValue:
    """Extracted slot value with confidence"""
    value: Any
    confidence: float
    source: str  # "user_input" | "context" | "default"


class IntentSlotMapper:
    """
    Maps intents to required slots with multilingual prompting.

    Supports Nigerian languages: English, Pidgin, Yoruba, Hausa, Igbo
    """

    INTENT_SLOTS: Dict[str, List[SlotDefinition]] = {
        "transfer_money": [
            SlotDefinition(
                name="recipient",
                prompt={
                    "en": "💸 Who would you like to send money to? (phone number or contact name)",
                    "pidgin": "💸 Who you wan send money give? (phone number or contact name)",
                    "yo": "💸 Ta ni o fẹ fi owo ranṣẹ si? (nọmba foonu tabi orukọ)",
                    "ha": "💸 Wa kake son aika kuɗi? (lambar waya ko suna)",
                    "ig": "💸 Ònye ị chọrọ iziga ego? (nọmba ekwentị ma ọ bụ aha)"
                },
                examples={
                    "en": ["Raqib Omotosho", "08012345678", "+2348012345678", "My brother"],
                    "pidgin": ["My mama", "08012345678"],
                }
            ),
            SlotDefinition(
                name="amount",
                prompt={
                    "en": "💰 How much would you like to send from your Finclusion wallet? (in Naira)",
                    "pidgin": "💰 How much you wan send from your Finclusion wallet? (for Naira)",
                    "yo": "💰 Elo ni o fẹ ranṣẹ lati Finclusion wallet rẹ? (ni Naira)",
                    "ha": "💰 Nawa kake son aika daga Finclusion wallet ku? (a Naira)",
                    "ig": "💰 Ego ole ka ị chọrọ iziga site na Finclusion wallet gị? (na Naira)"
                },
                validation_pattern=r"^\d+(\.\d{1,2})?$",
                examples={
                    "en": ["5000", "10000.50", "NGN 2500", "2k"],
                }
            ),
            SlotDefinition(
                name="pin",
                prompt={
                    "en": "🔒 Please enter your 4-digit Finclusion PIN to authorize this transfer:",
                    "pidgin": "🔒 Abeg enter your 4-digit Finclusion PIN to confirm this transfer:",
                    "yo": "🔒 Jọwọ tẹ PIN 4-digit Finclusion rẹ lati jẹrisi ifiranṣẹ yii:",
                    "ha": "🔒 Don Allah shigar da PIN 4-digit Finclusion ku don tabbatar da wannan aikawa:",
                    "ig": "🔒 Biko tinye PIN 4-digit Finclusion gị iji kwado mbufe a:"
                },
                validation_pattern=r"^\d{4}$",
                examples={
                    "en": ["1234", "5678"],
                }
            )
        ],

        "check_balance": [
            SlotDefinition(
                name="account_type",
                prompt={
                    "en": "Which account balance? (savings, current, or all)",
                    "pidgin": "Which account balance? (savings, current, or all)",
                    "yo": "Iru akọọlẹ wo ni o fẹ wo? (ifowopamọ, lọwọlọwọ, tabi gbogbo)",
                    "ha": "Wane asusun kake son duba? (ajiya, na yanzu, ko duka)",
                    "ig": "Kedu akaụntụ ka ị chọrọ ịhụ? (nchekwa, ugbu a, ma ọ bụ niile)"
                },
                examples={
                    "en": ["savings", "current", "all"],
                }
            )
        ],

        "pay_bill": [
            SlotDefinition(
                name="biller",
                prompt={
                    "en": "Who would you like to pay? (electricity, water, airtime, data)",
                    "pidgin": "Who you wan pay? (light, water, airtime, data)",
                    "yo": "Ta ni o fẹ sanwo fun? (ina, omi, airtime, data)",
                    "ha": "Wa kake son biya? (wutar lantarki, ruwa, airtime, data)",
                    "ig": "Ònye ka ị chọrọ ịkwụ ụgwọ? (ọkụ, mmiri, airtime, data)"
                },
                examples={
                    "en": ["IKEDC", "DSTV", "MTN", "Glo", "Airtel"],
                }
            ),
            SlotDefinition(
                name="amount",
                prompt={
                    "en": "How much would you like to pay? (in Naira)",
                    "pidgin": "How much you wan pay? (for Naira)",
                    "yo": "Elo ni o fẹ sanwo? (ni Naira)",
                    "ha": "Nawa kake son biya? (a Naira)",
                    "ig": "Ego ole ka ị chọrọ ịkwụ? (na Naira)"
                },
                validation_pattern=r"^\d+(\.\d{1,2})?$"
            ),
            SlotDefinition(
                name="account",
                prompt={
                    "en": "From which account? (savings, current)",
                    "pidgin": "From which account? (savings, current)",
                    "yo": "Lati iru akọọlẹ wo? (ifowopamọ, lọwọlọwọ)",
                    "ha": "Daga wane asusun? (ajiya, na yanzu)",
                    "ig": "Site n'akaụntụ ole? (nchekwa, ugbu a)"
                },
                examples={
                    "en": ["savings", "current"],
                }
            )
        ],

        "loan_inquiry": [
            SlotDefinition(
                name="loan_type",
                prompt={
                    "en": "What type of loan? (personal, business, education, agriculture)",
                    "pidgin": "Wetin type of loan? (personal, business, education, agriculture)",
                    "yo": "Iru awin wo? (ti ara ẹni, iṣowo, ẹkọ, ogbin)",
                    "ha": "Wane irin bashi? (na mutum, kasuwanci, ilimi, noma)",
                    "ig": "Ụdị mbinye ego ole? (nke onwe, azụmahịa, agụmakwụkwọ, ọrụ ugbo)"
                },
                examples={
                    "en": ["personal", "business", "education", "agriculture"],
                }
            ),
            SlotDefinition(
                name="amount",
                prompt={
                    "en": "How much do you need? (in Naira)",
                    "pidgin": "How much you need? (for Naira)",
                    "yo": "Elo ni o nilo? (ni Naira)",
                    "ha": "Nawa kake bukata? (a Naira)",
                    "ig": "Ego ole ka ị chọrọ? (na Naira)"
                },
                validation_pattern=r"^\d+(\.\d{1,2})?$"
            ),
            SlotDefinition(
                name="duration",
                prompt={
                    "en": "For how long? (e.g., 6 months, 1 year, 2 years)",
                    "pidgin": "For how long? (like 6 months, 1 year, 2 years)",
                    "yo": "Fun iye akoko wo? (bii oṣu mẹfa, ọdun kan, ọdun meji)",
                    "ha": "Har zuwa yaushe? (kamar wata 6, shekara 1, shekara 2)",
                    "ig": "Ruo ogologo oge ole? (dị ka ọnwa 6, afọ 1, afọ 2)"
                },
                examples={
                    "en": ["6 months", "1 year", "2 years", "12 months"],
                }
            )
        ],

        "deposit_funds": [
            SlotDefinition(
                name="amount",
                prompt={
                    "en": "💰 How much would you like to deposit into your Finclusion wallet? (in Naira)",
                    "pidgin": "💰 How much you wan put for your Finclusion wallet? (for Naira)",
                    "yo": "💰 Elo ni o fẹ fi sii ninu Finclusion wallet rẹ? (ni Naira)",
                    "ha": "💰 Nawa kake son saka a cikin Finclusion wallet ku? (a Naira)",
                    "ig": "💰 Ego ole ka ị chọrọ itinye n'ime Finclusion wallet gị? (na Naira)"
                },
                validation_pattern=r"^\d+(\.\d{1,2})?$",
                examples={
                    "en": ["5000", "10000.50", "NGN 2500", "50000"],
                }
            ),
            SlotDefinition(
                name="pin",
                prompt={
                    "en": "🔒 Please enter your 4-digit PIN to authorize this deposit:",
                    "pidgin": "🔒 Abeg enter your 4-digit PIN to confirm this deposit:",
                    "yo": "🔒 Jọwọ tẹ PIN 4-digit rẹ lati jẹrisi ifipamọ yii:",
                    "ha": "🔒 Don Allah shigar da PIN 4-digit ku don tabbatar da wannan ajiya:",
                    "ig": "🔒 Biko tinye PIN 4-digit gị iji kwado nkwụnye ego a:"
                },
                validation_pattern=r"^\d{4}$",
                examples={
                    "en": ["1234", "5678"],
                }
            )
        ]
    }

    @classmethod
    def get_required_slots(cls, intent: str) -> List[SlotDefinition]:
        """Get required slots for an intent"""
        return cls.INTENT_SLOTS.get(intent, [])

    @classmethod
    def get_slot_prompt(cls, intent: str, slot_name: str, language: str = "en") -> str:
        """Get localized prompt for a slot"""
        slots = cls.get_required_slots(intent)
        for slot in slots:
            if slot.name == slot_name:
                return slot.prompt.get(language, slot.prompt.get("en", ""))
        return ""


class SlotExtractor:
    """Extracts slot values from user input using patterns and context"""

    @staticmethod
    def extract_amount(text: str) -> Optional[SlotValue]:
        """Extract monetary amount from text"""
        shorthand_pattern = r'(\d+(?:\.\d+)?)\s*k\b'
        match = re.search(shorthand_pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1)) * 1000
            return SlotValue(
                value=amount,
                confidence=0.95,
                source="user_input"
            )

        patterns = [
            r'(?:NGN|₦|N)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:naira|NGN|₦)',
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    return SlotValue(
                        value=amount,
                        confidence=0.9,
                        source="user_input"
                    )
                except ValueError:
                    continue
        return None

    @staticmethod
    def extract_account_type(text: str) -> Optional[SlotValue]:
        """Extract account type from text"""
        text_lower = text.lower()

        account_keywords = {
            "savings": ["savings", "save", "saving", "ifowopamọ", "ajiya", "nchekwa"],
            "current": ["current", "checking", "lọwọlọwọ", "na yanzu", "ugbu a"],
        }

        for account_type, keywords in account_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return SlotValue(
                    value=account_type,
                    confidence=0.85,
                    source="user_input"
                )

        if "all" in text_lower or "both" in text_lower:
            return SlotValue(value="all", confidence=0.9, source="user_input")

        return None

    @staticmethod
    def extract_recipient(text: str) -> Optional[SlotValue]:
        """Extract recipient (phone number or name) from text

        Handles formats like:
        - "Chidera Ozigbo, 9025965477"
        - "08012345678"
        - "+2348012345678"
        - "John Doe"
        - "my brother 07025965477"
        """
        # Clean text - remove extra whitespace
        text = text.strip()

        # Phone patterns - expanded to handle more formats
        phone_patterns = [
            r'\b(\+234[7-9]\d{9})\b',           # +2347012345678
            r'\b(0[7-9]\d{9})\b',               # 07012345678 (11 digits)
            r'\b(234[7-9]\d{9})\b',             # 2347012345678
            r'\b([7-9][0-1]\d{8})\b',           # 9025965477 (10 digits without leading 0)
        ]

        phone_found = None
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                phone = match.group(1)
                # Normalize to +234 format
                if phone.startswith('+'):
                    phone_found = phone
                elif phone.startswith('0'):
                    phone_found = '+234' + phone[1:]
                elif phone.startswith('234'):
                    phone_found = '+' + phone
                elif len(phone) == 10 and phone[0] in '789':
                    # 10-digit number like 9025965477 -> +2349025965477
                    phone_found = '+234' + phone
                else:
                    phone_found = phone
                break

        # Name extraction - clean punctuation and find capitalized words
        # Remove the phone number from text first to get clean name
        text_for_name = text
        if phone_found:
            # Remove phone from text to extract name
            for pattern in phone_patterns:
                text_for_name = re.sub(pattern, '', text_for_name)

        # Clean punctuation and split
        text_for_name = re.sub(r'[,;:]+', ' ', text_for_name)  # Replace commas/semicolons with space
        text_for_name = text_for_name.strip()

        name_words = []
        common_names = ['john', 'mary', 'david', 'sarah', 'ade', 'bola', 'musa', 'amina',
                       'chidi', 'ngozi', 'chidera', 'ozigbo', 'raqib', 'emeka', 'kunle',
                       'tunde', 'yemi', 'sade', 'chinedu', 'tola', 'femi', 'bisi']

        for word in text_for_name.split():
            # Clean word of remaining punctuation
            word = re.sub(r'[^\w]', '', word)
            if not word:
                continue
            # Accept if: starts with uppercase OR is a known name OR looks like a name (2+ chars, alphabetic)
            if (word[0].isupper() or
                word.lower() in common_names or
                (len(word) >= 2 and word.isalpha() and word[0].isupper())):
                name_words.append(word.capitalize())

        # Combine name and phone if both found
        if name_words and phone_found:
            # Return combined: "Chidera Ozigbo (+2349025965477)"
            name = " ".join(name_words)
            return SlotValue(
                value=f"{name} ({phone_found})",
                confidence=0.95,
                source="user_input"
            )
        elif phone_found:
            return SlotValue(
                value=phone_found,
                confidence=0.95,
                source="user_input"
            )
        elif name_words:
            return SlotValue(
                value=" ".join(name_words),
                confidence=0.7,
                source="user_input"
            )

        # Last resort: if text looks like just a name (2+ words, no numbers)
        if not re.search(r'\d', text) and len(text.split()) >= 1:
            return SlotValue(
                value=text.strip(),
                confidence=0.5,
                source="user_input"
            )

        return None

    @staticmethod
    def extract_pin(text: str) -> Optional[SlotValue]:
        """Extract 4-digit PIN from text"""
        pin_pattern = r'\b(\d{4})\b'
        match = re.search(pin_pattern, text)
        if match:
            return SlotValue(
                value=match.group(1),
                confidence=0.95,
                source="user_input"
            )
        return None

    @staticmethod
    def extract_duration(text: str) -> Optional[SlotValue]:
        """Extract loan duration from text"""
        patterns = [
            (r'(\d+)\s*(?:months?|moons?|oṣu|wata|ọnwa)', lambda m: f"{m.group(1)} months"),
            (r'(\d+)\s*(?:years?|ọdun|shekara|afọ)', lambda m: f"{m.group(1)} years"),
        ]

        for pattern, formatter in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return SlotValue(
                    value=formatter(match),
                    confidence=0.9,
                    source="user_input"
                )

        return None


class DialogueManager:
    """
    Main dialogue manager with slot-filling state machine.

    Manages multi-turn conversations to collect required information.
    """

    def __init__(self, slot_filling_repo, memory_system=None):
        """
        Initialize dialogue manager.

        Args:
            slot_filling_repo: Repository for slot-filling state persistence
            memory_system: Optional memory system for context-aware extraction
        """
        self.slot_repo = slot_filling_repo
        self.memory_system = memory_system
        self.extractor = SlotExtractor()
        self.max_prompts_per_slot = 3

    async def start_slot_filling(
        self,
        session_id: str,
        intent: str
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Start slot-filling process for an intent.

        Returns:
            (success, message, state_info)
        """
        required_slots = IntentSlotMapper.get_required_slots(intent)

        if not required_slots:
            return False, f"No slots defined for intent: {intent}", None

        slot_names = [slot.name for slot in required_slots]

        try:
            state = await self.slot_repo.create_slot_state(
                session_id=session_id,
                intent=intent,
                required_slots=slot_names
            )

            first_slot = slot_names[0]

            return True, "", {
                "state_id": state.state_id,
                "intent": intent,
                "current_slot": first_slot,
                "required_slots": slot_names,
                "filled_slots": {},
                "status": "active"
            }

        except Exception as e:
            logger.error(f"Failed to start slot-filling: {e}")
            return False, f"Error starting slot-filling: {str(e)}", None

    async def process_slot_response(
        self,
        session_id: str,
        user_input: str,
        language: str = "en",
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Process user response for active slot-filling.

        Returns:
            (is_complete, next_prompt, state_info)
        """
        try:
            state = await self.slot_repo.get_active_slot_state(session_id)

            if not state:
                return False, "No active slot-filling session", None

            # 🔥 CRITICAL FIX: Database stores "in_progress", NOT "active"
            if state.status != "in_progress":
                return False, f"Slot-filling session is {state.status}", None

            current_slot = state.current_slot
            intent = state.intent

            extracted_value = await self._extract_slot_value(
                intent=intent,
                slot_name=current_slot,
                user_input=user_input,
                context=context
            )

            if extracted_value:
                # Parse filled_slots from JSON string (stored as TEXT in database)
                try:
                    filled_slots = json.loads(state.filled_slots) if state.filled_slots else {}
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Invalid filled_slots JSON: {state.filled_slots}, Error: {e}")
                    filled_slots = {}

                filled_slots[current_slot] = {
                    "value": extracted_value.value,
                    "confidence": extracted_value.confidence,
                    "source": extracted_value.source
                }

                # 🔥 CRITICAL FIX: Parse required_slots from JSON string (stored as TEXT in database)
                # Without this, iterating over `state.required_slots` iterates over characters, not slot names!
                try:
                    required_slots_list = json.loads(state.required_slots) if isinstance(state.required_slots, str) else state.required_slots
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Invalid required_slots JSON: {state.required_slots}, Error: {e}")
                    required_slots_list = []

                remaining_slots = [
                    s for s in required_slots_list
                    if s not in filled_slots
                ]

                if remaining_slots:
                    next_slot = remaining_slots[0]

                    await self.slot_repo.update_slot_state(
                        state_id=state.state_id,
                        filled_slots=filled_slots,
                        current_slot=next_slot,
                        status="in_progress"  # 🔥 FIX: Use "in_progress" not "active"
                    )

                    next_prompt = IntentSlotMapper.get_slot_prompt(
                        intent=intent,
                        slot_name=next_slot,
                        language=language
                    )

                    return False, next_prompt, {
                        "state_id": state.state_id,
                        "intent": intent,
                        "current_slot": next_slot,
                        "filled_slots": filled_slots,
                        "required_slots": required_slots_list,  # Use parsed list, not JSON string
                        "status": "active"
                    }
                else:
                    # All slots filled - mark as awaiting_confirmation instead of completing for transfers
                    if intent == "transfer_money":
                        await self.slot_repo.update_slot_state(
                            state_id=state.state_id,
                            filled_slots=filled_slots,
                            current_slot="awaiting_confirmation",  # Special marker
                            status="in_progress"  # 🔥 FIX: Keep as "in_progress" for confirmation
                        )

                        return True, "", {
                            "state_id": state.state_id,
                            "intent": intent,
                            "filled_slots": filled_slots,
                            "current_slot": "awaiting_confirmation",
                            "status": "awaiting_confirmation"  # Special status for chat_with_memory
                        }
                    else:
                        # For other intents, complete immediately
                        await self.slot_repo.complete_slot_state(state.state_id)

                        return True, "", {
                            "state_id": state.state_id,
                            "intent": intent,
                            "filled_slots": filled_slots,
                            "status": "completed"
                        }
            else:
                # 🔥 FIX: slot_prompts_count is an integer, not a dict
                prompts_count = state.slot_prompts_count + 1

                if prompts_count >= self.max_prompts_per_slot:
                    await self.slot_repo.abandon_slot_state(state.state_id)

                    return False, "I'm having trouble understanding. Let's try again later.", {
                        "state_id": state.state_id,
                        "status": "abandoned",
                        "reason": "max_prompts_exceeded"
                    }

                # Parse filled_slots to pass as dict (not JSON string)
                try:
                    filled_slots_dict = json.loads(state.filled_slots) if state.filled_slots else {}
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Invalid filled_slots JSON during retry: {state.filled_slots}, Error: {e}")
                    filled_slots_dict = {}

                await self.slot_repo.update_slot_state(
                    state_id=state.state_id,
                    filled_slots=filled_slots_dict,
                    current_slot=current_slot,
                    status="in_progress",  # 🔥 FIX: Use "in_progress" not "active"
                    increment_prompts=True
                )

                retry_prompt = self._generate_retry_prompt(
                    intent=intent,
                    slot_name=current_slot,
                    language=language,
                    attempt=prompts_count
                )

                return False, retry_prompt, {
                    "state_id": state.state_id,
                    "intent": intent,
                    "current_slot": current_slot,
                    "status": "active",
                    "prompts_count": prompts_count
                }

        except Exception as e:
            logger.error(f"Error processing slot response: {e}")
            return False, f"Error: {str(e)}", None

    async def _extract_slot_value(
        self,
        intent: str,
        slot_name: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[SlotValue]:
        """Extract slot value using appropriate extraction method"""

        if slot_name == "amount":
            return self.extractor.extract_amount(user_input)
        elif slot_name == "account_type" or slot_name == "source_account" or slot_name == "account":
            return self.extractor.extract_account_type(user_input)
        elif slot_name == "recipient":
            return self.extractor.extract_recipient(user_input)
        elif slot_name == "pin":
            return self.extractor.extract_pin(user_input)
        elif slot_name == "duration":
            return self.extractor.extract_duration(user_input)
        elif slot_name in ["biller", "loan_type"]:
            return SlotValue(
                value=user_input.strip(),
                confidence=0.8,
                source="user_input"
            )

        return None

    def _generate_retry_prompt(
        self,
        intent: str,
        slot_name: str,
        language: str,
        attempt: int
    ) -> str:
        """Generate helpful retry prompt"""
        base_prompt = IntentSlotMapper.get_slot_prompt(intent, slot_name, language)

        retry_prefixes = {
            "en": [
                "I didn't quite understand that. ",
                "Could you please clarify? ",
                "I need a bit more information. "
            ],
            "pidgin": [
                "I no really understand. ",
                "Abeg explain am well. ",
                "I need small more information. "
            ]
        }

        prefixes = retry_prefixes.get(language, retry_prefixes["en"])
        prefix = prefixes[min(attempt - 1, len(prefixes) - 1)]

        return f"{prefix}{base_prompt}"

    async def abandon_slot_filling(self, session_id: str) -> bool:
        """Abandon active slot-filling session"""
        try:
            state = await self.slot_repo.get_active_slot_state(session_id)
            if state:
                await self.slot_repo.abandon_slot_state(state.state_id)
                return True
            return False
        except Exception as e:
            logger.error(f"Error abandoning slot-filling: {e}")
            return False


async def create_dialogue_manager(slot_filling_repo, memory_system=None):
    """Factory function to create dialogue manager"""
    return DialogueManager(slot_filling_repo, memory_system)
