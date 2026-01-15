"""
Memory Extraction Service for LLM-based memory extraction.

Analyzes conversation turns to identify:
- Episodic memories (specific events, transactions)
- Preference memories (user likes, habits, methods)
- Goal memories (financial objectives, targets)
"""

import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractedMemory:
    """Represents an extracted memory from conversation"""
    memory_type: str
    content: str
    confidence: float
    context: Optional[str] = None


class MemoryExtractionService:
    """
    Service for extracting important memories from conversations using LLM.

    Memory Types:
    - episodic: Specific events, transactions, timeline references
    - preference: User likes/dislikes, habits, preferred methods
    - goal: Financial objectives, saving targets, future plans
    """

    def __init__(self, llm_service=None):
        """
        Initialize memory extraction service.

        Args:
            llm_service: LLM service for extraction (injected dependency)
        """
        self.llm_service = llm_service
        self.extraction_count = 0

    async def extract_memories(
        self,
        user_message: str,
        ai_response: str,
        session_id: Optional[str] = None,
        detected_intent: Optional[str] = None
    ) -> List[ExtractedMemory]:
        """
        Extract important memories from a conversation turn.

        Args:
            user_message: User's message text
            ai_response: AI's response text
            session_id: Optional session identifier
            detected_intent: Optional detected intent

        Returns:
            List of ExtractedMemory objects
        """
        try:
            if self.llm_service:
                memories = await self._extract_with_llm(
                    user_message,
                    ai_response,
                    detected_intent
                )
            else:
                memories = self._extract_with_rules(
                    user_message,
                    ai_response,
                    detected_intent
                )

            self.extraction_count += 1

            if memories:
                logger.info(
                    f"Extracted {len(memories)} memories from conversation "
                    f"(intent={detected_intent}, session={session_id})"
                )

            return memories

        except Exception as e:
            logger.error(f"Error extracting memories: {e}", exc_info=True)
            return []

    async def _extract_with_llm(
        self,
        user_message: str,
        ai_response: str,
        detected_intent: Optional[str]
    ) -> List[ExtractedMemory]:
        """
        Extract memories using LLM analysis.

        Args:
            user_message: User's message
            ai_response: AI's response
            detected_intent: Detected intent

        Returns:
            List of extracted memories
        """
        extraction_prompt = self._build_extraction_prompt(
            user_message,
            ai_response,
            detected_intent
        )

        try:
            response = await self.llm_service.generateResponse(
                message=extraction_prompt,
                systemPrompt=(
                    "You are a memory extraction system. Analyze conversations and extract "
                    "important long-term memories. Return JSON array only. "
                    "Be selective - only extract truly important information."
                ),
                conversationHistory=[]
            )

            response_clean = response.strip()

            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]

            response_clean = response_clean.strip()

            memories_data = json.loads(response_clean)

            if not isinstance(memories_data, list):
                logger.warning("LLM response is not a list")
                return []

            memories = []
            for mem_data in memories_data:
                if not isinstance(mem_data, dict):
                    continue

                memory_type = mem_data.get("type", "episodic")
                content = mem_data.get("content", "")
                confidence = float(mem_data.get("confidence", 0.5))

                if not content:
                    continue

                if confidence < 0.4:
                    continue

                memories.append(ExtractedMemory(
                    memory_type=memory_type,
                    content=content,
                    confidence=min(confidence, 1.0),
                    context=mem_data.get("context")
                ))

            return memories

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM memory extraction response: {e}")
            return self._extract_with_rules(user_message, ai_response, detected_intent)

        except Exception as e:
            logger.error(f"Error in LLM memory extraction: {e}")
            return []

    def _extract_with_rules(
        self,
        user_message: str,
        ai_response: str,
        detected_intent: Optional[str]
    ) -> List[ExtractedMemory]:
        """
        Extract memories using rule-based patterns (fallback).

        Args:
            user_message: User's message
            ai_response: AI's response
            detected_intent: Detected intent

        Returns:
            List of extracted memories
        """
        memories = []
        user_lower = user_message.lower()

        preference_indicators = [
            "i prefer", "i like", "i love", "i hate", "i don't like",
            "i always", "i usually", "i never", "my favorite"
        ]

        for indicator in preference_indicators:
            if indicator in user_lower:
                memories.append(ExtractedMemory(
                    memory_type="preference",
                    content=user_message,
                    confidence=0.7,
                    context=f"Detected preference indicator: '{indicator}'"
                ))
                break

        goal_indicators = [
            "i want to save", "my goal is", "i'm planning to", "i need to",
            "i'm saving for", "i want to buy", "my target is"
        ]

        for indicator in goal_indicators:
            if indicator in user_lower:
                memories.append(ExtractedMemory(
                    memory_type="goal",
                    content=user_message,
                    confidence=0.75,
                    context=f"Detected goal indicator: '{indicator}'"
                ))
                break

        episodic_intents = [
            "transfer_money", "pay_bill", "loan_inquiry", "airtime_purchase"
        ]

        if detected_intent in episodic_intents:
            memories.append(ExtractedMemory(
                memory_type="episodic",
                content=f"User performed {detected_intent}: {user_message}",
                confidence=0.8,
                context=f"Transaction intent: {detected_intent}"
            ))

        amount_pattern = r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:naira|ngn|₦)?'
        import re
        if re.search(amount_pattern, user_lower, re.IGNORECASE):
            if not any(m.memory_type == "episodic" for m in memories):
                memories.append(ExtractedMemory(
                    memory_type="episodic",
                    content=user_message,
                    confidence=0.65,
                    context="Contains financial amount"
                ))

        return memories

    def _build_extraction_prompt(
        self,
        user_message: str,
        ai_response: str,
        detected_intent: Optional[str]
    ) -> str:
        """
        Build prompt for LLM memory extraction.

        Args:
            user_message: User's message
            ai_response: AI's response
            detected_intent: Detected intent

        Returns:
            Extraction prompt
        """
        intent_context = f"\nDetected Intent: {detected_intent}" if detected_intent else ""

        return f"""Analyze this conversation turn and extract important long-term memories.

User: "{user_message}"
AI: "{ai_response}"{intent_context}

Extract memories in these categories:

1. **Preference** - User likes/dislikes, habits, preferred methods
   Example: "User prefers mobile banking over USSD"

2. **Goal** - Financial objectives, saving targets, future plans
   Example: "User wants to save 100,000 NGN for daughter's school fees by December"

3. **Episodic** - Specific events, transactions, timeline references
   Example: "User sends 10,000 NGN to Lagos monthly for family support"

Rules:
- Only extract if truly important for future conversations
- Assign confidence: 0.4-0.6 (uncertain), 0.7-0.8 (probable), 0.9-1.0 (certain)
- Skip generic chat, greetings, confirmations
- Extract at most 2-3 memories per turn
- Be specific and actionable

Return JSON array:
[
  {{"type": "preference", "content": "...", "confidence": 0.9}},
  {{"type": "goal", "content": "...", "confidence": 0.85}}
]

If no important memories, return: []"""

    def get_stats(self) -> Dict[str, Any]:
        """
        Get extraction statistics.

        Returns:
            Dict with extraction count
        """
        return {
            "total_extractions": self.extraction_count,
            "llm_enabled": self.llm_service is not None
        }

    def reset_stats(self) -> None:
        """Reset extraction counter"""
        self.extraction_count = 0


def create_memory_extraction_service(llm_service=None) -> MemoryExtractionService:
    """
    Factory function to create memory extraction service.

    Args:
        llm_service: Optional LLM service for extraction

    Returns:
        MemoryExtractionService instance
    """
    return MemoryExtractionService(llm_service=llm_service)
