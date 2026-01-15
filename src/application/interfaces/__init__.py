"""
Application interfaces and abstractions
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """Abstract interface for LLM providers"""

    @abstractmethod
    async def generateResponse(self, message: str, systemPrompt: str,
                              conversationHistory: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response from LLM"""
        pass

class MemoryInterface(ABC):
    """Abstract interface for memory providers"""

    @abstractmethod
    async def saveMessage(self, userId: str, message: str, role: str) -> None:
        """Save message to memory"""
        pass

    @abstractmethod
    async def getConversationHistory(self, userId: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieve conversation history"""
        pass

class SpeechInterface(ABC):
    """Abstract interface for speech services"""

    @abstractmethod
    async def transcribe(self, audioFile: bytes, language: Optional[str] = None) -> str:
        """Transcribe audio to text"""
        pass

    @abstractmethod
    async def synthesize(self, text: str, language: str = "en-US") -> bytes:
        """Synthesize text to speech"""
        pass

class TranslationInterface(ABC):
    """Abstract interface for translation services"""

    @abstractmethod
    async def translate(self, text: str, targetLang: str, sourceLang: Optional[str] = None) -> str:
        """Translate text"""
        pass

    @abstractmethod
    async def detectLanguage(self, text: str) -> str:
        """Detect language"""
        pass
