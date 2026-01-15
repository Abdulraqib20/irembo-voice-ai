"""
LLM Service - Groq and Gemini integration with task-specific model selection
"""

from typing import List, Dict, Any, Optional
import logging
from groq import AsyncGroq
import google.generativeai as genai
from src.config.settings import GROQ_API_KEY, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for LLM operations using Groq and Gemini with optimal model selection.

    Model Selection Strategy:
    - Chat:
    - SQL Generation:
    - Intent Classification:
    - Summarization:
    """

    # Model constants for task-specific routing
    MODEL_CHAT = "meta-llama/llama-4-maverick-17b-128e-instruct"
    MODEL_SQL = "openai/gpt-oss-120b"
    MODEL_INTENT = "qwen/qwen3-32b"
    MODEL_SUMMARY = "moonshotai/kimi-k2-instruct-0905"
    MODEL_DEFAULT = "meta-llama/llama-4-maverick-17b-128e-instruct"

    def __init__(self):
        self.groqClient = AsyncGroq(api_key=GROQ_API_KEY)
        genai.configure(api_key=GOOGLE_API_KEY)
        self.geminiModel = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.primaryProvider = "groq"

    async def generateResponse(self, message: str, systemPrompt: str,
                              conversationHistory: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using primary LLM (uses chat-optimized model)"""
        try:
            if self.primaryProvider == "groq":
                return await self.generateChatResponse(
                    message=message,
                    systemPrompt=systemPrompt,
                    conversationHistory=conversationHistory
                )
            else:
                return await self._generateGeminiResponse(message, systemPrompt, conversationHistory)
        except Exception as e:
            logger.error(f"Primary LLM failed: {str(e)}")
            if self.primaryProvider == "groq":
                logger.info("Falling back to Gemini")
                return await self._generateGeminiResponse(message, systemPrompt, conversationHistory)
            else:
                logger.info("Falling back to Groq")
                return await self.generateChatResponse(message, systemPrompt, conversationHistory)

    async def generateChatResponse(
        self,
        message: str,
        systemPrompt: str,
        conversationHistory: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.8
    ) -> str:
        return await self._generateGroqResponse(
            message=message,
            systemPrompt=systemPrompt,
            conversationHistory=conversationHistory,
            model=self.MODEL_CHAT,
            temperature=temperature,
            max_tokens=2048
        )

    async def generateSQLQuery(
        self,
        prompt: str,
        systemPrompt: str
    ) -> str:
        return await self._generateGroqResponse(
            message=prompt,
            systemPrompt=systemPrompt,
            conversationHistory=None,
            model=self.MODEL_SQL,
            temperature=0.2,  # Low temperature for deterministic code
            max_tokens=1024
        )

    async def summarizeConversation(
        self,
        text: str,
        systemPrompt: str
    ) -> str:
        return await self._generateGroqResponse(
            message=text,
            systemPrompt=systemPrompt,
            conversationHistory=None,
            model=self.MODEL_SUMMARY,
            temperature=0.5,
            max_tokens=512
        )

    async def _generateGroqResponse(
        self,
        message: str,
        systemPrompt: str,
        conversationHistory: Optional[List[Dict[str, str]]] = None,
        model: str = MODEL_DEFAULT,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Core Groq API call with configurable model selection.

        Args:
            message: User message
            systemPrompt: System prompt
            conversationHistory: Conversation history
            model: Groq model ID (e.g., "llama-3.3-70b-versatile")
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
        """
        messages = [{"role": "system", "content": systemPrompt}]

        if conversationHistory:
            messages.extend(conversationHistory[-5:])

        messages.append({"role": "user", "content": message})

        try:
            logger.info(f"Groq API call: model={model}, temp={temperature}, max_tokens={max_tokens}")

            completion = await self.groqClient.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content

        except Exception as e:
            logger.error(f"Groq API error with model {model}: {str(e)}")
            raise

    async def _generateGeminiResponse(self, message: str, systemPrompt: str,
                                     conversationHistory: Optional[List[Dict[str, str]]] = None) -> str:
        """Generate response using Gemini"""
        prompt = f"{systemPrompt}\n\n"

        if conversationHistory:
            for msg in conversationHistory[-5:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"

        prompt += f"User: {message}\nAssistant:"

        try:
            response = await self.geminiModel.generate_content_async(prompt)
            return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise

    async def classifyIntent(self, message: str) -> Dict[str, Any]:
        """Use LLM to classify intent"""
        systemPrompt = """You are an intent classifier for a Nigerian financial assistant.
Classify the user's intent into one of these categories:
- balance_inquiry
- transfer
- bill_payment
- airtime
- loan
- savings
- advice
- general_inquiry

Respond with JSON: {"intent": "category", "confidence": 0.0-1.0, "entities": {}}"""

        try:
            response = await self._generateGroqResponse(message, systemPrompt)
            import json
            return json.loads(response)
        except Exception as e:
            logger.error(f"Intent classification error: {str(e)}")
            return {"intent": "general_inquiry", "confidence": 0.5, "entities": {}}
