"""
Speech-to-Text Service - Groq Whisper integration
Uses Groq's Whisper-large-v3 model (free tier alternative to OpenAI)
"""

from typing import Optional, Dict, Any
import logging
import io
from groq import AsyncGroq
from src.config.settings import GROQ_API_KEY  # type: ignore

logger = logging.getLogger(__name__)

class SpeechToTextService:
    """Service for speech-to-text using Groq Whisper-large-v3"""

    def __init__(self):
        self.client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.model = "whisper-large-v3"

        if not self.client:
            logger.warning("Groq API key not configured - STT unavailable")

    async def transcribe(self, audioFile: bytes, language: Optional[str] = None) -> str:
        """
        Transcribe audio to text using Groq Whisper

        Args:
            audioFile: Audio file bytes (WAV, MP3, M4A, etc.)
            language: Optional language hint (en, yo, ha, ig, pcm)

        Returns:
            Transcribed text string
        """
        try:
            if not self.client:
                raise ValueError("Groq API key not configured")

            # Groq expects file-like object with name attribute
            audioBuffer = io.BytesIO(audioFile)
            audioBuffer.name = "audio.wav"

            # Build request params
            requestParams: Dict[str, Any] = {
                "file": audioBuffer,
                "model": self.model,
                "response_format": "json",
                "temperature": 0.0
            }

            # Only add language if valid (Groq Whisper supports these Nigerian langs)
            if language and language in ["en", "yo", "ha", "ig"]:
                requestParams["language"] = language

            response = await self.client.audio.transcriptions.create(**requestParams)

            transcribedText = response.text
            logger.info(f"Audio transcribed: {len(transcribedText)} chars")
            return transcribedText

        except Exception as e:
            logger.error(f"Groq STT error: {str(e)}")
            raise

    async def transcribeWithLanguageDetection(self, audioFile: bytes) -> Dict[str, Any]:
        """
        Transcribe audio with automatic language detection

        Args:
            audioFile: Audio file bytes

        Returns:
            {
                "text": str,
                "language": str,
                "duration": float
            }
        """
        try:
            if not self.client:
                raise ValueError("Groq API key not configured")

            audioBuffer = io.BytesIO(audioFile)
            audioBuffer.name = "audio.wav"

            response = await self.client.audio.transcriptions.create(
                file=audioBuffer,
                model=self.model,
                response_format="verbose_json",
                temperature=0.0
            )

            result = {
                "text": response.text,
                "language": getattr(response, 'language', 'en'),
                "duration": getattr(response, 'duration', 0.0)
            }

            logger.info(
                f"Audio transcribed with detection: {result['language']} "
                f"({result['duration']:.2f}s)"
            )

            return result

        except Exception as e:
            logger.error(f"Groq STT with detection error: {str(e)}")
            raise
