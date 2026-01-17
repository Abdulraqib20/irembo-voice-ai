"""
Enhanced language detector for Kinyarwanda/English/mixed utterances.

This is a lightweight heuristic detector intended for production routing
when external language ID is unavailable. It avoids heavy dependencies
and works on short, code-switched utterances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import re


@dataclass
class LanguageDetectionResult:
    """Language detection result payload."""

    language: str
    confidence: float
    scores: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable dictionary representation."""
        return {
            "language": self.language,
            "confidence": self.confidence,
            "scores": self.scores,
        }


class EnhancedLanguageDetector:
    """
    Simple heuristic language detector for rw/en/mixed.

    Parameters:
        useGoogleTranslate (bool): Placeholder flag for compatibility with
            earlier prototypes that used Google Translate detection.
    """

    _rw_markers: List[str] = [
        "nd", "nshaka", "urakoze", "murakoze", "aho", "igeze", "ni",
        "iki", "kumenya", "amafaranga", "gusaba", "gusaba",
        "gufata", "igihe", "rendez", "nkeneye", "ibisabwa",
        "ngomba", "nishyuye",
    ]
    _en_markers: List[str] = [
        "status", "apply", "application", "fee", "cost", "payment",
        "schedule", "appointment", "requirements", "documents",
        "reset", "password", "help", "track",
    ]

    def __init__(self, useGoogleTranslate: bool = False, **_: Any) -> None:
        self.use_google_translate = useGoogleTranslate

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect language category from text.

        Args:
            text (str): User utterance.

        Returns:
            Dict[str, Any]: Detection result with language code and confidence.
        """
        normalized = self._normalize(text)
        tokens = normalized.split()

        if not tokens:
            result = LanguageDetectionResult(
                language="unknown",
                confidence=0.0,
                scores={"rw": 0.0, "en": 0.0, "mixed": 0.0},
            )
            return result.to_dict()

        rw_score = self._score(tokens, self._rw_markers)
        en_score = self._score(tokens, self._en_markers)

        if rw_score > 0 and en_score > 0:
            language = "mixed"
        elif rw_score >= en_score:
            language = "rw" if rw_score > 0 else "unknown"
        else:
            language = "en" if en_score > 0 else "unknown"

        total = rw_score + en_score
        confidence = (max(rw_score, en_score) / total) if total else 0.0

        result = LanguageDetectionResult(
            language=language,
            confidence=round(confidence, 3),
            scores={
                "rw": round(rw_score, 3),
                "en": round(en_score, 3),
                "mixed": 1.0 if language == "mixed" else 0.0,
            },
        )
        return result.to_dict()

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for token matching."""
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s-]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _score(tokens: List[str], markers: List[str]) -> float:
        """Compute a simple score based on marker matches."""
        hits = 0
        for token in tokens:
            for marker in markers:
                if marker in token:
                    hits += 1
                    break
        return hits / max(len(tokens), 1)
