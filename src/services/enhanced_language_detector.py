"""
Enhanced Language Detector for Irembo Voice AI
Detects Kinyarwanda, English, and code-switched (mixed) utterances
"""

import os
import re
from typing import Dict, Optional, Any, Tuple
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0


class EnhancedLanguageDetector:
    """
    Language detector optimized for Kinyarwanda/English Voice AI.
    Handles code-switching detection which is common in Rwandan context.
    """
    
    def __init__(self, useGoogleTranslate: bool = False):
        self.useGoogleTranslate = useGoogleTranslate
        self.googleTranslateClient = None

        if self.useGoogleTranslate:
            try:
                from google.cloud import translate_v2 as translate
                self.googleTranslateClient = translate.Client()
            except Exception as e:
                print(f"Warning: Google Translate not available: {e}")
                self.useGoogleTranslate = False

        self.supportedLanguages = {'en', 'rw', 'mixed'}
        self.languageNames = {
            'en': 'English',
            'rw': 'Kinyarwanda',
            'mixed': 'Kinyarwanda-English Mixed',
            'fr': 'French'
        }

        # Kinyarwanda language indicators
        # Common words, verb prefixes, and grammatical patterns
        self.kinyarwandaIndicators = [
            # Common verbs and phrases
            'ndashaka', 'nashaka', 'nshaka', 'nkeneye', 'mumfashishe',
            'ndabashaka', 'urashaka', 'turashaka',
            # Question words
            'ese', 'ni iki', 'ni gute', 'ni ryari', 'ni he', 'ni nde',
            # Common nouns
            'amafaranga', 'amabwiriza', 'ikibazo', 'ubufasha', 'igihe',
            # Verb prefixes/suffixes
            'gusaba', 'gukora', 'gufata', 'kureba', 'kumenya', 'gutanga',
            'guhindura', 'kwishyura', 'kwinjira', 'gushyiraho', 'kongera',
            # Negation and connectors
            'ntabwo', 'ariko', 'ntibyemejwe', 'sinzi', 'ntabasha',
            # Common phrases in dataset
            'igeze he', 'aho igeze', 'yarangiye', 'natanze', 'nabonye',
            # Service-related
            'indangamuntu', 'pasiporo', 'icyemezo', 'icyangombwa', 'permis',
            'attestation', 'rendez-vous', 'appointment',
            # Locations (common in Rwanda)
            'i kigali', 'i huye', 'i rusizi', 'i muhanga', 'i nyagatare',
            'i rwamagana', 'i kicukiro',
            # Pronouns and possessives
            'yanjye', 'yawe', 'ye', 'yabo', 'yacu',
            # Status words
            'checka', 'niba', 'kuri', 'muri',
        ]

        # Strong English indicators
        self.englishIndicators = [
            # Question starters
            'what is', 'what are', 'how do', 'how can', 'how much',
            'where is', 'who can', 'can i', 'am i', 'do i',
            # Common phrases
            'i want to', 'i need to', 'i submitted', 'i paid',
            'please tell', 'help me', 'i cannot', 'i made',
            # Service words
            'application', 'status', 'requirements', 'documents',
            'appointment', 'payment', 'password', 'login', 'account',
            'upload', 'fee', 'eligibility', 'complaint',
            # Connectors
            'but', 'and', 'the', 'for', 'with',
        ]

        # Code-switching indicators (mixed Kinyarwanda + English)
        self.codeSwitchPatterns = [
            # Kinyarwanda + English noun
            r'ndashaka\s+\w*status',
            r'kureba\s+status',
            r'gukora\s+new\s+application',
            r'help\s+me\s+gu\w+',  # help me + Kinyarwanda verb
            r'ni\s+izihe\s+requirements',
            r'eligibility\s+ya\s+\w+',
            r'what\s+\w+\s+nkeneye',
            r'\w+\s+ariko\s+\w+',  # X ariko Y pattern
            r'please\s+\w+\s+(yanjye|yawe)',
            r'(application|status|password)\s+yanjye',
        ]
        
        # French borrowings common in Kinyarwanda context
        self.frenchBorrowings = [
            'rendez-vous', 'attestation', 'certificat', 'permis',
        ]

    def detectWithKeywords(self, text: str) -> Tuple[Optional[str], float]:
        """Detect language using keyword matching"""
        textLower = text.lower()
        
        # Count Kinyarwanda indicators
        rwScore = sum(1 for indicator in self.kinyarwandaIndicators 
                      if indicator in textLower)
        
        # Count English indicators
        enScore = sum(1 for indicator in self.englishIndicators 
                      if indicator in textLower)
        
        # Check for code-switching patterns
        mixedScore = sum(1 for pattern in self.codeSwitchPatterns 
                         if re.search(pattern, textLower))
        
        # French borrowings (neutral - common in both rw context)
        frScore = sum(1 for word in self.frenchBorrowings if word in textLower)
        
        # Boost mixed if both languages detected
        if rwScore >= 1 and enScore >= 1:
            mixedScore += 2
        
        # Add French borrowings to Kinyarwanda context
        rwScore += frScore * 0.5
        
        # Determine language
        scores = {
            'rw': rwScore * 1.2,  # Boost Kinyarwanda slightly
            'en': enScore * 1.0,
            'mixed': mixedScore * 1.5,  # Boost mixed detection
        }
        
        maxLang = max(scores.keys(), key=lambda k: scores[k])
        maxScore = scores[maxLang]
        
        # If mixed score is significant, prefer it
        if scores['mixed'] >= 2.0:
            confidence = min(0.95, 0.65 + (scores['mixed'] * 0.08))
            return 'mixed', confidence
        
        # Strong Kinyarwanda signal
        if scores['rw'] >= 2.0 and scores['rw'] > scores['en']:
            confidence = min(0.95, 0.6 + (scores['rw'] * 0.1))
            return 'rw', confidence
        
        # Strong English signal
        if scores['en'] >= 2.0 and scores['en'] > scores['rw']:
            confidence = min(0.95, 0.6 + (scores['en'] * 0.1))
            return 'en', confidence
        
        # Low confidence
        if maxScore >= 1.0:
            confidence = min(0.85, 0.5 + (maxScore * 0.1))
            return maxLang, confidence
        
        return None, 0.0

    def detectWithGoogle(self, text: str) -> Tuple[Optional[str], float]:
        """Use Google Translate API for detection (optional)"""
        if not self.googleTranslateClient:
            return None, 0.0

        try:
            result = self.googleTranslateClient.detect_language(text)
            detectedLang = result['language']
            confidence = result.get('confidence', 0.5)

            # Map to our language codes
            if detectedLang == 'rw':
                return 'rw', confidence
            elif detectedLang == 'en':
                return 'en', confidence
            elif detectedLang == 'fr':
                # French often detected for Kinyarwanda with French borrowings
                return 'rw', confidence * 0.7

            return detectedLang, confidence

        except Exception as e:
            print(f"Google Translate detection failed: {e}")
            return None, 0.0

    def detectWithLangdetect(self, text: str) -> Tuple[Optional[str], float]:
        """Use langdetect library for fallback detection"""
        try:
            detectedLang = detect(text)

            # langdetect maps: rw = Kinyarwanda (if available)
            if detectedLang == 'rw':
                return 'rw', 0.75
            elif detectedLang == 'en':
                return 'en', 0.80
            elif detectedLang == 'fr':
                # French often detected for Kinyarwanda text
                return 'rw', 0.60
            elif detectedLang == 'sw':
                # Swahili sometimes confused with Kinyarwanda
                return 'rw', 0.55

            return 'en', 0.50  # Default fallback

        except LangDetectException:
            return 'en', 0.50

    def detectLanguage(self, text: str) -> str:
        """
        Main detection method - returns language code.
        
        Returns:
            'en' for English
            'rw' for Kinyarwanda  
            'mixed' for code-switched
        """
        if not text or len(text.strip()) < 2:
            return 'en'

        # First try keyword-based detection (most reliable for our context)
        keywordLang, keywordConf = self.detectWithKeywords(text)
        if keywordConf >= 0.70 and keywordLang:
            return keywordLang

        # Try Google Translate if available
        if self.useGoogleTranslate:
            googleLang, googleConf = self.detectWithGoogle(text)
            if googleConf >= 0.75 and googleLang:
                return googleLang

        # Medium confidence from keywords
        if keywordConf >= 0.55 and keywordLang:
            return keywordLang

        # Fallback to langdetect
        langdetectLang, langdetectConf = self.detectWithLangdetect(text)
        return langdetectLang if langdetectLang else 'en'

    def getLanguageName(self, langCode: str) -> str:
        """Get human-readable language name"""
        return self.languageNames.get(langCode, 'English')

    def detectWithConfidence(self, text: str) -> Dict[str, Any]:
        """
        Detect language with full confidence details.
        
        Returns dict with:
            - language: Language code
            - language_name: Human-readable name
            - confidence: Detection confidence (0-1)
            - detection_method: Which method was used
        """
        if not text or len(text.strip()) < 2:
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'detection_method': 'default'
            }

        # Collect all detection results
        keywordLang, keywordConf = self.detectWithKeywords(text)
        googleLang, googleConf = None, 0.0
        langdetectLang, langdetectConf = None, 0.0

        if self.useGoogleTranslate:
            googleLang, googleConf = self.detectWithGoogle(text)

        langdetectLang, langdetectConf = self.detectWithLangdetect(text)

        detections = []
        if keywordConf > 0:
            detections.append(('keyword', keywordLang, keywordConf))
        if googleConf > 0:
            detections.append(('google', googleLang, googleConf))
        if langdetectConf > 0:
            detections.append(('langdetect', langdetectLang, langdetectConf))

        if not detections:
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'detection_method': 'fallback'
            }

        # Select best detection
        bestDetection = max(detections, key=lambda x: x[2])
        method, language, confidence = bestDetection

        # Prefer keyword detection for our domain
        if keywordConf >= 0.70 and keywordLang:
            language = keywordLang
            confidence = keywordConf
            method = 'keyword'
        elif googleConf >= 0.80 and googleLang:
            language = googleLang
            confidence = googleConf
            method = 'google_translate'

        finalLanguage = language if language else 'en'

        return {
            'language': finalLanguage,
            'language_name': self.getLanguageName(finalLanguage),
            'confidence': confidence,
            'detection_method': method
        }

    def isSupportedLanguage(self, langCode: str) -> bool:
        """Check if language is supported"""
        return langCode in self.supportedLanguages

    def isCodeSwitched(self, text: str) -> bool:
        """Check if text contains code-switching"""
        lang = self.detectLanguage(text)
        return lang == 'mixed'

    def translate(self, text: str, targetLanguage: str = 'en') -> Optional[str]:
        """Translate text using Google Translate (if available)"""
        if not self.googleTranslateClient:
            return None

        try:
            result = self.googleTranslateClient.translate(
                text,
                target_language=targetLanguage
            )
            return result['translatedText']
        except Exception as e:
            print(f"Translation failed: {e}")
            return None
