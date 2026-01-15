import os
from typing import Dict, Optional, Any
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

class EnhancedLanguageDetector:
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

        self.supportedLanguages = {'en', 'yo', 'ha', 'ig'}
        self.languageNames = {
            'en': 'English',
            'yo': 'Yoruba',
            'ha': 'Hausa',
            'ig': 'Igbo',
            'pcm': 'Nigerian Pidgin'
        }

        self.pidginIndicators = [
            'wetin', 'dey', 'abeg', 'abi', 'wahala', 'how far',
            'no be', 'i wan', 'wan', 'make', 'fit', 'sabi', 'chop', 'pikin',
            'oga', 'madam', 'kuku', 'shey', 'wey', 'go come', 'be my',
            'i go', 'no go', 'e be', 'e no', 'comot', 'carry', 'na so',
            'light bill', 'spend', 'yesterday', 'help me', 'pay light',
            'show me', 'for me', 'na wetin', 'which one'
        ]

        self.yorubaIndicators = [
            'ẹ', 'ọ', 'ṣ', 'bawo', 'pẹlẹ', 'owo', 'dara', 'jowo',
            'fẹ', 'ranse', 'káàárọ̀', 'káàsán', 'ìtàn', 'owó', 'ránsẹ́',
            'san bill', 'itan owo', 'gbe owo', 'bawo ni', 'balance mi',
            'mo fẹ́', 'ránsẹ́ owó', 'mo fẹ gbe'
        ]

        self.hausaIndicators = [
            'ina', 'sannu', 'yaya', 'kuɗi', 'kudi', 'amma', 'allah',
            'aika', 'son in', 'taimaka', 'taimako', 'kowane', 'hakuri',
            'barka', 'biya', 'hayaki', 'tarihi', 'tarihin', 'tura'
        ]

        self.igboIndicators = [
            'kedu', 'ndeewọ', 'biko', 'ego', 'ọma', 'ya', 'ị',
            'nke', 'nwa', 'ụ', 'nye', 'ziga', 'nwere', 'achọrọ',
            'kwụọ', 'ọkụ', 'latrik', 'jiri', 'enyemaka', 'ihe',
            'nkọwa', 'akaụntụ', 'm', 'gị', 'nwoke', 'nwanyi',
            'ụnyaahụ', 'ndewo', 'chọrọ'
        ]

        self.strongEnglishWords = [
            'what', 'is', 'my', 'the', 'balance', 'account', 'hello',
            'how', 'are', 'you', 'need', 'help', 'show', 'get', 'can',
            'good morning', 'send money to', 'electricity', 'with my',
            'i need', 'do i', 'save money', 'transactions'
        ]

    def detectWithKeywords(self, text: str) -> tuple[Optional[str], float]:
        textLower = text.lower()
        words = textLower.split()

        # Count unique matches to avoid over-counting
        pidginScore = sum(1 for indicator in self.pidginIndicators if indicator in textLower)
        yorubaScore = sum(1 for indicator in self.yorubaIndicators if indicator in text or indicator in textLower)
        hausaScore = sum(1 for indicator in self.hausaIndicators if indicator in text or indicator in textLower)
        igboScore = sum(1 for indicator in self.igboIndicators if indicator in text or indicator in textLower)
        englishScore = sum(1 for word in self.strongEnglishWords if word in textLower)

        # Remove very short common words from Yoruba that overlap with English
        yorubaShortWords = ['mi', 'ni', 'ti', 'ko', 'se', 're', 'wa', 'bi', 'na', 'mo', 'ya']
        yorubaShortMatches = sum(1 for word in yorubaShortWords if word in textLower)

        # If mostly short Yoruba words but also English words, likely English
        if englishScore >= 2 and yorubaShortMatches > yorubaScore * 0.5:
            yorubaScore = max(0, yorubaScore - yorubaShortMatches)

        # Scoring with adjusted weights
        scores = {
            'pcm': pidginScore * 1.3,  # Boost Pidgin
            'yo': yorubaScore * 1.0,
            'ha': hausaScore * 1.0,
            'ig': igboScore * 1.0,
            'en': englishScore * 1.1   # Boost English to compete with Yoruba
        }

        maxLang = max(scores.keys(), key=lambda k: scores[k])
        maxScore = scores[maxLang]

        # If English score is strong, prefer it
        if scores['en'] >= 2.5 and scores['en'] >= maxScore * 0.8:
            return 'en', min(0.95, 0.7 + (scores['en'] * 0.05))

        if maxScore >= 1.5:
            confidence = min(0.95, 0.6 + (maxScore * 0.1))
            return maxLang, confidence

        return None, 0.0

    def detectWithGoogle(self, text: str) -> tuple[Optional[str], float]:
        if not self.googleTranslateClient:
            return None, 0.0

        try:
            result = self.googleTranslateClient.detect_language(text)
            detectedLang = result['language']
            confidence = result.get('confidence', 0.5)

            if detectedLang == 'yo':
                return 'yo', confidence
            elif detectedLang == 'ha':
                return 'ha', confidence
            elif detectedLang == 'ig':
                return 'ig', confidence
            elif detectedLang == 'en':
                return 'en', confidence

            return detectedLang, confidence

        except Exception as e:
            print(f"Google Translate detection failed: {e}")
            return None, 0.0

    def detectWithLangdetect(self, text: str) -> tuple[Optional[str], float]:
        try:
            detectedLang = detect(text)

            if detectedLang in self.supportedLanguages:
                return detectedLang, 0.75
            elif detectedLang == 'en':
                return 'en', 0.80

            return detectedLang, 0.60

        except LangDetectException:
            return 'en', 0.50

    def detectLanguage(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return 'en'

        keywordLang, keywordConf = self.detectWithKeywords(text)
        if keywordConf >= 0.75 and keywordLang:
            return keywordLang

        if self.useGoogleTranslate:
            googleLang, googleConf = self.detectWithGoogle(text)
            if googleConf >= 0.70 and googleLang:
                return googleLang

        if keywordConf >= 0.60 and keywordLang:
            return keywordLang

        langdetectLang, langdetectConf = self.detectWithLangdetect(text)
        return langdetectLang if langdetectLang else 'en'

    def getLanguageName(self, langCode: str) -> str:
        return self.languageNames.get(langCode, 'English')

    def detectWithConfidence(self, text: str) -> Dict[str, Any]:
        if not text or len(text.strip()) < 2:
            return {
                'language': 'en',
                'language_name': 'English',
                'confidence': 0.5,
                'detection_method': 'default'
            }

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

        bestDetection = max(detections, key=lambda x: x[2])
        method, language, confidence = bestDetection

        if keywordConf >= 0.75 and keywordLang:
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
        return langCode in self.supportedLanguages or langCode == 'pcm'

    def translate(self, text: str, targetLanguage: str = 'en') -> Optional[str]:
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
