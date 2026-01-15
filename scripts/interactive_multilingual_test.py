#!/usr/bin/env python3
"""
Interactive Multilingual Classifier Tester
Test queries in real-time with language detection and intent classification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.services.hybrid_classifier import HybridClassifier
from src.services.enhanced_language_detector import EnhancedLanguageDetector
from colorama import Fore, Style, init

init(autoreset=True)

def printBanner():
    print("\n" + "=" * 70)
    print(f"{Fore.CYAN}🌍 INTERACTIVE MULTILINGUAL INTENT CLASSIFIER{Style.RESET_ALL}")
    print("=" * 70)
    print(f"{Fore.YELLOW}Supported Languages:{Style.RESET_ALL}")
    print("  • English (en)")
    print("  • Yoruba (yo)")
    print("  • Hausa (ha)")
    print("  • Igbo (ig)")
    print("  • Nigerian Pidgin (pcm)")
    print()
    print(f"{Fore.YELLOW}Commands:{Style.RESET_ALL}")
    print("  • Type your query in any language")
    print("  • 'quit' or 'exit' to stop")
    print("  • 'help' for examples")
    print("=" * 70 + "\n")

def printHelp():
    print(f"\n{Fore.CYAN}📚 Example Queries:{Style.RESET_ALL}")
    print(f"\n{Fore.GREEN}English:{Style.RESET_ALL}")
    print("  • What is my balance?")
    print("  • Send money to 08012345678")
    print("  • I need a loan")

    print(f"\n{Fore.GREEN}Yoruba:{Style.RESET_ALL}")
    print("  • Bawo ni balance mi?")
    print("  • Mo fẹ́ ránsẹ́ owó")
    print("  • Ẹ káàárọ̀")

    print(f"\n{Fore.GREEN}Hausa:{Style.RESET_ALL}")
    print("  • Yaya kuɗin da nake da shi?")
    print("  • Ina son in aika kuɗi")
    print("  • Don Allah taimaka ni")

    print(f"\n{Fore.GREEN}Igbo:{Style.RESET_ALL}")
    print("  • Kedu ego m nwere?")
    print("  • Biko ziga ego")
    print("  • Ndeewọ, achọrọ m enyemaka")

    print(f"\n{Fore.GREEN}Nigerian Pidgin:{Style.RESET_ALL}")
    print("  • Wetin be my balance?")
    print("  • I wan send money")
    print("  • Abeg help me\n")

def formatResult(query, langResult, intentResult):
    print(f"\n{Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Query:{Style.RESET_ALL} {query}")
    print(f"{Fore.CYAN}{'─' * 70}{Style.RESET_ALL}")

    langCode = langResult['language']
    langName = langResult['language_name']
    langConf = langResult['confidence']

    langColor = Fore.GREEN if langConf >= 0.7 else Fore.YELLOW if langConf >= 0.5 else Fore.RED
    print(f"\n{Fore.MAGENTA}🌐 Language Detection:{Style.RESET_ALL}")
    print(f"  Language: {langColor}{langName} ({langCode}){Style.RESET_ALL}")
    print(f"  Confidence: {langColor}{langConf:.1%}{Style.RESET_ALL}")

    intent = intentResult.intent.value
    confidence = intentResult.confidence
    method = intentResult.method
    latency = intentResult.latency_ms

    confColor = Fore.GREEN if confidence >= 0.7 else Fore.YELLOW if confidence >= 0.5 else Fore.RED
    print(f"\n{Fore.MAGENTA}🎯 Intent Classification:{Style.RESET_ALL}")
    print(f"  Intent: {confColor}{intent}{Style.RESET_ALL}")
    print(f"  Confidence: {confColor}{confidence:.1%}{Style.RESET_ALL}")
    print(f"  Method: {Fore.BLUE}{method}{Style.RESET_ALL}")
    print(f"  Latency: {Fore.BLUE}{latency:.0f}ms{Style.RESET_ALL}")

    if confidence < 0.5:
        print(f"\n{Fore.RED}⚠️  Low confidence - consider rephrasing{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{'─' * 70}{Style.RESET_ALL}\n")

def main():
    print(f"\n{Fore.YELLOW}⚙️  Initializing classifiers with Google Translate...{Style.RESET_ALL}")

    try:
        languageDetector = EnhancedLanguageDetector(useGoogleTranslate=True)
        intentClassifier = HybridClassifier(useGroq=True)
        print(f"{Fore.GREEN}✓ Ready! Using Google Translate + Groq LLM{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}✗ Failed to initialize: {e}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Falling back to basic detector...{Style.RESET_ALL}")
        try:
            from src.services.language_detector import LanguageDetector
            languageDetector = LanguageDetector()
            intentClassifier = HybridClassifier(useGroq=True)
            print(f"{Fore.GREEN}✓ Ready with basic detector{Style.RESET_ALL}")
        except Exception as e2:
            print(f"{Fore.RED}✗ Complete failure: {e2}{Style.RESET_ALL}")
            return

    printBanner()

    queryCount = 0

    while True:
        try:
            query = input(f"{Fore.CYAN}💬 Enter query:{Style.RESET_ALL} ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print(f"\n{Fore.YELLOW}👋 Tested {queryCount} queries. Goodbye!{Style.RESET_ALL}\n")
                break

            if query.lower() == 'help':
                printHelp()
                continue

            queryCount += 1

            langResult = languageDetector.detectWithConfidence(query)
            intentResult = intentClassifier.classify(query, langResult['language'])

            formatResult(query, langResult, intentResult)

        except KeyboardInterrupt:
            print(f"\n\n{Fore.YELLOW}👋 Interrupted. Tested {queryCount} queries. Goodbye!{Style.RESET_ALL}\n")
            break
        except Exception as e:
            print(f"{Fore.RED}✗ Error: {e}{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
