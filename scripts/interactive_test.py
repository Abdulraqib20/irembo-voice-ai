import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def classify_query(query, show_details=True):
    """Classify a single query and display results"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/classify",
            json={"query": query, "language": "en"},
            timeout=5
        )

        if response.status_code == 200:
            result = response.json()

            if show_details:
                print(f"\n{'='*60}")
                print(f"Query: {result['query']}")
                print(f"{'='*60}")
                print(f"✅ Intent: {result['intent']}")
                print(f"📊 Confidence: {result['confidence']:.2%}")
                print(f"⚙️  Method: {result['method']}")
                print(f"{'='*60}\n")
            else:
                print(f"→ {result['intent']} ({result['confidence']:.2%})")

            return result
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server. Is it running?")
        print("   Start server with: conda activate datathon && cd flask_app && python app.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def interactive_mode():
    """Interactive testing mode"""
    print("\n" + "="*60)
    print("🎯 INTERACTIVE INTENT CLASSIFIER")
    print("="*60)
    print("Type your queries below. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("💬 Your query: ").strip()

            if not query:
                print("⚠️  Empty query, please try again.\n")
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!\n")
                break

            classify_query(query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

def batch_mode(queries):
    """Test multiple queries at once"""
    print("\n" + "="*60)
    print(f"🔄 BATCH MODE - Testing {len(queries)} queries")
    print("="*60 + "\n")

    results = []
    for i, query in enumerate(queries, 1):
        print(f"{i}. '{query}'")
        result = classify_query(query, show_details=False)
        if result:
            results.append(result)
        print()

    print("="*60)
    print(f"✅ Completed: {len(results)}/{len(queries)} successful")
    print("="*60 + "\n")

    return results

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print("✅ Server is running!\n")
    except:
        print("\n⚠️  WARNING: Server appears to be offline!")
        print("   Start it with: conda activate datathon && cd flask_app && python app.py\n")

    # Check command line arguments
    if len(sys.argv) > 1:
        # Batch mode: python interactive_test.py "query 1" "query 2" "query 3"
        queries = sys.argv[1:]
        batch_mode(queries)
    else:
        # Interactive mode
        interactive_mode()
