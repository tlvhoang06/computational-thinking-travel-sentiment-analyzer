# src/test_env.py
"""
Environment test for MyTravelHelper.
Checks the dependencies used by the current API-based implementation.
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test if required packages are installed."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)

    packages = [
        ("streamlit", "Streamlit"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("requests", "Requests"),
    ]

    all_ok = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"[OK] {display_name} installed")
        except ImportError:
            print(f"[FAIL] {display_name} not installed")
            all_ok = False

    print()
    return all_ok


def test_local_modules():
    """Test local module imports."""
    print("=" * 60)
    print("TESTING LOCAL MODULES")
    print("=" * 60)

    try:
        from src.models import (
            analyze_aspect_sentiment,
            analyze_sentiment,
            detect_intents,
            detect_topics,
            extract_entities,
            set_hf_token,
        )
        print("[OK] Successfully imported models module")
    except ImportError as e:
        print(f"[FAIL] Failed to import models module: {e}")
        return False

    try:
        from src.utils import (
            analyze_text_statistics,
            clean_text,
            count_aspect_mentions,
            extract_aspects,
            extract_sentences_with_aspect,
            format_sentiment_result,
        )
        print("[OK] Successfully imported utils module")
    except ImportError as e:
        print(f"[FAIL] Failed to import utils module: {e}")
        return False

    print()
    return True


def test_text_utilities():
    """Test local text utility functions that do not require an API token."""
    print("=" * 60)
    print("TESTING TEXT UTILITIES")
    print("=" * 60)

    from src.utils import analyze_text_statistics, clean_text, extract_aspects

    text = "  The hotel   was AMAZING!!! Great location and friendly staff.  "
    cleaned = clean_text(text)
    aspects = extract_aspects(cleaned, ["location", "staff", "food"])
    stats = analyze_text_statistics(cleaned)

    print(f"Original: {text!r}")
    print(f"Cleaned:  {cleaned!r}")
    print(f"Aspects:  {aspects}")
    print(f"Stats:    {stats}")
    print("\n[OK] Text utilities test passed\n")
    return True


def test_entity_extraction():
    """Test rule-based entity extraction without requiring an API token."""
    print("=" * 60)
    print("TESTING ENTITY EXTRACTION")
    print("=" * 60)

    from src.models import extract_entities
    from src.utils import clean_text

    sample = """
    Amazing service! I want a hotel near the beach with a pool and free wifi.
    Is breakfast included? I am on a budget ($80 per night).
    """
    entities = extract_entities(clean_text(sample))

    print(f"Entities: {entities}")

    checks = [
        "Amazing" not in entities["locations"],
        "beach" in entities["locations"],
        "$80" in entities["prices"],
        "wifi" in entities["amenities"],
        "pool" in entities["amenities"],
        "breakfast" in entities["services"],
    ]

    if all(checks):
        print("\n[OK] Entity extraction test passed\n")
        return True

    print("\n[FAIL] Entity extraction test failed\n")
    return False


def test_hf_token_optional():
    """Validate Hugging Face token setup only when HF_TOKEN is available."""
    print("=" * 60)
    print("TESTING HUGGING FACE TOKEN SETUP")
    print("=" * 60)

    token = os.getenv("HF_TOKEN")
    if not token:
        print("[SKIP] HF_TOKEN is not set. API calls are tested in the Streamlit app after entering a token.\n")
        return True

    try:
        from src.models import set_hf_token

        set_hf_token(token)
        print("[OK] HF_TOKEN loaded successfully\n")
        return True
    except Exception as e:
        print(f"[FAIL] HF token setup failed: {e}\n")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("=" * 60)
    print("MyTravelHelper Environment Test")
    print("=" * 60)
    print()

    results = [
        ("Imports", test_imports()),
        ("Local Modules", test_local_modules()),
        ("Text Utilities", test_text_utilities()),
        ("Entity Extraction", test_entity_extraction()),
        ("HF Token Setup", test_hf_token_optional()),
    ]

    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name + '.':40s} {status}")

    print("=" * 60)

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("[OK] Environment is ready.")
    else:
        print("[FAIL] Some tests failed. Please check your installation.")

    print()
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
