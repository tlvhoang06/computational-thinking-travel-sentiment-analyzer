# src/test_env.py
"""
Comprehensive environment test for MyTravelHelper
Tests all components and dependencies
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test if all required packages are installed."""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    try:
        import streamlit
        print("[OK] Streamlit installed")
    except ImportError:
        print("[FAIL] Streamlit not installed")
        return False
    
    try:
        import transformers
        print("[OK] Transformers installed")
    except ImportError:
        print("[FAIL] Transformers not installed")
        return False
    
    try:
        import torch
        print("[OK] PyTorch installed")
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False
    
    try:
        import numpy
        print("[OK] NumPy installed")
    except ImportError:
        print("[FAIL] NumPy not installed")
        return False
    
    print()
    return True

def test_sentiment_analysis():
    """Test sentiment analysis pipeline."""
    print("=" * 60)
    print("TESTING SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        # Load model
        print("Loading sentiment analysis model...")
        pipe = pipeline("sentiment-analysis", 
                       model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # Test cases
        test_texts = [
            "I love this place! The staff is very friendly.",
            "The service was bad and the food was cold.",
            "Amazing hotel with beautiful views."
        ]
        
        print("Running sentiment analysis on test texts:\n")
        for text in test_texts:
            result = pipe(text, truncation=True)
            print("Text: %s" % text)
            print("Result: %s (score: %.4f)\n" % (result[0]['label'], result[0]['score']))
        
        print("[OK] Sentiment analysis test passed\n")
        return True
    except Exception as e:
        print(f"✗ Sentiment analysis test failed: {e}\n")
        return False

def test_zero_shot_classification():
    """Test zero-shot classification for topic detection."""
    print("=" * 60)
    print("TESTING ZERO-SHOT CLASSIFICATION (Topic Detection)")
    print("=" * 60)
    
    try:
        from transformers import pipeline
        
        print("Loading zero-shot classification model...")
        pipe = pipeline("zero-shot-classification", 
                       model="facebook/bart-large-mnli")
        
        # Test case
        text = "The hotel is clean and the staff is friendly, but the price is too high."
        candidate_labels = ["cleanliness", "service", "price", "location", "food"]
        
        print("\nText: %s" % text)
        print("Candidate topics: %s\n" % str(candidate_labels))
        
        result = pipe(text, candidate_labels, multi_class=True)
        
        print("Topic detection results:")
        for label, score in zip(result["labels"], result["scores"]):
            print("  - %s: %.4f" % (label, score))
        
        print("\n[OK] Zero-shot classification test passed\n")
        return True
    except Exception as e:
        print(f"✗ Zero-shot classification test failed: {e}\n")
        return False

def test_local_modules():
    """Test local module imports."""
    print("=" * 60)
    print("TESTING LOCAL MODULES")
    print("=" * 60)
    
    try:
        from src.models import (
            get_sentiment_pipeline,
            get_zero_shot_pipeline,
            analyze_sentiment,
            detect_topics,
            analyze_aspect_sentiment
        )
        print("[OK] Successfully imported models module")
    except ImportError as e:
        print("[FAIL] Failed to import models module: %s" % str(e))
        return False
    
    try:
        from src.utils import (
            clean_text,
            extract_aspects,
            extract_sentences_with_aspect,
            count_aspect_mentions,
            analyze_text_statistics
        )
        print("[OK] Successfully imported utils module\n")
    except ImportError as e:
        print("[FAIL] Failed to import utils module: %s" % str(e))
        return False
    
    return True

def test_aspect_based_sentiment():
    """Test aspect-based sentiment analysis."""
    print("=" * 60)
    print("TESTING ASPECT-BASED SENTIMENT ANALYSIS")
    print("=" * 60)
    
    try:
        from src.models import analyze_aspect_sentiment
        from src.utils import clean_text
        
        review = """
        The hotel had excellent cleanliness and professional staff. 
        The location was perfect for sightseeing. 
        However, the price was quite high and the food quality was average.
        """
        
        aspects = ["cleanliness", "staff", "location", "price", "food"]
        
        print("Review: %s\n" % review.strip())
        print("Analyzing aspects...\n")
        
        result = analyze_aspect_sentiment(review, aspects)
        
        for aspect, analysis in result.items():
            if analysis["mentioned"]:
                print("[FOUND] %s: %s (score: %.2f%%)" % (
                    aspect.upper(), analysis['sentiment'], 
                    analysis['confidence_score'] * 100))
            else:
                print("[NOT FOUND] %s: Not mentioned" % aspect.upper())
        
        print("\n[OK] Aspect-based sentiment analysis test passed\n")
        return True
    except Exception as e:
        print("[FAIL] Aspect-based sentiment analysis test failed: %s\n" % str(e))
        return False

def run_all_tests():
    """Run all tests and report results."""
    print("\n")
    print("=" * 60)
    print("MyTravelHelper Environment Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Sentiment Analysis", test_sentiment_analysis()))
    results.append(("Zero-shot Classification", test_zero_shot_classification()))
    results.append(("Local Modules", test_local_modules()))
    results.append(("Aspect-based Sentiment", test_aspect_based_sentiment()))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print("%-40s %s" % (test_name + ".", status))
    
    print("=" * 60)
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("[OK] All tests passed! Environment is ready.")
    else:
        print("[FAIL] Some tests failed. Please check your installation.")
    
    print()
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
