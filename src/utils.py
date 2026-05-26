# src/utils.py
"""
Utility functions for preprocessing and postprocessing in MyTravelHelper.
Includes text cleaning, aspect extraction, and data formatting.
"""
import re
from collections import Counter

def clean_text(text):
    """
    Clean and normalize text.
    - Remove extra whitespace
    - Remove special characters
    - Convert to lowercase-friendly format
    
    Args:
        text (str): Raw input text
    
    Returns:
        str: Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing spaces
    text = text.strip()
    # Remove extra punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text

def extract_aspects(review, aspects):
    """
    Extract aspects mentioned in a review.
    
    Args:
        review (str): Input review text
        aspects (list): List of candidate aspects
    
    Returns:
        dict: Aspects found with their frequency
    """
    review_lower = review.lower()
    found = {}
    
    for aspect in aspects:
        aspect_lower = aspect.lower()
        count = review_lower.count(aspect_lower)
        if count > 0:
            found[aspect] = count
    
    return found if found else {"No aspects mentioned": 0}

def extract_sentences_with_aspect(review, aspect):
    """
    Extract sentences containing a specific aspect.
    
    Args:
        review (str): Input review text
        aspect (str): Target aspect
    
    Returns:
        list: Sentences containing the aspect
    """
    sentences = re.split(r'[.!?]+', review)
    aspect_sentences = [
        s.strip() for s in sentences 
        if aspect.lower() in s.lower() and s.strip()
    ]
    return aspect_sentences

def count_aspect_mentions(review, aspect):
    """
    Count how many times an aspect is mentioned.
    
    Args:
        review (str): Input review
        aspect (str): Aspect to count
    
    Returns:
        int: Number of mentions
    """
    return review.lower().count(aspect.lower())

def analyze_text_statistics(text):
    """
    Analyze basic statistics of the text.
    
    Args:
        text (str): Input text
    
    Returns:
        dict: Text statistics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_words_per_sentence": round(len(words) / len(sentences), 2) if sentences else 0,
        "char_count": len(text)
    }

def format_sentiment_result(label, score):
    """
    Format sentiment result for display.
    
    Args:
        label (str): Sentiment label
        score (float): Confidence score
    
    Returns:
        str: Formatted result
    """
    confidence = "High" if score > 0.9 else "Medium" if score > 0.7 else "Low"
    return f"{label} ({score:.2%} confidence - {confidence})"
