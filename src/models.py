# src/models.py
"""
This module contains model inference logic for MyTravelHelper.
Uses Hugging Face Inference API (no local model downloads needed).
Requires HF_TOKEN environment variable or token parameter.
"""
from huggingface_hub import InferenceClient

# Global client (will be initialized with token in Streamlit app)
_client = None

def set_hf_token(token):
    """
    Set Hugging Face token for API access.
    Call this once at the beginning of your app.
    
    Args:
        token (str): Your Hugging Face API token
        
    Raises:
        ValueError: If token is empty
    """
    global _client
    if not token or not token.strip():
        raise ValueError("HF token cannot be empty!")
    _client = InferenceClient(api_key=token)

def get_hf_client():
    """
    Get the current Hugging Face Inference client.
    Raises error if token not set.
    
    Returns:
        InferenceClient: The Hugging Face client
        
    Raises:
        ValueError: If token not set
    """
    if _client is None:
        raise ValueError("HF token not set! Enter your Hugging Face token in the sidebar.")
    return _client

def analyze_sentiment(text):
    """
    Analyze sentiment of given text using Hugging Face Inference API.
    Uses zero-shot classification with BART model.
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        dict: Sentiment analysis result with label and score
    """
    try:
        client = get_hf_client()
        
        # Truncate to 512 tokens (API limit)
        text_truncated = text[:512] if len(text) > 512 else text
        
        # Use zero-shot classification for sentiment with BART model
        result = client.zero_shot_classification(
            text=text_truncated,
            candidate_labels=["positive", "negative"],
            model="facebook/bart-large-mnli"
        )
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Extract labels and scores
        labels = result.get("labels") or result.get("label")
        scores = result.get("scores") or result.get("score")
        
        if not labels or not scores:
            return {"error": "Invalid response format from API"}
        
        # Ensure they're lists
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(scores, list):
            scores = [scores]
        
        if len(labels) == 0 or len(scores) == 0:
            return {"error": "No results from sentiment analysis"}
        
        # Find label with highest score (don't assume order!)
        label_score_pairs = list(zip(labels, scores))
        top_label, top_score = max(label_score_pairs, key=lambda x: float(x[1]))
        top_score_float = float(top_score)
        
        return {
            "label": top_label.upper() if isinstance(top_label, str) else "POSITIVE",
            "score": round(top_score_float, 4),
            "confidence": "High" if top_score_float > 0.9 else "Medium" if top_score_float > 0.7 else "Low"
        }
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}

def detect_topics(text, topics):
    """
    Detect topics in text using zero-shot classification.
    
    Args:
        text (str): Input text
        topics (list): List of candidate topics
    
    Returns:
        dict: Topic detection results with scores
    """
    try:
        client = get_hf_client()
        model = "facebook/bart-large-mnli"
        
        # Truncate to 512 tokens
        text_truncated = text[:512] if len(text) > 512 else text
        
        result = client.zero_shot_classification(
            text=text_truncated,
            candidate_labels=topics,
            model=model
        )
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        
        # Extract labels and scores
        labels = result.get("labels") or result.get("label")
        scores = result.get("scores") or result.get("score")
        
        if not labels or not scores:
            return {"error": "Invalid response format from API"}
        
        # Ensure they're lists
        if not isinstance(labels, list):
            labels = [labels]
        if not isinstance(scores, list):
            scores = [scores]
        
        return {
            "topics": labels,
            "scores": [round(float(s), 4) for s in scores],
            "top_topic": labels[0] if len(labels) > 0 else None
        }
    except Exception as e:
        return {"error": f"Topic detection failed: {str(e)}"}

def analyze_aspect_sentiment(text, aspects):
    """
    Analyze sentiment for each aspect mentioned in text.
    Splits text by conjunctions (but, and, however) to analyze aspect-specific clauses.
    
    Args:
        text (str): Input review text
        aspects (list): List of aspects to check
    
    Returns:
        dict: Aspect-based sentiment analysis
    """
    import re
    
    results = {}
    
    # Split by conjunctions to isolate independent clauses
    clauses = re.split(
        r'\b(but|and|however|yet|though|although|while|since|because)\b',
        text,
        flags=re.IGNORECASE
    )
    
    # Filter to get only the actual clause text (remove conjunctions)
    clauses = [c.strip() for c in clauses if c.strip() and not re.match(r'\b(but|and|however|yet|though|although|while|since|because)\b', c, re.IGNORECASE)]
    
    for aspect in aspects:
        aspect_lower = aspect.lower()
        
        # Find clauses mentioning this aspect
        relevant_clauses = [c for c in clauses if aspect_lower in c.lower()]
        
        if relevant_clauses:
            sentiments = []
            for clause in relevant_clauses:
                try:
                    sent = analyze_sentiment(clause)
                    if "error" not in sent:
                        sentiments.append(sent)
                except:
                    pass
            
            if sentiments:
                # Average the sentiment scores
                avg_score = sum(float(s["score"]) for s in sentiments) / len(sentiments)
                # Use majority sentiment label
                labels_count = {}
                for s in sentiments:
                    label = s["label"]
                    labels_count[label] = labels_count.get(label, 0) + 1
                avg_label = max(labels_count, key=labels_count.get)
                
                results[aspect] = {
                    "mentioned": True,
                    "sentiment": avg_label,
                    "confidence_score": round(avg_score, 4),
                    "instances": len(sentiments)
                }
                continue
        
        results[aspect] = {
            "mentioned": False,
            "sentiment": None,
            "confidence_score": None
        }
    
    return results
