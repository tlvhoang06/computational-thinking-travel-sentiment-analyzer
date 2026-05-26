# src/models.py
"""
This module contains model loading and inference logic for MyTravelHelper.
Includes sentiment analysis, topic detection, and aspect-based analysis.
"""
from transformers import pipeline

# Cache for pipelines to avoid reloading
_sentiment_pipeline = None
_topic_pipeline = None
_zero_shot_pipeline = None

def get_sentiment_pipeline(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load sentiment analysis pipeline.
    Uses DistilBERT fine-tuned on SST-2 for sentiment classification.
    Returns: pipeline object
    """
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        _sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
    return _sentiment_pipeline

def get_zero_shot_pipeline(model_name="facebook/bart-large-mnli"):
    """
    Load zero-shot classification pipeline for topic detection.
    Uses BART model for flexible topic classification.
    Returns: pipeline object
    """
    global _zero_shot_pipeline
    if _zero_shot_pipeline is None:
        _zero_shot_pipeline = pipeline("zero-shot-classification", model=model_name)
    return _zero_shot_pipeline

def analyze_sentiment(text, pipeline_obj=None):
    """
    Analyze sentiment of given text.
    
    Args:
        text (str): Input text to analyze
        pipeline_obj: Optional pipeline object (if None, will load)
    
    Returns:
        dict: Sentiment analysis result with label and score
    """
    if pipeline_obj is None:
        pipeline_obj = get_sentiment_pipeline()
    
    try:
        result = pipeline_obj(text, truncation=True)
        return {
            "label": result[0]["label"],
            "score": round(result[0]["score"], 4),
            "confidence": "High" if result[0]["score"] > 0.9 else "Medium" if result[0]["score"] > 0.7 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}

def detect_topics(text, topics, pipeline_obj=None):
    """
    Detect topics in text using zero-shot classification.
    
    Args:
        text (str): Input text
        topics (list): List of candidate topics
        pipeline_obj: Optional pipeline object
    
    Returns:
        dict: Topic detection results with scores
    """
    if pipeline_obj is None:
        pipeline_obj = get_zero_shot_pipeline()
    
    try:
        result = pipeline_obj(text, topics, multi_class=True)
        return {
            "topics": result["labels"],
            "scores": [round(score, 4) for score in result["scores"]],
            "top_topic": result["labels"][0] if result["labels"] else None
        }
    except Exception as e:
        return {"error": str(e)}

def analyze_aspect_sentiment(text, aspects, sentiment_pipeline_obj=None, topic_pipeline_obj=None):
    """
    Analyze sentiment for each aspect mentioned in text.
    
    Args:
        text (str): Input review text
        aspects (list): List of aspects to check
        sentiment_pipeline_obj: Optional sentiment pipeline
        topic_pipeline_obj: Optional topic pipeline
    
    Returns:
        dict: Aspect-based sentiment analysis
    """
    sentiment_pipe = sentiment_pipeline_obj or get_sentiment_pipeline()
    topic_pipe = topic_pipeline_obj or get_zero_shot_pipeline()
    
    results = {}
    
    for aspect in aspects:
        aspect_lower = aspect.lower()
        # Check if aspect is mentioned in text
        if aspect_lower in text.lower():
            # Extract sentences mentioning this aspect
            sentences = text.split(". ")
            aspect_sentences = [s for s in sentences if aspect_lower in s.lower()]
            
            if aspect_sentences:
                # Analyze sentiment for each sentence mentioning the aspect
                sentiments = []
                for sentence in aspect_sentences:
                    try:
                        sent = analyze_sentiment(sentence, sentiment_pipe)
                        sentiments.append(sent)
                    except:
                        pass
                
                if sentiments:
                    avg_score = sum(s["score"] for s in sentiments if "score" in s) / len(sentiments)
                    avg_label = sentiments[0]["label"] if sentiments else "NEUTRAL"
                    
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
