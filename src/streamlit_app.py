# src/streamlit_app.py
"""
Streamlit UI for MyTravelHelper - AI-powered Travel Review Analyzer
Provides comprehensive analysis of travel reviews including sentiment, aspects, and topics.
"""

import streamlit as st
from models import (
    set_hf_token,
    analyze_sentiment,
    detect_topics,
    analyze_aspect_sentiment
)
from models import detect_intents, extract_entities
from utils import (
    clean_text,
    extract_aspects,
    extract_sentences_with_aspect,
    analyze_text_statistics
)

# Page configuration
st.set_page_config(
    page_title="MyTravelHelper",
    page_icon="✈️",
    layout="wide"
)

# Title and description
st.title("✈️ MyTravelHelper: AI-powered Travel Review Analyzer")
st.markdown("""
An intelligent application for analyzing travel reviews using artificial intelligence.
Discover sentiment, identify aspects, and detect topics in travel feedback.
""")

# Sidebar configuration
st.sidebar.title("Settings")
st.sidebar.markdown("---")

# Hugging Face Token Input (at the top of sidebar)
st.sidebar.subheader("🔑 Hugging Face API Configuration")
hf_token = st.sidebar.text_input(
    "Enter your Hugging Face API Token:",
    type="password",
    help="Get your token from https://huggingface.co/settings/tokens"
)

if hf_token:
    try:
        set_hf_token(hf_token)
        st.sidebar.success("✓ Token set successfully!")
    except ValueError as e:
        st.sidebar.error(f"Error: {str(e)}")
else:
    st.sidebar.warning("⚠️ Please enter your Hugging Face API token to proceed")

st.sidebar.markdown("---")

# Define travel aspects
travel_aspects = [
    "cleanliness",
    "staff",
    "service",
    "location",
    "price",
    "food",
    "room",
    "amenities",
    "comfort",
    "views"
]

# Select aspects to analyze
selected_aspects = st.sidebar.multiselect(
    "Select aspects to analyze:",
    travel_aspects,
    default=travel_aspects[:5]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**About MyTravelHelper:**\n\n"
    "This tool uses Hugging Face Inference API to:\n"
    "- Analyze overall sentiment\n"
    "- Detect specific aspects mentioned\n"
    "- Identify sentiment for each aspect\n"
    "- Extract key topics\n\n"
    "**Get HF Token:**\n"
    "1. Visit huggingface.co\n"
    "2. Sign up/Login\n"
    "3. Go to Settings > Tokens\n"
    "4. Create new token\n"
    "5. Copy and paste here"
)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Enter Your Travel Review")
    user_review = st.text_area(
        "Paste a travel review or share your experience:",
        height=150,
        placeholder="Example: The hotel was amazing! Great location and friendly staff. "
                    "Food was delicious but prices were quite high..."
    )

with col2:
    st.subheader("📊 Analysis Options")
    analyze_button = st.button("🔍 Analyze Review", key="analyze", use_container_width=True)
    clear_button = st.button("🗑️ Clear", key="clear", use_container_width=True)

# Analysis logic
if clear_button:
    st.rerun()

if analyze_button:
    if not hf_token:
        st.error("❌ Please enter your Hugging Face API token in the sidebar first!")
    elif not user_review.strip():
        st.warning("⚠️ Please enter a review or query.")
    else:
        # Clean the text
        cleaned_review = clean_text(user_review)
        
        st.markdown("---")
        st.subheader("📈 Analysis Results")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overall Sentiment",
            "Aspect-Based Analysis",
            "Intent & Entities",
            "Topic Detection",
            "Text Statistics"
        ])
        
        # Tab 1: Overall Sentiment
        with tab1:
            st.write("### Overall Sentiment Analysis")
            try:
                sentiment = analyze_sentiment(cleaned_review)
                
                if "error" not in sentiment:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        sentiment_color = "🟢" if sentiment["label"] == "POSITIVE" else ""
                        st.metric(
                            "Sentiment",
                            f"{sentiment_color} {sentiment['label']}"
                        )
                    
                    with col2:
                        st.metric(
                            "Confidence Score",
                            f"{sentiment['score']:.2%}"
                        )
                    
                    with col3:
                        st.metric(
                            "Confidence Level",
                            sentiment["confidence"]
                        )
                    
                    # Sentiment gauge
                    st.divider()
                    st.write("**Sentiment Intensity:**")
                    st.progress(sentiment["score"], text=f"{sentiment['score']:.1%}")
                else:
                    st.error(f"Error analyzing sentiment: {sentiment['error']}")
            except Exception as e:
                st.error(f"Error during sentiment analysis: {str(e)}")
        
        # Tab 2: Aspect-Based Analysis
        with tab2:
            st.write("### Aspect-Based Sentiment Analysis")
            
            if selected_aspects:
                try:
                    aspect_results = analyze_aspect_sentiment(
                        cleaned_review,
                        selected_aspects
                    )
                    
                    # Display results
                    mentioned_count = sum(1 for a in aspect_results.values() if a["mentioned"])
                    
                    st.info(f"**Found {mentioned_count}/{len(selected_aspects)} aspects mentioned**")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    for idx, (aspect, result) in enumerate(aspect_results.items()):
                        col = col1 if idx % 2 == 0 else col2
                        
                        with col:
                            with st.container(border=True):
                                if result["mentioned"]:
                                    sentiment_icon = "😊" if result["sentiment"] == "POSITIVE" else "😞"
                                    st.write(f"#### {aspect.upper()} {sentiment_icon}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.metric("Sentiment", result["sentiment"])
                                    with col_b:
                                        st.metric("Score", f"{result['confidence_score']:.2%}")
                                    
                                    # Extract sentences about this aspect
                                    aspect_sentences = extract_sentences_with_aspect(
                                        cleaned_review,
                                        aspect
                                    )
                                    if aspect_sentences:
                                        st.write("**Related sentences:**")
                                        for sent in aspect_sentences[:2]:  # Show first 2
                                            st.caption(f"💬 {sent}")
                                else:
                                    st.write(f"#### {aspect.upper()}")
                                    st.caption("Not mentioned in the review")
                
                except Exception as e:
                    st.error(f"Error during aspect analysis: {str(e)}")
            else:
                st.warning("Please select at least one aspect to analyze")
        
        # Tab 3: Intent Classification & Entity Extraction
        with tab3:
            st.write("### Intent Classification & Entity Extraction")

            try:
                intents = detect_intents(cleaned_review)
                entities = extract_entities(cleaned_review)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Detected Intents")
                    if "error" not in intents and intents.get("primary_intent"):
                        st.metric("Primary Intent", intents["primary_intent"], f"{intents['primary_score']:.0%}")
                        if intents.get("intents"):
                            st.write("**All detected intents:**")
                            for intent, score in intents["intents"].items():
                                st.caption(f"• {intent}: {score:.0%}")
                    else:
                        st.info("No clear intent detected")

                with col2:
                    st.subheader("🏷️ Extracted Entities")
                    if "error" not in entities:
                        found_entities = False
                        if entities.get("prices"):
                            found_entities = True
                            st.write(f"**💰 Prices:** {', '.join(entities['prices'])}")
                        if entities.get("amenities"):
                            found_entities = True
                            st.write(f"**🏨 Amenities:** {', '.join(entities['amenities'])}")
                        if entities.get("services"):
                            found_entities = True
                            st.write(f"**🔧 Services:** {', '.join(entities['services'])}")
                        if entities.get("room_types"):
                            found_entities = True
                            st.write(f"**🛏️ Room Types:** {', '.join(entities['room_types'])}")
                        if entities.get("locations"):
                            found_entities = True
                            st.write(f"**📍 Locations:** {', '.join(entities['locations'][:3])}")
                        if not found_entities:
                            st.info("No entities found")
                    else:
                        st.error(entities["error"])

            except Exception as e:
                st.error(f"Error during intent/entity analysis: {str(e)}")

        # Tab 4: Topic Detection
        with tab4:
            st.write("### Topic Detection")
            
            try:
                topic_labels = ["accommodation", "transportation", "food", "attractions", 
                               "experience", "value", "cleanliness", "service"]
                
                topics = detect_topics(cleaned_review, topic_labels)
                
                if "error" not in topics:
                    st.write("**Detected Topics and Relevance Scores:**")
                    
                    # Create a visual representation
                    data_dict = {}
                    for topic, score in zip(topics["topics"], topics["scores"]):
                        if score > 0.1:  # Only show topics with >10% relevance
                            data_dict[topic] = score
                    
                    if data_dict:
                        # Sort by score
                        sorted_topics = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)
                        
                        for topic, score in sorted_topics:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.progress(score, text=topic)
                            with col2:
                                st.write(f"{score:.1%}")
                    else:
                        st.info("No prominent topics detected")
                else:
                    st.error(f"Error detecting topics: {topics['error']}")
            except Exception as e:
                st.error(f"Error during topic detection: {str(e)}")
        
        # Tab 5: Text Statistics
        with tab5:
            st.write("### Text Statistics")
            
            try:
                stats = analyze_text_statistics(cleaned_review)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", stats["word_count"])
                
                with col2:
                    st.metric("Sentence Count", stats["sentence_count"])
                
                with col3:
                    st.metric("Avg Words/Sentence", stats["avg_words_per_sentence"])
                
                with col4:
                    st.metric("Character Count", stats["char_count"])
                
                st.divider()
                
                # Aspect frequency
                st.write("**Aspect Frequency in Review:**")
                aspect_mentions = extract_aspects(cleaned_review, selected_aspects)
                
                if aspect_mentions and "No aspects mentioned" not in aspect_mentions:
                    for aspect, count in sorted(aspect_mentions.items(), 
                                               key=lambda x: x[1], reverse=True):
                        st.write(f"- **{aspect}**: mentioned {count} time(s)")
                else:
                    st.info("No selected aspects mentioned in the review")
                
            except Exception as e:
                st.error(f"Error calculating statistics: {str(e)}")
        
        st.markdown("---")
        
        # Original review display (collapsed)
        with st.expander("📄 View Original & Cleaned Text"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Review:**")
                st.text(user_review)
            with col2:
                st.write("**Cleaned Review:**")
                st.text(cleaned_review)
