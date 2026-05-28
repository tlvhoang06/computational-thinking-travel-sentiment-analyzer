# MyTravelHelper: AI-powered Travel Review Analyzer

## Overview

MyTravelHelper is an intelligent Streamlit application for analyzing travel reviews and user travel queries with Hugging Face Inference API. The app provides overall sentiment analysis, aspect-based sentiment analysis, intent classification with entity extraction, topic detection, and text statistics.

## Features

### Core Features

- **Overall Sentiment Analysis**: Classifies reviews as positive or negative with confidence scores.
- **Aspect-Based Sentiment Analysis**: Analyzes sentiment for selected aspects such as cleanliness, staff, service, location, food, price, and amenities.
- **Intent Classification & Entity Extraction**: Detects user intent and extracts practical entities such as prices, amenities, services, room types, and locations.
- **Topic Detection**: Identifies main topics mentioned in reviews using zero-shot classification.
- **Text Statistics**: Shows word count, sentence count, average words per sentence, and character count.
- **Interactive Web Interface**: Built with Streamlit.

### Advanced Features

1. **Aspect-Based Sentiment Analysis**
   - Splits text by conjunctions such as `but`, `and`, `however`, and `though`.
   - Analyzes sentiment for clauses that mention each aspect.
   - Handles mixed reviews like: "The service is perfect, but the location is too far."

2. **Intent Classification & Entity Extraction**
   - Uses zero-shot classification to detect user intent.
   - Extracts prices, amenities, services, room types, and locations.
   - Location extraction now uses location context patterns such as `near the beach`, `from the city center`, and multi-word proper nouns.
   - Avoids incorrectly treating sentiment words like `Amazing` as locations.

3. **Topic Detection in Reviews**
   - Uses zero-shot classification to rank candidate topics.
   - Supports custom topic lists without retraining.
   - Returns relevance scores for each detected topic.

## Project Structure

```text
24120056_12/
|-- notebook.ipynb           # Project notebook with explanations and demos
|-- requirements.txt         # Python dependencies
|-- README.md                # Project documentation
`-- src/
    |-- __init__.py          # Package initializer
    |-- models.py            # Hugging Face inference and NLP logic
    |-- utils.py             # Text preprocessing utilities
    |-- streamlit_app.py     # Main Streamlit application
    `-- test_env.py          # Environment test script
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip
- Internet connection
- Hugging Face account and API token

### Step 1: Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python src/test_env.py
```

## Usage

Run the app:

```bash
streamlit run src/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

### Web Interface Flow

1. Paste your Hugging Face token in the sidebar.
2. Paste a travel review or query into the text area.
3. Select the aspects you want to analyze.
4. Click **Analyze Review**.
5. View results in 5 tabs:
   - **Overall Sentiment**
   - **Aspect-Based Analysis**
   - **Intent & Entities**
   - **Topic Detection**
   - **Text Statistics**

## Models Used

### `facebook/bart-large-mnli`

The project uses BART MNLI through Hugging Face Inference API for zero-shot classification tasks:

- Sentiment analysis with labels: `positive`, `negative`
- Topic detection with custom candidate topics
- Intent classification with travel-related intent labels

### Rule-Based Entity Extraction

Entity extraction is implemented in `src/models.py` with regex and keyword matching:

- **Prices**: `$80`, `EUR 100`, `VND 500000`
- **Amenities**: `wifi`, `pool`, `gym`, `spa`, `parking`
- **Services**: `breakfast`, `room service`, `reception`, `shuttle`
- **Room types**: `suite`, `double`, `single`, `deluxe`
- **Locations**: extracted only from location context or multi-word proper nouns

Example:

```text
Input: "Amazing service! I want a hotel near the beach with a pool and free wifi."

Extracted entities:
- locations: ["beach"]
- amenities: ["wifi", "pool"]
- services: []

"Amazing" is not extracted as a location.
```

## Core Modules

### `src/models.py`

- `set_hf_token(token)`: Initializes Hugging Face InferenceClient.
- `get_hf_client()`: Returns the active Hugging Face client.
- `analyze_sentiment(text)`: Runs overall sentiment analysis.
- `analyze_aspect_sentiment(text, aspects)`: Runs aspect-based sentiment analysis with clause splitting.
- `detect_intents(text)`: Detects user intent with zero-shot classification.
- `extract_entities(text)`: Extracts prices, amenities, services, room types, and locations.
- `detect_topics(text, topics)`: Detects review topics with zero-shot classification.

### `src/utils.py`

- `clean_text()`: Cleans and normalizes text.
- `extract_aspects()`: Finds mentioned aspects.
- `extract_sentences_with_aspect()`: Gets sentences related to a specific aspect.
- `analyze_text_statistics()`: Calculates basic text statistics.
- `format_sentiment_result()`: Formats sentiment output.

### `src/streamlit_app.py`

- Sidebar token input
- Review/query text area
- Aspect selector
- 5 result tabs
- Error handling for token, API, and analysis failures

## Advanced Features Explained

### Advanced Feature 1: Aspect-Based Sentiment Analysis

The app avoids analyzing the entire review as one sentiment when the review contains mixed opinions. It splits the text into clauses and analyzes only the clause related to each aspect.

```text
Review: "The service is perfect, but the location is too far."

Results:
- service: POSITIVE
- location: NEGATIVE
```

### Advanced Feature 2: Intent Classification & Entity Extraction

The app detects what the user is trying to do and extracts useful requirements.

```text
Input:
"I want a family-friendly hotel near the beach with a pool and free wifi.
Is breakfast included? I'm on a budget (~$80 per night)."

Entities:
- locations: beach
- prices: $80
- amenities: wifi, pool
- services: breakfast
```

### Advanced Feature 3: Topic Detection in Reviews

The app ranks candidate topics by relevance using zero-shot classification.

```text
Review: "Amazing location near beach! Perfect for vacation."

Detected topics:
- location: 95%
- amenities: 40%
- food: 30%
```

## Performance Notes

- Hugging Face API calls usually take a few seconds depending on network and API load.
- Aspect-based sentiment can take longer because it may call sentiment analysis for multiple clauses and aspects.
- No local model download is required for the main app workflow.

## Troubleshooting

### Token not set

Paste a valid Hugging Face token into the sidebar.

### Invalid or expired token

Create a new token at:

```text
https://huggingface.co/settings/tokens
```

### Slow response

Wait a few seconds and retry. Response time depends on Hugging Face API load and network speed.

### No entities found

The app only displays entities when it detects a clear price, amenity, service, room type, or location context.

## Future Enhancements

- Multi-language support
- Batch review processing
- Export results to CSV/JSON
- Dashboard for historical trend analysis
- Fine-tuned travel-domain models
- Hotel name extraction
- More robust location extraction with a dedicated NER model

## Dependencies

See `requirements.txt` for the complete list.

- streamlit
- huggingface_hub
- requests

## Resources

- [Hugging Face Models](https://huggingface.co/models)
- [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

Educational project for assignment completion.
