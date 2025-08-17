# 📚 API Documentation

## Overview

The SMS Spam Filter provides both a web interface and programmatic API for spam detection. This document covers the core classes, methods, and usage patterns for developers who want to integrate the spam filter into their applications.

## Table of Contents

- [Core Classes](#core-classes)
- [SMSSpamFilter](#smsspamfilter)
- [SMSPreprocessor](#smspreprocessor)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [REST API (Future)](#rest-api-future)

## Core Classes

### SMSSpamFilter

The main class that handles model training, prediction, and persistence.

#### Constructor

```python
class SMSSpamFilter:
    def __init__(self):
        """
        Initialize the SMS Spam Filter.
        
        Attributes:
            preprocessor (SMSPreprocessor): Text preprocessing instance
            vectorizer (TfidfVectorizer): TF-IDF vectorizer for feature extraction
            models (dict): Dictionary containing trained models
            model_names (list): Available model names
        """
```

#### Methods

##### `train_models(df=None)`

Trains all classification models with the provided dataset.

**Parameters:**
- `df` (pandas.DataFrame, optional): Training dataset with columns:
  - `message` (str): SMS message text
  - `label` (str): Classification label ('spam' or 'ham')
- If `df` is None, uses built-in sample data

**Returns:**
- `bool`: True if training successful, False otherwise

**Example:**
```python
import pandas as pd

# With custom dataset
df = pd.read_csv('spam_dataset.csv')
success = spam_filter.train_models(df)

# With sample data
success = spam_filter.train_models()
```

**Raises:**
- `Exception`: If training fails due to invalid data or model errors

---

##### `predict(message, model_name='Naive Bayes')`

Classifies a message as spam or ham using the specified model.

**Parameters:**
- `message` (str): SMS message to classify
- `model_name` (str): Model to use for prediction
  - Options: 'Naive Bayes', 'Logistic Regression', 'Random Forest'
  - Default: 'Naive Bayes'

**Returns:**
- `tuple`: (classification_result, confidence_score)
  - `classification_result` (str): 'spam' or 'ham'
  - `confidence_score` (float): Prediction confidence (0.0 to 1.0)

**Example:**
```python
message = "Congratulations! You've won £1000!"
result, confidence = spam_filter.predict(message, 'Naive Bayes')
print(f"Classification: {result}")
print(f"Confidence: {confidence:.2%}")
```

**Edge Cases:**
- Empty message returns ('ham', 0.5)
- Invalid model name returns (None, 0.0)
- Preprocessing errors return (None, 0.0)

---

##### `load_models()`

Loads pre-trained models and vectorizer from disk.

**Returns:**
- `bool`: True if models loaded successfully, False otherwise

**File Structure Expected:**
```
models/
├── vectorizer.pkl
├── naive_bayes.pkl
├── logistic_regression.pkl
└── random_forest.pkl
```

**Example:**
```python
if spam_filter.load_models():
    print("Models loaded successfully")
else:
    print("Failed to load models, need to train")
    spam_filter.train_models()
```

---

##### `save_models()`

Saves trained models and vectorizer to disk.

**Creates:**
- `models/` directory if it doesn't exist
- Individual pickle files for each model and vectorizer

**Example:**
```python
spam_filter.train_models()
spam_filter.save_models()  # Models saved to models/ directory
```

**Raises:**
- `Exception`: If saving fails due to permissions or disk space

---

##### `load_sample_data()`

Creates sample training data for demonstration purposes.

**Returns:**
- `pandas.DataFrame`: Sample dataset with spam and ham messages

**Sample Data Structure:**
```python
{
    'message': ['WINNER!! Prize reward!', 'Hey, how are you?', ...],
    'label': ['spam', 'ham', ...]
}
```

### SMSPreprocessor

Handles text preprocessing and cleaning for SMS messages.

#### Constructor

```python
class SMSPreprocessor:
    def __init__(self):
        """
        Initialize the preprocessor.
        
        Attributes:
            stop_words (set): English stopwords for removal
            stemmer (PorterStemmer): Stemming algorithm
        """
```

#### Methods

##### `clean_text(text)`

Comprehensive text cleaning and preprocessing.

**Parameters:**
- `text` (str): Raw SMS message text

**Returns:**
- `str`: Cleaned and preprocessed text

**Processing Steps:**
1. Convert to lowercase
2. Remove URLs (http/https/www patterns)
3. Remove phone numbers (various formats)
4. Remove special characters and digits
5. Remove extra whitespace
6. Remove stopwords
7. Apply stemming
8. Filter short words (< 3 characters)

**Example:**
```python
preprocessor = SMSPreprocessor()
raw_text = "WINNER!! Call 123-456-7890 NOW! Visit http://example.com"
cleaned = preprocessor.clean_text(raw_text)
print(cleaned)  # Output: "winner call visit"
```

**Edge Cases:**
- `None` or `NaN` input returns empty string
- Empty text returns empty string
- Text with only special characters returns empty string

## Usage Examples

### Basic Classification

```python
from app import SMSSpamFilter

# Initialize and setup
spam_filter = SMSSpamFilter()

# Load existing models or train new ones
if not spam_filter.load_models():
    print("Training new models...")
    spam_filter.train_models()

# Single message prediction
message = "Free entry! Win £500 cash prize!"
result, confidence = spam_filter.predict(message)

print(f"Message: {message}")
print(f"Classification: {result}")
print(f"Confidence: {confidence:.2%}")
```

### Batch Processing

```python
messages = [
    "Hey, are you coming to the party tonight?",
    "URGENT! You've won a lottery! Call now!",
    "Meeting postponed to 3 PM tomorrow",
    "FREE ringtone! Text STOP to opt out"
]

results = []
for msg in messages:
    result, confidence = spam_filter.predict(msg)
    results.append({
        'message': msg,
        'classification': result,
        'confidence': confidence
    })

# Display results
for item in results:
    print(f"{item['classification'].upper()}: {item['message'][:50]}... ({item['confidence']:.2%})")
```

### Custom Dataset Training

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('your_sms_dataset.csv')

# Ensure correct column names
df = df.rename(columns={'text': 'message', 'class': 'label'})

# Map labels if necessary
label_map = {'spam': 'spam', 'ham': 'ham', 'legitimate': 'ham'}
df['label'] = df['label'].map(label_map)

# Train with custom data
spam_filter = SMSSpamFilter()
success = spam_filter.train_models(df)

if success:
    spam_filter.save_models()
    print("Custom models trained and saved!")
```

### Model Comparison

```python
test_message = "Congratulations! You've won £1000 cash prize!"

models = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
results = {}

for model in models:
    result, confidence = spam_filter.predict(test_message, model)
    results[model] = {'result': result, 'confidence': confidence}

# Display comparison
print(f"Message: {test_message}")
print("\nModel Predictions:")
for model, prediction in results.items():
    print(f"{model}: {prediction['result']} ({prediction['confidence']:.2%})")
```

## Error Handling

### Common Exceptions

```python
from app import SMSSpamFilter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

spam_filter = SMSSpamFilter()

try:
    # Attempt to load models
    if not spam_filter.load_models():
        logger.warning("No existing models found, training new models...")
        spam_filter.train_models()
    
    # Make prediction
    result, confidence = spam_filter.predict("Test message")
    
    if result is None:
        logger.error("Prediction failed")
    else:
        logger.info(f"Prediction successful: {result} ({confidence:.2%})")
        
except Exception as e:
    logger.error(f"Application error: {str(e)}")
```

### Validation Helpers

```python
def validate_message(message):
    """Validate input message"""
    if not isinstance(message, str):
        raise TypeError("Message must be a string")
    
    if len(message.strip()) == 0:
        raise ValueError("Message cannot be empty")
    
    if len(message) > 1000:  # SMS length limit
        raise ValueError("Message too long (max 1000 characters)")
    
    return message.strip()

def validate_model_name(model_name, available_models):
    """Validate model name"""
    if model_name not in available_models:
        raise ValueError(f"Invalid model: {model_name}. Available: {available_models}")
    
    return model_name
```

## Configuration

### Environment Variables

```python
import os

# Model configuration
MAX_FEATURES = int(os.getenv('MAX_FEATURES', 5000))
TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))

# File paths
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
DATA_DIR = os.getenv('DATA_DIR', 'data')

# Logging level
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
```

### Custom Configuration Class

```python
class SpamFilterConfig:
    """Configuration class for SMS Spam Filter"""
    
    # Model parameters
    MAX_FEATURES = 5000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # File paths
    MODEL_DIR = 'models'
    VECTORIZER_FILE = 'vectorizer.pkl'
    
    # Model names
    NAIVE_BAYES = 'Naive Bayes'
    LOGISTIC_REGRESSION = 'Logistic Regression'
    RANDOM_FOREST = 'Random Forest'
    
    # Text processing
    MIN_WORD_LENGTH = 3
    STOPWORDS_LANG = 'english'
    
    @classmethod
    def get_model_path(cls, model_name):
        """Get file path for a model"""
        filename = model_name.lower().replace(' ', '_') + '.pkl'
        return os.path.join(cls.MODEL_DIR, filename)
```

## Performance Optimization

### Memory Management

```python
import gc
from sklearn.feature_extraction.text import TfidfVectorizer

# Optimize TF-IDF parameters for large datasets
vectorizer = TfidfVectorizer(
    max_features=5000,      # Limit vocabulary size
    max_df=0.95,           # Remove very common words
    min_df=2,              # Remove very rare words
    stop_words='english',   # Remove stopwords
    ngram_range=(1, 2)     # Use unigrams and bigrams
)

# Clear memory after training
def cleanup_after_training():
    """Clean up memory after model training"""
    gc.collect()
```

### Batch Prediction

```python
def predict_batch(spam_filter, messages, batch_size=100):
    """
    Predict multiple messages efficiently
    
    Args:
        spam_filter: SMSSpamFilter instance
        messages: List of messages
        batch_size: Number of messages to process at once
    
    Returns:
        List of (result, confidence) tuples
    """
    results = []
    
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        batch_results = []
        
        for message in batch:
            result, confidence = spam_filter.predict(message)
            batch_results.append((result, confidence))
        
        results.extend(batch_results)
        
        # Optional: progress tracking
        print(f"Processed {min(i + batch_size, len(messages))}/{len(messages)} messages")
    
    return results
```

## REST API (Future)

### Planned Endpoints

```python
# Future REST API endpoints (not yet implemented)

# POST /api/v1/predict
{
    "message": "Your SMS message here",
    "model": "Naive Bayes"  # optional
}

# Response
{
    "classification": "spam",
    "confidence": 0.95,
    "model_used": "Naive Bayes",
    "timestamp": "2024-01-01T12:00:00Z"
}

# POST /api/v1/predict/batch
{
    "messages": ["Message 1", "Message 2", ...],
    "model": "Naive Bayes"  # optional
}

# GET /api/v1/models
{
    "models": [
        {
            "name": "Naive Bayes",
            "accuracy": 0.964,
            "last_trained": "2024-01-01T10:00:00Z"
        }
    ]
}

# POST /api/v1/retrain
{
    "dataset_url": "https://example.com/dataset.csv"  # optional
}
```

### Integration Example

```python
# Future integration example
import requests

def predict_via_api(message, api_url="http://localhost:8000"):
    """Predict using REST API (future implementation)"""
    response = requests.post(
        f"{api_url}/api/v1/predict",
        json={"message": message}
    )
    return response.json()

# Usage
result = predict_via_api("Win £1000 now!")
print(f"Classification: {result['classification']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Best Practices

### 1. Input Validation

```python
def safe_predict(spam_filter, message, model_name='Naive Bayes'):
    """Safe prediction with validation"""
    try:
        # Validate inputs
        message = validate_message(message)
        model_name = validate_model_name(model_name, spam_filter.model_names)
        
        # Make prediction
        return spam_filter.predict(message, model_name)
        
    except (TypeError, ValueError) as e:
        logger.error(f"Validation error: {e}")
        return None, 0.0
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None, 0.0
```

### 2. Model Versioning

```python
import datetime
import json

def save_model_metadata(spam_filter, version, accuracy_scores):
    """Save model metadata for versioning"""
    metadata = {
        'version': version,
        'timestamp': datetime.datetime.now().isoformat(),
        'models': accuracy_scores,
        'vectorizer_features': spam_filter.vectorizer.get_feature_names_out().shape[0]
    }
    
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
```

### 3. Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

# Usage
@monitor_performance
def predict_with_monitoring(spam_filter, message):
    return spam_filter.predict(message)
```

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check if `models/` directory exists
   - Verify pickle files are not corrupted
   - Ensure Python version compatibility

2. **Prediction Returns None**
   - Verify model is properly loaded
   - Check if message preprocessing fails
   - Validate model name is correct

3. **Low Accuracy**
   - Increase training data size
   - Improve text preprocessing
   - Try different model parameters

4. **Memory Issues**
   - Reduce `max_features` in TF-IDF
   - Use batch processing for large datasets
   - Clear unused variables with `gc.collect()`

For additional support, please refer to the [main documentation](../README.md) or open an issue on GitHub.