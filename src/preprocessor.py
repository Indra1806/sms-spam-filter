# src/preprocessor.py
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import logging

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

logger = logging.getLogger(__name__)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Custom text preprocessor for SMS messages
    
    Steps performed:
    1. Lowercase conversion
    2. URL/Email/Phone removal
    3. Special character removal
    4. Tokenization
    5. Stopword removal
    6. Stemming
    """
    
    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stemmer = PorterStemmer() if use_stemming else None
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        
    def fit(self, X, y=None):
        """Fit method for sklearn compatibility"""
        return self
        
    def transform(self, X):
        """Transform text data"""
        if isinstance(X, str):
            return [self._preprocess_text(X)]
        return [self._preprocess_text(text) for text in X]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess individual text message
        
        Args:
            text: Raw SMS text
            
        Returns:
            Preprocessed text string
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10,}\b', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Remove short tokens (less than 2 characters)
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)
        
    def preprocess_batch(self, texts: list) -> list:
        """Preprocess a batch of texts with progress tracking"""
        processed_texts = []
        for i, text in enumerate(texts):
            processed_texts.append(self._preprocess_text(text))
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        return processed_texts