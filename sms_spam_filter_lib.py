import re
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

class SMSPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.fitted = False
    
    def clean_text(self, text):
        if text is None or text == "":
            return ""
        text = str(text).lower()
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\b\d{10,}\b', '', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        words = [self.stemmer.stem(word) for word in words]
        return ' '.join(words)
    
    def fit(self, texts, labels):
        self.label_encoder.fit(labels)
        self.fitted = True
        return self
    
    def transform(self, texts, labels=None):
        if not self.fitted and labels is not None:
            raise ValueError("Preprocessor must be fitted first!")
        cleaned_texts = [self.clean_text(text) for text in texts]
        if labels is not None:
            encoded_labels = self.label_encoder.transform(labels)
            return cleaned_texts, encoded_labels
        return cleaned_texts
    
    def fit_transform(self, texts, labels):
        return self.fit(texts, labels).transform(texts, labels)

class SMSVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            stop_words=None,
            lowercase=False,
            token_pattern=r'\b\w+\b'
        )
        self.fitted = False
    
    def fit(self, texts):
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, texts):
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted first!")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        result = self.vectorizer.fit_transform(texts)
        self.fitted = True
        return result
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

class SMSSpamModel:
    def __init__(self, model_type='naive_bayes'):
        self.model_type = model_type
        if model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0
            )
        else:
            raise ValueError("Unsupported model type")
        self.fitted = False
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.fitted = True
        return self
    
    def predict(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

from sms_spam_filter_lib import SMSPreprocessor, SMSVectorizer, SMSSpamModel