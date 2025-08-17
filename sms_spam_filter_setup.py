#!/usr/bin/env python3
"""
SMS Spam Filter - Complete Setup Script
This script creates all necessary files and trains the models
"""

import os
import pickle
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

class SMSPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.fitted = False
    
    def clean_text(self, text):
        """Clean and preprocess SMS text"""
        if text is None or text == "":
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{10,}\b', '', text)
        text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '', text)
        
        # Remove extra spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Tokenize and remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Stem words
        words = [self.stemmer.stem(word) for word in words]
        
        return ' '.join(words)
    
    def fit(self, texts, labels):
        """Fit the preprocessor"""
        self.label_encoder.fit(labels)
        self.fitted = True
        return self
    
    def transform(self, texts, labels=None):
        """Transform texts and optionally labels"""
        if not self.fitted and labels is not None:
            raise ValueError("Preprocessor must be fitted first!")
        
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        if labels is not None:
            encoded_labels = self.label_encoder.transform(labels)
            return cleaned_texts, encoded_labels
        
        return cleaned_texts
    
    def fit_transform(self, texts, labels):
        """Fit and transform in one step"""
        return self.fit(texts, labels).transform(texts, labels)

class SMSVectorizer:
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95,
            stop_words=None,  # We already removed stopwords in preprocessing
            lowercase=False,  # Already lowercased in preprocessing
            token_pattern=r'\b\w+\b'
        )
        self.fitted = False
    
    def fit(self, texts):
        """Fit the vectorizer"""
        self.vectorizer.fit(texts)
        self.fitted = True
        return self
    
    def transform(self, texts):
        """Transform texts to vectors"""
        if not self.fitted:
            raise ValueError("Vectorizer must be fitted first!")
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform in one step"""
        result = self.vectorizer.fit_transform(texts)
        self.fitted = True  # <-- Fix: set fitted to True
        return result
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()

class SMSSpamModel:
    def __init__(self, model_type='naive_bayes'):
        """Initialize the SMS spam model"""
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
        """Train the model"""
        self.model.fit(X_train, y_train)
        self.fitted = True
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.fitted:
            raise ValueError("Model must be fitted first!")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Get model score"""
        return self.model.score(X, y)

# Comprehensive SMS dataset
sms_dataset = [
    # Spam messages
    ("Win a £1000 cash prize! Text WIN to 12345 now!", "spam"),
    ("URGENT! Your account will be suspended. Click link now!", "spam"),
    ("FREE entry in 2 a weekly comp for a chance to win an iPhone", "spam"),
    ("Congratulations! You've won a lottery of $50000", "spam"),
    ("LIMITED TIME OFFER! Buy now and save 50%", "spam"),
    ("WINNER! As a valued network customer you have been selected", "spam"),
    ("Free entry in 2 a wkly comp to win FA Cup final tkts", "spam"),
    ("URGENT! We've tried to contact u. This is our 2nd attempt", "spam"),
    ("SIX chances to win CASH! From 100 to 20,000 pounds!", "spam"),
    ("PRIVATE! Your 2003 Account Statement shows 800 points", "spam"),
    ("You have won a Nokia 7250i. To claim call now!", "spam"),
    ("Claim your free mobile phone camera worth £100", "spam"),
    ("Get a FREE camera phone + 1000 free texts", "spam"),
    ("XXXMobileMovieClub: To use your credit click the WAP link", "spam"),
    ("CALL FREE 08707509020 to claim 3hrs chit-chat", "spam"),
    ("Txt JOKE to 80878 for adult jokes! Cost 25p per msg", "spam"),
    ("STOP! Win a holiday to Euro 2004! Txt EURO to 80077", "spam"),
    ("WIN cash prizes up to £5000! Just call 09061743806", "spam"),
    ("Congratulations ur awarded 500 of CD vouchers", "spam"),
    ("GUARANTEED LOAN approval! Bad credit OK! Call now", "spam"),
    ("You are chosen to receive $1000 cash or $2000 gift card", "spam"),
    ("Act now! Don't miss this limited time offer!", "spam"),
    ("Your mobile number won our monthly draw! Claim prize", "spam"),
    ("Discount pharmacy! Viagra, Cialis at lowest prices", "spam"),
    ("Make money from home! $5000/week guaranteed!", "spam"),
    
    # Ham messages
    ("Hey, are we still meeting for dinner tonight?", "ham"),
    ("Thanks for the birthday wishes!", "ham"),
    ("I'll pick you up at 7pm", "ham"),
    ("Can you send me the report?", "ham"),
    ("See you tomorrow at the office", "ham"),
    ("Good morning! How was your weekend?", "ham"),
    ("Don't forget we have a meeting at 3pm", "ham"),
    ("Could you please review this document?", "ham"),
    ("Happy anniversary! Hope you have a great day", "ham"),
    ("Let me know when you're free to talk", "ham"),
    ("The weather is really nice today", "ham"),
    ("I'm running a bit late, be there in 10 minutes", "ham"),
    ("Thanks for your help with the project", "ham"),
    ("Did you watch the game last night?", "ham"),
    ("Looking forward to seeing you soon", "ham"),
    ("Please call me when you get this message", "ham"),
    ("Great job on the presentation!", "ham"),
    ("What time does the store close?", "ham"),
    ("I'll send you the details by email", "ham"),
    ("Have a safe trip!", "ham"),
    ("Mom, I'll be home for dinner", "ham"),
    ("Can you pick up some milk on your way home?", "ham"),
    ("The meeting has been moved to tomorrow", "ham"),
    ("Your order is ready for pickup", "ham"),
    ("Reminder: Doctor appointment at 2pm", "ham"),
    ("Flight delayed by 30 minutes", "ham"),
    ("Happy birthday! Hope you have a wonderful day", "ham"),
    ("Thanks for the lunch recommendation", "ham"),
    ("I lost my keys, can you let me in?", "ham"),
    ("The movie starts at 8pm", "ham"),
    ("Congratulations on your promotion!", "ham"),
    ("Can we reschedule our meeting?", "ham"),
    ("I'll be working from home today", "ham"),
    ("The package was delivered successfully", "ham"),
    ("Don't forget to bring the documents", "ham")
]

def create_models():
    """Create and train all models"""
    print("SMS Spam Filter - Model Training")
    print("=" * 40)
    
    # Prepare data
    texts = [sms[0] for sms in sms_dataset]
    labels = [sms[1] for sms in sms_dataset]
    
    print(f"Dataset size: {len(texts)} samples")
    print(f"Spam messages: {labels.count('spam')}")
    print(f"Ham messages: {labels.count('ham')}")
    
    # Step 1: Create and save preprocessor
    print("\n1. Creating preprocessor...")
    preprocessor = SMSPreprocessor()
    cleaned_texts, encoded_labels = preprocessor.fit_transform(texts, labels)
    
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("✓ Preprocessor saved to preprocessor.pkl")
    
    # Step 2: Create and save vectorizer
    print("\n2. Creating vectorizer...")
    vectorizer = SMSVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_texts)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved to vectorizer.pkl")
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names())}")
    
    # Step 3: Train and save model
    print("\n3. Training models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Train different models and select the best
    models = {
        'Naive Bayes': SMSSpamModel('naive_bayes'),
        'Random Forest': SMSSpamModel('random_forest'),
        'Logistic Regression': SMSSpamModel('logistic_regression')
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Evaluate
        test_score = model.score(X_test, y_test)
        cv_scores = cross_val_score(model.model, X_train, y_train, cv=3)
        
        print(f"  Test Score: {test_score:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    # Save the best model
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f"\n✓ Best model ({best_name}) saved to model.pkl")
    print(f"  Best score: {best_score:.4f}")
    
    # Test the complete pipeline
    print("\n4. Testing complete pipeline...")
    test_messages = [
        "Congratulations! You won $1000! Click here now!",
        "Hey, can you pick me up at 6pm?",
        "FREE iPhone! Limited offer! Call now!",
        "Meeting at 3pm tomorrow"
    ]
    
    for msg in test_messages:
        cleaned = preprocessor.transform([msg])
        vectorized = vectorizer.transform(cleaned)
        prediction = best_model.predict(vectorized)[0]
        probabilities = best_model.predict_proba(vectorized)[0]
        
        pred_label = preprocessor.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        status = "🚨 SPAM" if pred_label == 'spam' else "✅ HAM"
        print(f"{status} ({confidence:.3f}): {msg[:50]}{'...' if len(msg) > 50 else ''}")
    
    return True

def create_predictor_file():
    """Create a simple predictor file"""
    predictor_code = '''import pickle

def predict_sms(message):
    """
    Predict if an SMS is spam or ham
    
    Args:
        message (str): The SMS message to classify
        
    Returns:
        dict: Prediction results
    """
    try:
        # Load models
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Process message
        cleaned_message = preprocessor.transform([message])
        vectorized_message = vectorizer.transform(cleaned_message)
        
        # Make prediction
        prediction = model.predict(vectorized_message)[0]
        probabilities = model.predict_proba(vectorized_message)[0]
        
        # Convert prediction back to label
        pred_label = preprocessor.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Get class probabilities
        classes = preprocessor.label_encoder.classes_
        prob_dict = dict(zip(classes, probabilities))
        
        return {
            'message': message,
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'is_spam': pred_label == 'spam'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'message': message
        }

if __name__ == "__main__":
    # Test messages
    test_messages = [
        "Congratulations! You have won $5000!",
        "Hey, are we still on for lunch?",
        "FREE iPhone! Click now!",
        "Can you review this document?"
    ]
    
    print("SMS Spam Predictor Test")
    print("=" * 25)
    
    for i, msg in enumerate(test_messages, 1):
        result = predict_sms(msg)
        
        if 'error' in result:
            print(f"{i}. ERROR: {result['error']}")
        else:
            status = "🚨 SPAM" if result['is_spam'] else "✅ HAM"
            print(f"{i}. {status} ({result['confidence']:.3f}): {msg}")
'''
    
    with open('predict_sms.py', 'w', encoding='utf-8') as f:
        f.write(predictor_code)
    
    print("\n✓ Predictor file created: predict_sms.py")

if __name__ == "__main__":
    try:
        # Create all models
        success = create_models()
        
        if success:
            # Create predictor file
            create_predictor_file()
            
            print(f"\n🎉 SMS Spam Filter setup complete!")
            print(f"\nFiles created:")
            for filename in ['preprocessor.pkl', 'vectorizer.pkl', 'model.pkl', 'predict_sms.py']:
                if os.path.exists(filename):
                    size = os.path.getsize(filename)
                    print(f"  ✓ {filename} ({size:,} bytes)")
            
            print(f"\nUsage:")
            print(f"  python predict_sms.py")
            print(f"\nOr import in your code:")
            print(f"  from predict_sms import predict_sms")
            print(f"  result = predict_sms('Your message here')")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Please make sure you have the required packages installed:")
        print("pip install scikit-learn nltk pandas numpy")