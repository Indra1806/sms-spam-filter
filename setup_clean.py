#!/usr/bin/env python3
"""
Clean SMS Spam Filter Setup Script
This script creates all files with proper encoding and line endings
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

# Try to import nltk
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    # Download required data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
        
except ImportError:
    print("NLTK not found. Installing...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'nltk'])
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

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
        return self.vectorizer.fit_transform(texts)
    
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

# Dataset
SMS_DATASET = [
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
    ("The movie starts at 8pm", "ham")
]

def clean_setup():
    print("SMS Spam Filter - Clean Setup")
    print("=" * 35)
    
    # Remove existing pickle files
    for file in ['preprocessor.pkl', 'vectorizer.pkl', 'model.pkl']:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed existing {file}")
    
    # Prepare data
    texts = [sms[0] for sms in SMS_DATASET]
    labels = [sms[1] for sms in SMS_DATASET]
    
    print(f"\nDataset: {len(texts)} messages ({labels.count('spam')} spam, {labels.count('ham')} ham)")
    
    # Step 1: Preprocessor
    print("\n1. Creating preprocessor...")
    preprocessor = SMSPreprocessor()
    cleaned_texts, encoded_labels = preprocessor.fit_transform(texts, labels)
    
    with open('preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    print("✓ Preprocessor saved")
    
    # Step 2: Vectorizer
    print("\n2. Creating vectorizer...")
    vectorizer = SMSVectorizer(max_features=3000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(cleaned_texts)
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved (vocab: {len(vectorizer.get_feature_names())})")
    
    # Step 3: Model
    print("\n3. Training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    # Try different models
    models = {
        'Naive Bayes': SMSSpamModel('naive_bayes'),
        'Random Forest': SMSSpamModel('random_forest'),
        'Logistic Regression': SMSSpamModel('logistic_regression')
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"  {name}: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model = model
            best_name = name
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model saved: {best_name} ({best_score:.4f})")
    
    # Create Flask app
    create_flask_app()
    
    # Create folder structure
    create_folder_structure()
    
    print("\n🎉 Setup complete!")
    print("\nFiles created:")
    for file in ['preprocessor.pkl', 'vectorizer.pkl', 'model.pkl', 'app.py']:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✓ {file} ({size:,} bytes)")
    
    print("\nTo run the app:")
    print("  python app.py")
    print("  Then open: http://localhost:5000")

def create_flask_app():
    app_code = '''import pickle
import os
import logging
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class SMSSpamPredictor:
    def __init__(self):
        self.preprocessor = None
        self.vectorizer = None
        self.model = None
        self.loaded = False
        
    def load_models(self):
        try:
            required_files = ['preprocessor.pkl', 'vectorizer.pkl', 'model.pkl']
            for file in required_files:
                if not os.path.exists(file):
                    logger.error(f"Required file missing: {file}")
                    return False
            
            with open('preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            logger.info("Preprocessor loaded successfully")
            
            with open('vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info("Vectorizer loaded successfully")
            
            with open('model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            logger.info("Model loaded successfully")
            
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict(self, message):
        if not self.loaded:
            if not self.load_models():
                return None
        
        try:
            cleaned_message = self.preprocessor.transform([message])
            vectorized_message = self.vectorizer.transform(cleaned_message)
            
            prediction = self.model.predict(vectorized_message)[0]
            probabilities = self.model.predict_proba(vectorized_message)[0]
            
            pred_label = self.preprocessor.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            classes = self.preprocessor.label_encoder.classes_
            prob_dict = dict(zip(classes, [float(p) for p in probabilities]))
            
            return {
                'prediction': pred_label,
                'confidence': confidence,
                'probabilities': prob_dict,
                'is_spam': pred_label == 'spam',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

predictor = SMSSpamPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            }), 400
        
        message = data['message'].strip()
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'Empty message'
            }), 400
        
        result = predictor.predict(message)
        
        if result is None:
            return jsonify({
                'success': False,
                'error': 'Failed to load models'
            }), 500
        
        if not result.get('success', False):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Prediction failed')
            }), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error'
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor.loaded
    })

if __name__ == '__main__':
    logger.info("Starting SMS Spam Filter App...")
    
    if predictor.load_models():
        logger.info("All models loaded successfully!")
    else:
        logger.warning("Failed to load models. Please ensure all pickle files exist.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''
    
    with open('app.py', 'w', encoding='utf-8', newline='\n') as f:
        f.write(app_code)
    
    print("✓ Flask app created: app.py")

def create_folder_structure():
    # Create templates folder
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Spam Filter</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            width: 100%;
            max-width: 600px;
            text-align: center;
        }
        
        .title {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1rem;
            margin-bottom: 30px;
        }
        
        .input-group {
            margin-bottom: 25px;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            resize: vertical;
            min-height: 120px;
            transition: border-color 0.3s ease;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            transition: transform 0.2s ease;
            font-weight: 600;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 20px;
            border-radius: 15px;
            font-size: 18px;
            font-weight: 600;
            display: none;
        }
        
        .result.spam {
            background-color: #ffebee;
            color: #c62828;
            border: 2px solid #ffcdd2;
        }
        
        .result.ham {
            background-color: #e8f5e8;
            color: #2e7d32;
            border: 2px solid #a5d6a7;
        }
        
        .result.error {
            background-color: #fff3e0;
            color: #ef6c00;
            border: 2px solid #ffcc02;
        }
        
        .confidence {
            margin-top: 10px;
            font-size: 14px;
            opacity: 0.8;
        }
        
        .loading {
            display: none;
            margin-top: 20px;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .examples {
            margin-top: 30px;
            text-align: left;
        }
        
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .example {
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        
        .example:hover {
            background: #e9ecef;
        }
        
        .example-label {
            font-weight: 600;
            color: #667eea;
            font-size: 12px;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="title">📱 SMS Spam Filter</h1>
        <p class="subtitle">Enter an SMS message to check if it's spam or legitimate</p>
        
        <div class="input-group">
            <textarea 
                id="messageInput" 
                placeholder="Enter your SMS message here..."
                maxlength="500"
            ></textarea>
        </div>
        
        <button class="btn" id="analyzeBtn">🔍 Analyze Message</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing message...</p>
        </div>
        
        <div class="result" id="result"></div>
        
        <div class="examples">
            <h3>Try These Examples:</h3>
            
            <div class="example" onclick="setExample('Congratulations! You have won $5000! Click here to claim your prize now!')">
                <div class="example-label">Likely Spam</div>
                <div>Congratulations! You have won $5000! Click here to claim your prize now!</div>
            </div>
            
            <div class="example" onclick="setExample('Hey mom, I will be home late tonight. Don\\'t wait up for dinner.')">
                <div class="example-label">Legitimate Message</div>
                <div>Hey mom, I will be home late tonight. Don't wait up for dinner.</div>
            </div>
            
            <div class="example" onclick="setExample('URGENT! Your bank account will be suspended. Verify your details immediately!')">
                <div class="example-label">Likely Spam</div>
                <div>URGENT! Your bank account will be suspended. Verify your details immediately!</div>
            </div>
            
            <div class="example" onclick="setExample('Meeting rescheduled to 3pm tomorrow. Conference room B.')">
                <div class="example-label">Legitimate Message</div>
                <div>Meeting rescheduled to 3pm tomorrow. Conference room B.</div>
            </div>
        </div>
    </div>

    <script>
        const messageInput = document.getElementById('messageInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const result = document.getElementById('result');
        const loading = document.getElementById('loading');

        function setExample(text) {
            messageInput.value = text;
            messageInput.focus();
        }

        function showLoading() {
            loading.style.display = 'block';
            result.style.display = 'none';
            analyzeBtn.disabled = true;
        }

        function hideLoading() {
            loading.style.display = 'none';
            analyzeBtn.disabled = false;
        }

        function showResult(data) {
            result.style.display = 'block';
            
            if (data.success) {
                if (data.is_spam) {
                    result.className = 'result spam';
                    result.innerHTML = `
                        <div>🚨 <strong>SPAM DETECTED</strong></div>
                        <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                        <div class="confidence">Spam probability: ${(data.probabilities.spam * 100).toFixed(1)}%</div>
                    `;
                } else {
                    result.className = 'result ham';
                    result.innerHTML = `
                        <div>✅ <strong>LEGITIMATE MESSAGE</strong></div>
                        <div class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                        <div class="confidence">Ham probability: ${(data.probabilities.ham * 100).toFixed(1)}%</div>
                    `;
                }
            } else {
                result.className = 'result error';
                result.innerHTML = `
                    <div>⚠️ <strong>ERROR</strong></div>
                    <div class="confidence">${data.error || 'Failed to analyze the message. Please try again.'}</div>
                `;
            }
        }

        analyzeBtn.addEventListener('click', async () => {
            const message = messageInput.value.trim();
            
            if (!message) {
                alert('Please enter a message to analyze.');
                return;
            }
            
            showLoading();
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                showResult(data);
                
            } catch (error) {
                console.error('Error:', error);
                showResult({
                    success: false,
                    error: 'Network error. Please check your connection and try again.'
                });
            } finally {
                hideLoading();
            }
        });

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.ctrlKey) {
                e.preventDefault();
                analyzeBtn.click();
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8', newline='\n') as f:
        f.write(html_template)
    
    print("✓ HTML template created: templates/index.html")