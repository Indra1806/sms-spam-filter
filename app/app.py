import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
except:
    logger.warning("Could not download NLTK stopwords")

class SMSPreprocessor:
    """SMS text preprocessing class"""
    
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        self.stemmer = PorterStemmer()
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove phone numbers
        text = re.sub(r'\+?1?\d{9,15}', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and stem
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)

class SMSSpamFilter:
    """Main SMS Spam Filter class"""
    
    def __init__(self):
        self.preprocessor = SMSPreprocessor()
        self.vectorizer = None
        self.models = {}
        self.model_names = ['Naive Bayes', 'Logistic Regression', 'Random Forest']
        
    def load_sample_data(self):
        """Create sample training data if no dataset is available"""
        spam_messages = [
            "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
            "Urgent! You have won a 1 week FREE membership in our £100,000 prize Jackpot!",
            "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121",
            "PRIVATE! Your 2003 Account Statement for shows 800 un-claimed pounds",
            "Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed",
            "Click the link and win £1000 cash or prize equivalent T&Cs apply",
            "You have won £1000 cash! Call now to claim your prize",
            "URGENT! We have been trying to contact u. Last weekfor claiming ur £100 reward",
            "Win a £1000 or a new car! Call 09061701461. T&Cs apply",
            "FREE ringtone! Reply SUPER to 85555 now! More premium content to follow"
        ]
        
        ham_messages = [
            "Hey, how are you doing today?",
            "Can you pick up some milk on your way home?",
            "Meeting is scheduled for 3 PM in conference room",
            "Happy birthday! Hope you have a great day",
            "Thanks for helping me with the project yesterday",
            "Are we still on for dinner tonight?",
            "Could you send me the report when you get a chance?",
            "Great job on the presentation today!",
            "Don't forget about the doctor's appointment tomorrow",
            "Love you, see you soon!"
        ]
        
        messages = spam_messages + ham_messages
        labels = ['spam'] * len(spam_messages) + ['ham'] * len(ham_messages)
        
        return pd.DataFrame({'message': messages, 'label': labels})
    
    def train_models(self, df=None):
        """Train the spam filter models"""
        try:
            if df is None:
                logger.info("Loading sample training data...")
                df = self.load_sample_data()
            
            logger.info("Preprocessing messages...")
            df['cleaned_message'] = df['message'].apply(self.preprocessor.clean_text)
            
            # Remove empty messages
            df = df[df['cleaned_message'].str.len() > 0]
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X = self.vectorizer.fit_transform(df['cleaned_message'])
            y = df['label'].map({'spam': 1, 'ham': 0})
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models
            models_to_train = {
                'Naive Bayes': MultinomialNB(),
                'Logistic Regression': LogisticRegression(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
            }
            
            logger.info("Training models...")
            for name, model in models_to_train.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(y_test, predictions)
                self.models[name] = {'model': model, 'accuracy': accuracy}
                logger.info(f"{name} accuracy: {accuracy:.4f}")
            
            self.save_models()
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False
    
    def save_models(self):
        """Save trained models and vectorizer"""
        try:
            os.makedirs('models', exist_ok=True)
            
            # Save vectorizer
            with open('models/vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save models
            for name, model_data in self.models.items():
                filename = f"models/{name.lower().replace(' ', '_')}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model_data, f)
            
            logger.info("Models saved successfully!")
            
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load vectorizer
            with open('models/vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load models
            for name in self.model_names:
                filename = f"models/{name.lower().replace(' ', '_')}.pkl"
                if os.path.exists(filename):
                    with open(filename, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            if self.models:
                logger.info("Models loaded successfully!")
                return True
            else:
                logger.warning("No models found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, message, model_name='Naive Bayes'):
        """Predict if a message is spam or ham"""
        try:
            if not self.vectorizer or model_name not in self.models:
                return None, 0.0
            
            # Preprocess message
            cleaned_message = self.preprocessor.clean_text(message)
            if not cleaned_message:
                return "ham", 0.5
            
            # Vectorize message
            message_vector = self.vectorizer.transform([cleaned_message])
            
            # Get prediction and probability
            model = self.models[model_name]['model']
            prediction = model.predict(message_vector)[0]
            
            try:
                probabilities = model.predict_proba(message_vector)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.5
            
            result = "spam" if prediction == 1 else "ham"
            return result, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return None, 0.0

def main():
    """Streamlit app main function"""
    st.set_page_config(
        page_title="SMS Spam Filter",
        page_icon="📱",
        layout="wide"
    )
    
    st.title("📱 SMS Spam Filter")
    st.markdown("---")
    
    # Initialize the spam filter
    if 'spam_filter' not in st.session_state:
        st.session_state.spam_filter = None  # or your default model

    # Now you can safely access st.session_state["spam_filter"]
    spam_filter = st.session_state.spam_filter
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    if spam_filter.models:
        model_options = list(spam_filter.models.keys())
        selected_model = st.sidebar.selectbox("Choose Model:", model_options)
        
        # Display model accuracies
        st.sidebar.subheader("Model Accuracies:")
        for name, model_data in spam_filter.models.items():
            accuracy = model_data.get('accuracy', 0)
            st.sidebar.write(f"{name}: {accuracy:.4f}")
    else:
        st.sidebar.error("No models available")
        selected_model = 'Naive Bayes'
    
    # Retrain option
    if st.sidebar.button("Retrain Models"):
        with st.spinner("Retraining models..."):
            if spam_filter.train_models():
                st.sidebar.success("Models retrained successfully!")
                st.rerun()
            else:
                st.sidebar.error("Failed to retrain models")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Message Classification")
        
        # Text input
        message = st.text_area(
            "Enter your SMS message:",
            placeholder="Type your message here...",
            height=100
        )
        
        # Predict button
        if st.button("🔍 Analyze Message", type="primary"):
            if message.strip():
                with st.spinner("Analyzing message..."):
                    result, confidence = spam_filter.predict(message, selected_model)
                
                if result is not None:
                    # Display result
                    if result == "spam":
                        st.error(f"🚨 **SPAM DETECTED**")
                        st.write(f"**Confidence:** {confidence:.2%}")
                    else:
                        st.success(f"✅ **LEGITIMATE MESSAGE**")
                        st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Show message details
                    with st.expander("Message Details"):
                        cleaned_msg = spam_filter.preprocessor.clean_text(message)
                        st.write(f"**Original:** {message}")
                        st.write(f"**Cleaned:** {cleaned_msg}")
                        st.write(f"**Model Used:** {selected_model}")
                else:
                    st.error("Error analyzing message. Please try again.")
            else:
                st.warning("Please enter a message to analyze.")
        
        # Example messages
        st.subheader("📋 Try These Examples")
        
        col_spam, col_ham = st.columns(2)
        
        with col_spam:
            st.write("**Spam Examples:**")
            spam_examples = [
                "WINNER!! You have won £1000! Call now!",
                "FREE entry to win £500! Text WIN to 12345",
                "Urgent! Click here to claim your prize"
            ]
            for i, example in enumerate(spam_examples):
                if st.button(f"Example {i+1}", key=f"spam_{i}"):
                    st.session_state.example_message = example
        
        with col_ham:
            st.write("**Ham Examples:**")
            ham_examples = [
                "Hey, are you free for lunch today?",
                "Meeting rescheduled to 3 PM tomorrow",
                "Thanks for your help with the project!"
            ]
            for i, example in enumerate(ham_examples):
                if st.button(f"Example {i+1}", key=f"ham_{i}"):
                    st.session_state.example_message = example
        
        # Use example message
        if hasattr(st.session_state, 'example_message'):
            st.text_area(
                "Selected example:",
                value=st.session_state.example_message,
                key="example_display",
                disabled=True
            )
            if st.button("🔍 Analyze Example", type="secondary"):
                result, confidence = spam_filter.predict(st.session_state.example_message, selected_model)
                if result == "spam":
                    st.error(f"🚨 **SPAM DETECTED** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ **LEGITIMATE MESSAGE** (Confidence: {confidence:.2%})")
    
    with col2:
        st.subheader("ℹ️ About")
        st.info(
            "This SMS spam filter uses machine learning to classify messages as spam or legitimate. "
            "It employs text preprocessing and TF-IDF vectorization with multiple classification algorithms."
        )
        
        st.subheader("🔧 Features")
        st.markdown("""
        - **Multiple Models**: Naive Bayes, Logistic Regression, Random Forest
        - **Text Preprocessing**: Cleaning, stemming, stopword removal
        - **TF-IDF Vectorization**: Advanced text feature extraction
        - **Confidence Scores**: Probability-based predictions
        - **Real-time Analysis**: Instant message classification
        """)
        
        st.subheader("📊 Model Info")
        if spam_filter.models:
            best_model = max(spam_filter.models.items(), key=lambda x: x[1].get('accuracy', 0))
            st.metric("Best Model", best_model[0], f"{best_model[1].get('accuracy', 0):.2%}")
            st.metric("Total Models", len(spam_filter.models))
        
        st.subheader("⚠️ Disclaimer")
        st.caption(
            "This is a demo application. For production use, train with a larger, "
            "more diverse dataset and implement additional security measures."
        )

if __name__ == "__main__":
    try:
        logger.info("Starting SMS Spam Filter App...")
        main()
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"Application error: {str(e)}")