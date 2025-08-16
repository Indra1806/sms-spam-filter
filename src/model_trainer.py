# src/model_trainer.py
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class SpamClassifierTrainer:
    """
    Handles training and evaluation of SMS spam classification models
    """
    
    def __init__(self, model_type: str = 'naive_bayes'):
        """
        Initialize trainer with specified model type
        
        Args:
            model_type: 'naive_bayes', 'logistic_regression', or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.pipeline = None
        
        # Initialize model based on type
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the specified model"""
        models = {
            'naive_bayes': MultinomialNB(alpha=1.0),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        if self.model_type not in models:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model = models[self.model_type]
        logger.info(f"Initialized {self.model_type} model")
        
    def create_pipeline(self, preprocessor, max_features: int = 5000):
        """
        Create ML pipeline with preprocessor, vectorizer, and classifier
        
        Args:
            preprocessor: Text preprocessing object
            max_features: Maximum number of TF-IDF features
        """
        # TF-IDF Vectorizer configuration
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,                    # Ignore terms in less than 2 documents
            max_df=0.95,                 # Ignore terms in more than 95% of documents
            ngram_range=(1, 2),          # Use unigrams and bigrams
            stop_words=None,             # We handle stopwords in preprocessing
            lowercase=False,             # Already handled in preprocessing
            token_pattern=r'\b\w+\b'     # Word tokens only
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        logger.info("Created ML pipeline")
        
    def train(self, X_train, y_train):
        """Train the model"""
        if self.pipeline is None:
            raise ValueError("Pipeline not created. Call create_pipeline() first.")
            
        logger.info("Starting model training...")
        self.pipeline.fit(X_train, y_train)
        logger.info("Model training completed")
        
    def evaluate(self, X_test, y_test, verbose: bool = True):
        """
        Evaluate model performance
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.pipeline is None:
            raise ValueError("Model not trained")
            
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='spam', average='binary'),
            'recall': recall_score(y_test, y_pred, pos_label='spam', average='binary'),
            'f1_score': f1_score(y_test, y_pred, pos_label='spam', average='binary')
        }
        
        if verbose:
            print("\n" + "="*50)
            print("MODEL EVALUATION RESULTS")
            print("="*50)
            print(f"Model Type: {self.model_type}")
            print(f"Accuracy:  {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall:    {metrics['recall']:.4f}")
            print(f"F1-Score:  {metrics['f1_score']:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        # Store predictions for visualization
        metrics['y_true'] = y_test
        metrics['y_pred'] = y_pred
        
        return metrics
        
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.title(f'Confusion Matrix - {self.model_type.replace("_", " ").title()}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_feature_importance(self, top_n: int = 20):
        """Get most important features for spam detection"""
        if self.model_type == 'naive_bayes':
            # For Naive Bayes, use log probability ratios
            feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
            log_prob_ratio = (self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0])
            
            # Get top spam indicators
            top_spam_indices = log_prob_ratio.argsort()[-top_n:][::-1]
            top_spam_features = [(feature_names[i], log_prob_ratio[i]) for i in top_spam_indices]
            
            return top_spam_features
            
        elif self.model_type == 'logistic_regression':
            # For Logistic Regression, use coefficients
            feature_names = self.pipeline.named_steps['vectorizer'].get_feature_names_out()
            coefficients = self.model.coef_[0]
            
            # Get top features (both positive and negative)
            top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]
            top_features = [(feature_names[i], coefficients[i]) for i in top_indices]
            
            return top_features
            
        return None
        
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """Plot feature importance"""
        features = self.get_feature_importance(top_n)
        
        if features is None:
            print(f"Feature importance not available for {self.model_type}")
            return
            
        words, scores = zip(*features)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if score > 0 else 'blue' for _, score in features]
        bars = plt.barh(range(len(words)), [score for _, score in features], color=colors)
        plt.yticks(range(len(words)), words)
        plt.xlabel('Feature Importance Score')
        plt.title(f'Top {top_n} Features - {self.model_type.replace("_", " ").title()}')
        plt.grid(axis='x', alpha=0.3)
        
        # Add legend
        import matplotlib.patches as patches
        red_patch = patches.Patch(color='red', label='Spam indicators')
        blue_patch = patches.Patch(color='blue', label='Ham indicators')
        plt.legend(handles=[red_patch, blue_patch])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_model(self, model_path: str, vectorizer_path: str = None, preprocessor_path: str = None):
        """Save trained model and components"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
            
        # Save complete pipeline
        joblib.dump(self.pipeline, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Optionally save components separately
        if vectorizer_path:
            joblib.dump(self.pipeline.named_steps['vectorizer'], vectorizer_path)
            
        if preprocessor_path:
            joblib.dump(self.pipeline.named_steps['preprocessor'], preprocessor_path)