# src/predictor.py
import joblib
import logging
import pickle
from typing import Union, List, Dict

logger = logging.getLogger(__name__)

class SpamPredictor:
    """
    Handles spam prediction for new SMS messages
    """
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with saved model
        
        Args:
            model_path: Path to saved model pipeline
        """
        self.model_path = model_path
        self.pipeline = None
        self.load_model()
        
    def load_model(self):
        """Load saved model pipeline"""
        try:
            self.pipeline = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def predict(self, message: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict spam/ham for single message or list of messages
        
        Args:
            message: SMS message(s) to classify
            
        Returns:
            Prediction(s): 'spam' or 'ham'
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")
            
        # Handle single message
        if isinstance(message, str):
            prediction = self.pipeline.predict([message])[0]
            return prediction
            
        # Handle list of messages
        predictions = self.pipeline.predict(message)
        return predictions.tolist()
        
    def predict_proba(self, message: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        Get prediction probabilities
        
        Returns:
            Dictionary with probabilities for each class
        """
        if self.pipeline is None:
            raise ValueError("Model not loaded")
            
        # Get class labels
        classes = self.pipeline.classes_
        
        # Handle single message
        if isinstance(message, str):
            proba = self.pipeline.predict_proba([message])[0]
            return dict(zip(classes, proba))
            
        # Handle list of messages
        probabilities = self.pipeline.predict_proba(message)
        return [dict(zip(classes, proba)) for proba in probabilities]
        
    def explain_prediction(self, message: str, top_features: int = 10) -> Dict:
        """
        Provide explanation for prediction (basic implementation)
        
        Args:
            message: SMS message to explain
            top_features: Number of top features to show
            
        Returns:
            Dictionary with prediction details
        """
        prediction = self.predict(message)
        probabilities = self.predict_proba(message)
        
        # Get processed message
        preprocessor = self.pipeline.named_steps['preprocessor']
        processed_message = preprocessor.transform([message])[0]
        
        explanation = {
            'message': message,
            'processed_message': processed_message,
            'prediction': prediction,
            'confidence': max(probabilities.values()),
            'probabilities': probabilities,
            'top_words': processed_message.split()[:top_features]
        }
        
        return explanation