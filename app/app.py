# app/app.py
from flask import Flask, render_template, request, jsonify
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import SpamPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize predictor (update path as needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'spam_classifier.pkl')
predictor = None

try:
    predictor = SpamPredictor(MODEL_PATH)
    logger.info("Spam predictor initialized successfully")
except Exception as e:
    logger.error(f"Error initializing predictor: {str(e)}")

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for spam prediction
    
    Expected JSON:
    {
        "message": "Your SMS text here"
    }
    
    Returns:
    {
        "message": "input message",
        "prediction": "spam" or "ham",
        "confidence": 0.95,
        "probabilities": {"ham": 0.05, "spam": 0.95}
    }
    """
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # Get message from request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        message = data['message'].strip()
        
        if not message:
            return jsonify({'error': 'Empty message'}), 400
            
        # Make prediction
        prediction = predictor.predict(message)
        probabilities = predictor.predict_proba(message)
        confidence = max(probabilities.values())
        
        result = {
            'message': message,
            'prediction': prediction,
            'confidence': round(confidence, 4),
            'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
            'status': 'success'
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    API endpoint for batch prediction
    
    Expected JSON:
    {
        "messages": ["message1", "message2", ...]
    }
    """
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.get_json()
        
        if not data or 'messages' not in data:
            return jsonify({'error': 'No messages provided'}), 400
            
        messages = data['messages']
        
        if not isinstance(messages, list) or len(messages) == 0:
            return jsonify({'error': 'Invalid messages format'}), 400
            
        # Make predictions
        predictions = predictor.predict(messages)
        probabilities = predictor.predict_proba(messages)
        
        results = []
        for i, message in enumerate(messages):
            result = {
                'message': message,
                'prediction': predictions[i],
                'probabilities': {k: round(v, 4) for k, v in probabilities[i].items()}
            }
            results.append(result)
            
        return jsonify({
            'results': results,
            'total_processed': len(messages),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Batch prediction failed', 'details': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = 'healthy' if predictor is not None else 'unhealthy'
    return jsonify({'status': status})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)