import pickle
from sms_spam_filter_lib import SMSPreprocessor, SMSVectorizer, SMSSpamModel

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
