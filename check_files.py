#!/usr/bin/env python3
"""
Diagnostic script to check SMS spam filter files
"""

import os
import pickle
import sys

def check_files():
    print("SMS Spam Filter - File Diagnostic")
    print("=" * 35)
    
    # Check required files
    required_files = ['preprocessor.pkl', 'vectorizer.pkl', 'model.pkl', 'app.py']
    
    print("1. Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - NOT FOUND")
    
    # Check templates folder
    print("\n2. Checking templates folder...")
    if os.path.exists('templates'):
        if os.path.exists('templates/index.html'):
            print("  ✅ templates/index.html")
        else:
            print("  ❌ templates/index.html - NOT FOUND")
    else:
        print("  ❌ templates folder - NOT FOUND")
    
    # Try to load pickle files
    print("\n3. Testing pickle files...")
    
    try:
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        print("  ✅ preprocessor.pkl loads successfully")
    except FileNotFoundError:
        print("  ❌ preprocessor.pkl - File not found")
    except Exception as e:
        print(f"  ❌ preprocessor.pkl - Error: {e}")
    
    try:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("  ✅ vectorizer.pkl loads successfully")
    except FileNotFoundError:
        print("  ❌ vectorizer.pkl - File not found")
    except Exception as e:
        print(f"  ❌ vectorizer.pkl - Error: {e}")
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("  ✅ model.pkl loads successfully")
    except FileNotFoundError:
        print("  ❌ model.pkl - File not found")
    except Exception as e:
        print(f"  ❌ model.pkl - Error: {e}")
    
    # Test complete pipeline
    print("\n4. Testing complete pipeline...")
    try:
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Test prediction
        test_message = "FREE iPhone! Click now!"
        cleaned = preprocessor.transform([test_message])
        vectorized = vectorizer.transform(cleaned)
        prediction = model.predict(vectorized)[0]
        probabilities = model.predict_proba(vectorized)[0]
        
        pred_label = preprocessor.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        print(f"  ✅ Pipeline test successful!")
        print(f"  📧 Test message: '{test_message}'")
        print(f"  🔍 Prediction: {pred_label.upper()} ({confidence:.3f})")
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {e}")
    
    # Check Python version and packages
    print("\n5. Environment check...")
    print(f"  Python version: {sys.version}")
    
    packages = ['sklearn', 'flask', 'nltk', 'numpy', 'pandas']
    for package in packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT INSTALLED")
    
    print("\n" + "=" * 35)
    print("Diagnostic complete!")

if __name__ == "__main__":
    check_files()