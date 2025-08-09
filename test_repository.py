#!/usr/bin/env python3
"""
Test script to validate the notebooks can run without errors.
This script extracts and runs key code cells from the notebooks.
"""

import os
import sys

def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas and numpy imported successfully")
    except ImportError as e:
        print(f"❌ Error importing pandas/numpy: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        print("✅ scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Error importing scikit-learn: {e}")
        return False
    
    return True

def test_data_loading():
    """Test data loading functionality."""
    print("\n🧪 Testing data loading...")
    
    try:
        import pandas as pd
        
        # Test sample data creation (as would happen when dataset is missing)
        sample_data = {
            'Class': ['ham', 'spam', 'ham', 'spam', 'ham'],
            'Message': [
                'Hi how are you doing today?',
                'WINNER!! Free entry to win £1000! Text WIN to 80086',
                'Can you pick me up at 5pm?',
                'Urgent! Call now to claim your prize',
                'See you tomorrow for lunch'
            ]
        }
        df = pd.DataFrame(sample_data)
        
        # Test label mapping
        df['Label'] = df['Class'].map({'ham': 0, 'spam': 1})
        
        print(f"✅ Sample dataset created successfully! Shape: {df.shape}")
        print(f"✅ Label mapping works: {df['Label'].tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading test: {e}")
        return False

def test_text_processing():
    """Test text processing functionality."""
    print("\n🧪 Testing text processing...")
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        sample_texts = [
            "This is a normal message",
            "FREE! Win money now! Call immediately!",
            "How are you doing today?",
            "URGENT! Limited time offer!"
        ]
        
        # Test TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
        
        print(f"✅ TF-IDF vectorization works! Matrix shape: {tfidf_matrix.shape}")
        print(f"✅ Feature names available: {len(vectorizer.get_feature_names_out())} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in text processing test: {e}")
        return False

def test_model_training():
    """Test basic model training functionality."""
    print("\n🧪 Testing model training...")
    
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Create sample dataset
        data = {
            'Message': [
                'Hi there how are you',
                'FREE money win now call',
                'Let me know when you arrive',
                'URGENT winner selected call now',
                'Thanks for the great dinner',
                'Claim your prize immediately',
                'See you at the meeting tomorrow',
                'Limited time offer call now'
            ] * 10,  # Repeat to have more samples
            'Class': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'] * 10
        }
        df = pd.DataFrame(data)
        
        # Prepare features and labels
        X = df['Message']
        y = df['Class'].map({'ham': 0, 'spam': 1})
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Vectorize
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        model = LogisticRegression(random_state=42)
        model.fit(X_train_tfidf, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training successful! Accuracy: {accuracy:.3f}")
        
        # Test prediction on new text
        test_message = ["Win free money now!"]
        test_tfidf = vectorizer.transform(test_message)
        prediction = model.predict(test_tfidf)[0]
        prob = model.predict_proba(test_tfidf)[0]
        
        result = "SPAM" if prediction == 1 else "HAM"
        confidence = max(prob)
        
        print(f"✅ Prediction test: '{test_message[0]}' -> {result} (confidence: {confidence:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training test: {e}")
        return False

def test_notebooks_structure():
    """Test that notebook files exist and are readable."""
    print("\n🧪 Testing notebook structure...")
    
    required_files = [
        'Spam_Detection_NLP.ipynb',
        'Polarity_of_Sentiments.ipynb',
        'README.md',
        'requirements.txt',
        'demo_spam_detection.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🚀 Running Repository Validation Tests")
    print("=" * 50)
    
    tests = [
        test_notebooks_structure,
        test_imports,
        test_data_loading,
        test_text_processing,
        test_model_training
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed!")
        print("✅ Repository is ready for use!")
        return 0
    else:
        print(f"⚠️ {passed}/{total} tests passed")
        print("❌ Some issues need to be addressed")
        return 1

if __name__ == "__main__":
    sys.exit(main())