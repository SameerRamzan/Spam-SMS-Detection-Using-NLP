#!/usr/bin/env python3
"""
SMS Spam Detection Demo Script

This script demonstrates how to use the trained models for spam detection.
Run this script to test the spam detection models on sample messages.

Usage:
    python demo_spam_detection.py

Requirements:
    - Install dependencies: pip install -r requirements.txt
    - Have the dataset 'Spam_SMS.csv' in the project directory
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

def load_dataset():
    """Load the SMS spam dataset with error handling."""
    dataset_path = "Spam_SMS.csv"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset file 'Spam_SMS.csv' not found!")
        print("📥 Please download the SMS Spam Collection dataset and place it in the project directory.")
        print("🔗 Dataset URL: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        
        # Create a sample dataset for demonstration
        print("\n🔄 Creating a sample dataset for demonstration...")
        sample_data = {
            'Class': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam'] * 20,
            'Message': [
                'Hi how are you doing today?',
                'WINNER!! Free entry to win £1000! Text WIN to 80086',
                'Can you pick me up at 5pm?',
                'Urgent! Call now to claim your prize money',
                'See you tomorrow for lunch',
                'Free mobile ringtones! Reply STOP to unsubscribe'
            ] * 20
        }
        df = pd.DataFrame(sample_data)
        print(f"✅ Sample dataset created with {len(df)} rows for demonstration.")
        return df
    else:
        df = pd.read_csv(dataset_path)
        print(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        return df

def train_model(df):
    """Train a simple TF-IDF + Logistic Regression model."""
    print("\n🤖 Training spam detection model...")
    
    # Prepare data
    X = df['Message']
    y = df['Class'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.3f}")
    
    return model, vectorizer

def predict_message(message, model, vectorizer):
    """Predict if a message is spam or ham."""
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]
    probability = model.predict_proba(message_tfidf)[0]
    
    result = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    confidence = max(probability)
    
    return result, confidence

def demo_predictions(model, vectorizer):
    """Demonstrate predictions on sample messages."""
    print("\n🔍 Testing model on sample messages:")
    print("=" * 50)
    
    sample_messages = [
        "Hi there! How was your day?",
        "FREE! Win a brand new iPhone! Click here now!",
        "Can you pick me up from the airport tomorrow?",
        "URGENT! You've won £1000! Call 09012345678 now!",
        "Thanks for the dinner last night, it was great",
        "Congratulations! You've been selected for a special offer",
        "Are we still meeting for coffee at 3pm?",
        "Limited time offer! Get 50% off now! Text STOP to opt out",
        "Happy birthday! Hope you have a wonderful day",
        "Your account will be suspended unless you verify now"
    ]
    
    for i, message in enumerate(sample_messages, 1):
        result, confidence = predict_message(message, model, vectorizer)
        print(f"{i:2d}. {result} ({confidence:.2f}) | {message}")
    
    print("=" * 50)

def interactive_mode(model, vectorizer):
    """Interactive mode for testing custom messages."""
    print("\n🎯 Interactive Spam Detection Mode")
    print("Enter messages to test (type 'quit' to exit):")
    print("-" * 40)
    
    while True:
        try:
            message = input("\nEnter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break
        
        if message.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not message:
            print("⚠️ Please enter a message.")
            continue
        
        result, confidence = predict_message(message, model, vectorizer)
        print(f"Result: {result} (Confidence: {confidence:.2f})")

def main():
    """Main function to run the demo."""
    print("🚀 SMS Spam Detection Demo")
    print("=" * 30)
    
    # Load dataset
    df = load_dataset()
    
    # Train model
    model, vectorizer = train_model(df)
    
    # Demo predictions
    demo_predictions(model, vectorizer)
    
    # Interactive mode
    interactive_mode(model, vectorizer)

if __name__ == "__main__":
    main()