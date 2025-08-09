# Spam SMS Detection Using Natural Language Processing

A comprehensive machine learning project for detecting spam SMS messages using advanced NLP techniques including transformer models and traditional machine learning approaches.

## 🎯 Project Overview

This project implements multiple approaches to classify SMS messages as either "Ham" (legitimate) or "Spam" (unwanted) messages:

1. **Transformer-based Classification**: Uses DistilBERT, a lightweight version of BERT, for state-of-the-art spam detection
2. **Sentiment Analysis**: Analyzes the polarity and sentiment patterns in SMS messages
3. **Traditional ML Approaches**: Implements TF-IDF with Logistic Regression for comparison

## 📊 Dataset

The project uses an SMS spam dataset containing labeled messages. The dataset includes:
- **Ham messages**: 4,827 legitimate SMS messages (86.6%)
- **Spam messages**: 747 spam SMS messages (13.4%)
- **Total**: 5,574 SMS messages

### Data Features
- **Class**: Label indicating 'ham' or 'spam'
- **Message**: The actual SMS message content
- **Message Length**: Character count of each message

## 🚀 Features

### Spam Detection Notebook (`Spam_Detection_NLP.ipynb`)
- **Exploratory Data Analysis**: Comprehensive data visualization and statistics
- **Text Preprocessing**: Advanced text cleaning and preparation
- **Transformer Model**: DistilBERT-based classification with Hugging Face Transformers
- **Model Training**: Fine-tuning with proper validation and evaluation
- **Visualization**: Confusion matrices, word clouds, and performance metrics
- **Model Persistence**: Save and load trained models

### Sentiment Analysis Notebook (`Polarity_of_Sentiments.ipynb`)
- **Sentiment Analysis**: TextBlob-based polarity detection
- **Text Processing**: NLTK-based preprocessing (tokenization, stemming, lemmatization)
- **Feature Engineering**: TF-IDF vectorization
- **Traditional ML**: Logistic Regression implementation
- **Statistical Analysis**: Word frequency and n-gram analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SameerRamzan/Spam-SMS-Detection-Using-NLP.git
   cd Spam-SMS-Detection-Using-NLP
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Place your `Spam_SMS.csv` file in the project root directory
   - The dataset should have columns: `Class` and `Message`
   - Alternatively, you can use the SMS Spam Collection Dataset from UCI ML Repository

## 📋 Usage

### Running the Notebooks

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Spam Detection with Transformers**:
   - Open `Spam_Detection_NLP.ipynb`
   - Follow the step-by-step implementation
   - Train the DistilBERT model on your dataset

3. **Sentiment Analysis**:
   - Open `Polarity_of_Sentiments.ipynb`
   - Explore sentiment patterns in SMS messages
   - Compare traditional ML approaches

### Quick Start Example

```python
# Load and preprocess data
import pandas as pd
df = pd.read_csv("Spam_SMS.csv")
df['Label'] = df['Class'].map({'ham': 0, 'spam': 1})

# Using the transformer model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Predict new messages
def predict_spam(message):
    inputs = tokenizer(message, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(-1).item()
    return "Spam" if prediction == 1 else "Ham"
```

## 📈 Model Performance

### Transformer Model (DistilBERT)
- **Architecture**: DistilBERT-base-uncased
- **Training**: 3 epochs with learning rate 2e-5
- **Validation**: Accuracy and F1-score metrics
- **Advantages**: State-of-the-art performance, contextual understanding

### Traditional ML Model (Logistic Regression)
- **Accuracy**: 97.2%
- **Precision (Ham)**: 97%
- **Precision (Spam)**: 99%
- **Recall (Ham)**: 100%
- **Recall (Spam)**: 81%

## 🔍 Key Insights

1. **Message Length**: Spam messages are typically longer (avg: 138 chars) than ham messages (avg: 71 chars)
2. **Common Spam Words**: "call", "free", "text", "claim", "reply"
3. **Sentiment Patterns**: Spam messages tend to have higher positive polarity scores
4. **N-gram Analysis**: Spam messages contain phrases like "please call", "free entry", "prize guaranteed"

## 📊 Visualizations

The notebooks include various visualizations:
- Class distribution plots
- Message length histograms
- Word clouds for spam vs ham messages
- Confusion matrices
- Training loss and accuracy curves
- Sentiment distribution analysis

## 🏗️ Project Structure

```
Spam-SMS-Detection-Using-NLP/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── Spam_Detection_NLP.ipynb           # Main transformer-based implementation
├── Polarity_of_Sentiments.ipynb       # Sentiment analysis implementation
├── logs/                              # Training logs and metrics
│   ├── events.out.tfevents.*          # TensorBoard logs
└── .gitignore                         # Git ignore file
```

## 🔧 Technical Details

### Dependencies
- **Deep Learning**: torch, transformers, datasets
- **Data Processing**: pandas, numpy, scikit-learn
- **NLP**: nltk, textblob, wordcloud
- **Visualization**: matplotlib, seaborn
- **Evaluation**: evaluate

### Hardware Requirements
- **CPU**: Multi-core processor recommended
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster training

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and commit: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Sameer Ramzan**
- GitHub: [@SameerRamzan](https://github.com/SameerRamzan)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection Dataset
- Hugging Face for the Transformers library
- The open-source community for various NLP tools and libraries

## 📚 References

1. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
2. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. SMS Spam Collection Dataset - UCI ML Repository

---

⭐ **Star this repository if you found it helpful!**