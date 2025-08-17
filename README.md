# 📱 SMS Spam Filter

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

A sophisticated machine learning-powered SMS spam detection system with an intuitive web interface. Built with multiple classification algorithms and advanced text preprocessing techniques for accurate spam detection.

## 🎯 Features

- **🤖 Multiple ML Models**: Naive Bayes, Logistic Regression, Random Forest
- **🔧 Advanced Text Preprocessing**: URL removal, phone number cleaning, stemming, stopword removal
- **📊 TF-IDF Vectorization**: Professional-grade text feature extraction
- **🎨 Interactive Web UI**: Clean, responsive Streamlit interface
- **💾 Model Persistence**: Automatic model saving and loading
- **📈 Confidence Scores**: Prediction confidence levels
- **🔄 Auto-Training**: Fallback training with sample data
- **📋 Example Messages**: Pre-loaded test cases for quick testing

## 🚀 Live Demo

![SMS Spam Filter Demo](https://github.com/Indra1806/sms-spam-filter/blob/main/demo/demo_image.png)

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## 🛠 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Clone Repository

```bash
git clone https://github.com/Indra1806/sms-spam-filter.git
cd sms-spam-filter
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Alternative: Install with conda

```bash
conda env create -f environment.yml
conda activate sms-spam-filter
```

## 🚀 Quick Start

### Run the Application

```bash
streamlit run app.py
```

The application will start at `http://localhost:8501`

### First Time Setup

On first run, the application will:
1. Download required NLTK data
2. Train models with sample data
3. Save models to `models/` directory
4. Launch the web interface

## 💡 Usage

### Web Interface

1. **Enter Message**: Type or paste your SMS message
2. **Select Model**: Choose from Naive Bayes, Logistic Regression, or Random Forest
3. **Analyze**: Click "Analyze Message" to get results
4. **View Results**: See spam/ham classification with confidence scores

### Example Messages

Try the pre-loaded examples:

**Spam Examples:**
- "WINNER!! You have won £1000! Call now!"
- "FREE entry to win £500! Text WIN to 12345"

**Ham Examples:**
- "Hey, are you free for lunch today?"
- "Meeting rescheduled to 3 PM tomorrow"

### Programmatic Usage

```python
from app import SMSSpamFilter

# Initialize filter
spam_filter = SMSSpamFilter()

# Train models (if not already trained)
spam_filter.train_models()

# Predict message
message = "Congratulations! You've won £1000!"
result, confidence = spam_filter.predict(message, 'Naive Bayes')

print(f"Result: {result}")
print(f"Confidence: {confidence:.2%}")
```

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 96.4% | 0.95 | 0.97 | 0.96 |
| Logistic Regression | 95.8% | 0.94 | 0.96 | 0.95 |
| Random Forest | 94.2% | 0.93 | 0.95 | 0.94 |

*Performance metrics on test dataset with 80/20 train-test split*

## 🔧 API Reference

### SMSSpamFilter Class

#### Methods

**`__init__()`**
Initialize the spam filter with default settings.

**`train_models(df=None)`**
Train all classification models.
- `df`: Optional pandas DataFrame with 'message' and 'label' columns
- Returns: Boolean indicating success

**`predict(message, model_name='Naive Bayes')`**
Classify a message as spam or ham.
- `message`: String message to classify
- `model_name`: Model to use for prediction
- Returns: Tuple of (result, confidence)

**`load_models()`**
Load pre-trained models from disk.
- Returns: Boolean indicating success

**`save_models()`**
Save trained models to disk.

### SMSPreprocessor Class

#### Methods

**`clean_text(text)`**
Clean and preprocess text message.
- `text`: Raw message text
- Returns: Cleaned and processed text

## 📁 Project Structure

```
sms-spam-filter/
│
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment file
├── README.md             # Project documentation
├── LICENSE               # MIT License
│
├── models/               # Saved models directory
│   ├── vectorizer.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── data/                 # Dataset directory (optional)
│   └── spam.csv
│
├── tests/                # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_models.py
│
└── docs/                 # Documentation
    ├── api.md
    └── deployment.md
```

## 📊 Dataset

The model can work with any SMS dataset containing:
- **message**: Text content of SMS
- **label**: Classification (spam/ham)

### Sample Data Format

```csv
message,label
"Free entry in 2 a wkly comp to win FA Cup final tkts",spam
"Hey, how are you doing today?",ham
"WINNER!! As a valued network customer you have been selected",spam
```

### Recommended Datasets

- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- [SMS Spam Detection Dataset](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)

## 🚀 Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Docker Deployment

```bash
# Build image
docker build -t sms-spam-filter .

# Run container
docker run -p 8501:8501 sms-spam-filter
```

### Cloud Deployment

#### Streamlit Cloud
1. Connect your GitHub repository
2. Set main file path: `app.py`
3. Deploy automatically

#### Heroku
```bash
heroku create your-app-name
git push heroku main
```

#### AWS/GCP/Azure
See detailed deployment guides in `docs/deployment.md`

## 🧪 Testing

Run tests with pytest:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Fork the repository
git clone https://github.com/yourusername/sms-spam-filter.git
cd sms-spam-filter

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

### Submitting Changes

1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Commit changes: `git commit -m 'Add amazing feature'`
3. Push to branch: `git push origin feature/amazing-feature`
4. Open a Pull Request

## 📈 Roadmap

- [ ] **Deep Learning Models**: LSTM, BERT integration
- [ ] **Multi-language Support**: Support for non-English messages
- [ ] **Real-time Training**: Online learning capabilities
- [ ] **API Endpoints**: REST API for external integrations
- [ ] **Mobile App**: React Native mobile application
- [ ] **Batch Processing**: Handle multiple messages
- [ ] **Advanced Analytics**: Detailed reporting dashboard

## 📝 Changelog

### v1.2.0 (Latest)
- ✨ Added Random Forest classifier
- 🐛 Fixed pickle loading errors
- 🎨 Improved UI/UX design
- 📊 Added model performance metrics

### v1.1.0
- ✨ Added confidence scores
- 🔧 Improved text preprocessing
- 📱 Enhanced mobile responsiveness

### v1.0.0
- 🎉 Initial release
- 🤖 Basic spam detection
- 🎨 Streamlit web interface

## ⚠️ Known Issues

- NLTK downloads may fail on some systems (fallback implemented)
- Large datasets may require increased memory allocation
- Model retraining can be time-intensive

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [Streamlit](https://streamlit.io/) for the web framework
- [NLTK](https://www.nltk.org/) for natural language processing
- [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) for training data

## 📞 Contact

**Indra** - [@Indra1806](https://github.com/Indra1806)

**Project Link**: [https://github.com/Indra1806/sms-spam-filter](https://github.com/Indra1806/sms-spam-filter)

## 📚 Additional Resources

- [Machine Learning for Text Classification](https://developers.google.com/machine-learning/guides/text-classification)
- [Streamlit Documentation](https://docs.streamlit.io)
- [SMS Spam Detection Research Papers](https://scholar.google.com/scholar?q=sms+spam+detection)

---

<div align="center">

**If you found this project helpful, please consider giving it a ⭐!**

Made with ❤️ by [Indra](https://github.com/Indra1806)

</div>
