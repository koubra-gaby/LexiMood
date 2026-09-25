# LexiMood 🇳🇬

**LexiMood** is a comprehensive sentiment analysis framework tailored for **Nigerian Pidgin**. It evaluates textual data through a tripartite approach, leveraging distinct linguistic and algorithmic models to accurately classify emotional intent.

This repository implements and compares three different methodologies:
1. **Traditional ML:** TF-IDF feature extraction combined with Logistic Regression.
2. **Deep Learning:** Neural networks built using the Keras framework.
3. **State-of-the-Art NLP:** Fine-tuned Transformer models for advanced language understanding.

---

## 📁 Repository Structure

```text
├── models/                         # Directory designated for saved/trained model weights
│   └── .gitkeep
├── notebooks/                      # Jupyter notebooks for experimentation and analysis
│   ├── starter.ipynb               # Initial data exploration and baseline setup
│   ├── tfidf_logistic_regression.ipynb
│   ├── keras_sentiment_analysis.ipynb
│   └── transformer_sentiment_analysis.ipynb
├── src/                            # Source code for reusable modular components
│   ├── __init__.py
│   ├── preprocessing.py            # Text cleaning and preprocessing pipelines for Nigerian Pidgin
│   └── utils.py                    # Helper functions and evaluation metrics
├── .gitignore
└── requirements.txt                # Project dependencies
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com
cd LexiMood
```

### 2. Install dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the analysis
Navigate to the `notebooks/` directory and open any Jupyter notebook to experiment with the different models.

---

## 🛠️ Tech Stack & Methods

* **Text Processing:** Custom preprocessing scripts designed to handle the specific syntactic and lexical nuances of Nigerian Pidgin.
* **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency).
* **Machine Learning & Deep Learning:** Scikit-learn, Keras, TensorFlow.
* **Transformers:** Hugging Face Transformers library.
