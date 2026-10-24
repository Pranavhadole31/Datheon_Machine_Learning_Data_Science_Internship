# Datheon Task 4 — NLP Sentiment Classification

TF-IDF text classification comparing Multinomial Naive Bayes and Logistic Regression.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook sentiment_tfidf_classifier.ipynb
```

The notebook tries the supplied CSV first, then the NLTK movie_reviews corpus as a fallback. The notebook prints which source was loaded.

## Methods
- HTML/URL/punctuation cleanup and lowercasing
- Stopword removal while preserving negation terms
- TF-IDF with `max_features=5000` and unigram/bigram features
- Multinomial Naive Bayes and Logistic Regression
- Accuracy, macro precision/recall/F1, classification reports and confusion matrices
- `predict_sentiment()` helper for custom phrases

Run all notebook cells and report the measured results. Dataset performance may vary by source and split.
