# 🛡️ Fake News Detection System

A multi-page Streamlit dashboard that classifies news text as **REAL** or **FAKE** using classic NLP
(regex cleaning, stopword removal, stemming, n-gram TF-IDF) and compares Naive Bayes, Logistic Regression
and a Passive-Aggressive classifier. The best model by F1-score is used for predictions.

## Pages
- **Home**: overview of the pipeline
- **News Detection**: paste a headline or article, get a label and confidence
- **Analytics & EDA**: model benchmark table, metric chart, confusion matrix
- **Dataset Explorer**: profile the training data or upload your own CSV

## Quick start
```bash
git clone <your-repo-url>
cd fake-news-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```
The app opens at http://localhost:8501.

## Dataset
`train.py` looks for data in this order:
1. `dataset/news_dataset.csv` with columns `text`, `label` (1 = fake, 0 = real) and optionally `subject`
2. `dataset/Fake.csv` + `dataset/True.csv` (the Kaggle *Fake and Real News* dataset)
3. Otherwise it generates a **synthetic demo dataset**. This only verifies the pipeline works;
   its accuracy numbers are meaningless. Use real data for anything serious.

## Project structure
```
app.py                     Streamlit dashboard
train.py                   Training + evaluation script
utils/text_preprocessor.py Shared text cleaning
models/                    Generated artifacts (model, vectorizer, metrics)
dataset/                   Training data
```

## Deploying on Streamlit Community Cloud
Train locally on real data, commit the generated `models/` folder, push to GitHub, then create the app
at https://share.streamlit.io pointing to `app.py`.

## Disclaimer
Predictions reflect linguistic patterns in the training data and are probabilistic. They do not replace
independent fact-checking.
