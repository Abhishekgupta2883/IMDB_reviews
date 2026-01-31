# IMDB Review Sentiment Checker

A web app that classifies movie reviews as **positive** or **negative** using NLP and a Logistic Regression model. Built with Flask and a simple, styled frontend.

## What it does

- Paste a review (or any text) and get an instant **positive** or **negative** label.
- Text is cleaned (lowercasing, tokenization, stopword removal, stemming) before prediction.
- Prediction uses a pre-trained **TF-IDF** vectorizer and **Logistic Regression** model (`.pkl` files in the repo).

## Tech stack

- **Backend:** Python, Flask  
- **NLP:** NLTK (tokenization, stopwords, Porter stemmer), regex  
- **Model:** scikit-learn (TF-IDF + Logistic Regression), pickle for loading

## Setup

1. **Clone or download** this repo.

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (run once; needs internet):
   ```bash
   python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
   ```

5. **Run the app:**
   ```bash
   python app.py
   ```

6. Open **http://127.0.0.1:5000** in your browser.

## Project structure

```
IMDB reviews/
├── app.py                 # Flask app, text cleaning, model loading, /predict
├── lr_model.pkl           # Trained Logistic Regression model
├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
├── requirements.txt
├── README.md
├── static/
│   └── style.css          # Styling for the web UI
└── templates/
    ├── base.html          # Base layout
    └── index.html         # Main page and form
```

## API

- **POST `/predict`**  
  - Body: `{"review": "Your review text here"}` (JSON) or form field `review`.  
  - Response: `{"sentiment": "positive"|"negative", "label": 0|1, "review": "..."}`.

## License

Use and modify as you like.
