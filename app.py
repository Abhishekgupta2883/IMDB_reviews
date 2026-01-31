import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pickle

# Download NLTK data if not present (run once)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = re.sub(r"<br />", "", text)  # Remove <br /> tags
    text = word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Load TF-IDF vectorizer and Logistic Regression model
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vec = pickle.load(f)

with open("lr_model.pkl", "rb") as f:
    lr = pickle.load(f)


def predict_sentiment(review_text):
    """Predict whether a review is positive (1) or negative (0)."""
    cleaned = transform_text(review_text)
    X = tfidf_vec.transform([cleaned])
    pred = lr.predict(X)[0]
    return "positive" if pred == 1 else "negative", int(pred)


# --- Flask app ---
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(force=True, silent=True) or request.form
    review = (data.get("review") or "").strip()
    if not review:
        return jsonify({"error": "Please enter a review."}), 400
    try:
        sentiment, label = predict_sentiment(review)
        return jsonify({"sentiment": sentiment, "label": label, "review": review})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
