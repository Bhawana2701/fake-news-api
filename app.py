
from flask import Flask, request, jsonify
import joblib

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return "Fake News Detection API Running"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get JSON request
        data = request.get_json()

        # Extract text
        text = data['text']

        # Convert text to TF-IDF
        text_tfidf = vectorizer.transform([text])

        # Predict
        prediction = model.predict(text_tfidf)[0]

        # Confidence
        probabilities = model.predict_proba(text_tfidf)[0]
        confidence = max(probabilities)

        # Label mapping
        if prediction == 0:
            label = "Fake News"
        else:
            label = "Real News"

        # Return response
        return jsonify({
            "prediction": label,
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# Run app
if __name__ == '__main__':
    app.run()
