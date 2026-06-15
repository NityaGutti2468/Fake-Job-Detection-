from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load saved model and TF-IDF vectorizer
model = joblib.load("model_fake_job.pkl")
vectorizer = joblib.load("vectorizer_fake_job.pkl")


def clean_text(text):
    """Clean user input text before prediction."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    confidence_text = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("job_description", "")

        if user_input.strip() == "":
            prediction_text = "Please enter a job description."
        else:
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            prediction = model.predict(vectorized_input)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(vectorized_input)[0]
                confidence = max(probability) * 100
                confidence_text = f"Confidence: {confidence:.2f}%"

            if prediction == 1:
                prediction_text = "Fake Job Posting"
            else:
                prediction_text = "Real Job Posting"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        confidence_text=confidence_text,
        user_input=user_input,
    )


if __name__ == "__main__":
    app.run(debug=True)
