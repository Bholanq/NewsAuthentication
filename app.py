from flask import Flask, render_template, request
import joblib
import shap
import numpy as np
from scipy.sparse import csr_matrix
from newsapi import NewsApiClient
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from dotenv import load_dotenv
import os
load_dotenv()  # ✅ Load .env file
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # ✅ Read from environment

# Set Matplotlib to use the 'Agg' backend (non-GUI)
matplotlib.use('Agg')  # Must be set before importing pyplot

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# News API Client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# SHAP Explainer with increased max_evals
background_text = ["background text"]
background_matrix = vectorizer.transform(background_text).toarray()
explainer = shap.Explainer(lambda x: model.predict_proba(x), background_matrix, max_evals=1000)  # Increase max_evals

# Threshold for filtering low-importance words
SHAP_THRESHOLD = 0.01  # Only include words with SHAP values >= 0.05


@app.route("/")
def home():
    """Render the home page with news headlines."""
    articles = newsapi.get_top_headlines(language="en", country="us")["articles"]
    return render_template("index.html", articles=articles)


@app.route("/check", methods=["POST"])
def check_news():
    """Predict if the news is fake or real and provide explainability."""
    text = request.form["news_text"]

    # Transform input text
    input_text = vectorizer.transform([text])
    input_array = input_text.toarray()  # Convert sparse matrix to dense array

    # Get model prediction
    prediction = model.predict(input_text)
    result = "Fake News" if prediction[0] == 1 else "Real News"

    # Get SHAP values to explain the prediction
    shap_values = explainer(input_array)
    feature_names = vectorizer.get_feature_names_out()

    # Filter words based on SHAP threshold
    shap_values_filtered = []
    input_array_filtered = []
    feature_names_filtered = []

    for i in range(len(feature_names)):
        if abs(shap_values.values[0, i, 1]) >= SHAP_THRESHOLD:  # Check if SHAP value meets the threshold
            shap_values_filtered.append(shap_values.values[0, i, 1])
            input_array_filtered.append(input_array[0, i])
            feature_names_filtered.append(feature_names[i])

    # Convert filtered data back to arrays
    shap_values_filtered = np.array(shap_values_filtered).reshape(1, -1)
    input_array_filtered = np.array(input_array_filtered).reshape(1, -1)
    feature_names_filtered = np.array(feature_names_filtered)

    # Generate beeswarm plot with filtered data
    plt.figure()
    shap.summary_plot(shap_values_filtered, input_array_filtered, feature_names=feature_names_filtered, show=False)
    plt.title("SHAP Beeswarm Plot for Prediction")
    plt.tight_layout()

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Close the plot to free memory
    plt.close()

    # Generate a summary of the beeswarm plot
    summary = generate_beeswarm_summary(shap_values_filtered, feature_names_filtered, result)

    return render_template("result.html", text=text, result=result, plot_url=plot_url, summary=summary)


def generate_beeswarm_summary(shap_values, feature_names, prediction):
    """Generate a summary of the beeswarm plot."""
    # Get the indices of the top 5 most influential words
    top_indices = np.argsort(np.abs(shap_values[0]))[-5:][::-1]
    top_words = feature_names[top_indices]
    top_shap_values = shap_values[0][top_indices]

    # Determine the impact direction (positive or negative)
    impact_direction = []
    for value in top_shap_values:
        if value > 0:
            impact_direction.append("supports 'Fake News'")
        else:
            impact_direction.append("supports 'Real News'")

    # Create the summary
    summary = f"The following words had the most influence on the prediction:\n"
    for word, direction in zip(top_words, impact_direction):
        summary += f"- '{word}' ({direction})\n"

    return summary


if __name__ == "__main__":
    app.run(debug=True)