import os
import joblib
import requests
import pandas as pd
from newsapi import NewsApiClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from dotenv import load_dotenv
load_dotenv()  # ✅ Load .env file
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # ✅ Read from environment
# Load existing model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Initialize NewsAPI client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Define CSV file to store new data
NEW_DATA_FILE = "realtime_news_data.csv"
TRAIN_THRESHOLD = 100  # Retrain when 50+ new articles are collected
CONFIDENCE_THRESHOLD = 0.9  # Only use predictions with >90% confidence


def fetch_latest_news(language="en", num_articles=100):
    response = newsapi.get_everything(
        q="pakistan",  # Broaden search keywords
        language=language,
        sort_by="publishedAt",  # Get the latest articles first
        page_size=num_articles  # Increase the number of articles
    )

    # Debugging: Check the total results
    print(f"🔍 Total articles found: {response.get('totalResults', 0)}")

    articles = response.get("articles", [])

    return [(article["title"] + " " + (article["description"] or "")) for article in articles if "title" in article]


def classify_news(news_list):
    """Classifies news articles, filters by confidence, and prints accuracy."""
    transformed_news = vectorizer.transform(news_list)
    probabilities = model.predict_proba(transformed_news)  # Get probability scores
    predictions = model.predict(transformed_news)

    # Fake (1) or Real (0) classification with confidence filtering
    high_confidence_data = []
    for news, pred, prob in zip(news_list, predictions, probabilities):
        confidence = max(prob)
        if confidence >= CONFIDENCE_THRESHOLD:
            high_confidence_data.append({"text": news, "label": pred, "confidence": confidence})

    return high_confidence_data  # Only return high-confidence classifications



def save_new_data(news_list):
    """Saves automatically classified high-confidence news to CSV file."""
    df_new = pd.DataFrame(news_list, columns=["text", "label", "confidence"])

    if os.path.exists(NEW_DATA_FILE):
        df_new.to_csv(NEW_DATA_FILE, mode="a", header=False, index=False)
    else:
        df_new.to_csv(NEW_DATA_FILE, mode="w", header=True, index=False)

    print(f"✅ Saved {len(df_new)} new articles to {NEW_DATA_FILE}")  # Debugging



def should_retrain():
    """Check if new data has reached the retraining threshold."""
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        return len(df_new) >= TRAIN_THRESHOLD
    return False


def retrain_model():
    """Retrains the model with old + new high-confidence data and displays accuracy."""
    print("\n🔄 Retraining model with new data...")

    # Load Kaggle dataset
    df_fake = pd.read_csv("Fake.csv")
    df_real = pd.read_csv("True.csv")
    df_fake["label"] = 1
    df_real["label"] = 0
    df = pd.concat([df_fake, df_real])

    # Load new high-confidence labeled data
    if os.path.exists(NEW_DATA_FILE):
        df_new = pd.read_csv(NEW_DATA_FILE)
        df_new = df_new.drop(columns=["confidence"])  # Remove confidence column
        df = pd.concat([df, df_new])

    df = df[['text', 'label']]
    df.dropna(inplace=True)

    # Convert text into numerical features
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train new model
    new_model = MultinomialNB()
    new_model.fit(X_train, y_train)

    # Save the updated model and vectorizer
    joblib.dump(new_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    y_pred = new_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n✅ Model updated! New Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:\n", report)


if __name__ == "__main__":
    print("Fetching live news...")

    # Remove 'query' parameter, only fetching top headlines
    latest_news = fetch_latest_news(num_articles=100)

    if not latest_news:
        print("No news articles found.")
    else:
        print("\nClassifying news articles...\n")
        high_confidence_news = classify_news(latest_news)

        if high_confidence_news:
            print(f"✅ {len(high_confidence_news)} high-confidence articles saved for retraining.")
            save_new_data(high_confidence_news)

        # Automatically retrain if enough high-confidence data is collected
        if should_retrain():
            retrain_model()
            os.remove(NEW_DATA_FILE)  # Clear the new data file after retraining
            print("🔄 Training data reset. Ready for new samples!")

        print("\nProcess completed.")