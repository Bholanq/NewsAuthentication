import os
import zipfile
import kaggle
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Set Kaggle dataset directory
dataset = "clmentbisaillon/fake-and-real-news-dataset"

# Download dataset using Kaggle API
if not os.path.exists("fake-and-real-news-dataset.zip"):
    os.system(f"kaggle datasets download -d {dataset}")

# Extract dataset if not already extracted
if not os.path.exists("Fake.csv") or not os.path.exists("True.csv"):
    with zipfile.ZipFile("fake-and-real-news-dataset.zip", 'r') as zip_ref:
        zip_ref.extractall(".")

# Load Fake and Real news datasets
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels: 1 for Fake, 0 for Real
df_fake["label"] = 1
df_real["label"] = 0

# Merge both datasets
df = pd.concat([df_fake, df_real])

# Select necessary columns
df = df[['text', 'label']]
df.dropna(inplace=True)

# Define Features (X) and Labels (y)
X, y = df['text'], df['label']

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved!")
