import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the original dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels: 1 for Fake, 0 for Real
df_fake["label"] = 1
df_real["label"] = 0

# Merge datasets
df = pd.concat([df_fake, df_real])

# Select necessary columns
df = df[['text', 'label']]
df.dropna(inplace=True)

# *Check if real-time news data exists and is valid*
realtime_file = "realtime_news_data.csv"
if os.path.exists(realtime_file):
    try:
        df_realtime = pd.read_csv(realtime_file)

        # Ensure "label" column exists and is numeric
        if "label" in df_realtime.columns:
            df_realtime["label"] = pd.to_numeric(df_realtime["label"], errors="coerce")
            df_realtime.dropna(subset=["label"], inplace=True)  # Remove invalid labels

            if not df_realtime.empty:
                df = pd.concat([df, df_realtime])  # Merge datasets
                print("✅ Real-time news data added for evaluation!")
            else:
                print("⚠ Realtime news file exists but contains no valid labeled data.")
        else:
            print("⚠ Realtime news file exists but does not contain a 'label' column.")

    except Exception as e:
        print(f"⚠ Error reading real-time news data: {e}")
else:
    print("⚠ No real-time news data found. Evaluating only original dataset.")

# Define Features (X) and Labels (y)
X, y = df['text'], df['label']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the saved vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Transform ONLY the test set (not the entire dataset)
X_test_transformed = vectorizer.transform(X_test)

# Load the trained model
model = joblib.load("model.pkl")

# Predict on test data
y_pred = model.predict(X_test_transformed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the evaluation results
print(f"🟢 Model Accuracy (Including Real-Time News): {accuracy:.4f}")
print("\n📊 Classification Report:\n", report)