import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from wordcloud import STOPWORDS, WordCloud

# =====================================================================
# 1. DATA LOADING & EXPLORATORY DATA ANALYSIS
# =====================================================================
print("Loading dataset...")
data = pd.read_csv("datasetc.csv")

print("\nDataset Info:")
print(f"Shape of dataset: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values check:")
print(data.isnull().sum())

print("\nDataset Columns:")
print(data.columns)

# Visualizing target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=data, x="Label", order=data["Label"].value_counts().index)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()

# =====================================================================
# 2. WORD CLOUD GENERATION
# =====================================================================
stopwords = STOPWORDS

# Real News Word Cloud (Assuming Class 0 = Real)
if 0 in data["Class"].values:
    real_text = " ".join(data["Text"][data["Class"] == 0].astype(str))
    wcreal = WordCloud(
        background_color="white", stopwords=stopwords, height=800, width=800
    ).generate(real_text)
    wcreal.to_file("wcreal_output.png")

# Fake News Word Cloud (Assuming Class 1 = Fake)
if 1 in data["Class"].values:
    fake_text = " ".join(data["Text"][data["Class"] == 1].astype(str))
    wcfake = WordCloud(
        background_color="black", stopwords=stopwords, height=800, width=800
    ).generate(fake_text)
    wcfake.to_file("wcfake_output.png")

# =====================================================================
# 3. FEATURE EXTRACTION & MODEL TRAINING
# =====================================================================
X_raw = data["Text"].astype(str).values
y = data["Class"].values

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

print("\n" + "=" * 40)
print("RANDOM FOREST MODEL RESULTS")
print("=" * 40)
print(f"Training Accuracy: {accuracy_score(y_train, train_preds) * 100:.2f}%")
print(f"Testing Accuracy:  {accuracy_score(y_test, test_preds) * 100:.2f}%")
print("=" * 40 + "\n")

# =====================================================================
# 4. CLI INTERACTIVE INFERENCE (Optional Test Sample Inspection)
# =====================================================================
try:
    user_idx = int(
        input(f"Enter an index number between 0 and {X_test.shape[0]-1}: ")
    )
    if 0 <= user_idx < X_test.shape[0]:
        sample_input = X_test[user_idx]
        sample_pred = model.predict(sample_input)[0]
        label = "Fake News" if sample_pred == 1 else "Real News"
        print(f"\nPrediction for test index {user_idx}: {label}")
    else:
        print("Index out of range. Skipping CLI inference.")
except ValueError:
    print("Invalid input. Skipping CLI inference.")

# =====================================================================
# 5. STREAMLIT WEB APP INTERFACE
# =====================================================================
# To run as a web app, execute: streamlit run your_script_name.py
st.set_page_config(
    page_title="Fake News Detector", page_icon="📰", layout="centered"
)

st.title("📰 Fake News Detection System")
st.write(
    "Enter a news headline or article excerpt below to verify its authenticity using the trained Random Forest NLP model."
)

user_article = st.text_area(
    "News Article Text",
    height=150,
    placeholder="Paste news content here...",
)

if st.button("Detect News Authenticity"):
    if user_article.strip():
        # Transform input using the already fitted vectorizer
        transformed_input = vectorizer.transform([user_article])
        prediction = model.predict(transformed_input)[0]

        st.markdown("---")
        if prediction == 1:
            st.error("🚨 **Prediction:** The given news is likely **Fake News**.")
        else:
            st.success(
                "✅ **Prediction:** The given news is likely **Real News**."
            )
    else:
        st.warning("Please enter some text before clicking detect.")