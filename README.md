An end-to-end NLP and machine learning project for automated fake news detection, featuring robust preprocessing, exploratory data analysis, and a real-time Streamlit web application.

🚀 Key Features
Exploratory Data Analysis (EDA): Automated dataset inspection, missing-value analysis, class distribution visualization (seaborn), and customized stopword-filtered WordClouds for real vs. fake news.

Text Mining & Feature Engineering: Converts unstructured article text into weighted numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

Robust Model Training: Utilizes a stratified train-test split (75/25 or 70/30) to preserve class balance, training an ensemble Random Forest Classifier optimized for generalization.

Dual Inference Interface: Supports both command-line index-based test sample evaluation and a modern, real-time Streamlit web app for live text classification.

🛠️ Tech Stack
Language: Python

Data Science & ML: Pandas, NumPy, Scikit-Learn (TF-IDF, Random Forest, Metrics)

Text Mining & Visualization: NLTK, WordCloud, Matplotlib, Seaborn

Deployment & UI: Streamlit
