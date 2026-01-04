import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -------------------------------
# 1. Load data from Step 2
# -------------------------------
df = pd.read_csv("data/reviews_with_rules.csv")

print("Dataset loaded for ML")

# -------------------------------
# 2. Prepare labels
# -------------------------------
# Convert labels to binary
# Genuine -> 0, Suspicious -> 1
df['label'] = df['review_label'].map({
    'Genuine': 0,
    'Suspicious': 1
})

# Drop rows with missing text
df = df.dropna(subset=['Text'])

X = df['Text']
y = df['label']

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train-test split completed")

# -------------------------------
# 4. Text vectorization (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Text vectorization completed")

# -------------------------------
# 5. Train ML model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

print("Model training completed")

# -------------------------------
# 6. Evaluate model
# -------------------------------
y_pred = model.predict(X_test_vec)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Compare with rule-based system
# -------------------------------
print("\nML Enhancement completed successfully")
