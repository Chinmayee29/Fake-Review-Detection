# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression

# # ----------------------------------
# # Page Configuration
# # ----------------------------------
# st.set_page_config(page_title="Fake Review Detection", layout="centered")

# st.title("🔐 Fake Review Detection & Trust Scoring System")
# st.write("Detect suspicious reviews and view trust scores using ML + rules")

# # ----------------------------------
# # Load Dataset
# # ----------------------------------
# @st.cache_data
# def load_data():
#     return pd.read_csv("data/reviews_with_rules.csv")

# df = load_data()

# # ----------------------------------
# # Train ML Model
# # ----------------------------------
# @st.cache_resource
# def train_model():
#     vectorizer = TfidfVectorizer(
#         stop_words="english",
#         max_features=5000
#     )

#     X = vectorizer.fit_transform(df["Text"])
#     y = df["review_label"].map({"Genuine": 0, "Suspicious": 1})

#     # Balanced model improves fake review recall
#     model = LogisticRegression(
#         max_iter=1000,
#         class_weight="balanced"
#     )

#     model.fit(X, y)
#     return vectorizer, model

# vectorizer, model = train_model()

# # ----------------------------------
# # User Input
# # ----------------------------------
# st.subheader("📝 Enter a Review")
# review = st.text_area("Paste a product review here")

# if st.button("Analyze"):
#     if review.strip() == "":
#         st.warning("Please enter a review")
#     else:
#         # ML prediction
#         vec = vectorizer.transform([review])
#         pred = model.predict(vec)[0]
#         prob = model.predict_proba(vec)[0][pred]

#         # ----------------------------------
#         # Rule-Based Penalties (Hybrid Logic)
#         # ----------------------------------
#         penalty = 0

#         # Rule 1: Very short review
#         if len(review) < 100:
#             penalty += 20

#         # Rule 2: Excessive exclamation marks
#         if review.count("!") > 3:
#             penalty += 20

#         # Rule 3: Overly positive keyword spam
#         positive_words = [
#             "amazing", "best", "excellent",
#             "perfect", "worth it", "highly recommended"
#         ]
#         positive_count = sum(word in review.lower() for word in positive_words)

#         if positive_count >= 3:
#             penalty += 30

#         # Final Trust Score
#         trust_score = max(0, int(prob * 100) - penalty)

#         # ----------------------------------
#         # Output
#         # ----------------------------------
#         if trust_score < 50:
#             st.error("🚨 Suspicious Review Detected")
#         else:
#             st.success("✅ Genuine Review")

#         st.write(f"**Trust Score:** {trust_score}/100")

#         # Explainability (optional but powerful)
#         with st.expander("🔍 Explanation"):
#             st.write(f"Review Length: {len(review)} characters")
#             st.write(f"Exclamation Marks: {review.count('!')}")
#             st.write(f"Positive Keywords Used: {positive_count}")
#             st.write(f"ML Confidence: {int(prob * 100)}%")
#             st.write(f"Penalty Applied: {penalty}")

# # ----------------------------------
# # Dataset Preview
# # ----------------------------------
# st.subheader("📊 Sample Dataset")
# st.dataframe(df.head())
# st.subheader("📈 Review Analytics")

# # Review label distribution
# label_counts = df["review_label"].value_counts()

# st.write("### Review Type Distribution")
# st.bar_chart(label_counts)

# # Review length distribution
# df["review_length"] = df["Text"].apply(len)

# st.write("### Review Length Distribution")
# st.line_chart(df["review_length"].head(50))





import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Fake Review Detection",
    layout="centered"
)

st.title("🔐 Fake Review Detection & Trust Scoring System")
st.write("Detect suspicious reviews and view trust scores using ML + rules")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    return pd.read_csv("data/sample_reviews.csv")

df = load_data()

# ---------- Train Model ----------
@st.cache_resource
def train_model(data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000
    )
    X = vectorizer.fit_transform(data["Text"])
    y = data["review_label"].map({"Genuine": 0, "Suspicious": 1})

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return vectorizer, model

vectorizer, model = train_model(df)

# ---------- UI ----------
st.subheader("📝 Enter a Review")
review = st.text_area("Paste a product review here")

if st.button("Analyze"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        vec = vectorizer.transform([review])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0][prediction]
        trust_score = int(probability * 100)

        if prediction == 1:
            st.error("🚨 Suspicious Review")
        else:
            st.success("✅ Genuine Review")

        st.write(f"### 🔐 Trust Score: **{trust_score}/100**")

# ---------- Dataset Preview ----------
st.subheader("📊 Sample Dataset Used for Training")
st.dataframe(df)
