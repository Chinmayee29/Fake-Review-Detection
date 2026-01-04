import pandas as pd

# -------------------------------
# 1. Load cleaned dataset
# -------------------------------
df = pd.read_csv("data/Reviews.csv")

# Keep only required columns
df = df[['UserId', 'Score', 'Text', 'Time']]
df = df.dropna(subset=['Text'])

# Create review length feature
df['review_length'] = df['Text'].apply(len)

print("Dataset loaded for rule-based detection")

# -------------------------------
# 2. Rule 1: Short review + high rating
# -------------------------------
def rule_short_high_rating(row):
    if row['Score'] >= 4 and row['review_length'] < 30:
        return 1
    return 0

df['rule_short_high_rating'] = df.apply(rule_short_high_rating, axis=1)

# -------------------------------
# 3. Rule 2: Too many reviews by same user
# -------------------------------
review_count = df['UserId'].value_counts()

# Threshold: more than 50 reviews = suspicious
df['user_review_count'] = df['UserId'].map(review_count)
df['rule_many_reviews'] = df['user_review_count'].apply(
    lambda x: 1 if x > 50 else 0
)

# -------------------------------
# 4. Rule 3: Duplicate review text
# -------------------------------
text_counts = df['Text'].value_counts()
df['duplicate_text_count'] = df['Text'].map(text_counts)
df['rule_duplicate_text'] = df['duplicate_text_count'].apply(
    lambda x: 1 if x > 1 else 0
)

# -------------------------------
# 5. Combine rules to get fake score
# -------------------------------
df['fake_score'] = (
    df['rule_short_high_rating'] +
    df['rule_many_reviews'] +
    df['rule_duplicate_text']
)

# -------------------------------
# 6. Final classification
# -------------------------------
def classify_review(score):
    if score >= 2:
        return "Suspicious"
    else:
        return "Genuine"

df['review_label'] = df['fake_score'].apply(classify_review)

# -------------------------------
# 7. Results summary
# -------------------------------
print("\nReview classification count:")
print(df['review_label'].value_counts())

print("\nSample suspicious reviews:")
print(df[df['review_label'] == "Suspicious"][[
    'UserId', 'Score', 'review_length', 'fake_score'
]].head())

# -------------------------------
# 8. Save output for next step
# -------------------------------
df.to_csv("data/reviews_with_rules.csv", index=False)
print("\nRule-based detection completed. Output saved.")
