import pandas as pd

# -------------------------------
# 1. Load rule-based output
# -------------------------------
df = pd.read_csv("data/reviews_with_rules.csv")

print("Loaded data from Step 2")

# -------------------------------
# 2. Initialize Trust Score
# -------------------------------
df['trust_score'] = 100

# -------------------------------
# 3. Apply penalties
# -------------------------------

# Rule 1 penalty
df.loc[df['rule_short_high_rating'] == 1, 'trust_score'] -= 25

# Rule 2 penalty
df.loc[df['rule_many_reviews'] == 1, 'trust_score'] -= 35

# Rule 3 penalty
df.loc[df['rule_duplicate_text'] == 1, 'trust_score'] -= 30

# Ensure score stays between 0 and 100
df['trust_score'] = df['trust_score'].clip(0, 100)

# -------------------------------
# 4. Trust Category
# -------------------------------
def trust_category(score):
    if score >= 80:
        return "Highly Trusted"
    elif score >= 50:
        return "Moderately Trusted"
    else:
        return "Low Trust / Fake"

df['trust_category'] = df['trust_score'].apply(trust_category)

# -------------------------------
# 5. Summary statistics
# -------------------------------
print("\nTrust category distribution:")
print(df['trust_category'].value_counts())

print("\nSample low-trust reviews:")
print(df[df['trust_category'] == "Low Trust / Fake"][
    ['UserId', 'Score', 'review_length', 'trust_score']
].head())

# -------------------------------
# 6. Save final output
# -------------------------------
df.to_csv("data/final_reviews_with_trust_score.csv", index=False)
print("\nTrust score system completed. Final output saved.")
