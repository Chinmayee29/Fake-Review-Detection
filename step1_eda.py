import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Load the dataset
# -------------------------------
# Change file name if needed
df = pd.read_csv("data/Reviews.csv")

print("Dataset loaded successfully!")
print("Shape of dataset:", df.shape)

# -------------------------------
# 2. View first few records
# -------------------------------
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# 3. Check column names
# -------------------------------
print("\nColumn names:")
print(df.columns)

# -------------------------------
# 4. Select only required columns
# -------------------------------
df = df[['UserId', 'Score', 'Text', 'Time']]

print("\nSelected columns:")
print(df.head())

# -------------------------------
# 5. Check missing values
# -------------------------------
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing review text
df = df.dropna(subset=['Text'])

# -------------------------------
# 6. Create new useful features
# -------------------------------

# Review length (number of characters)
df['review_length'] = df['Text'].apply(len)

# Convert time to datetime
df['review_time'] = pd.to_datetime(df['Time'], unit='s')

print("\nNew features added:")
print(df[['Score', 'review_length', 'review_time']].head())

# -------------------------------
# 7. Basic statistics
# -------------------------------
print("\nRating distribution:")
print(df['Score'].value_counts())

print("\nReview length stats:")
print(df['review_length'].describe())

# -------------------------------
# 8. Visualization - Rating Distribution
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Score', data=df)
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 9. Visualization - Review Length vs Rating
# -------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(x='Score', y='review_length', data=df)
plt.title("Review Length vs Rating")
plt.xlabel("Rating")
plt.ylabel("Review Length")
plt.show()

# -------------------------------
# 10. Reviews per User (Behavior Analysis)
# -------------------------------
reviews_per_user = df['UserId'].value_counts()

print("\nTop 5 most active reviewers:")
print(reviews_per_user.head())

plt.figure(figsize=(6,4))
sns.histplot(reviews_per_user, bins=30)
plt.title("Reviews per User Distribution")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Users")
plt.show()
