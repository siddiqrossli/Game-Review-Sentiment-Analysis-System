import pandas as pd
import pickle
import re
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "steam_reviews.csv")

# ================= LOAD DATA =================
print("Loading dataset...")
df = pd.read_csv(csv_path)

# Keep only required columns
df = df[["review_text", "review_score"]]
df = df.dropna()

print("\nLabel distribution BEFORE mapping:")
print(df["review_score"].value_counts())

# ================= NORMALIZE LABELS =================
# Convert all label formats to: 1 = positive, 0 = negative

def normalize_label(x):
    if isinstance(x, str):
        x = x.lower()
        if x in ["positive", "pos", "1", "true", "yes"]:
            return 1
        else:
            return 0
    if x == 1:
        return 1
    if x == 0:
        return 0
    if x == -1:
        return 0
    if x is True:
        return 1
    if x is False:
        return 0
    return 0

df["label"] = df["review_score"].apply(normalize_label)

print("\nLabel distribution AFTER mapping:")
print(df["label"].value_counts())

# ================= BALANCE DATASET =================
print("\nBalancing dataset...")

pos = df[df["label"] == 1]
neg = df[df["label"] == 0]

min_size = min(len(pos), len(neg), 50000)

pos = pos.sample(min_size, random_state=42)
neg = neg.sample(min_size, random_state=42)

df = pd.concat([pos, neg]).sample(frac=1, random_state=42)

texts = df["review_text"].astype(str)
labels = df["label"].values

print("Positive:", sum(labels == 1))
print("Negative:", sum(labels == 0))
print("Total samples:", len(df))

# ================= PREPROCESS =================
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    return " ".join(
        stemmer.stem(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    )

print("\nPreprocessing text...")
texts_clean = texts.apply(preprocess)

# ================= TF-IDF WITH BIGRAMS =================
print("Vectorizing...")
vectorizer = TfidfVectorizer(
    max_features=12000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(texts_clean)
y = labels

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= TRAIN =================
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained")

# ================= EVALUATE =================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
plt.close()

# ================= SAVE =================
model_data = {
    "model": model,
    "vectorizer": vectorizer,
    "stop_words": stop_words,
    "stemmer": stemmer
}

with open(os.path.join(BASE_DIR, "game_sentiment_model.pkl"), "wb") as f:
    pickle.dump(model_data, f)

print("\nSaved: game_sentiment_model.pkl")
