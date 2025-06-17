import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline

nltk.download('stopwords')

# Global stemmer and stopword list
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalpha() or char.isspace())
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def visualize_sentiment_distribution(data):
    sns.countplot(data['sentiment'])
    plt.title('Distribution of Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

def print_data_info(data):
    print("First few rows of the dataset:")
    print(data.head())
    print("\nDataset Info:")
    print(data.info())
    print("\nDataset Description:")
    print(data.describe())

def build_model():
    return make_pipeline(TfidfVectorizer(max_features=5000), LogisticRegression())

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

def predict_new_reviews(model, reviews):
    print("\nPredictions for new reviews:")
    for review in reviews:
        cleaned = preprocess_text(review)
        prediction = model.predict([cleaned])[0]
        print(f"Review: {review}")
        print(f"Predicted Sentiment: {'✅ Positive' if prediction == 1 else '❌ Negative'}\n")

if __name__ == "__main__":
    # ✅ Your original read_csv line remains unchanged
    data = pd.read_csv(r'C:\Users\jthul\Dropbox\My PC (LAPTOP-1NRT3JF9)\Desktop\projects\Sentiment Analysis of Movie Reviews\IMDB_Dataset.csv')

    print_data_info(data)
    visualize_sentiment_distribution(data)

    print("Cleaning and preprocessing text...")
    data['cleaned_review'] = data['review'].apply(preprocess_text)
    data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_review'], data['sentiment'], test_size=0.2, random_state=42)

    print("Training model...")
    model = build_model()
    model.fit(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    new_reviews = [
        "I absolutely loved this movie! The storyline was fantastic and the acting was superb.",
        "This was the worst film I have ever seen. It was a total waste of time."
    ]
    predict_new_reviews(model, new_reviews)
