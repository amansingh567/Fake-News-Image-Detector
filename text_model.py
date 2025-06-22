import pandas as pd
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
import re
import nltk
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('punkt') # used by word_tokenize
nltk.download('wordnet') # used by WordNetLemmatizer
nltk.download('stopwords') 

def preprocess_text(text, lemmatizer, stpwrds):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stpwrds]
    return ' '.join(cleaned)

def train_text_model():
    dataset = pd.read_csv("datasets/train.csv")
    dataset = dataset.dropna()

    # Drop unnecessary columns
    dataset = dataset.drop(columns=["author", "title", "id"])

    y = dataset['label']
    X = dataset['text']  # X is now a Series

    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()
    stpwrds = set(stopwords.words('english'))

    # Apply preprocessing
    X = X.apply(lambda text: preprocess_text(text, lemmatizer, stpwrds))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # TF-IDF vectorization
    tfidf_v = TfidfVectorizer()  # Converts text into weighted word features and tells how important a word is whereas countVectorizer count the frequency how many times it appear
    tfidf_X_train = tfidf_v.fit_transform(X_train)
    tfidf_X_test = tfidf_v.transform(X_test)

    # Train classifier
    classifier = PassiveAggressiveClassifier()
    classifier.fit(tfidf_X_train, y_train)

    # Evaluate
    y_pred = classifier.predict(tfidf_X_test)
    acc = accuracy_score(y_test, y_pred)
    print("accuracy: %.2f%%" % (acc * 100))

    # Save model
    joblib.dump((classifier, tfidf_v), "models/text_classifier.pkl")

if __name__ == "__main__":
    train_text_model()
