import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model():
    # Sample dataset with texts and labels (1: spam, 0: not spam)
    data = {
        "text": [
            "Congratulations, you've won a lottery!",
            "Hi, how are you doing today?",
            "Claim your free prize now!",
            "Let's have a meeting tomorrow.",
            "Free money awaits you, click now!",
            "Please review the attached document."
        ],
        "label": [1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    return model, vectorizer

if __name__ == "__main__":
    model, vectorizer = train_model()
    email = input("Enter an email text to classify: ")
    vectorized_email = vectorizer.transform([email])
    prediction = model.predict(vectorized_email)[0]
    print("Prediction:", "Spam" if prediction == 1 else "Not Spam")
