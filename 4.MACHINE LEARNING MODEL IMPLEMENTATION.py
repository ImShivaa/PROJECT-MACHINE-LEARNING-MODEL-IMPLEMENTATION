import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

name = input("Enter your name: ")
print(f"Welcome, {name}! Training the spam detection model...")

data = pd.DataFrame({
    "text": ["Win a lottery now!", "Hello friend, how are you?", "Claim your free prize!", "Let's meet for coffee"],
    "label": ["spam", "ham", "spam", "ham"]
})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

test_email = ["Congratulations! You won a free trip"]
test_vector = vectorizer.transform(test_email)
print("Prediction:", model.predict(test_vector)[0])

print(f"Thank you for using, {name}!")
