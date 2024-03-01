import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the IMDb movie reviews dataset
df = pd.read_csv('imdb_reviews.csv')

# Data preprocessing
# (perform necessary text preprocessing steps)

# Split data into features (X) and labels (y)
X = df['review']
y = df['sentiment']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical format using TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the Support Vector Machine classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = svm_classifier.predict(X_test_tfidf)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example: Predict sentiment for a new movie review
new_review = ["Happy to see you"]
new_review_tfidf = vectorizer.transform(new_review)
predicted_sentiment = svm_classifier.predict(new_review_tfidf)
print("\nPredicted Sentiment:", predicted_sentiment[0])
