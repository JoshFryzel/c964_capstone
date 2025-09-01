import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data from the csv
email_data = pd.read_csv("data/spam_or_not_spam.csv")
#Removes any empty rows
email_data = email_data.dropna(subset=["email"])

# Clean the Data
def clean_text(text):
    #Convert the input to a string
    text = str(text).lower()
    #Remove html tags
    text = re.sub(r'<.*?>', '', text)
    #Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Remove numbers, commented due to spam often containing monetary offers and numeric values
    #Text = re.sub(r'\d+', '', text)
    #Remove whitespace
    return text.strip()
#Apply clean_text function to all emails
email_data["cleaned"] = email_data["email"].apply(clean_text)

# Convert cleaned text into a matrix ignoring stopwords
vectors = CountVectorizer(stop_words='english')
X = vectors.fit_transform(email_data["cleaned"])
y = email_data["label"]

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naive Bayes classifier
spam_filter = MultinomialNB()
#Train the model
spam_filter.fit(X_train, y_train)

# Evaluate
y_pred = spam_filter.predict(X_test)
print(classification_report(y_test, y_pred))

# Use ML to predict spam or ham based on user input
while True:
    user_input = input("Enter an email message (or 'quit'): ")
    if user_input.lower() == "quit":
        break
    cleaned_input = clean_text(user_input)
    vectorized_input = vectors.transform([cleaned_input])
    prediction = spam_filter.predict(vectorized_input)[0]
    print("Prediction:", "SPAM" if prediction == 1 else "HAM")