import re
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model('D:\Working_proj\essay-shiny-master\main\model\model.h5')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Tokenize the text
def tokenize_text(text):
    return word_tokenize(text)

# Use TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Load and prepare data to fit the vectorizer (you may need to adjust this part to your actual data)
sample_data = ["This is a sample essay text."] * 1000  # Replace with your actual data
X_sentences = [clean_text(essay) for essay in sample_data]
vectorizer.fit(X_sentences)

# Padding settings
max_len = 50  # Adjust as necessary

# Function to preprocess and predict the score of a new essay
def predict_essay_score(essay_text):
    cleaned_text = clean_text(essay_text)
    tokenized_text = word_tokenize(cleaned_text)
    sentence = ' '.join(tokenized_text)
    vector = vectorizer.transform([sentence]).toarray()
    padded_text = np.zeros((1, max_len))
    length = min(len(vector[0]), max_len)
    padded_text[0, :length] = vector[0][:length]
    prediction = model.predict(padded_text)
    predicted_score = np.clip(np.round(prediction), 1, 6)
    return print('the predicted score is:', predicted_score[0][0])

essay = str(input("Enter your essay here: \n"))
tokenize_text(essay)
predict_essay_score(essay)