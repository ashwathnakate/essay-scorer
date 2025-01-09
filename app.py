
import re
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer



# Load the trained model
model = joblib.load('model/new_model.joblib')

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
    return predicted_score[0][0]

# ---------------------main app ends here--------------------

# Streamlit app 
st.set_page_config(page_title='Automated Essay Scorer', page_icon='✍️') 


st.markdown( """ 
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap'); 
            * { 
                font-family: 'Poppins', sans-serif !important; 
            }
            
            .main {
                background-color: #f5f5f5; 
                padding: 20px; 
                border-radius: 10px; 
                max-width: 800px; 
                margin: 0 auto; 
            } 
            
            h1 { 
                font-family: 'Poppins';
                color: #000000; 
            }
            .stMarkdown{
                font-family: 'Poppins';
                font-weight: 400;
            }
            p{
                font-family: 'Poppins';
                font-weight: 400;
            } 
            .footer { 
                position: fixed; 
                bottom: 0; 
                width: 100%; 
                text-align: center; 
                color: rgba(255, 255, 255, 0.5); 
            } 
            .feedback { 
                font-size: 1.2em; 
                margin-top: 20px; 
                padding: 10px; 
                border-radius: 5px; 
            } 
            .stTextArea textarea { 
                padding: 30px !important;
                background-color: rgba(255, 255, 255, 0.05) !important;  
                border: 2px solid #fff !important; /* Added white border */ 
                padding: 10px !important; 
                border-radius: 5px !important; 
            }
            
            .custom-progress-bar { 
                width: 100%; 
                background-color: #f3f3f3; 
                border-radius: 5px; 
            } 
            
            .custom-progress-bar-fill { 
                height: 25px; 
                width: 0; 
                background-color: #4CAF50; 
                border-radius: 5px; 
                text-align: center; 
                line-height: 25px; 
                color: white; 
                font-weight: bold; 
                animation: fillProgress 2s linear forwards; 
            } 
            
            @keyframes fillProgress { 
                from { width: 0; } to { width: 100%; } 
            }
            </style> """, 
            unsafe_allow_html=True ) 




st.title('📚 Automated Essay Scorer') 
st.write('**Input your essay below and get a predicted score!**') 
st.write('This is an AI based scorer. Maximum rating is 6') 

essay_text = st.text_area('✍️ Essay Text', '') 

if st.button('Predict Score'): 
    with st.spinner('Analyzing your essay...'): 
        score = predict_essay_score(essay_text)
        progress_bar = st.progress(0) 
        for i in range(100): 
            time.sleep(0.02) 
            progress_bar.progress(i + 1)
        st.success('Prediction complete!') 
        st.write(f'## 🏆 Predicted Essay Score: {score}') 
        
        # Generate the word cloud
        word_cloud = WordCloud(background_color='white').generate(clean_text(essay_text))
        st.write("### Word Cloud of your essay")
        fig, ax = plt.subplots()
        ax.imshow(word_cloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Provide Feedback
        feedback = {
            1: '😟 Needs significant improvement.',
            2: '🙁 Below average. Work on clarity and structure.',
            3: '😐 Average. Improve on details and depth.',
            4: "😊 Above average. Good job, but there's room for improvement.",
            5: "😀 Great job! Focus on polishing",
            6: "🏆 Excellent! Keep up the great work"
        }
        feedback_class = f"feedback-{score}"
        st.markdown(f'<div class="feedback {feedback_class}">{feedback[score]}</div>', unsafe_allow_html=True)
        
st.markdown( """ 
                <div class="footer"> 
                    <p>Created with ❤️ by ashwathnakate</p> 
                </div> """, 
                unsafe_allow_html=True)
