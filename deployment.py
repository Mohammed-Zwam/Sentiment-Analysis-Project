import streamlit_lottie as st_lotte 
import requests
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
import pickle
import PIL as img

# Upload Data
model = pickle.load(open('D:\Programming\Coding\Python [AI]\Sentiment Analysis Project\SentimentFile.sav','rb'))
st.title('Sentiment Analysis Project')
st.info('This project focuses on developing a sentiment analysis system using advanced machine learning techniques. The goal is to analyze textual data to determine the underlying sentiment, which can be classified as positive, negative, or neutral.')
st.subheader('Enter your comment or review to classify your sentiment')
comment = st.text_input('Comment: ')

btn = st.button('Predict')
if btn and comment:
    comment_list = [comment]
    sentiment = model.predict(comment_list)    
    if sentiment == 1:
        st.success('Positive Sentiment 😊')
    elif sentiment == -1:
        st.error('Negative Sentiment 😠')
    else:
        st.warning('Neutral Sentiment 😐')
elif btn and not comment: 
    st.write('Please enter a comment 🚨')