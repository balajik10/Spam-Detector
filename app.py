import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    # Join the list into a string
    return ' '.join(y)

# Load the TF-IDF vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Load the Multinomial Naive Bayes classifier
mnb = pickle.load(open('model.pkl', 'rb'))


st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = mnb.predict(vector_input)[0]

    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')