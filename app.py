import streamlit as st
import pickle
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

ps = PorterStemmer()

def preprocess(text):
    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        ps.stem(word)
        for word in words
        if word not in stopwords.words('english')
    ]

    return " ".join(words)

st.title("📧 AI Spam Email Detector")

email = st.text_area("Paste email text here")

if st.button("Detect"):

    cleaned = preprocess(email)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector).max() * 100

    if prediction == 1:
        st.error(f"🚨 Spam Detected ({probability:.2f}%)")
    else:
        st.success(f"✅ Ham Email ({probability:.2f}%)")