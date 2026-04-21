import streamlit as st
import joblib
import spacy

# Load model + vectorizer
model = joblib.load(open("C:/Users/admin/Downloads/logistic_model.pkl", "rb"))
vectorizer = joblib.load(open("C:/Users/admin/Downloads/vectorizer_model.pkl", "rb"))

# Load spacy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct]
    return " ".join(tokens)

# UI
st.set_page_config(page_title="News Classifier", layout="centered")

st.title("📰 News Category Classifier")
st.write("Enter a news headline and get its category")

user_input = st.text_input("Enter headline:")

if st.button("Predict"):
    if user_input:
        processed = preprocess(user_input)
        vector = vectorizer.transform([processed])

        prediction = model.predict(vector)[0]
        confidence = model.predict_proba(vector).max()

        st.success(f"Category: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")
    else:
        st.warning("Please enter a headline")
