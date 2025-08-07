import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GEMINI_API_KEY = "AIzaSyBx38MG0txhX7iGM8jGgxsQ0av-xJlakLg"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return "\n".join([para.get_text() for para in paragraphs])
    except Exception as e:
        return f"Error fetching content: {e}"


def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

def retrieve_relevant_context(sentences, question, top_n=5):
    vectorizer = TfidfVectorizer().fit(sentences + [question])
    vectors = vectorizer.transform(sentences + [question])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    return " ".join([sentences[i] for i in top_indices])

def generate_answer(context, question):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    prompt = f"Answer the question using the context below:\n\nContext: {context}\n\nQuestion: {question}"
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    response = requests.post(GEMINI_URL, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "Gemini response format error."
    else:
        return f"Error: {response.status_code} - {response.text}"

st.title(" Blog Q&A")

blog_url = st.text_input("Enter Blog URL")
question = st.text_input("Ask a Question")

if st.button("Get Answer"):
    if blog_url and question:
        with st.spinner("Fetching and analyzing blog..."):
            text = get_text_from_url(blog_url)
            if "Error" in text:
                st.error(text)
            else:
                sentences = split_into_sentences(text)
                context = retrieve_relevant_context(sentences, question)
                answer = generate_answer(context, question)
                st.markdown("###  Answer")
                st.success(answer)
    else:
        st.warning("Please enter both a URL and a question.")

