import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")
GEMINI_API_KEY = "AIzaSyBx38MG0txhX7iGM8jGgxsQ0av-xJlakLg"
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

def get_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error fetching content: {e}")
        return ""

def split_into_sentences(text):
    return nltk.sent_tokenize(text)

def retrieve_relevant_context(sentences, question, top_n=5):
    vectorizer = TfidfVectorizer().fit(sentences + [question])
    vectors = vectorizer.transform(sentences + [question])
    similarity = cosine_similarity(vectors[-1], vectors[:-1]).flatten()
    top_indices = similarity.argsort()[-top_n:][::-1]
    top_sentences = [sentences[i] for i in top_indices]
    return " ".join(top_sentences)

def generate_answer_gemini(context, question):
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }

    prompt = f"Answer the question using the context below:\n\nContext: {context}\n\nQuestion: {question}"

    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    response = requests.post(GEMINI_ENDPOINT, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f" Gemini API Error {response.status_code}: {response.text}"

st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="🧠")

st.title(" RAG Chatbot ")
st.markdown("Ask questions from any blog/article URL.")

url = st.text_input("🔗 Enter blog/article URL:")

if url:
    with st.spinner("Fetching and processing content..."):
        text = get_text_from_url(url)
        sentences = split_into_sentences(text)

    question = st.text_area("❓ Ask your question here:")

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Generating answer..."):
                context = retrieve_relevant_context(sentences, question)
                answer = generate_answer_gemini(context, question)
                st.success(" Answer:")
                st.write(answer)


