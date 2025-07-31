import faiss
import numpy as np
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')

class SimpleVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.sentences = []
        self.index = None

    def build_index(self, text):
        self.sentences = sent_tokenize(text)
        X = self.vectorizer.fit_transform(self.sentences).toarray().astype('float32')
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X)

    def query(self, q, k=5):
        q_vec = self.vectorizer.transform([q]).toarray().astype('float32')
        _, indices = self.index.search(q_vec, k)
        return [self.sentences[i] for i in indices[0]]
