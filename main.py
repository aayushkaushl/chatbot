from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
from langchain.docstore.document import Document
load_dotenv()
api_keys = os.environ["api_key"]
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
st.title("BLOG APP")

option = st.selectbox("Choose input type", ["Web URL", "PDF File", "text File", "text input"])

url = None
pdf_file = None
text_file = None
manual_text = None
if option == "Web URL":
    url = st.text_input("Enter URL")
elif option == "PDF File":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
elif option == "text File":
    text_file = st.file_uploader("Upload text", type=["txt"])
elif option == "text input":
    manual_text = st.text_area("write here to find answer", height=200)

if st.button("Load Document"):
    with st.spinner("Loading document"):
        try:
            if option == "Web URL" and url:
                loader = WebBaseLoader(url)
                data = loader.load()

            elif option == "PDF File" and pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_file.read())
                    tmp_path = tmp_file.name
                loader = PyPDFLoader(tmp_path)
                data = loader.load()
                os.remove(tmp_path)  
               
            elif option == "text File" and text_file:
                txt_content = text_file.read().decode("utf-8")
                data = [Document(page_content=txt_content)]

            elif option == "text input" and manual_text:
              data = [Document(page_content=manual_text)]
            else:
                 st.warning("not a valid input")
                 st.stop()
                                                                
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(data)

            embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            store = FAISS.from_documents(split_docs, embed)
            st.session_state.retriever = store.as_retriever()
            st.session_state.doc_loaded = True
            st.success("document uploads sucessfully")

        except Exception as e:
            st.error(f"Error loading document {e}")
            st.session_state.doc_loaded = False
            st.session_state.retriever = None

if st.session_state.doc_loaded:
    question = st.text_input("Write your question")
    if st.button("get answer") and question:
        with st.spinner("getting answer"):
            try:
                prompt_template = PromptTemplate(
                    template="""
                     Help me to answer the question based on the blog:

                     Context: {context}
                     Question: {question}
                    """,
                    input_variables=["question", "context"]
                )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    api_key=api_keys
                )

                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.retriever,
                    chain_type_kwargs={"prompt": prompt_template},
                )

                answer = chain.run(question)
                st.success(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                
