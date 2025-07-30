# app.py
import streamlit as st
import uuid
import os
import shutil
import pickle
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Config
GROQ_APIKEY = os.getenv("GROQ_APIKEY")
if not GROQ_APIKEY:
    st.error("❌ GROQ_APIKEY is missing in .env file")
    st.stop()

st.set_page_config(page_title="IntelDocs 📄", layout="centered")
st.title("📄 Chat with Your Document")
st.caption("Powered by LangChain + Groq")

# Session ID
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

user_id = st.session_state.user_id
UPLOAD_DIR = "uploads"
FAISS_INDEX_PATH = f"faiss_indexes/{user_id}.pkl"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("faiss_indexes", exist_ok=True)

# Upload and Process PDF
with st.form("upload-form"):
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    submitted = st.form_submit_button("Upload & Embed")

    if submitted and uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, f"{user_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = FAISS.from_documents(chunks, embedding=embeddings)

            with open(FAISS_INDEX_PATH, "wb") as f:
                pickle.dump(db, f)

            st.success("✅ Document processed and indexed with FAISS!")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")

# Ask Questions
if os.path.exists(FAISS_INDEX_PATH):
    question = st.text_input("Ask something from your document:")
    if question:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            with open(FAISS_INDEX_PATH, "rb") as f:
                db = pickle.load(f)

            llm = ChatGroq(
                model="qwen/qwen3-32b",
                temperature=0.3,
                api_key=GROQ_APIKEY,
                reasoning_format="parsed"
            )

            qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
            answer = qa.run(question)

            st.markdown("### ✅ Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
