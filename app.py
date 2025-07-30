# app.py
import streamlit as st
import uuid
import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Config
GROQ_APIKEY =st.secrets["GROQ_APIKEY"]

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
VECTORSTORE_DIR = f"chroma_db/{user_id}"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Upload and Process PDF
with st.form("upload-form"):
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    submitted = st.form_submit_button("Upload & Embed")

    if submitted and uploaded_file:
        pdf_path = os.path.join(UPLOAD_DIR, f"{user_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Load, split, embed
        try:
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-small-v2")
            if os.path.exists(VECTORSTORE_DIR):
                shutil.rmtree(VECTORSTORE_DIR)

            db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
            db.persist()
            st.success("✅ Document processed successfully!")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")

# Ask Questions
if os.path.exists(VECTORSTORE_DIR):
    question = st.text_input("Ask something from your document:")
    if question:
        try:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            db = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)

            llm = ChatGroq(
                model="gemma-7b-it",  # Smaller model than Mixtral
                temperature=0.3,
                api_key=GROQ_APIKEY,
            )

            qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
            answer = qa.run(question)
            st.markdown("### ✅ Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
