import streamlit as st
from rag_pipeline import retrieve_docs, answer_query, llm_model
from vector_database import upload_pdf, process_pdf, vacant_folder
import os
import hashlib
from datetime import datetime
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from fpdf import FPDF

# Set up Streamlit page layout
st.set_page_config(page_title="AI Lawyer", layout="wide")

# Directory to store cached vector databases
CACHE_DIR = "vectorstore/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Session state for persistent chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Generate a unique cache key for a PDF file using MD5 hash
def get_cache_key(file_path):
    with open(file_path, 'rb') as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

# Return the embedding model (needed for FAISS load)
def get_embedding_model(model_name="deepseek-r1:1.5b"):
    return OllamaEmbeddings(model=model_name)

# Save FAISS vector store using FAISS native method
def save_cached_db(key, db):
    path = os.path.join(CACHE_DIR, key)
    db.save_local(path)

# Load FAISS vector store with explicit deserialization consent
def load_cached_db(key):
    path = os.path.join(CACHE_DIR, key)
    if os.path.exists(path):
        return FAISS.load_local(
            folder_path=path,
            embeddings=get_embedding_model("deepseek-r1:1.5b"),
            allow_dangerous_deserialization=True
        )
    return None

# Save conversation to a PDF file
def save_conversation_as_pdf(log):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for entry in log:
        role = entry['role']
        text = entry['text']
        pdf.multi_cell(0, 10, f"{role}: {text}\n")
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# Sidebar with enhanced visuals
with st.sidebar:
    st.markdown("""
    
    """, unsafe_allow_html=True)
    st.markdown("## ⚙️ Options")
    clear_chat = st.button("🧹 Clear Chat")
    download_log = st.button("💾 Download Chat as TXT")
    download_pdf = st.button("📄 Download Chat as PDF")
    upload = st.file_uploader("📄 Upload PDF", type="pdf", accept_multiple_files=False)

if clear_chat:
    st.session_state.chat_history = []
    st.rerun()

# Main chat area
st.title("🤖 AI Lawyer - Ask Me Anything About Law")

# Text area for user query
user_query = st.chat_input("Type your legal question here...")

# Main logic
if user_query and upload:
    file_path = upload_pdf(upload)
    cache_key = get_cache_key(file_path)

    db = load_cached_db(cache_key)
    if db is None:
        db = process_pdf(file_path)
        save_cached_db(cache_key, db)

    retrieved_docs = retrieve_docs(user_query, db)
    response = answer_query(documents=retrieved_docs, model=llm_model, query=user_query)
    clean_answer = response.content.replace("<think>", "").strip()

    st.session_state.chat_history.append({"role": "User", "text": user_query})
    st.session_state.chat_history.append({"role": "AI Lawyer", "text": clean_answer})

    vacant_folder(file_path)

elif user_query and not upload:
    st.error("⚠️ Please upload a PDF first to enable answering.")

# Custom layout styling for dark mode and compact appearance
st.markdown("""
    
""", unsafe_allow_html=True)

# Display full chat history
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "User" else "🧠"
    with st.chat_message(msg["role"]):
        st.markdown(f"{avatar} {msg['text']}", unsafe_allow_html=True)

# Download buttons
if download_log and st.session_state.chat_history:
    filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in st.session_state.chat_history:
            f.write(f"{entry['role']}: {entry['text']}\n\n")
    with open(filename, "rb") as f:
        st.sidebar.download_button("⬇️ Download TXT", f, file_name=filename, mime="text/plain")

if download_pdf and st.session_state.chat_history:
    pdf_file = save_conversation_as_pdf(st.session_state.chat_history)
    with open(pdf_file, "rb") as f:
        st.sidebar.download_button("⬇️ Download PDF", f, file_name=pdf_file, mime="application/pdf")


# Inject custom CSS
st.markdown("""
    <style>
    .small-font {
        font-size:12px !important;
    }
    .small-input input {
        font-size: 12px !important;
    }
    .small-textarea textarea {
        font-size: 12px !important;
    }
    </style>
""", unsafe_allow_html=True)