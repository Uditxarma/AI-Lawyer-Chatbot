from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Create PDF storage directory
pdf_directories = "pdfs/"
os.makedirs(pdf_directories, exist_ok=True)

# Save uploaded PDF to disk
def upload_pdf(file):
    save_path = os.path.join(pdf_directories, file.name)
    with open(save_path, 'wb') as f:
        f.write(file.getbuffer())
    return save_path

# Load PDF, split into chunks, and generate vector DB
def process_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    text_chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
    db = FAISS.from_documents(text_chunks, embeddings)
    return db

# Delete the uploaded file after processing
def vacant_folder(file_path):
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Failed to delete file. Reason: {e}")