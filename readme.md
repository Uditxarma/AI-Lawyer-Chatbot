# 🧑‍⚖️ AI Lawyer – RAG-Powered PDF Q&A Chatbot

A powerful RAG (Retrieval-Augmented Generation) system using **Deepseek**, **LangChain**, and **Streamlit**. This setup allows you to chat with PDFs and get accurate answers to complex questions about your local documents.

This project walks through the entire process step by step:
- Setting up **Ollama’s Deepseek-r1** LLM model (known for its strong reasoning abilities)
- Integrating it with a LangChain-powered RAG pipeline
- Building a user-friendly **Streamlit** interface for real-time querying of your PDFs

Whether you're exploring Deepseek, reasoning models, LangChain, or building your own document-aware AI chatbot, this project has you covered.

---

## 📦 Features

- 📤 Upload any legal PDF  
- 🤖 Ask natural language questions  
- 🧠 RAG-based answers using DeepSeek + Groq LLM  
- ⚡ FAISS-powered similarity search  
- ♻️ Intelligent caching of PDF embeddings  
- 💬 Clear chat or download full conversation logs  

---

## 📁 Project Structure

```
.
├── main.py                # Streamlit frontend and app logic
├── vector_database.py     # PDF processing, embeddings, vector DB logic
├── rag_pipeline.py        # Prompt template, LLM response generation
├── pdfs/                  # Stores uploaded PDFs
├── vectorstore/cache/     # Caches FAISS vector stores
└── README.md              # You're here!
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-lawyer-chatbot.git
cd ai-lawyer-chatbot
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
streamlit
langchain
langchain-community
langchain-core
langchain-groq
faiss-cpu
pdfplumber
```

> ⚠️ Ensure Ollama is installed and the DeepSeek model is available locally or via Groq.

---

### 4. Run the App

```bash
streamlit run main.py
```

---

## 🧠 How It Works

1. **Upload PDF** → PDF is saved and processed  
2. **Vectorization** → Text chunks are embedded using `deepseek-r1:14b`  
3. **Caching** → Embeddings are cached locally using a hash of PDF content  
4. **Query** → User's question is matched with similar chunks via FAISS  
5. **LLM Answer** → Only relevant context is passed to the LLM to generate an answer

---

## ✅ Example Use Case

Upload a legal document (e.g., a contract or act), then ask:

> “What are the conditions for termination?”

The bot will retrieve only the relevant section and answer accordingly.

---

## 🧪 Future Enhancements

- 🔁 Streamed response  
- 🔍 Support for multi-PDF search  
- ☁️ Upload result to GitHub/Google Drive  
- 🌐 Web deployment with Docker  

---

## 🤝 Contributing

Pull requests and feedback are welcome! Let's make legal AI simpler together.

---

## 📜 License

MIT License. Use at your own legal discretion 😄
