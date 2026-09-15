# 🎓 JBIET AI Chatbot

An AI-powered chatbot for J.B. Institute of Engineering and Technology (JBIET) using Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Answers questions about JBIET using college documents
- PDF text extraction and OCR
- Document cleaning and chunking
- Semantic search using embeddings
- ChromaDB vector database
- Groq LLM integration
- Question routing between JBIET and general questions
- Follow-up question rewriting using conversation history
- Streamlit chat interface

## 🏗️ Architecture

User
↓
Question Rewriter
↓
Question Router
├── JBIET → ChromaDB → Retriever → RAG → LLM
└── General → LLM
↓
Answer

## 🛠️ Technologies

- Python
- LangChain
- Hugging Face Embeddings
- ChromaDB
- Groq
- Streamlit
- PyMuPDF
- Tesseract OCR

## 📁 Project Structure

```text
Jbiet_collage_chatbot/
├── app.py
├── collage_data.ipynb
├── pdf_urls.csv
├── requirements.txt
├── pyproject.toml
├── src/
│   └── jbiet_chatbot.py
├── data/
│   ├── pdfs/
│   └── chroma_db/
└── README.md
