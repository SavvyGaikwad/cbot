#!/usr/bin/env python3
"""
Convert ChromaDB to FAISS for Streamlit Cloud compatibility
Run this script locally where ChromaDB works, then upload the FAISS files
"""

import os
import sys
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import shutil

def convert_chroma_to_faiss():
    """Convert existing ChromaDB to FAISS format"""
    
    # Paths
    chroma_path = r"C:\Users\hp\Downloads\final\comprehensive_vector_db\chroma.sqlite3"
    faiss_path = "faiss_vector_db"
    
    print("🔄 Starting ChromaDB to FAISS conversion...")
    
    # Check if ChromaDB exists
    if not os.path.exists(chroma_path):
        print(f"❌ ChromaDB not found at {chroma_path}")
        return False
    
    # Initialize embedding model
    print("📦 Loading embedding model...")
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embedding model loaded")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        return False
    
    # Load ChromaDB
    print("📂 Loading ChromaDB...")
    try:
        chroma_db = Chroma(
            persist_directory=chroma_path,
            embedding_function=embedding_model
        )
        print("✅ ChromaDB loaded")
    except Exception as e:
        print(f"❌ Failed to load ChromaDB: {e}")
        return False
    
    # Get all documents from ChromaDB
# Get all documents from ChromaDB
    print("📄 Extracting documents from ChromaDB...")
    try:
        # Replace this with actual method call to get documents from ChromaDB instance
        all_docs = chroma_db.get()  # or chroma_db.get_all_documents() if applicable
        print(f"✅ Extracted {len(all_docs)} documents")
    except Exception as e:
        print(f"❌ Failed to extract documents: {e}")
        return False
