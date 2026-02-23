import os
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from database import DocumentChunk, save_chunk, UploadedFile

# Configuration
USE_MOCK = True
EMBED_DIM = 384

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(EMBED_DIM).astype('float32')

def retrieve(query, db: Session, user_id: int, top_k=2):
    q = get_embedding(query).astype('float32')
    
    # Fetch only chunks belonging to the current user
    db_chunks = db.query(DocumentChunk).filter(DocumentChunk.user_id == user_id).all()
    if not db_chunks:
        return []

    docs_with_scores = []
    for chunk in db_chunks:
        embedding = np.frombuffer(chunk.embedding, dtype='float32')
        dist = np.linalg.norm(embedding - q)
        docs_with_scores.append((chunk.content, dist))
    
    docs_with_scores.sort(key=lambda x: x[1])
    return [doc for doc, score in docs_with_scores[:top_k]]

def generate_answer(query, db: Session, user_id: int):
    docs = retrieve(query, db, user_id)
    if not docs:
        return "No documents found in your history. Please upload a PDF."
    
    if USE_MOCK:
        return f"Answer based on your history: {', '.join(docs)}"
    return f"(LLM) Answer based on: {', '.join(docs)}"

def process_and_save_pdf_text(text, filename, db: Session, user_id: int):
    # 1. Record the file
    new_file = UploadedFile(
        filename=filename, 
        upload_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        user_id=user_id
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)
    
    # 2. Save chunks linked to this file
    chunks = chunk_text(text)
    for chunk in chunks:
        emb = get_embedding(chunk)
        embedding_bytes = emb.tobytes()
        db_chunk = DocumentChunk(
            content=chunk, 
            embedding=embedding_bytes, 
            user_id=user_id,
            file_id=new_file.id
        )
        db.add(db_chunk)
    db.commit()
    return len(chunks)
