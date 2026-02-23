import os
from sqlalchemy import create_engine, Column, Integer, Text, LargeBinary, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import numpy as np

# --- DATABASE SELECTION ---
USE_SQLITE = True 

if USE_SQLITE:
    # Use v3 to include the new file history tables/columns
    DATABASE_URL = "sqlite:///./rag_v3.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/rag_db")
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    chunks = relationship("DocumentChunk", back_populates="owner")
    files = relationship("UploadedFile", back_populates="owner")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    upload_date = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="files")
    chunks = relationship("DocumentChunk", back_populates="file")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(LargeBinary, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"))
    file_id = Column(Integer, ForeignKey("uploaded_files.id"))
    
    owner = relationship("User", back_populates="chunks")
    file = relationship("UploadedFile", back_populates="chunks")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)

def save_chunk(db, text, embedding_vec, user_id, file_id):
    embedding_bytes = embedding_vec.tobytes()
    db_chunk = DocumentChunk(content=text, embedding=embedding_bytes, user_id=user_id, file_id=file_id)
    db.add(db_chunk)
    db.commit()
