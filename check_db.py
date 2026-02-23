from database import SessionLocal, DocumentChunk, User
import numpy as np

def check_database():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        chunks = db.query(DocumentChunk).all()
        print(f"\n--- Multiuser Database Check ---")
        print(f"Total Users: {len(users)}")
        for u in users:
            chunk_count = db.query(DocumentChunk).filter(DocumentChunk.user_id == u.id).count()
            print(f"- User '{u.username}' (ID: {u.id}): {chunk_count} chunks")
        
        print(f"\nTotal chunks across all users: {len(chunks)}")
        
        if len(chunks) > 0:
            print("\nRecent Chunks (First 3):")
            for i, chunk in enumerate(chunks[:3]):
                print(f"- [{chunk.id}] Owner ID: {chunk.user_id} | Content: {chunk.content[:60]}...")
        print(f"--------------------------------\n")
    except Exception as e:
        print(f"Error checking database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    check_database()
