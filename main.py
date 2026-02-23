import os
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
import PyPDF2
from typing import List

# Configuration
hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
USE_MOCK = True  # use deterministic mock embeddings for local runs

EMBED_DIM = 384

# App state
app = FastAPI()
documents: List[str] = []
document_embeddings: List[np.ndarray] = []

#-----------------chunking-----------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text):
    # Deterministic mock embedding based on text hash
    np.random.seed(hash(text) % (2**32))
    return np.random.randn(EMBED_DIM).astype('float32')
    
def faiss_indexing(chunks):
    # Build in-memory embeddings list (faiss optional, not required)
    global document_embeddings
    document_embeddings = [get_embedding(doc) for doc in chunks]
    return chunks

def retrieve(query, top_k=2):
    if not document_embeddings:
        return []
    q = get_embedding(query).astype('float32')
    # compute L2 distances
    dists = [np.linalg.norm(vec - q) for vec in document_embeddings]
    sorted_idx = sorted(range(len(dists)), key=lambda i: dists[i])[:top_k]
    return [documents[i] for i in sorted_idx]

def generate_answer(query):
    docs = retrieve(query)
    context = "\n".join(docs)
    
    if USE_MOCK:
        return f"Answer based on: {', '.join(docs)}"
    try:
        # Placeholder for real LLM call; keep mock for now
        return f"(LLM) Answer based on: {', '.join(docs)}"
    except:
        return f"Error. Docs: {', '.join(docs)}"
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Document Assistant (RAG)</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary: #6366f1;
                --primary-hover: #4f46e5;
                --bg: #0f172a;
                --card: #1e293b;
                --text: #f8fafc;
                --text-muted: #94a3b8;
            }
            body { 
                font-family: 'Outfit', sans-serif; 
                background: var(--bg); 
                color: var(--text); 
                margin: 0; 
                display: flex; 
                justify-content: center; 
                align-items: center; 
                min-height: 100vh;
            }
            .container { 
                background: var(--card); 
                padding: 2.5rem; 
                border-radius: 1.5rem; 
                box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); 
                width: 100%; 
                max-width: 500px;
                border: 1px solid rgba(255,255,255,0.1);
            }
            h1 { font-weight: 600; margin-bottom: 0.5rem; color: var(--primary); }
            p { color: var(--text-muted); margin-bottom: 2rem; }
            .section { margin-bottom: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.03); border-radius: 1rem; }
            h2 { font-size: 1.1rem; margin-top: 0; }
            input, button { 
                width: 100%; 
                padding: 0.8rem; 
                margin-top: 0.5rem; 
                border-radius: 0.5rem; 
                border: 1px solid rgba(255,255,255,0.1); 
                background: rgba(255,255,255,0.05); 
                color: white;
                box-sizing: border-box;
            }
            button { 
                background: var(--primary); 
                font-weight: 600; 
                cursor: pointer; 
                border: none;
                transition: background 0.2s, transform 0.1s;
                margin-top: 1rem;
            }
            button:hover { background: var(--primary-hover); }
            button:active { transform: translateY(1px); }
            #result { 
                margin-top: 1.5rem; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                background: rgba(0,0,0,0.2); 
                font-size: 0.9rem; 
                line-height: 1.5;
                white-space: pre-wrap;
                display: none;
            }
            .status { font-size: 0.8rem; color: #10b981; margin-top: 0.5rem; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Assistant</h1>
            <p>Upload a PDF and chat with your documents.</p>
            
            <div class="section">
                <h2>1. Upload PDF</h2>
                <input type="file" id="pdfFile" accept=".pdf">
                <button onclick="uploadPDF()">Update Index</button>
                <div id="uStatus" class="status">PDF Indexed successfully!</div>
            </div>

            <div class="section">
                <h2>2. Ask Question</h2>
                <input type="text" id="question" placeholder="What is this document about?">
                <button onclick="askQuestion()">Ask AI</button>
            </div>

            <div id="result"></div>
        </div>

        <script>
            async function uploadPDF() {
                const fileInput = document.getElementById('pdfFile');
                if (!fileInput.files[0]) return alert("Please select a file");
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                document.getElementById('uStatus').style.display = 'none';
                try {
                    const res = await fetch('/upload-pdf', { method: 'POST', body: formData });
                    const data = await res.json();
                    document.getElementById('uStatus').innerText = `Indexed ${data.chunks} chunks!`;
                    document.getElementById('uStatus').style.display = 'block';
                } catch (e) { alert("Upload failed"); }
            }

            async function askQuestion() {
                const q = document.getElementById('question').value;
                if (!q) return alert("Please enter a question");

                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerText = "Thinking...";

                const formData = new FormData();
                formData.append('question', q);

                try {
                    const res = await fetch('/query', { method: 'POST', body: formData });
                    const data = await res.json();
                    resultDiv.innerText = data.answer || data.error;
                } catch (e) { resultDiv.innerText = "Error querying API"; }
            }
        </script>
    </body>
    </html>
    """

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global documents

    reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = chunk_text(text)

    documents[:] = faiss_indexing(chunks)
    return {"message": "PDF uploaded and indexed successfully.", "chunks": len(documents)}


@app.post("/query")
async def query_endpoint(question: str = Form(...)):
    if not documents:
        return {"error": "No documents indexed. Please upload a PDF first."}
    answer = generate_answer(question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
