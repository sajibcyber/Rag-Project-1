from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import PyPDF2
from typing import List
import os

from database import init_db, get_db, User, UploadedFile
from rag import generate_answer, process_and_save_pdf_text
from auth import get_password_hash, verify_password, create_access_token, get_current_user

# App state
app = FastAPI()

# Initialize DB tables on startup
@app.on_event("startup")
def on_startup():
    try:
        init_db()
        print("Database connected and tables created.")
    except Exception as e:
        print(f"Database connection failed: {e}")

# --- AUTH ENDPOINTS ---

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# --- RAG ENDPOINTS ---

@app.get("/documents")
async def get_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    files = db.query(UploadedFile).filter(UploadedFile.user_id == current_user.id).all()
    return [{"id": f.id, "filename": f.filename, "date": f.upload_date} for f in files]

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...), 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        if not text.strip():
            return JSONResponse(status_code=400, content={"detail": "PDF is empty."})

        num_chunks = process_and_save_pdf_text(text, file.filename, db, current_user.id)
        return {"message": f"Successfully indexed '{file.filename}'", "chunks": num_chunks}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Server Error: {str(e)}"})

@app.post("/query")
async def query_endpoint(
    question: str = Form(...), 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    answer = generate_answer(question, db, current_user.id)
    return {"answer": answer}

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Assistant - Multiuser</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            :root { --primary: #6366f1; --primary-hover: #4f46e5; --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --text-muted: #94a3b8; }
            body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); margin: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; padding: 20px; box-sizing: border-box; }
            .container { background: var(--card); padding: 2.5rem; border-radius: 1.5rem; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.5); width: 100%; max-width: 600px; border: 1px solid rgba(255,255,255,0.1); transition: all 0.3s ease; }
            h1 { font-weight: 600; text-align: center; color: var(--primary); margin-top: 0; margin-bottom: 0.5rem; }
            .form-group { margin-bottom: 1.2rem; }
            input, button { width: 100%; padding: 0.9rem; border-radius: 0.6rem; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: white; box-sizing: border-box; font-size: 1rem; }
            button { background: var(--primary); font-weight: 600; cursor: pointer; border: none; margin-top: 0.8rem; }
            button:hover { background: var(--primary-hover); transform: translateY(-1px); }
            .hidden { display: none !important; }
            .section { background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 1rem; margin-top: 1.5rem; border: 1px solid rgba(255,255,255,0.05); }
            h3 { margin-top: 0; font-size: 1.1rem; color: var(--primary); border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 0.5rem; margin-bottom: 1rem; }
            #historyList { max-height: 150px; overflow-y: auto; margin-bottom: 1rem; border-radius: 0.5rem; background: rgba(0,0,0,0.2); border: 1px solid rgba(255,255,255,0.05); }
            .history-item { padding: 0.6rem 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.9rem; display: flex; justify-content: space-between; align-items: center; }
            .history-item:last-child { border-bottom: none; }
            .history-date { font-size: 0.75rem; color: var(--text-muted); }
            #result { margin-top: 1.2rem; padding: 1rem; background: rgba(0,0,0,0.3); border-radius: 0.6rem; font-size: 0.95rem; line-height: 1.6; white-space: pre-wrap; display: none; border: 1px solid rgba(255,255,255,0.1); }
            .user-info { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); }
            .logout-btn { width: auto; padding: 0.4rem 0.8rem; background: #ef4444; font-size: 0.8rem; margin: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Multiuser</h1>
            
            <div id="authSection">
                <div style="display: flex; gap: 10px; margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 1rem;">
                    <button type="button" onclick="toggleAuth('login')" id="btnLoginTab" style="flex: 1; margin: 0;">Login</button>
                    <button type="button" onclick="toggleAuth('register')" id="btnRegTab" style="flex: 1; margin: 0; background: rgba(255,255,255,0.05);">Register</button>
                </div>
                <div class="form-group"><input type="text" id="username" placeholder="Username"></div>
                <div class="form-group"><input type="password" id="password" placeholder="Password"></div>
                <button type="button" id="authSubmitBtn" onclick="handleAuth()">Login</button>
            </div>

            <div id="appSection" class="hidden">
                <div class="user-info">
                    <span id="displayUser" style="font-weight: 600;"></span>
                    <button type="button" class="logout-btn" onclick="logout()">Logout</button>
                </div>
                
                <div class="section">
                    <h3>My Documents</h3>
                    <div id="historyList"><div style="padding: 1rem; text-align: center; color: var(--text-muted);">No documents yet</div></div>
                    <input type="file" id="pdfFile" accept=".pdf" style="margin-bottom: 10px;">
                    <button type="button" onclick="uploadPDF()">Add New PDF</button>
                </div>

                <div class="section">
                    <h3>Query Assistant</h3>
                    <input type="text" id="question" placeholder="Ask about your knowledge base...">
                    <button type="button" onclick="askQuestion()">Get Answer</button>
                    <div id="result"></div>
                </div>
            </div>
        </div>

        <script>
            let mode = 'login';
            let token = localStorage.getItem('token');
            let user = localStorage.getItem('username');

            if (token) showApp();

            function toggleAuth(m) {
                mode = m;
                document.getElementById('btnLoginTab').style.background = mode === 'login' ? 'var(--primary)' : 'rgba(255,255,255,0.05)';
                document.getElementById('btnRegTab').style.background = mode === 'register' ? 'var(--primary)' : 'rgba(255,255,255,0.05)';
                document.getElementById('authSubmitBtn').innerText = mode === 'login' ? 'Login' : 'Create Account';
            }

            async function handleAuth() {
                const u = document.getElementById('username').value.trim();
                const p = document.getElementById('password').value.trim();
                if (!u || !p) return alert("Fill credentials");

                if (mode === 'register') {
                    const fd = new FormData();
                    fd.append('username', u); fd.append('password', p);
                    const res = await fetch('/register', { method: 'POST', body: fd });
                    const data = await res.json();
                    if (res.ok) { alert("Account created! Now login."); toggleAuth('login'); } 
                    else alert(data.detail);
                } else {
                    const params = new URLSearchParams();
                    params.append('username', u); params.append('password', p);
                    const res = await fetch('/login', { method: 'POST', body: params });
                    const data = await res.json();
                    if (res.ok && data.access_token) {
                        token = data.access_token; user = u;
                        localStorage.setItem('token', token); localStorage.setItem('username', user);
                        showApp();
                    } else alert(data.detail);
                }
            }

            function showApp() {
                document.getElementById('authSection').classList.add('hidden');
                document.getElementById('appSection').classList.remove('hidden');
                document.getElementById('displayUser').innerText = `User: ${user}`;
                loadHistory();
            }

            async function loadHistory() {
                try {
                    const res = await fetch('/documents', { headers: { 'Authorization': `Bearer ${token}` } });
                    const files = await res.json();
                    const list = document.getElementById('historyList');
                    if (files.length === 0) {
                        list.innerHTML = '<div style="padding: 1rem; text-align: center; color: var(--text-muted);">No documents indexed</div>';
                        return;
                    }
                    list.innerHTML = files.map(f => `
                        <div class="history-item">
                            <span>${f.filename}</span>
                            <span class="history-date">${f.date}</span>
                        </div>
                    `).join('');
                } catch(e) { console.error(e); }
            }

            async function uploadPDF() {
                const file = document.getElementById('pdfFile').files[0];
                if (!file) return alert("Select a PDF");
                const fd = new FormData(); fd.append('file', file);
                try {
                    const res = await fetch('/upload-pdf', { 
                        method: 'POST', body: fd,
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    const data = await res.json();
                    if (res.ok) { alert(data.message); loadHistory(); } else alert(data.detail);
                } catch(e) { alert("Upload failed"); }
            }

            async function askQuestion() {
                const q = document.getElementById('question').value;
                if (!q) return;
                const d = document.getElementById('result');
                d.style.display = 'block'; d.innerText = "Thinking...";
                const fd = new FormData(); fd.append('question', q);
                try {
                    const res = await fetch('/query', { 
                        method: 'POST', body: fd, 
                        headers: { 'Authorization': `Bearer ${token}` }
                    });
                    const data = await res.json();
                    d.innerText = data.answer || data.detail;
                } catch(e) { d.innerText = "Query error"; }
            }

            function logout() { localStorage.clear(); window.location.reload(); }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
