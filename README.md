# AI Document Assistant (Mini-RAG)

A lightweight Retrieval-Augmented Generation (RAG) system built with FastAPI, PyPDF2, and NumPy. This project allows you to upload PDF documents, index their content, and query them through a simple web interface.

## Features
- **PDF Processing**: Extracts text and chunks it for indexing.
- **In-memory Vector Storage**: Uses local embeddings (mocked by default for easy setup) and L2 distance for retrieval.
- **Interactive UI**: Modern web dashboard for uploading and querying.
- **FastAPI Backend**: Efficient and easy-to-use API endpoints.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd project_1
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python main.py
```
Open your browser and navigate to `http://127.0.0.1:8000`.

## API Endpoints
- `GET /`: Interactive web dashboard.
- `POST /upload-pdf`: Upload a PDF file to the index.
- `POST /query`: Query the document index (form data: `question`).

## License
MIT
