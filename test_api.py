import requests

def test_app():
    # 1. Create a dummy PDF
    from reportlab.pdfgen import canvas
    pdf_path = "test.pdf"
    c = canvas.Canvas(pdf_path)
    c.drawString(100, 750, "This is a test document about artificial intelligence and RAG systems.")
    c.drawString(100, 730, "RAG stands for Retrieval-Augmented Generation, which combines search and LLMs.")
    c.save()

    print("Uploading PDF...")
    with open(pdf_path, 'rb') as f:
        files = {'file': (pdf_path, f, 'application/pdf')}
        r = requests.post("http://127.0.0.1:8000/upload-pdf", files=files)
        print(r.json())

    print("\nQuerying...")
    data = {'question': 'What is RAG?'}
    r = requests.post("http://127.0.0.1:8000/query", data=data)
    print(r.json())

if __name__ == "__main__":
    try:
        import reportlab
    except ImportError:
        import os
        os.system(".venv\\Scripts\\pip install reportlab")
    
    test_app()
