import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz
from docx import Document
import tempfile

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception:
            text = ""
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = Document(tmp_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        except Exception:
            text = ""
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return text

    elif file.type == "text/plain":
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception:
            return ""

    return ""

def chunk_text(text, chunk_size=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += (" " if current else "") + para
        else:
            if current.strip():
                chunks.append(current.strip())
            current = para
    if current.strip():
        chunks.append(current.strip())
    return chunks

def embed_chunks(chunks):
    return model.encode(chunks, show_progress_bar=False)

def save_to_faiss(chunks, embeddings, faiss_path):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_path + ".index")
    with open(faiss_path + "_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

def search_faiss(query, faiss_path, top_k=5):
    index = faiss.read_index(faiss_path + ".index")
    with open(faiss_path + "_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    query_embedding = model.encode([query])
    _, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]
