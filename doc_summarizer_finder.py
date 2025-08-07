import streamlit as st
import os
import tempfile
import fitz
from openai import OpenAI
from dotenv import load_dotenv
from docx import Document
from rag import chunk_text, embed_chunks, save_to_faiss, search_faiss
import numpy as np 
import asyncio
from openai import AsyncOpenAI
import hashlib
import json

# Cache setup
CACHE_DIR = ".cache_summaries"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def load_cached_summary(file_hash):
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def save_cached_summary(file_hash, summary):
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(summary)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Legal Summarizer")
st.title("Legal Document Summarizer")
st.markdown("Upload a legal document (PDF, DOCX, or TXT), and receive a concise summary or ask questions about it.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"], accept_multiple_files=False)

def extract_text(file):
    if file.type == "application/pdf":
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", dir=tempfile.gettempdir()) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            st.warning(f"Error extracting PDF text: {e}")
            text = None
        os.remove(tmp_path)
        return text

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=tempfile.gettempdir()) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = Document(tmp_path)
            text = "\n".join(para.text for para in doc.paragraphs)
        except Exception as e:
            st.warning(f"Error reading DOCX: {e}")
            text = None
        os.remove(tmp_path)
        return text

    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    return None

async def summarize_chunk(start_idx, chunk_group):
    chunk_texts = "\n\n".join(chunk_group)
    prompt = f"""
You are a helpful legal assistant. Summarize the following portion of a legal document in clear and concise language.
Focus on key events, involved parties, dates, and outcomes if mentioned.
DO NOT number sections, do not label them, and do not include 'Document Chunk' in your response.
Write only the plain summary sentences.

{chunk_texts}
"""
    try:
        resp = await async_client.chat.completions.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=60
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to summarize: {e}"

def summarize_text(text):
    if not text or len(text.strip()) < 100:
        return "No usable content was extracted from the file."
    
    chunks = chunk_text(text, chunk_size=1000)
    chunks = [c for c in chunks if len(c.strip()) > 50]
    max_chunks = min(15, len(chunks))

    batch_size = 2
    batches = [chunks[i:i+batch_size] for i in range(0, max_chunks, batch_size)]

    async def process_all():
        tasks = [summarize_chunk(i*batch_size, batch) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks)
        return "".join(results)

    chunk_summaries = asyncio.run(process_all())

    final_prompt = f"""
You are a legal assistant. Combine the following text into one cohesive paragraph.
Focus on the main events, people, dates, treatments, and outcomes.
Avoid repetitions and do not use bullet points, numbers, or labels.
Write it as a flowing narrative in one paragraph.

{chunk_summaries}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4-0613",
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Failed to generate final summary: {e}"

def summarize_text_with_cache(text, file_bytes):
    file_hash = get_file_hash(file_bytes)
    cached = load_cached_summary(file_hash)
    if cached:
        return cached
    summary = summarize_text(text)
    save_cached_summary(file_hash, summary)
    return summary

def find_answer(question, file_name):
    base_name = os.path.splitext(file_name)[0]
    temp_index_path = os.path.join(tempfile.gettempdir(), base_name)
    top_chunks = search_faiss(question, faiss_path=temp_index_path)
    context = "\n\n".join(top_chunks)

    prompt = f"""
You are a legal assistant. Use the context below to answer the user's question.

Context:
\"\"\"{context}\"\"\"

Question:
\"\"\"{question}\"\"\"

Answer clearly and concisely using only the document contents.
"""
    response = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

if uploaded_file:
    summary_blocks = []
    block = f"**File: {uploaded_file.name}**\n\n"
    file_bytes = uploaded_file.getvalue()

    
    text = extract_text(uploaded_file)
    if not text:
        block += "Could not extract text from this file. \n---"
        summary_blocks.append(block)
    else:
        with st.spinner("Preparing for question answering..."):
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            base_name = os.path.splitext(uploaded_file.name)[0]
            temp_index_path = os.path.join(tempfile.gettempdir(), base_name)
            save_to_faiss(chunks, np.array(embeddings), faiss_path=temp_index_path)

            action = st.radio(f"What would you like to do with **{uploaded_file.name}**?", ["Summarize", "Find Something"])

            if action == "Summarize":
                if st.button(f"Summarize {uploaded_file.name}"):
                    with st.spinner("Summarizing..."):
                        summary = summarize_text_with_cache(text, file_bytes)
                        block += summary.replace("\n", " \n") + "\n\n---"
                        summary_blocks.append(block)

            elif action == "Find Something":
                user_question = st.text_input(f"Enter your question about {uploaded_file.name}:")
                if st.button(f"Find answer in {uploaded_file.name}"):
                    if user_question.strip():
                        with st.spinner("Searching..."):
                            answer = find_answer(user_question, uploaded_file.name)
                            block += f"**Question:** {user_question} \n\n"
                            block += answer + "\n\n----"
                            summary_blocks.append(block)
                    else:
                        block += "Please enter a question before clicking. \n---"
                        summary_blocks.append(block)

    for block in reversed(summary_blocks):
        st.markdown(block)
