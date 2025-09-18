import streamlit as st
import os
import tempfile
from openai import OpenAI
from dotenv import load_dotenv
from rag import chunk_text, embed_chunks, save_to_faiss, search_faiss, extract_text
import numpy as np
import hashlib
from concurrent.futures import ThreadPoolExecutor
import base64
import time


CACHE_DIR = ".cache_summaries"
CACHE_TTL_DAYS = 2  
os.makedirs(CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def is_stale(path, ttl_days=CACHE_TTL_DAYS):
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds > ttl_days * 24 * 60 * 60

def load_cached_summary(file_hash):
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.txt")
    if os.path.exists(cache_path):
        if is_stale(cache_path):
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    return None

def save_cached_summary(file_hash, summary):
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.txt")
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(summary)

def cleanup_cache(ttl_days=CACHE_TTL_DAYS):
    for name in os.listdir(CACHE_DIR):
        path = os.path.join(CACHE_DIR, name)
        if os.path.isfile(path) and is_stale(path, ttl_days):
            try:
                os.remove(path)
            except OSError:
                pass

cleanup_cache()

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  


def set_background(image_file):
    try:
        with open(image_file, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()
        st.markdown(
            f"""
            <style>
            html, body, .stApp {{
                height: 100%;
                margin: 0;
                padding: -50;
                padding-bottom: 15vh;
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center calc(0% - 70px);
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: black !important;
            }}
            .stApp {{
                padding-top: 3.5rem;
            }}
            .stMarkdown, .stTextInput, .stExpander, .stTextInput > div > input {{
                color: black !important;
            }}
            .streamlit-expanderHeader {{
                color: black !important;
            }}
            header[data-testid="stHeader"] {{
                background-color: rgba(255, 255, 255, 0);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception:
        pass

set_background("Lakers.png")

st.set_page_config(page_title="Legal Summarizer")
st.title("Legal Document Summarizer")
st.markdown("Upload a legal document (PDF, DOCX, or TXT), and receive a concise summary or ask questions about it.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"], accept_multiple_files=False)


_CONCURRENCY_LIMIT = 5

def _group(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _needs_reduce(texts, max_chars=12000):
    return sum(len(t) for t in texts) > max_chars

def _ask_llm(prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()

def _summarize_group_sync(group_text: str) -> str:
    prompt = f"""
You are a helpful legal assistant. Summarize the following portion of a legal document in clear and concise language.
Focus on key events, involved parties, dates, and outcomes if mentioned.
DO NOT number sections, do not label them, and do not include 'Document Chunk' in your response.
Write only the plain summary sentences.

{group_text}
"""
    return _ask_llm(prompt)

def summarize_text(text, file_hash_for_cache: str):
    if not text or len(text.strip()) < 100:
        return "No usable content was extracted from the file."

    chunks = chunk_text(text, chunk_size=1000)
    chunks = [c for c in chunks if len(c.strip()) > 50]

    group_size = 2
    groups = ["\n\n".join(g) for g in _group(chunks, group_size)]
    with ThreadPoolExecutor(max_workers=_CONCURRENCY_LIMIT) as ex:
        group_summaries = list(ex.map(_summarize_group_sync, groups))

    while _needs_reduce(group_summaries):
        reduce_bundle_size = 8
        reduced = []
        for bundle in _group(group_summaries, reduce_bundle_size):
            bundle_text = "\n\n".join(bundle)
            reduce_prompt = f"""
You are a legal assistant. Merge the following partial summaries into a single cohesive paragraph.
Focus on the main events, people, dates, treatments, and outcomes.
Avoid repetitions and do not use bullet points, numbers, or labels.

{bundle_text}
"""
            reduced.append(_ask_llm(reduce_prompt))
        group_summaries = reduced

    final_prompt = f"""
You are a legal assistant. Combine the following text into one cohesive paragraph.
Focus on the main events, people, dates, treatments, and outcomes.
Avoid repetitions and do not use bullet points, numbers, or labels.
Write it as a flowing narrative in one paragraph.

{"\n\n".join(group_summaries)}
"""
    try:
        final_summary = _ask_llm(final_prompt)
        return final_summary
    except Exception as e:
        return f"Failed to generate final summary: {e}"

def summarize_text_with_cache(text, file_bytes, file_hash_for_cache: str):
    cached = load_cached_summary(file_hash_for_cache)
    if cached:
        return cached
    summary = summarize_text(text, file_hash_for_cache)
    save_cached_summary(file_hash_for_cache, summary)
    return summary

def find_answer(question: str, faiss_path: str):
    top_chunks = search_faiss(question, faiss_path=faiss_path)
    context = "\n\n".join(top_chunks)
    prompt = f"""
You are a legal assistant. Use the context below to answer the user's question.

Context:
\"\"\"{context}\"\"\"

Question:
\"\"\"{question}\"\"\"

Answer clearly and concisely using only the document contents.
"""
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0.2
    )
    return resp.output_text.strip()


if uploaded_file:
    summary_blocks = []
    block = f"**File: {uploaded_file.name}**\n\n"
    file_bytes = uploaded_file.getvalue()
    file_hash = get_file_hash(file_bytes)

    text = extract_text(uploaded_file)
    if not text:
        block += "Could not extract text from this file. \n---"
        summary_blocks.append(block)
    else:
        with st.spinner("Preparing for question answering..."):
            chunks = chunk_text(text)  
            embeddings = embed_chunks(chunks)

            base_name = os.path.splitext(uploaded_file.name)[0]
            temp_index_path = os.path.join(tempfile.gettempdir(), f"{base_name}_{file_hash[:8]}")
            save_to_faiss(chunks, np.array(embeddings), faiss_path=temp_index_path)

            action = st.radio(
                f"What would you like to do with **{uploaded_file.name}**?",
                ["Summarize", "Find Something"]
            )

            if action == "Summarize":
                with st.form(key=f"summarize_form_{uploaded_file.name}"):
                    submitted_sum = st.form_submit_button(f"Summarize {uploaded_file.name}")
                if submitted_sum:
                    with st.spinner("Summarizing..."):
                        summary = summarize_text_with_cache(text, file_bytes, file_hash)
                        block += summary.replace("\n", " \n") + "\n\n---"
                        summary_blocks.append(block)

            elif action == "Find Something":
                with st.form(key=f"find_form_{uploaded_file.name}"):
                    user_question = st.text_input(f"Enter your question about {uploaded_file.name}:")
                    submitted = st.form_submit_button("Find Answer")
                if submitted:
                    if user_question.strip():
                        with st.spinner("Searching..."):
                            answer = find_answer(user_question, temp_index_path)
                            block += f"**Question:** {user_question} \n\n"
                            block += answer + "\n\n----"
                            summary_blocks.append(block)
                    else:
                        block += "Please enter a question before submitting. \n---"
                        summary_blocks.append(block)

    for block in reversed(summary_blocks):
        st.markdown(block)
