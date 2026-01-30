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
import hashlib as _hashlib


CACHE_DIR = ".cache_summaries"        
TEXT_CACHE_DIR = ".cache_text"        
PARTIAL_CACHE_DIR = ".cache_partials" 

CACHE_TTL_DAYS = 2  
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TEXT_CACHE_DIR, exist_ok=True)
os.makedirs(PARTIAL_CACHE_DIR, exist_ok=True)

def get_file_hash(file_bytes):
    hasher = hashlib.md5()
    hasher.update(file_bytes)
    return hasher.hexdigest()

def is_stale(path, ttl_days=CACHE_TTL_DAYS):
    age_seconds = time.time() - os.path.getmtime(path)
    return age_seconds > ttl_days * 24 * 60 * 60

def _cleanup_dir(dir_path, ttl_days=CACHE_TTL_DAYS):
    try:
        for name in os.listdir(dir_path):
            p = os.path.join(dir_path, name)
            if os.path.isfile(p) and is_stale(p, ttl_days):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception:
        pass

_cleanup_dir(CACHE_DIR, CACHE_TTL_DAYS)
_cleanup_dir(TEXT_CACHE_DIR, CACHE_TTL_DAYS)
_cleanup_dir(PARTIAL_CACHE_DIR, CACHE_TTL_DAYS)

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

def load_cached_text(file_hash):
    p = os.path.join(TEXT_CACHE_DIR, f"{file_hash}.txt")
    if os.path.exists(p):
        if is_stale(p):
            try:
                os.remove(p)
            except OSError:
                pass
            return None
        try:
            return open(p, "r", encoding="utf-8").read()
        except Exception:
            return None
    return None

def save_cached_text(file_hash, text):
    p = os.path.join(TEXT_CACHE_DIR, f"{file_hash}.txt")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception:
        pass


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="Legal Summarizer")
st.title("Legal Document Summarizer")
st.markdown("Upload a legal document (PDF, DOCX, or TXT), and receive a concise summary or ask questions about it.")

uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"], accept_multiple_files=False)


_CONCURRENCY_LIMIT = int(os.getenv("SUM_CONCURRENCY", "8"))

_GROUP_SIZE = int(os.getenv("SUM_GROUP_SIZE", "4"))           
_REDUCE_BUNDLE_SIZE = int(os.getenv("SUM_REDUCE_BUNDLE", "16"))

def _needs_reduce(texts, max_chars=int(os.getenv("SUM_REDUCE_MAX_CHARS", "18000"))):
    return sum(len(t) for t in texts) > max_chars

def _group(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _ask_llm(prompt: str) -> str:
    resp = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0.2,
    )
    return resp.output_text.strip()


def _key_for_text(s: str) -> str:
    return _hashlib.sha1(s.encode("utf-8")).hexdigest()

def _load_partial(k: str) -> str | None:
    p = os.path.join(PARTIAL_CACHE_DIR, f"{k}.txt")
    if os.path.exists(p):
        if is_stale(p):
            try:
                os.remove(p)
            except OSError:
                pass
            return None
        try:
            return open(p, "r", encoding="utf-8").read()
        except Exception:
            return None
    return None

def _save_partial(k: str, summary: str):
    p = os.path.join(PARTIAL_CACHE_DIR, f"{k}.txt")
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(summary)
    except Exception:
        pass

def _summarize_group_sync(group_text: str) -> str:
    k = _key_for_text(group_text)
    cached = _load_partial(k)
    if cached:
        return cached

    prompt = f"""
You are a helpful legal assistant. Summarize the following portion of a legal document in clear and concise language.
Focus on key events, involved parties, dates, and outcomes if mentioned.
DO NOT number sections, do not label them, and do not include 'Document Chunk' in your response.
Write only the plain summary sentences.

{group_text}
"""
    out = _ask_llm(prompt)
    _save_partial(k, out)
    return out


def clean_text(t: str) -> str:
    lines = []
    seen = set()
    for raw in t.splitlines():
        s = raw.strip()
        if not s:
            continue
        low = s.lower()
        if s.isdigit():
            continue
        if low.startswith("page "):
            continue
        # drop exact duplicates
        if s in seen:
            continue
        seen.add(s)
        lines.append(s)
    return "\n".join(lines)


def summarize_text(text, file_hash_for_cache: str):
    if not text or len(text.strip()) < 100:
        return "No usable content was extracted from the file."


    chunks = chunk_text(text, chunk_size=1000)
    chunks = [c for c in chunks if len(c.strip()) > 50]

    groups = ["\n\n".join(g) for g in _group(chunks, _GROUP_SIZE)]

    with ThreadPoolExecutor(max_workers=_CONCURRENCY_LIMIT) as ex:
        group_summaries = list(ex.map(_summarize_group_sync, groups))

    while _needs_reduce(group_summaries):
        reduced = []
        for bundle in _group(group_summaries, _REDUCE_BUNDLE_SIZE):
            bundle_text = "\n\n".join(bundle)
            reduce_prompt = f"""
You are a legal assistant. Merge the following partial summaries into a single cohesive paragraph.
Focus on the main events, people, dates, treatments, and outcomes.
Avoid repetitions and do not use bullet points, numbers, or labels.

{bundle_text}
"""
            reduced.append(_ask_llm(reduce_prompt))
        group_summaries = reduced

    joined_summaries = "\n\n".join(group_summaries)

    final_prompt = f"""
You are a legal assistant. Combine the following text into one cohesive paragraph.
Focus on the main events, people, dates, treatments, and outcomes.
Avoid repetitions and do not use bullet points, numbers, or labels.
Write it as a flowing narrative in one paragraph.

{joined_summaries}
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

    text = load_cached_text(file_hash)
    if not text:
        raw_text = extract_text(uploaded_file)  
        if raw_text:
            text = clean_text(raw_text)        
            save_cached_text(file_hash, text)
        else:
            text = ""

    if not text:
        block += "Could not extract text from this file. \n---"
        summary_blocks.append(block)
    else:
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
                        base_name = os.path.splitext(uploaded_file.name)[0]
                        temp_index_path = os.path.join(
                            tempfile.gettempdir(),
                            f"{base_name}_{file_hash[:8]}"
                        )
                        chunks = chunk_text(text)  
                        embeddings = embed_chunks(chunks)
                        save_to_faiss(chunks, np.array(embeddings), faiss_path=temp_index_path)

                        answer = find_answer(user_question, temp_index_path)
                        block += f"**Question:** {user_question} \n\n"
                        block += answer + "\n\n----"
                        summary_blocks.append(block)
                else:
                    block += "Please enter a question before submitting. \n---"
                    summary_blocks.append(block)

    for block in reversed(summary_blocks):
        st.markdown(block)

