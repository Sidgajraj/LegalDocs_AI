Legal Document Summarizer

This project is a legal document summarizer and search assistant built with Streamlit and OpenAI. It reads long legal documents, summarizes them clearly, and can also answer specific questions about the content. It is designed to handle real-world files such as PDFs, Word documents, and scanned pages using OCR, embeddings, and caching to make the experience fast and reliable.

A demo video is included in the repository showing how the app works from upload to summary and search. It gives a quick overview of the workflow and helps visualize how the tool performs in real time.

What It Does
You upload a file and the app gets to work. If you choose Summarize, it breaks the document into smaller pieces, summarizes each part using GPT, and then merges everything into one clear and easy-to-read paragraph. If you choose Find Something, it converts the document into searchable embeddings using FAISS and Sentence Transformers. You can then ask any question like "Who was the defendant?" or "What was the final judgment?" and it finds the most relevant sections and answers based only on the actual content of the file.

How It’s Structured
doc_summarizer_finder.py is the main Streamlit app. It manages uploads, user actions, caching, and cleanup. It decides whether to summarize or search and handles grouping, merging, and saving results in cache so you do not have to reprocess the same file.
rag.py is the engine that does the heavy lifting. It extracts text from PDFs, DOCX, or TXT files, handles OCR when needed, chunks the text, generates embeddings, and creates the FAISS index for retrieval.

Tech Stack
Python
Streamlit
OpenAI (GPT-4 for summaries and answers)
FAISS
SentenceTransformers (all-MiniLM-L6-v2)
PyMuPDF and Tesseract OCR
NumPy and Pickle for embeddings and caching

Setup
To run the project, install the dependencies, set your OpenAI API key, and make sure Tesseract is properly configured. Once everything is ready, start the app with Streamlit and upload a document from your local machine.

How It Works
When a document is uploaded, the app first extracts its text. For PDFs, it tries to read the native text and switches to OCR if the document is scanned. Once extracted, the text is cleaned and split into manageable chunks. Each chunk is summarized in parallel, and those smaller summaries are merged until the final version reads like a concise narrative.
In search mode, the document is embedded into FAISS, allowing the system to find and return the most relevant passages to answer the user’s question while staying fully grounded in the document.

Caching
Every file uploaded is hashed and cached. If you upload the same file again, the summarization or search loads instantly from cache. The cache automatically clears every few days to keep things fresh.

Key Functions
extract_text retrieves text from PDFs, DOCX, and TXT, with OCR fallback
chunk_text splits large text into smaller sections
embed_chunks creates embeddings for semantic search
summarize_text_with_cache summarizes the document and saves results
find_answer retrieves and answers based on the document content

How It Feels
You upload a long contract, deposition, or case file. In seconds, you get a clear summary that feels like it was written by someone who actually read the whole thing. You can then ask a question, and it gives you a direct, grounded answer pulled straight from the document. It feels simple, intelligent, and genuinely useful.

Why It Matters
Legal work involves massive amounts of reading and analysis. This project reduces that workload dramatically. It makes it easier to find the key points in long files, saves hours of manual review, and ensures that responses stay factual and document-based. It is built to make legal reading faster, clearer, and more efficient.

[InternetShortcut]
URL=https://github.com/Sidgajraj/LegalDocs_AI/blob/main/Summarizer%20video.mp4

