import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz
from docx import Document
import tempfile
import pytesseract
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO


try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:
    _HAS_CUDA = False

_TESS_CMD = os.getenv("TESSERACT_CMD")
if _TESS_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESS_CMD

TESS_CONFIG = os.getenv("TESS_CONFIG", "--oem 1 --psm 6")


model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cuda" if _HAS_CUDA else "cpu"
)


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    g = img.convert("L")
    g = ImageOps.autocontrast(g)
    w, h = g.size
    if max(w, h) < 1600:
        scale = 2 if max(w, h) < 1200 else 1.5
        g = g.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    g = g.filter(ImageFilter.MedianFilter(size=3))
    g = g.filter(ImageFilter.UnsharpMask(radius=1.5, percent=125, threshold=3))
    g = g.point(lambda x: 255 if x > 180 else 0, mode="1")
    return g.convert("L")

def _pil_from_pixmap(pix: "fitz.Pixmap") -> Image.Image:
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if pix.colorspace and pix.colorspace.n == 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    img_bytes = pix.tobytes("png")
    return Image.open(BytesIO(img_bytes))

def _tess(img: Image.Image, lang: str) -> str:
    img_p = _preprocess_for_ocr(img)
    try:
        return pytesseract.image_to_string(img_p, lang=lang, config=TESS_CONFIG)
    except Exception:
        try:
            return pytesseract.image_to_string(img_p, lang=lang)
        except Exception:
            return ""

def _ocr_pixmap_to_text(pix: "fitz.Pixmap", lang="eng") -> str:
    try:
        pil = _pil_from_pixmap(pix)
        return _tess(pil, lang=lang)
    except Exception:
        return ""

def _tile_ocr(pil_img: Image.Image, lang="eng", tile_px=2000, overlap=100) -> str:
    w, h = pil_img.size
    if max(w, h) <= tile_px:
        return _tess(pil_img, lang=lang)
    out = []
    x = 0
    while x < w:
        y = 0
        x_end = min(w, x + tile_px)
        while y < h:
            y_end = min(h, y + tile_px)
            tile = pil_img.crop((x, y, x_end, y_end))
            out.append(_tess(tile, lang=lang))
            if y_end == h:
                break
            y = y_end - overlap
        if x_end == w:
            break
        x = x_end - overlap
    return "\n".join([t.strip() for t in out if t and t.strip()])


def extract_text(file, ocr_lang: str = "eng", ocr_min_chars_per_page: int = 10) -> str:
    if file.type == "application/pdf":
        texts = []
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = fitz.open(tmp_path)
            zoom = 300 / 72.0
            mat = fitz.Matrix(zoom, zoom)
            for page in doc:
                native = page.get_text("text") or ""
                native = native.strip()
                if len(native) < ocr_min_chars_per_page:
                    try:
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        pil = _pil_from_pixmap(pix)
                        if max(pil.size) > 2200:
                            ocr = _tile_ocr(pil, lang=ocr_lang, tile_px=2200, overlap=150)
                        else:
                            ocr = _tess(pil, lang=ocr_lang)
                        ocr = (ocr or "").strip()
                        texts.append(ocr if len(ocr) >= ocr_min_chars_per_page else native)
                    except Exception:
                        texts.append(native)
                else:
                    texts.append(native)
            doc.close()
        except Exception:
            texts = []
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return "\n".join([t for t in texts if t])

    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            doc = Document(tmp_path)
            text = "\n".join(p.text for p in doc.paragraphs)
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
    return model.encode(chunks, batch_size=64, show_progress_bar=False)

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
    query_embedding = model.encode([query], batch_size=1, show_progress_bar=False)
    _, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]
