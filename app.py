import streamlit as st
import os
import io
import tempfile
import requests
import time
import numpy as np
import re
from urllib.parse import urljoin

# Mistral 0.4.2 (legacy client)
from mistralai.client import MistralClient

import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader

# --- Tambahan import untuk dashboard ---
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import json
import string

# WordCloud opsional (fallback otomatis kalau belum terpasang)
try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False


# --------------------- Page config ---------------------
st.set_page_config(page_title="Document Intelligence Agent", layout="wide")
st.title("Document Intelligence Agent")
st.markdown("Upload documents or URL to extract information and ask questions")


# --------------------- Sidebar: API Keys ----------------
with st.sidebar:
    st.header("API Configuration")
    mistral_api_key = st.text_input("Mistral AI API Key (legacy 0.4.2)", type="password")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password")
    model_preference = st.selectbox(
        "Model preference",
        ["Auto (Pro→Flash)", "Flash only", "Pro only"],
        index=0,
        help="Fallback otomatis ke Flash saat Pro kena rate limit/kuota."
    )
    answer_language = st.selectbox(
        "Answer language",
        ["Bahasa Indonesia", "English"],
        index=0,
        help="Bahasa jawaban untuk fitur Q&A."
    )

    st.markdown("---")
    st.markdown("How To Get API Key Tutorials")
    st.markdown(
        """
- **Mistral AI API Key** — [YouTube Tutorial](https://youtu.be/NUCcUFwfhlA?si=iLrFFxVtcFUp657C)  
- **Google API Key (Gemini)** — [YouTube Tutorial](https://youtu.be/IHj7wF-8ry8?si=VKvhMM3pMeKwkXAv)
        """
    )


# Disimpan global agar helper bisa akses
MODEL_PREFERENCE = model_preference
ANSWER_LANGUAGE = answer_language

# MistralClient (legacy) — tidak untuk OCR di 0.4.2
mistral_client = None
if mistral_api_key:
    try:
        mistral_client = MistralClient(api_key=mistral_api_key)
        st.success("✅ Mistral API connected (legacy client 0.4.2)")
    except Exception as e:
        st.error(f"Failed to initialize Mistral client: {e}")

# Gemini untuk OCR & QnA
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        st.success("✅ Google API connected")
    except Exception as e:
        st.error(f"Failed to initialize Google API: {e}")

# --------------------- Helpers (Gemini OCR) -------------------------
def _is_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "429" in msg or "quota" in msg or "rate limit" in msg or "exceeded" in msg

def _get_model_candidates() -> list:
    if MODEL_PREFERENCE == "Flash only":
        return ["gemini-1.5-flash"]
    if MODEL_PREFERENCE == "Pro only":
        return ["gemini-1.5-pro"]
    return ["gemini-1.5-pro", "gemini-1.5-flash"]

def _generate_with_fallback(parts_or_prompt):
    last_error = None
    candidates = _get_model_candidates()
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(model_name=model_name)
            resp = model.generate_content(parts_or_prompt)
            text = getattr(resp, "text", "").strip()
            if text:
                if model_name != candidates[0]:
                    st.info(f"Using fallback model: {model_name}")
                return text
        except Exception as e:
            last_error = e
            if _is_quota_error(e):
                continue
            else:
                return f"Error generating response: {e}"
    if last_error and _is_quota_error(last_error):
        time.sleep(30)
        try:
            model = genai.GenerativeModel(model_name=candidates[-1])
            resp = model.generate_content(parts_or_prompt)
            return getattr(resp, "text", "").strip()
        except Exception as e2:
            return f"Error generating response: {e2}"
    return f"Error generating response: {last_error}"

def _truncate_context(text: str, max_chars: int = 20000) -> str:
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

# --------------------- Text chunking & Embeddings (RAG) ---------------------
EMBEDDING_MODEL = "models/text-embedding-004"

def _chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list:
    if not text:
        return []
    chunks = []
    start = 0
    end = max(chunk_size, 1)
    text_len = len(text)
    while start < text_len:
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= text_len:
            break
        start = max(end - overlap, 0)
        end = min(start + chunk_size, text_len)
    return chunks

def _extract_embedding_values(resp) -> list:
    emb = getattr(resp, "embedding", None)
    if emb is not None:
        values = getattr(emb, "values", emb)
        if isinstance(values, (list, tuple)):
            return list(values)
        if isinstance(values, dict) and "values" in values:
            return list(values["values"])
    if isinstance(resp, dict):
        emb = resp.get("embedding")
        if isinstance(emb, dict) and "values" in emb:
            return list(emb["values"])
        if isinstance(emb, (list, tuple)):
            return list(emb)
    return []

def _embed_text(text: str) -> np.ndarray:
    try:
        resp = genai.embed_content(model=EMBEDDING_MODEL, content=text)
        values = _extract_embedding_values(resp)
        if not values:
            return np.array([])
        return np.array(values, dtype=np.float32)
    except Exception:
        return np.array([])

def _build_retrieval_index(full_text: str):
    text_hash = str(hash(full_text))
    if (st.session_state.get("cached_text_hash") == text_hash and 
        st.session_state.get("retrieval_embeddings") is not None):
        return
    chunks = _chunk_text(full_text)
    embeddings = []

    max_chunks = min(len(chunks), 10)
    chunks = chunks[:max_chunks]
    
    for ch in chunks:
        try:
            vec = _embed_text(ch)
            if vec.size == 0:
                embeddings = []
                break
            embeddings.append(vec)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                st.warning("⚠️ Gemini embedding quota exceeded. Using full-context fallback.")
                embeddings = []
                break
            else:
                embeddings = []
                break
    
    if embeddings:
        matrix = np.vstack(embeddings)
        st.session_state.retrieval_chunks = chunks
        st.session_state.retrieval_embeddings = matrix
        st.session_state.retrieval_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        st.session_state.cached_text_hash = text_hash
    else:
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None
        st.session_state.cached_text_hash = None

def _retrieve_top_k(query: str, k: int = 5) -> list:
    if not query:
        return []
    chunks = st.session_state.get("retrieval_chunks")
    emb = st.session_state.get("retrieval_embeddings")
    norms = st.session_state.get("retrieval_norms")
    if not chunks or emb is None or norms is None:
        return []
    q_vec = _embed_text(query)
    if q_vec.size == 0:
        return []
    q_vec = q_vec.reshape(1, -1)
    q_norm = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-10
    sims = (emb @ q_vec.T) / (norms * q_norm)
    sims = sims.ravel()
    top_idx = np.argsort(-sims)[: max(1, k)]
    return [chunks[i] for i in top_idx]

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        return text
    except Exception:
        return ""

def gemini_ocr_image(image_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(image_bytes))
    prompt_doc = (
        "Convert this document image into clean Markdown. "
        "Preserve headings, lists, and tables (use Markdown tables). "
        "Maintain natural reading order."
    )
    text = _generate_with_fallback([prompt_doc, img])
    if not text or len(text.strip()) < 30:
        prompt_cap_id = (
            "Jelaskan gambar ini secara ringkas, jelas, dan akurat. "
            "Sebutkan objek utama, konteks, warna, teks (jika ada), dan hal penting lainnya."
        )
        prompt_cap_en = (
            "Describe this image concisely and accurately. "
            "Mention main objects, context, colors, any visible text, and other important details."
        )
        prompt_cap = prompt_cap_id if ANSWER_LANGUAGE == "Bahasa Indonesia" else prompt_cap_en
        text = _generate_with_fallback([prompt_cap, img])
    return text

def gemini_ocr_pdf(pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
    suffix = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        file_obj = genai.upload_file(
            path=tmp_path,
            mime_type="application/pdf",
            display_name=filename
        )
        try:
            for _ in range(30):
                f = genai.get_file(file_obj.name)
                state = getattr(getattr(f, "state", None), "name", getattr(f, "state", ""))
                if str(state).upper() == "ACTIVE":
                    break
                time.sleep(1)
            else:
                return "Error: File processing timed out. Please try again."
        except Exception:
            time.sleep(2)
        prompt = (
            "Extract the full content of this PDF as clean Markdown. "
            "Preserve headings and tables. If pages are scanned, perform OCR first."
        )
        return _generate_with_fallback([file_obj, prompt])
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def process_document_with_gemini(kind: str, name: str, data: bytes) -> str:
    if kind == "pdf":
        text = extract_text_from_pdf_bytes(data)
        if len(text) >= 200:
            return text
        return gemini_ocr_pdf(data, filename=name)
    else:  # image
        return gemini_ocr_image(data)

def answer_from_image(image_bytes: bytes, question: str) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if ANSWER_LANGUAGE == "Bahasa Indonesia":
            prompt = (
                "Anda adalah asisten analisis visual. Jawab pertanyaan pengguna hanya berdasarkan gambar ini.\n"
                "Jika informasi tidak terlihat pada gambar, katakan tidak ada.\n"
                f"Pertanyaan: {question}"
            )
        else:
            prompt = (
                "You are a visual analysis assistant. Answer the user's question based only on this image.\n"
                "If the information is not visible in the image, say so.\n"
                f"Question: {question}"
            )
        return _generate_with_fallback([prompt, img]) or "No response text."
    except Exception as e:
        return f"Error generating visual answer: {e}"

def generate_response(context: str, query: str) -> str:
    if not context or len(context) < 10:
        if "image_bytes" in st.session_state and isinstance(st.session_state.image_bytes, dict):
            for img_name, img_bytes in st.session_state.image_bytes.items():
                if any(word.lower() in img_name.lower() for word in query.lower().split()):
                    return answer_from_image(img_bytes, query)
            first_img = next(iter(st.session_state.image_bytes.values()))
            return answer_from_image(first_img, query)
        elif "image_bytes" in st.session_state:
            return answer_from_image(st.session_state.image_bytes, query)
        return "Error: Document context is empty or too short."
    
    try:
        retrieved_chunks = _retrieve_top_k(query, k=5)
        if retrieved_chunks:
            context_block = "\n\n---\n\n".join(retrieved_chunks)
        else:
            context_block = context
        
        doc_context = ""
        if st.session_state.get("documents") and len(st.session_state.documents) > 1:
            doc_names = [doc['name'] for doc in st.session_state.documents]
            doc_context = f"\n\nAvailable documents: {', '.join(doc_names)}"
        
        if ANSWER_LANGUAGE == "Bahasa Indonesia":
            prompt = f"""
Anda adalah asisten analisis dokumen. Gunakan konteks berikut untuk menjawab:

{context_block}{doc_context}

Pertanyaan pengguna:
{query}

Jawab dalam Bahasa Indonesia secara ringkas, jelas, dan akurat. Jika jawabannya tidak terdapat pada konteks, katakan: "Tidak ditemukan pada dokumen". Jika ada beberapa dokumen, sebutkan dari dokumen mana informasi berasal.
"""
        else:
            prompt = f"""
You are a document analysis assistant. Use only the context below to answer:

{context_block}{doc_context}

User question:
{query}

Respond in English concisely. If the answer is not in the context, say so. If there are multiple documents, mention which document the information comes from.
"""
        return _generate_with_fallback(prompt) or "No response text."
    except Exception as e:
        return f"Error generating response: {e}"

# --------------------- API Fallback & Caching ---------------------
def generate_with_mistral_fallback(prompt: str) -> str:
    if not mistral_client:
        return "Error: No Mistral API key configured for fallback."
    
    try:
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Mistral fallback: {e}"

def generate_response_with_fallback(context: str, query: str) -> str:
    try:
        return generate_response(context, query)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Gemini API quota exceeded. Falling back to Mistral API...")
            if ANSWER_LANGUAGE == "Bahasa Indonesia":
                mistral_prompt = f"""
Anda adalah asisten analisis dokumen. Gunakan konteks berikut untuk menjawab:

{context}

Pertanyaan pengguna:
{query}

Jawab dalam Bahasa Indonesia secara ringkas, jelas, dan akurat. Jika jawabannya tidak terdapat pada konteks, katakan: "Tidak ditemukan pada dokumen".
"""
            else:
                mistral_prompt = f"""
You are a document analysis assistant. Use only the context below to answer:

{context}

User question:
{query}

Respond in English concisely. If the answer is not in the context, say so.
"""
            return generate_with_mistral_fallback(mistral_prompt)
        else:
            return f"Error generating response: {e}"

# --------------------- Document Management Helpers ---------------------
def clear_all_document_state():
    st.session_state.documents = []
    st.session_state.ocr_content = None
    st.session_state.retrieval_chunks = None
    st.session_state.retrieval_embeddings = None
    st.session_state.retrieval_norms = None
    st.session_state.image_bytes = {}
    st.session_state.chat_history = []

def rebuild_document_content():
    if st.session_state.documents:
        all_content = [f"--- DOCUMENT: {d['name']} ---\n{d['content']}" for d in st.session_state.documents]
        st.session_state.ocr_content = "\n\n".join(all_content)
        _build_retrieval_index(st.session_state.ocr_content)
    else:
        st.session_state.ocr_content = None
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None


# ======================= DASHBOARD HELPERS =======================
_ID_STOPWORDS = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","ini","itu","ada","tidak","atau","karena","sebagai",
    "dalam","atas","oleh","sebuah","para","akan","juga","sudah","belum","saat","kami","kita","mereka","anda",
    "ia","dia","tersebut","rp","usd","pt","tbk","persero","co","ltd","inc"
}
_EN_STOPWORDS = {
    "the","and","of","to","in","for","on","at","by","with","from","as","is","are","was","were","be","been","a","an",
    "this","that","these","those","it","its","we","you","they","he","she","them","our","your","their",
    "or","not","but","if","then","so","than","such","per","vs"
}
STOPWORDS = _ID_STOPWORDS | _EN_STOPWORDS

def _tokenize(text: str) -> list:
    text = text.lower()
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    toks = [t for t in text.split() if t and t not in STOPWORDS and not t.isdigit() and len(t) > 2]
    return toks

def _compute_ocr_quality(text: str) -> float:
    if not text: return 0.0
    n = len(text)
    good = sum(ch.isalnum() or ch.isspace() or ch in ".,:;-%()[]|/+\n" for ch in text)
    bad = text.count("�")
    short_lines = sum(1 for ln in text.splitlines() if 0 < len(ln.strip()) < 3)
    score = (good / n) * 100.0
    score -= min(25, bad * 0.5)
    score -= min(15, short_lines * 0.2)
    return max(0.0, min(100.0, score))

def _structure_stats(md_text: str) -> dict:
    if not md_text:
        return {"text": 0, "tables": 0, "images": 0}
    lines = md_text.splitlines()
    table_lines = sum(1 for ln in lines if ln.strip().startswith("|") and ln.count("|") >= 2)
    image_tags = md_text.count("![") + md_text.lower().count("<img")
    text_chars = len(md_text)
    return {"text": text_chars, "tables": table_lines, "images": image_tags}

def _extract_sections(md_text: str):
    sections = []
    current_title = "Intro"
    current_buf = []
    for ln in md_text.splitlines():
        if ln.startswith("--- DOCUMENT:"):
            if current_buf:
                sections.append((current_title, "\n".join(current_buf).strip()))
                current_buf = []
            current_title = ln.replace("--- DOCUMENT:", "").strip()
        elif re.match(r"^\s{0,3}#{1,3}\s+\S", ln):
            if current_buf:
                sections.append((current_title, "\n".join(current_buf).strip()))
                current_buf = []
            current_title = re.sub(r"^\s{0,3}#{1,3}\s+", "", ln).strip()
        else:
            current_buf.append(ln)
    if current_buf:
        sections.append((current_title, "\n".join(current_buf).strip()))
    return sections[:12] if sections else [("All", md_text)]

def _missing_matrix(sections):
    cats = ["N/A/NA", "Empty Lines", "Dashes(-/—)", "Question(?)"]
    labels = [title[:40] + ("…" if len(title) > 40 else "") for title,_ in sections]
    matrix = []
    for _, txt in sections:
        lines = txt.splitlines()
        na = sum(bool(re.search(r"\b(n/?a|tidak tersedia|kosong)\b", ln, re.I)) for ln in lines)
        empty = sum(1 for ln in lines if not ln.strip())
        dashes = sum(ln.count("-") + ln.count("—") for ln in lines)
        ques = txt.count("?")
        matrix.append([na, empty, dashes, ques])
    return labels, cats, np.array(matrix).T

def _is_financial_report(text: str) -> bool:
    keys = ["revenue","pendapatan","penjualan","income","profit","laba","rugi",
            "neraca","balance sheet","arus kas","cash flow","laba kotor","gross","operating","net"]
    t = text.lower()
    return any(k in t for k in keys)

_num_pat = re.compile(r"([\-]?\d{1,3}(?:[\.,]\d{3})*(?:[\.,]\d{1,2})?)")

def _parse_number(s: str):
    s = s.strip()
    if not s: return None
    if "." in s and "," in s:
        if s.find(".") < s.find(","):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        if "," in s and "." not in s:
            s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        digits = re.sub(r"[^\d\-\.]", "", s)
        try:
            return float(digits)
        except Exception:
            return None

def _extract_financials(text: str):
    rev = {}
    prof = {}
    gross = {}
    op = {}
    net = {}
    assets = equity = curr_assets = curr_liab = debt = None

    lines = text.splitlines()
    for ln in lines:
        years = re.findall(r"\b(20\d{2}|19\d{2})\b", ln)
        if not years:
            continue
        y = int(years[0])

        if re.search(r"\b(revenue|pendapatan|penjualan)\b", ln, re.I):
            m = _num_pat.search(ln)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    rev[y] = val
        if re.search(r"\b(net income|laba bersih|laba/rugi bersih|profit|laba)\b", ln, re.I):
            m = _num_pat.search(ln)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    prof[y] = val
        if re.search(r"\b(gross profit|laba kotor)\b", ln, re.I):
            m = _num_pat.search(ln)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    gross[y] = val
        if re.search(r"\b(operating profit|laba usaha|laba operasi)\b", ln, re.I):
            m = _num_pat.search(ln)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    op[y] = val
        if re.search(r"\b(net profit|net income|laba bersih)\b", ln, re.I):
            m = _num_pat.search(ln)
            if m:
                val = _parse_number(m.group(1))
                if val is not None:
                    net[y] = val

        if assets is None and re.search(r"\b(total assets|jumlah aset)\b", ln, re.I):
            m = _num_pat.search(ln); assets = _parse_number(m.group(1)) if m else None
        if equity is None and re.search(r"\b(total equity|ekuitas)\b", ln, re.I):
            m = _num_pat.search(ln); equity = _parse_number(m.group(1)) if m else None
        if curr_assets is None and re.search(r"\b(current assets|aset lancar)\b", ln, re.I):
            m = _num_pat.search(ln); curr_assets = _parse_number(m.group(1)) if m else None
        if curr_liab is None and re.search(r"\b(current liab|liabilitas lancar|utang lancar)\b", ln, re.I):
            m = _num_pat.search(ln); curr_liab = _parse_number(m.group(1)) if m else None
        if debt is None and re.search(r"\b(total debt|utang|pinjaman)\b", ln, re.I):
            m = _num_pat.search(ln); debt = _parse_number(m.group(1)) if m else None

    return {
        "revenue_by_year": dict(sorted(rev.items())),
        "profit_by_year": dict(sorted(prof.items())),
        "gross_by_year": dict(sorted(gross.items())),
        "operating_by_year": dict(sorted(op.items())),
        "net_by_year": dict(sorted(net.items())),
        "assets": assets, "equity": equity,
        "current_assets": curr_assets, "current_liabilities": curr_liab, "debt": debt
    }

def _yoy_growth(series: dict) -> dict:
    ys = sorted(series.keys())
    growth = {}
    for i in range(1, len(ys)):
        y0, y1 = ys[i-1], ys[i]
        if series[y0] and series[y0] != 0:
            growth[y1] = (series[y1] - series[y0]) / abs(series[y0]) * 100.0
    return growth

def _readability_from_avg_sentence_len(text: str):
    if not text: return 0.0, 0.0
    sents = re.split(r"[.!?\n]+", text)
    sents = [s.strip() for s in sents if s.strip()]
    if not sents: return 0.0, 0.0
    words = sum(len(s.split()) for s in sents)
    avg = words / len(sents)
    score = 100.0 - ((avg - 8) / (40 - 8)) * 100.0
    score = max(0.0, min(100.0, score))
    return avg, score

def _ner_counts_with_gemini(text: str):
    if not google_api_key:
        return None
    try:
        prompt = """
Extract counts of named entities by coarse type from the text.
Only return a compact JSON like {"ORG": 12, "PERSON": 5, "LOC": 7}. Types limited to ORG, PERSON, LOC.
If none detected, use 0.
Text:
""" + _truncate_context(text, 20000)
        raw = _generate_with_fallback(prompt)
        data = json.loads(re.findall(r"\{.*\}", raw, re.S)[0])
        return {"ORG": int(data.get("ORG", 0)), "PERSON": int(data.get("PERSON", 0)), "LOC": int(data.get("LOC", 0))}
    except Exception:
        return None

def _ner_counts_naive(text: str):
    ORG = len(re.findall(r"\b(PT|Tbk|Persero|Inc\.?|Ltd\.?|LLC|Corp\.?)\b", text))
    PERSON = len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+(?: [A-Z][a-z]+)?\b", text))
    LOC = len(re.findall(r"\b(Jakarta|Bandung|Surabaya|Medan|Indonesia|Singapore|Malaysia|USA|Europe|Asia)\b", text))
    return {"ORG": ORG, "PERSON": PERSON, "LOC": LOC}


# --------------------- UI Layout -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload multiple documents (DOC, PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    url_input = st.text_input("Or enter a URL (web page or document):")
    st.session_state["url_input"] = url_input

    process_button = st.button("Process Documents")

    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "ocr_content" not in st.session_state:
        st.session_state.ocr_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --------------------- URL processing helper -----------------------
    def process_url_to_content(url: str) -> tuple:
        try:
            r = requests.get(
                url,
                timeout=30,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                },
                allow_redirects=True,
            )
            r.raise_for_status()
            content_type = r.headers.get("Content-Type", "").lower()
            clean_url = url.split("?")[0]
            ext = os.path.splitext(clean_url)[1].lower()
            data = r.content
            is_pdf_sig = data[:4] == b"%PDF"
            is_png_sig = data[:8] == b"\x89PNG\r\n\x1a\n"
            is_jpg_sig = data[:3] == b"\xff\xd8\xff"
            chosen_kind = None
            if is_pdf_sig or "pdf" in content_type or ext == ".pdf":
                chosen_kind = "pdf"
            elif is_png_sig or is_jpg_sig or any(img_ct in content_type for img_ct in ["image/png", "image/jpeg", "image/jpg"]) or ext in [".png", ".jpg", ".jpeg"]:
                chosen_kind = "image"
            if not chosen_kind and content_type.startswith("text/html"):
                html_text = r.text
                links = re.findall(r'href=[\"\']([^\"\']+\.(?:pdf|png|jpe?g))(?:[\#\?][^\"\']*)?[\"\']', html_text, flags=re.IGNORECASE)
                if links:
                    target_url = urljoin(url, links[0])
                    rr = requests.get(
                        target_url,
                        timeout=30,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                        },
                        allow_redirects=True,
                    )
                    rr.raise_for_status()
                    target_ct = rr.headers.get("Content-Type", "").lower()
                    tdata = rr.content
                    if tdata[:4] == b"%PDF" or "pdf" in target_ct:
                        chosen_kind = "pdf"
                        data = tdata
                        clean_url = target_url.split("?")[0]
                    elif tdata[:8] == b"\x89PNG\r\n\x1a\n" or tdata[:3] == b"\xff\xd8\xff" or any(ic in target_ct for ic in ["image/png", "image/jpeg", "image/jpg"]):
                        chosen_kind = "image"
                        data = tdata
                        clean_url = target_url.split("?")[0]
                        st.session_state.image_bytes = data
                if not chosen_kind:
                    stripped = re.sub(r"<script[\s\S]*?</script>", " ", html_text, flags=re.IGNORECASE)
                    stripped = re.sub(r"<style[\s\S]*?</style>", " ", stripped, flags=re.IGNORECASE)
                    text_only = re.sub(r"<[^>]+>", " ", stripped)
                    text_only = re.sub(r"\s+", " ", text_only).strip()
                    st.session_state.ocr_content = text_only
                    if st.session_state.ocr_content:
                        _build_retrieval_index(st.session_state.ocr_content)
                        return True, "Webpage processed as text."
                    return False, "No content extracted from webpage."
            if not chosen_kind:
                return False, f"Unsupported content type: {content_type or ext}"
            st.session_state.ocr_content = process_document_with_gemini(
                chosen_kind, os.path.basename(clean_url) or "download", data
            )
            if chosen_kind == "image":
                st.session_state.image_bytes = data
            if st.session_state.ocr_content:
                _build_retrieval_index(st.session_state.ocr_content)
                return True, "Document processed successfully!"
            return False, "No content extracted."
        except Exception as e:
            return False, f"Error processing document: {e}"

    if process_button:
        if not google_api_key:
            st.error("Please provide a valid Google API Key for OCR/processing.")
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_content = []
                for uploaded_file in uploaded_files:
                    try:
                        ext = os.path.splitext(uploaded_file.name)[1].lower()
                        kind = "pdf" if ext == ".pdf" else "image"
                        if kind == "image":
                            if "image_bytes" not in st.session_state:
                                st.session_state.image_bytes = {}
                            st.session_state.image_bytes[uploaded_file.name] = uploaded_file.getvalue()
                        
                        content = process_document_with_gemini(
                            kind, uploaded_file.name, uploaded_file.getvalue()
                        )
                        
                        if content:
                            doc_info = {
                                "name": uploaded_file.name,
                                "type": kind,
                                "content": content,
                                "size": len(uploaded_file.getvalue())
                            }
                            st.session_state.documents.append(doc_info)
                            all_content.append(f"--- DOCUMENT: {uploaded_file.name} ---\n{content}")
                            st.success(f"Document {uploaded_file.name} processed successfully!")
                        else:
                            st.warning(f"No content extracted from {uploaded_file.name}.")
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                if all_content:
                    st.session_state.ocr_content = "\n\n".join(all_content)
                    _build_retrieval_index(st.session_state.ocr_content)
                    st.success(f"All {len(uploaded_files)} documents processed and combined!")
        if url_input:
            with st.spinner("Downloading & processing from URL..."):
                success, msg = process_url_to_content(url_input)
                if success:
                    if st.session_state.get("ocr_content"):
                        url_doc_info = {
                            "name": f"URL: {url_input[:50]}...",
                            "type": "url",
                            "content": st.session_state.ocr_content,
                            "size": len(st.session_state.ocr_content)
                        }
                        st.session_state.documents.append(url_doc_info)
                    st.success(msg)
                else:
                    st.error(msg)
        if not uploaded_files and not url_input:
            st.warning("Please upload a document or provide a URL.")

with col2:
    st.header("Document Q&A")

    if st.session_state.documents:
        st.markdown(f"**{len(st.session_state.documents)} document(s) loaded.**")
        
        total_chars = sum(doc['size'] for doc in st.session_state.documents)
        total_mb = total_chars / (1024 * 1024)
        
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.metric("Total Documents", len(st.session_state.documents))
        with c2:
            st.metric("Total Content", f"{total_chars:,} chars")
        with c3:
            st.metric("Memory Usage", f"{total_mb:.1f} MB")
        
        if len(st.session_state.documents) > 10:
            st.warning("⚠️ Many documents loaded. Consider removing some to improve performance.")
        elif total_chars > 500000:
            st.warning("⚠️ Very large document collection. Consider removing some documents to avoid processing limits.")
        elif total_chars > 200000 and len(st.session_state.documents) > 3:
            st.warning("⚠️ Large document collection. Consider removing some documents to avoid processing limits.")
        
        with st.expander("📋 Document List"):
            for i, doc in enumerate(st.session_state.documents):
                cc1, cc2 = st.columns([4, 1])
                with cc1:
                    st.markdown(f"**{i+1}. {doc['name']}** ({doc['type']}) - {doc['size']} chars")
                with cc2:
                    if st.button(f"🗑️", key=f"del_{i}", help=f"Delete {doc['name']}"):
                        deleted_doc = st.session_state.documents.pop(i)
                        if deleted_doc['type'] == 'image' and 'image_bytes' in st.session_state:
                            if isinstance(st.session_state.image_bytes, dict):
                                st.session_state.image_bytes.pop(deleted_doc['name'], None)
                            else:
                                st.session_state.image_bytes = {}
                        rebuild_document_content()
                        st.rerun()
            
            if st.session_state.documents:
                if st.button("🔄 Reset All Documents", type="secondary"):
                    clear_all_document_state()
                    st.rerun()
        
        # ---------- Quick actions ----------
        if st.session_state.documents:
            st.markdown("**Quick Actions:**")
            q1, q2 = st.columns(2)
            with q1:
                if st.button("🗑️ Clear Chat", help="Clear chat history but keep documents"):
                    st.session_state.chat_history = []
                    st.rerun()
            with q2:
                if st.button("📊 Document Stats", help="Show dashboard with charts"):
                    st.session_state.show_stats = not st.session_state.get('show_stats', False)
                    st.rerun()

        # ---------- DASHBOARD ----------
        if st.session_state.get('show_stats', False):
            st.markdown("## 📊 Document Analytics Dashboard")

            # Data agregat per dokumen
            stats_data = []
            combined_texts = []
            struct_total = {"text":0, "tables":0, "images":0}
            for doc in st.session_state.documents:
                txt = doc['content'] or ""
                combined_texts.append(txt)
                ocr_q = _compute_ocr_quality(txt)
                words = len(_tokenize(txt))
                lines = len(txt.splitlines())
                stt = {
                    "Document": doc['name'],
                    "Type": doc['type'],
                    "Size (chars)": doc['size'],
                    "Words": words,
                    "Lines": lines,
                    "OCR Quality": round(ocr_q,1)
                }
                stats_data.append(stt)
                s = _structure_stats(txt)
                struct_total["text"] += s["text"]
                struct_total["tables"] += s["tables"]
                struct_total["images"] += s["images"]

            combined = "\n\n".join(combined_texts)

            st.dataframe(stats_data, use_container_width=True)

            # -------- 1) Document Health --------
            st.markdown("### Document Health")
            dh1, dh2 = st.columns([1,1])
            with dh1:
                ocr_score = _compute_ocr_quality(combined)
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=ocr_score,
                    number={'suffix': " /100"},
                    title={'text': "OCR Quality Score"},
                    gauge={'axis': {'range': [0,100]},
                           'bar': {'thickness': 0.3},
                           'steps': [
                               {'range': [0,50], 'color': "#fce4ec"},
                               {'range': [50,75], 'color': "#fff3e0"},
                               {'range': [75,100], 'color': "#e8f5e9"},
                           ]}
                ))
                st.plotly_chart(fig_g, use_container_width=True)

            with dh2:
                labels = ["Text", "Tables", "Images"]
                values = [max(1, struct_total["text"]), max(1, struct_total["tables"]), max(1, struct_total["images"])]
                fig_donut = px.pie(values=values, names=labels, hole=0.6, title="Document Structure Overview")
                st.plotly_chart(fig_donut, use_container_width=True)

            secs = _extract_sections(combined)
            sec_labels, cat_labels, mat = _missing_matrix(secs)
            fig_heat = px.imshow(mat, aspect="auto", color_continuous_scale="Viridis",
                                 labels=dict(x="Section", y="Missing Type", color="Count"),
                                 x=sec_labels, y=cat_labels)
            fig_heat.update_layout(title="Missing Data Heatmap")
            st.plotly_chart(fig_heat, use_container_width=True)

        # ---------- Chat history ----------
        for m in st.session_state.chat_history:
            role = "You" if m["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {m['content']}")

        # ---------- Q&A input (pakai form agar field auto kosong setelah submit) ----------
        with st.form("qa_form_docs", clear_on_submit=True):
            user_q = st.text_input(
                "Your question (can ask about specific documents or compare them):",
                key="qa_input"
            )
            submitted = st.form_submit_button("Ask")

        if submitted and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Generating response..."):
                if not google_api_key:
                    ans = "Please provide a valid Google API Key."
                else:
                    ans = generate_response_with_fallback(st.session_state.ocr_content, user_q)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()  # field sudah kosong otomatis oleh form

    else:
        st.info("No documents processed yet. You can either upload files or just type a URL below and press Ask.")
        with st.form("qa_form_nodocs", clear_on_submit=True):
            user_q = st.text_input(
                "Your question (can ask about specific documents or compare them):",
                key="qa_input"
            )
            submitted = st.form_submit_button("Ask")

        if submitted and user_q:
            url_candidate = st.session_state.get("url_input")
            if url_candidate and not st.session_state.get("ocr_content"):
                with st.spinner("Processing URL before answering..."):
                    success, msg = process_url_to_content(url_candidate)
                    if not success:
                        st.error(msg)
                        st.stop()
            if st.session_state.get("ocr_content"):
                st.session_state.chat_history.append({"role": "user", "content": user_q})
                with st.spinner("Generating response..."):
                    if not google_api_key:
                        ans = "Please provide a valid Google API Key."
                    else:
                        ans = generate_response_with_fallback(st.session_state.ocr_content, user_q)
                st.session_state.chat_history.append({"role": "assistant", "content": ans})
                st.rerun()  # field sudah kosong otomatis oleh form
            else:
                st.warning("Please provide a URL or upload a document first.")
# Tampilkan konten hasil OCR/ekstraksi
if st.session_state.get("documents"):
    with st.expander("📄 View All Document Contents"):
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"### {doc['name']} ({doc['type']})")
            st.markdown(doc['content'])
            st.markdown("---")
