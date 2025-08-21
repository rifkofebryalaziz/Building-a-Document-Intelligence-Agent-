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
                # coba model berikutnya segera
                continue
            else:
                return f"Error generating response: {e}"
    # Semua kandidat gagal; jika karena kuota, tunggu sebentar lalu coba terakhir
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
    """
    Split long text into overlapping chunks for retrieval.
    chunk_size and overlap are character-based for simplicity and speed.
    """
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
    """Make embedding response robust across SDK variations."""
    # Try attribute access first
    emb = getattr(resp, "embedding", None)
    if emb is not None:
        values = getattr(emb, "values", emb)
        if isinstance(values, (list, tuple)):
            return list(values)
        if isinstance(values, dict) and "values" in values:
            return list(values["values"])
    # Fallback to dict-like
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
    """
    Build an in-memory retrieval index (chunks + embeddings) and store in session.
    If embedding generation fails, the app will fall back to full-context QA.
    """
    # Check if we already have embeddings for this exact text (caching)
    text_hash = str(hash(full_text))
    if (st.session_state.get("cached_text_hash") == text_hash and 
        st.session_state.get("retrieval_embeddings") is not None):
        return  # Use cached embeddings
    
    chunks = _chunk_text(full_text)
    embeddings = []
    
    # Limit chunks to reduce API calls (max 10 chunks)
    max_chunks = min(len(chunks), 10)
    chunks = chunks[:max_chunks]
    
    for i, ch in enumerate(chunks):
        try:
            vec = _embed_text(ch)
            if vec.size == 0:
                embeddings = []  # invalidate index
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
        st.session_state.cached_text_hash = text_hash  # Cache the hash
    else:
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None
        st.session_state.cached_text_hash = None

def _retrieve_top_k(query: str, k: int = 5) -> list:
    """Return top-k most similar chunks to the query using cosine similarity."""
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
    """
    Ekstrak teks dari PDF non-scan via PyPDF2 (cepat & lokal).
    Jika kosong (kemungkinan scan), nanti fallback ke Gemini OCR.
    """
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
    """OCR gambar (PNG/JPG) ke Markdown via Gemini."""
    img = Image.open(io.BytesIO(image_bytes))
    # 1) Coba sebagai dokumen terlebih dahulu
    prompt_doc = (
        "Convert this document image into clean Markdown. "
        "Preserve headings, lists, and tables (use Markdown tables). "
        "Maintain natural reading order."
    )
    text = _generate_with_fallback([prompt_doc, img])
    if not text or len(text.strip()) < 30:
        # 2) Jika bukan dokumen, fallback: deskripsi/caption gambar
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
    """OCR + ekstraksi PDF (termasuk scan) ke Markdown via Gemini."""
    # Simpan sementara agar bisa di-upload ke Gemini
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
        # Pastikan file sudah siap diproses sebelum dipakai di generate_content
        try:
            for _ in range(30):  # tunggu hingga ~30 detik maksimal
                f = genai.get_file(file_obj.name)
                state = getattr(getattr(f, "state", None), "name", getattr(f, "state", ""))
                if str(state).upper() == "ACTIVE":
                    break
                time.sleep(1)
            else:
                return "Error: File processing timed out. Please try again."
        except Exception:
            # Jika pengecekan status gagal, beri sedikit jeda lalu lanjut mencoba
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
    """
    Router sederhana:
    - PDF: coba PyPDF2 dulu (cepat). Jika terlalu pendek/kosong → fallback Gemini OCR.
    - Image: langsung pakai Gemini OCR.
    """
    if kind == "pdf":
        text = extract_text_from_pdf_bytes(data)
        # Jika hasil terlalu pendek (kemungkinan scan), fallback OCR via Gemini
        if len(text) >= 200:
            return text
        return gemini_ocr_pdf(data, filename=name)
    else:  # image
        return gemini_ocr_image(data)

def answer_from_image(image_bytes: bytes, question: str) -> str:
    """Jawab pertanyaan langsung dari gambar (visual Q&A) dengan Gemini."""
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
    """Jawab pertanyaan berdasarkan dokumen menggunakan RAG bila tersedia, tanpa truncation artifisial."""
    if not context or len(context) < 10:
        # Coba jawab langsung dari gambar jika tersedia
        if "image_bytes" in st.session_state and isinstance(st.session_state.image_bytes, dict):
            # Multiple images - try to find relevant one based on query
            for img_name, img_bytes in st.session_state.image_bytes.items():
                if any(word.lower() in img_name.lower() for word in query.lower().split()):
                    return answer_from_image(img_bytes, query)
            # If no specific match, use first image
            first_img = next(iter(st.session_state.image_bytes.values()))
            return answer_from_image(first_img, query)
        elif "image_bytes" in st.session_state:
            # Single image (backward compatibility)
            return answer_from_image(st.session_state.image_bytes, query)
        return "Error: Document context is empty or too short."
    
    try:
        retrieved_chunks = _retrieve_top_k(query, k=5)
        if retrieved_chunks:
            context_block = "\n\n---\n\n".join(retrieved_chunks)
        else:
            # Fallback: gunakan seluruh konteks tanpa pemotongan manual
            context_block = context
        
        # Add document context if multiple documents
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
    """Fallback to Mistral API when Gemini hits quota limits."""
    if not mistral_client:
        return "Error: No Mistral API key configured for fallback."
    
    try:
        # Use Mistral's chat completion
        response = mistral_client.chat(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error with Mistral fallback: {e}"

def generate_response_with_fallback(context: str, query: str) -> str:
    """Generate response with Gemini fallback to Mistral if quota exceeded."""
    try:
        # Try Gemini first
        return generate_response(context, query)
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            st.warning("⚠️ Gemini API quota exceeded. Falling back to Mistral API...")
            # Fallback to Mistral
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
    """Clear all document-related session state variables."""
    st.session_state.documents = []
    st.session_state.ocr_content = None
    st.session_state.retrieval_chunks = None
    st.session_state.retrieval_embeddings = None
    st.session_state.retrieval_norms = None
    st.session_state.image_bytes = {}
    st.session_state.chat_history = []

def rebuild_document_content():
    """Rebuild combined content and RAG index from remaining documents."""
    if st.session_state.documents:
        all_content = [f"--- DOCUMENT: {d['name']} ---\n{d['content']}" for d in st.session_state.documents]
        st.session_state.ocr_content = "\n\n".join(all_content)
        _build_retrieval_index(st.session_state.ocr_content)
    else:
        st.session_state.ocr_content = None
        st.session_state.retrieval_chunks = None
        st.session_state.retrieval_embeddings = None
        st.session_state.retrieval_norms = None

# --------------------- UI Layout -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload multiple documents (PDF, PNG, JPG, JPEG)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    url_input = st.text_input("Or enter a URL (web page or document):")
    # Persist last URL for use from Q&A panel
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
        """
        Download a URL and convert it into document content in session_state.
        Returns (success: bool, message: str)
        """
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
            # Signature sniffing
            is_pdf_sig = data[:4] == b"%PDF"
            is_png_sig = data[:8] == b"\x89PNG\r\n\x1a\n"
            is_jpg_sig = data[:3] == b"\xff\xd8\xff"
            chosen_kind = None
            if is_pdf_sig or "pdf" in content_type or ext == ".pdf":
                chosen_kind = "pdf"
            elif is_png_sig or is_jpg_sig or any(img_ct in content_type for img_ct in ["image/png", "image/jpeg", "image/jpg"]) or ext in [".png", ".jpg", ".jpeg"]:
                chosen_kind = "image"
            # HTML page → try to find asset link
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
                        # Simpan untuk visual Q&A jika dibutuhkan
                        st.session_state.image_bytes = data
                # Still HTML → extract visible text
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
        # Karena OCR memakai Gemini, butuh Google API key
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
                            # Simpan image bytes untuk visual Q&A fallback
                            if "image_bytes" not in st.session_state:
                                st.session_state.image_bytes = {}
                            st.session_state.image_bytes[uploaded_file.name] = uploaded_file.getvalue()
                        
                        content = process_document_with_gemini(
                            kind, uploaded_file.name, uploaded_file.getvalue()
                        )
                        
                        if content:
                            # Store individual document
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
                
                # Combine all content for unified processing
                if all_content:
                    st.session_state.ocr_content = "\n\n".join(all_content)
                    # Build retrieval index for all documents combined
                    _build_retrieval_index(st.session_state.ocr_content)
                    st.success(f"All {len(uploaded_files)} documents processed and combined!")
        if url_input:
            with st.spinner("Downloading & processing from URL..."):
                success, msg = process_url_to_content(url_input)
                if success:
                    # Add URL content to documents list if it was processed
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
        
        # Show memory usage and limits
        total_chars = sum(doc['size'] for doc in st.session_state.documents)
        total_mb = total_chars / (1024 * 1024)  # Rough estimate: 1 char ≈ 1 byte
        
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.metric("Total Documents", len(st.session_state.documents))
        with col2:
            st.metric("Total Content", f"{total_chars:,} chars")
        with col3:
            st.metric("Memory Usage", f"{total_mb:.1f} MB")
        
        # Warning if approaching limits
        if len(st.session_state.documents) > 10:  # Warning for many documents
            st.warning("⚠️ Many documents loaded. Consider removing some to improve performance.")
        elif total_chars > 500000:  # Warning for very large content (500k chars ≈ 500KB)
            st.warning("⚠️ Very large document collection. Consider removing some documents to avoid processing limits.")
        elif total_chars > 200000 and len(st.session_state.documents) > 3:  # Warning for medium-large multi-doc
            st.warning("⚠️ Large document collection. Consider removing some documents to avoid processing limits.")
        
        # Show document summary
        with st.expander("📋 Document List"):
            for i, doc in enumerate(st.session_state.documents):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{i+1}. {doc['name']}** ({doc['type']}) - {doc['size']} chars")
                with col2:
                    if st.button(f"🗑️", key=f"del_{i}", help=f"Delete {doc['name']}"):
                        # Remove document from list
                        deleted_doc = st.session_state.documents.pop(i)
                        # Remove from image_bytes if it was an image
                        if deleted_doc['type'] == 'image' and 'image_bytes' in st.session_state:
                            if isinstance(st.session_state.image_bytes, dict):
                                st.session_state.image_bytes.pop(deleted_doc['name'], None)
                            else:
                                st.session_state.image_bytes = {}
                        
                        # Clear all content and rebuild from remaining documents
                        rebuild_document_content()
                        st.rerun()
            
            # Add reset button if there are documents
            if st.session_state.documents:
                if st.button("🔄 Reset All Documents", type="secondary"):
                    clear_all_document_state()
                    st.rerun()
        
        # Quick actions
        if st.session_state.documents:
            st.markdown("**Quick Actions:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("🗑️ Clear Chat", help="Clear chat history but keep documents"):
                    st.session_state.chat_history = []
                    st.rerun()
            with col2:
                if st.button("🧹 Clear Small Docs", help="Remove documents under 1000 chars"):
                    st.session_state.documents = [d for d in st.session_state.documents if d['size'] >= 1000]
                    if st.session_state.documents:
                        rebuild_document_content()
                    else:
                        clear_all_document_state()
                    st.rerun()
            with col3:
                if st.button("📊 Document Stats", help="Show detailed document statistics"):
                    st.session_state.show_stats = not st.session_state.get('show_stats', False)
                    st.rerun()
            with col4:
                if st.button("🐛 Debug Mode", help="Toggle debug information"):
                    st.session_state.show_debug = not st.session_state.get('show_debug', False)
                    st.rerun()
        
        # Show detailed statistics if requested
        if st.session_state.get('show_stats', False):
            st.markdown("**📊 Detailed Document Statistics:**")
            stats_data = []
            for doc in st.session_state.documents:
                stats_data.append({
                    "Document": doc['name'],
                    "Type": doc['type'],
                    "Size (chars)": doc['size'],
                    "Size (KB)": f"{doc['size']/1024:.1f}",
                    "Words": len(doc['content'].split()),
                    "Lines": len(doc['content'].split('\n'))
                })
            
            # Display as a nice table
            for stat in stats_data:
                st.markdown(f"**{stat['Document']}**")
                st.markdown(f"  - Type: {stat['Type']} | Size: {stat['Size (chars)']:,} chars ({stat['Size (KB)']} KB)")
                st.markdown(f"  - Words: {stat['Words']:,} | Lines: {stat['Lines']:,}")
                st.markdown("---")
        
        # Debug section (only show if needed)
        if st.session_state.get('show_debug', False):
            with st.expander("🐛 Debug Info"):
                st.markdown("**Current Session State:**")
                st.markdown(f"- Documents count: {len(st.session_state.get('documents', []))}")
                st.markdown(f"- OCR content length: {len(st.session_state.get('ocr_content', '') or '')}")
                st.markdown(f"- Retrieval chunks: {len(st.session_state.get('retrieval_chunks', []) or [])}")
                st.markdown(f"- Image bytes keys: {list(st.session_state.get('image_bytes', {}).keys())}")
                st.markdown(f"- Chat history length: {len(st.session_state.get('chat_history', []))}")
                
                if st.button("🧹 Force Cleanup"):
                    clear_all_document_state()
                    st.rerun()

        for m in st.session_state.chat_history:
            role = "You" if m["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {m['content']}")

        user_q = st.text_input("Your question (can ask about specific documents or compare them):")
        if st.button("Ask") and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Generating response..."):
                if not google_api_key:
                    ans = "Please provide a valid Google API Key."
                else:
                    ans = generate_response_with_fallback(st.session_state.ocr_content, user_q)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
    else:
        st.info("No documents processed yet. You can either upload files or just type a URL below and press Ask.")
        user_q = st.text_input("Your question (you can also paste a URL first):")
        if st.button("Ask") and user_q:
            # If there is a URL in session and no content yet, try to process it first
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
                st.rerun()
            else:
                st.warning("Please provide a URL or upload a document first.")

# Tampilkan konten hasil OCR/ekstraksi
if st.session_state.get("documents"):
    with st.expander("📄 View All Document Contents"):
        for i, doc in enumerate(st.session_state.documents):
            st.markdown(f"### {doc['name']} ({doc['type']})")
            st.markdown(doc['content'])
            st.markdown("---")