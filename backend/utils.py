from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


# --- Cleaning & chunking ---


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_tokens: int = 800, overlap_tokens: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap_tokens)
    return chunks


def compute_chunk_id(doc_id: str, index: int) -> str:
    return hashlib.md5(f"{doc_id}:{index}".encode("utf-8")).hexdigest()


# --- Fetch helpers ---


def fetch_remote(url: str, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def read_pdf_bytes(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    texts: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text:
            texts.append(page_text)
    return normalize_whitespace("\n".join(texts))


def read_html_bytes(data: bytes) -> str:
    soup = BeautifulSoup(data, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(" ")
    return normalize_whitespace(text)


def read_text_file(path: Path) -> str:
    return normalize_whitespace(path.read_text(encoding="utf-8", errors="ignore"))


def read_csv_records(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def resolve_source_to_text(path_or_url: str) -> Tuple[str, str]:
    """Return (doc_id, text) for a given local path or remote URL.

    - For CSV/TXT: load textual content (CSV content concatenated)
    - For PDF/HTML: parse bytes
    doc_id is a stable key derived from the basename or URL path.
    """
    if re.match(r"^https?://", path_or_url):
        data = fetch_remote(path_or_url)
        lower = path_or_url.lower()
        if lower.endswith(".pdf"):
            text = read_pdf_bytes(data)
        else:
            text = read_html_bytes(data)
        doc_id = os.path.basename(path_or_url.split("?")[0]) or hashlib.md5(path_or_url.encode()).hexdigest()
        return doc_id, text

    path = Path(path_or_url)
    if not path.exists():
        raise FileNotFoundError(f"Source not found: {path_or_url}")
    lower = path.name.lower()
    if lower.endswith(".pdf"):
        text = read_pdf_bytes(path.read_bytes())
    elif lower.endswith(".html") or lower.endswith(".htm"):
        text = read_html_bytes(path.read_bytes())
    elif lower.endswith(".csv"):
        df = read_csv_records(path)
        text = normalize_whitespace("\n".join(df.astype(str).agg(" ".join, axis=1).tolist()))
    else:
        text = read_text_file(path)
    doc_id = path.stem
    return doc_id, text



