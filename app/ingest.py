"""Turn uploaded files into text, then into retrieval-sized chunks.

Chunking choice: split on Markdown headings first, then group paragraphs up to
~MAX_WORDS. Each chunk is prefixed with its heading path ("Warranty > Deductible")
so a short paragraph like "A $100 deductible applies..." still carries the
context of which policy it belongs to. That noticeably improves retrieval for
questions that use the section's vocabulary rather than the paragraph's.
"""

import io
import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 140

SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class Chunk:
    section: str
    text: str  # includes the heading prefix; this is what gets embedded and shown


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Use .md, .txt, or .pdf.")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        if not text.strip():
            # Scanned PDFs have no text layer; OCR would be the next step here.
            raise ValueError("No extractable text found in PDF (it may be a scanned image).")
        return text
    return data.decode("utf-8", errors="replace")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Return (heading_path, body) pairs. Text before any heading gets the title or ''."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    body: list[str] = []

    def flush():
        content = "\n".join(body).strip()
        if content:
            sections.append((" > ".join(path), content))
        body.clear()

    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip()]
        else:
            body.append(line)
    flush()
    return sections


def chunk_text(text: str, max_words: int = MAX_WORDS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        group: list[str] = []
        words = 0
        for para in paragraphs:
            n = len(para.split())
            if group and words + n > max_words:
                chunks.append(_make_chunk(section, group))
                group, words = [], 0
            group.append(para)
            words += n
        if group:
            chunks.append(_make_chunk(section, group))
    return chunks


def _make_chunk(section: str, paragraphs: list[str]) -> Chunk:
    body = "\n\n".join(paragraphs)
    text = f"{section}\n{body}" if section else body
    return Chunk(section=section, text=text)
