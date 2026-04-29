"""Resume PDF → UserProfile.

W1 only does text extraction. The LLM-driven structured parse (filling out
`education`, `experience`, `skills`, `preferences`) lands in W2 once the LLM
client and the `parse_resume` SKILL are wired in.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader

from .schema import UserProfile


def load_resume_pdf(pdf_path: str | Path) -> UserProfile:
    """Read a PDF and return a UserProfile with `raw_resume_text` populated.

    Page text is extracted with pypdf and joined with double newlines. Empty
    pages contribute nothing. Whitespace is normalized at the boundaries only —
    inner formatting is preserved so the W2 LLM parser sees the original layout.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Resume PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = [page.extract_text() or "" for page in reader.pages]
    raw_text = "\n\n".join(p for p in pages if p.strip()).strip()
    return UserProfile(raw_resume_text=raw_text, source_pdf=str(pdf_path.resolve()))
