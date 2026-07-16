"""PDF-only master resume evidence and explicitly confirmed semantic text."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class MasterResumeSource(BaseModel):
    """Immutable identity and extracted text for one master resume PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_path: str
    sha256: str
    extracted_text: str

    @field_validator("source_path")
    @classmethod
    def _source_path_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_path must not be blank")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_must_be_hex(cls, value: str) -> str:
        normalized = value.casefold()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        return normalized

    @field_validator("extracted_text")
    @classmethod
    def _extracted_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("master resume PDF contains no extractable text")
        return value


class MasterResumeDocument(BaseModel):
    """Reusable semantic text explicitly checked by the user against its PDF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_sha256: str
    semantic_text: str
    confirmed_by_user: bool = False

    @field_validator("source_sha256")
    @classmethod
    def _source_sha256_must_be_hex(cls, value: str) -> str:
        normalized = value.casefold()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
        return normalized

    @field_validator("semantic_text")
    @classmethod
    def _semantic_text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("semantic_text must not be blank")
        return value


def load_resume_pdf(resume_path: str | Path) -> MasterResumeSource:
    """Extract one PDF without inferring structure or accepting empty text."""
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Resume path is not a file: {path}")
    if path.suffix.casefold() != ".pdf":
        raise ValueError("Unsupported resume format: master resume must be a PDF")

    resolved = path.resolve()
    extracted_text = _extract_pdf_text(resolved)
    if not extracted_text.strip():
        raise ValueError("Master resume PDF contains no extractable text")
    return MasterResumeSource(
        source_path=str(resolved),
        sha256=_sha256_file(resolved),
        extracted_text=extracted_text,
    )


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page for page in pages if page.strip()).strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["MasterResumeDocument", "MasterResumeSource", "load_resume_pdf"]
