"""Render a semantic resume document with the one deterministic Typst template."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from .models import ResumeDocument, RichText


class ResumeRenderError(RuntimeError):
    """The document could not be rendered without compromising the artifact."""


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    path: Path
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ResumePage:
    page_number: int
    image: ArtifactFile
    extracted_text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "page_number": self.page_number,
            "image": self.image.as_dict(),
            "extracted_text": self.extracted_text,
        }


@dataclass(frozen=True, slots=True)
class ResumeRenderResult:
    generated_on: date
    document_sha256: str
    template_sha256: str
    portrait_sha256: str | None
    render_key: str
    visible_characters: int
    pdf: ArtifactFile
    pages: tuple[ResumePage, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_on": self.generated_on.isoformat(),
            "document_sha256": self.document_sha256,
            "template_sha256": self.template_sha256,
            "portrait_sha256": self.portrait_sha256,
            "render_key": self.render_key,
            "visible_characters": self.visible_characters,
            "pdf": self.pdf.as_dict(),
            "pages": [page.as_dict() for page in self.pages],
        }


_SAFE_FILENAME_RE = re.compile(r"[^\w-]+", re.UNICODE)


def resume_template_path() -> Path:
    return Path(__file__).parent.parent / "resume_templates" / "engineering_resume.typ"


def resume_template_sha256() -> str:
    return _sha256_file(resume_template_path())


def render_resume(
    *,
    document: ResumeDocument,
    company: str,
    role: str,
    output_dir: str | Path,
    portrait_source_pdf: str | Path | None = None,
) -> ResumeRenderResult:
    """Render one content-addressed PDF and one PNG for every actual page."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"resume artifact output is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    canonical_document = document.model_dump_json(exclude_none=True)
    document_sha = hashlib.sha256(canonical_document.encode("utf-8")).hexdigest()
    template_sha = resume_template_sha256()
    generated_on = _server_date()

    typst = _find_typst()
    if not typst:
        raise ResumeRenderError("Typst is required to generate a resume PDF")

    with tempfile.TemporaryDirectory(prefix=".resume-render-", dir=destination) as raw_tmp:
        temporary = Path(raw_tmp)
        portrait = _extract_portrait(portrait_source_pdf, temporary / "assets")
        portrait_sha = _sha256_file(portrait) if portrait is not None else None
        render_key = _render_identity(document_sha, template_sha, portrait_sha)
        stem = _artifact_stem(company, role, generated_on, render_key)
        pdf_path = destination / f"{stem}.pdf"
        pages_dir = destination / "pages" / render_key[:16]
        relative_portrait = portrait.relative_to(temporary) if portrait is not None else None
        source = _render_typst_source(document, portrait_path=relative_portrait)
        if pdf_path.exists():
            _validate_existing_pdf(pdf_path)
        else:
            source_path = temporary / "resume.typ"
            source_path.write_text(source, encoding="utf-8")
            compiled_path = temporary / "resume.pdf"
            _compile_typst_pdf(source_path, compiled_path, typst)
            _validate_pdf_text_layout(compiled_path)
            try:
                os.link(compiled_path, pdf_path)
            except FileExistsError:
                _validate_existing_pdf(pdf_path)

        page_text = _extract_page_text(pdf_path)
        pages = _existing_page_images(pages_dir, page_text)
        if pages is None:
            if pages_dir.exists():
                raise ResumeRenderError(
                    "content-addressed resume page images are incomplete or corrupt"
                )
            staged_pages_dir = temporary / "rendered-pages"
            _render_page_images(pdf_path, staged_pages_dir, page_text)
            pages_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(staged_pages_dir, pages_dir)
            except OSError:
                pages = _existing_page_images(pages_dir, page_text)
                if pages is None:
                    raise
            else:
                pages = _existing_page_images(pages_dir, page_text)
                if pages is None:
                    raise ResumeRenderError("rendered resume page images could not be installed")

    assert pages is not None
    if not pages:
        raise ResumeRenderError("the rendered resume contains no pages")

    visible = len("".join(document_text(document).split()))
    return ResumeRenderResult(
        generated_on=generated_on,
        document_sha256=document_sha,
        template_sha256=template_sha,
        portrait_sha256=portrait_sha,
        render_key=render_key,
        visible_characters=visible,
        pdf=_artifact_file(pdf_path),
        pages=pages,
    )


def document_text(document: ResumeDocument) -> str:
    """Return the exact visible semantic text without adding inferred labels."""
    parts = [document.header.name.plain_text]
    parts.extend(line.plain_text for line in document.header.lines)
    for section in document.sections:
        parts.append(section.title.plain_text)
        for entry in section.entries:
            for row in entry.rows:
                parts.append(row.left.plain_text)
                if row.right is not None:
                    parts.append(row.right.plain_text)
            parts.extend(block.content.plain_text for block in entry.blocks)
    return "\n".join(parts)


def _render_typst_source(
    document: ResumeDocument,
    *,
    portrait_path: Path | None,
) -> str:
    template = resume_template_path().read_text(encoding="utf-8").rstrip()
    lines = [template]
    header_lines = ",".join(_typst_rich(line) for line in document.header.lines)
    portrait_arg = _typst_string(portrait_path.as_posix()) if portrait_path is not None else "none"
    lines.append(
        "#resume-header("
        f"{_typst_rich(document.header.name)}, "
        f"lines: ({header_lines}{',' if header_lines else ''}), "
        f"portrait: {portrait_arg})"
    )

    for section in document.sections:
        if section.page_break_before:
            lines.append("#pagebreak(weak: false)")
        lines.append(
            f"#resume-section({_typst_rich(section.title)}, "
            f"keep-with-next: {_typst_bool(section.keep_title_with_first_entry)})"
        )
        for entry_index, entry in enumerate(section.entries):
            if entry.page_break_before:
                lines.append("#pagebreak(weak: false)")
            row_values = ",".join(
                "("
                f"{_typst_rich(row.left)},"
                f"{_typst_rich(row.right) if row.right is not None else 'none'}"
                ")"
                for row in entry.rows
            )
            body = "".join(
                "#"
                + ("resume-bullet" if block.kind == "bullet" else "resume-paragraph")
                + f"({_typst_rich(block.content)})"
                for block in entry.blocks
            )
            lines.append(
                "#resume-entry("
                f"[{body}], "
                f"rows: ({row_values}{',' if row_values else ''}), "
                "keep-with-next: "
                f"{_typst_bool(entry.keep_header_with_first_block)}, "
                "divider-before: "
                f"{_typst_bool(entry.divider_before and entry_index > 0)})"
            )
    return "\n".join(lines) + "\n"


def _typst_rich(value: RichText) -> str:
    atoms: list[str] = []
    for span in value.spans:
        atom = f"#text({_typst_string(span.text)})"
        if span.emphasis:
            atom = f"#strong[{atom}]"
        atoms.append(atom)
    return "[" + "".join(atoms) + "]"


def _typst_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _typst_bool(value: bool) -> str:
    return "true" if value else "false"


def _extract_portrait(source_pdf: str | Path | None, assets_dir: Path) -> Path | None:
    """Extract the portrait only when the caller explicitly supplies its source PDF."""
    if source_pdf is None:
        return None
    source = Path(source_pdf).expanduser().resolve()
    pdfimages = shutil.which("pdfimages")
    if source.suffix.casefold() != ".pdf" or not source.is_file() or not pdfimages:
        return None
    assets_dir.mkdir(parents=True, exist_ok=True)
    prefix = assets_dir / "portrait"
    result = subprocess.run(
        [pdfimages, "-f", "1", "-l", "1", "-j", str(source), str(prefix)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        return None
    candidates = [
        path
        for path in assets_dir.glob("portrait-*")
        if path.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ]
    return max(candidates, key=lambda path: path.stat().st_size) if candidates else None


def _compile_typst_pdf(source_path: Path, pdf_path: Path, typst: str) -> None:
    command = [typst, "compile", "--root", str(source_path.parent)]
    for font_path in _typst_font_paths():
        command.extend(("--font-path", str(font_path)))
    command.extend(("--pdf-standard", "a-2u", str(source_path), str(pdf_path)))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResumeRenderError(f"Typst could not compile the resume: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Typst error").strip()
        raise ResumeRenderError(f"Typst compile failed: {detail[:800]}")
    warnings = (result.stderr or "").strip()
    if "unknown font family" in warnings.casefold():
        raise ResumeRenderError(f"Typst could not load the configured fonts: {warnings[:800]}")
    if not pdf_path.is_file() or pdf_path.stat().st_size < 100:
        raise ResumeRenderError("Typst did not create a valid PDF")
    if not pdf_path.read_bytes().startswith(b"%PDF-"):
        raise ResumeRenderError("Typst output is not a PDF")


def _render_page_images(
    pdf_path: Path,
    pages_dir: Path,
    page_text: list[str],
) -> tuple[ResumePage, ...]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        raise ResumeRenderError("pdftoppm is required to produce visual review pages")
    pages_dir.mkdir(parents=True, exist_ok=True)
    prefix = pages_dir / "page"
    for stale in pages_dir.glob("page-*.png"):
        stale.unlink()
    result = subprocess.run(
        [pdftoppm, "-png", "-r", "144", str(pdf_path), str(prefix)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Poppler error").strip()
        raise ResumeRenderError(f"PDF page rendering failed: {detail[:500]}")
    images = sorted(pages_dir.glob("page-*.png"), key=_page_number_from_path)
    pages: list[ResumePage] = []
    for image_path in images:
        page_number = _page_number_from_path(image_path)
        data = image_path.read_bytes()
        if len(data) < 100 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ResumeRenderError(f"page {page_number} is not a valid PNG")
        extracted = page_text[page_number - 1] if page_number <= len(page_text) else ""
        pages.append(
            ResumePage(
                page_number=page_number,
                image=_artifact_file(image_path),
                extracted_text=extracted,
            )
        )
    return tuple(pages)


def _existing_page_images(
    pages_dir: Path,
    page_text: list[str],
) -> tuple[ResumePage, ...] | None:
    if not pages_dir.is_dir():
        return None
    try:
        images = sorted(pages_dir.glob("page-*.png"), key=_page_number_from_path)
    except ResumeRenderError:
        return None
    if len(images) != len(page_text) or [_page_number_from_path(path) for path in images] != list(
        range(1, len(page_text) + 1)
    ):
        return None
    pages: list[ResumePage] = []
    for page_number, (image_path, extracted_text) in enumerate(
        zip(images, page_text, strict=True),
        start=1,
    ):
        data = image_path.read_bytes()
        if len(data) < 100 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        pages.append(
            ResumePage(
                page_number=page_number,
                image=_artifact_file(image_path),
                extracted_text=extracted_text,
            )
        )
    return tuple(pages)


def _extract_page_text(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def _validate_existing_pdf(pdf_path: Path) -> None:
    data = pdf_path.read_bytes()
    if len(data) < 100 or not data.startswith(b"%PDF-"):
        raise ResumeRenderError("content-addressed resume PDF is incomplete or corrupt")
    _validate_pdf_text_layout(pdf_path)


def _page_number_from_path(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    if not match:
        raise ResumeRenderError(f"unexpected page image name: {path.name}")
    return int(match.group(1))


def _validate_pdf_text_layout(pdf_path: Path) -> None:
    pdftohtml = shutil.which("pdftohtml")
    if not pdftohtml:
        return
    result = subprocess.run(
        [pdftohtml, "-xml", "-hidden", "-i", "-stdout", str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Poppler error").strip()
        raise ResumeRenderError(f"PDF layout inspection failed: {detail[:500]}")
    overlaps = _overlapping_text_lines(result.stdout)
    if overlaps:
        page, first_top, first_bottom, second_top, second_bottom = overlaps[0]
        raise ResumeRenderError(
            "PDF text lines overlap: "
            f"page {page}, {first_top}-{first_bottom} intersects "
            f"{second_top}-{second_bottom}"
        )


def _overlapping_text_lines(pdf_xml: str) -> list[tuple[int, int, int, int, int]]:
    root = ET.fromstring(pdf_xml)
    overlaps: list[tuple[int, int, int, int, int]] = []
    for page in root.findall("page"):
        fragments: list[tuple[int, int]] = []
        for node in page.findall("text"):
            if "".join(node.itertext()).strip():
                fragments.append((int(node.attrib["top"]), int(node.attrib["height"])))
        fragments.sort()
        lines: list[dict[str, int]] = []
        for top, height in fragments:
            line = next(
                (
                    candidate
                    for candidate in reversed(lines[-6:])
                    if abs(candidate["anchor"] - top) <= 4
                ),
                None,
            )
            if line is None:
                lines.append({"anchor": top, "top": top, "bottom": top + height})
            else:
                line["top"] = min(line["top"], top)
                line["bottom"] = max(line["bottom"], top + height)
        lines.sort(key=lambda line: line["top"])
        page_number = int(page.attrib.get("number", "0"))
        for first, second in itertools.pairwise(lines):
            # Poppler rounds text boxes to integer coordinates. Adjacent lines
            # can therefore share an edge without occupying the same area.
            if second["top"] < first["bottom"]:
                overlaps.append(
                    (
                        page_number,
                        first["top"],
                        first["bottom"],
                        second["top"],
                        second["bottom"],
                    )
                )
    return overlaps


def _find_typst() -> str | None:
    for candidate in (
        "/opt/homebrew/bin/typst",
        "/usr/local/bin/typst",
        shutil.which("typst"),
    ):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _typst_font_paths() -> tuple[Path, ...]:
    candidates = (Path("/Applications/Microsoft Word.app/Contents/Resources/DFonts"),)
    return tuple(path for path in candidates if path.is_dir())


def _server_date() -> date:
    return datetime.now().astimezone().date()


def _render_identity(
    document_sha256: str,
    template_sha256: str,
    portrait_sha256: str | None,
) -> str:
    payload = json.dumps(
        {
            "document_sha256": document_sha256,
            "portrait_sha256": portrait_sha256,
            "template_sha256": template_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _artifact_stem(company: str, role: str, generated_on: date, render_key: str) -> str:
    return (
        f"简历_{_filename_component(company, 'company')}_"
        f"{_filename_component(role, 'role')}_{generated_on.isoformat()}_{render_key[:10]}"
    )


def _filename_component(value: str, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    cleaned = re.sub(r"_+", "_", _SAFE_FILENAME_RE.sub("_", normalized)).strip("_-")
    return cleaned[:48].rstrip("_-") or fallback


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file(path: Path) -> ArtifactFile:
    return ArtifactFile(path=path, sha256=_sha256_file(path), size_bytes=path.stat().st_size)


__all__ = [
    "ArtifactFile",
    "ResumePage",
    "ResumeRenderError",
    "ResumeRenderResult",
    "document_text",
    "render_resume",
    "resume_template_path",
    "resume_template_sha256",
]
