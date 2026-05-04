"""W13.9 — docx → HTML preview for in-browser display.

The W12-fix-c ``docx_tailor`` produces a tailored .docx that preserves Word
formatting end-to-end. But until W13.9 the user could only **download**
the .docx — no way to see it before opening Word.

This module converts a .docx to a self-contained HTML page using
``mammoth``, which is a pure-Python lib that walks docx XML and emits
HTML+inline CSS approximating Word's rendering. Pixel-perfect it isn't,
but for "does my tailored resume look about right" preview, it's plenty.

For users who want a PDF: any modern browser (Chrome / Safari / Edge)
can ⌘P / Ctrl-P and "Save as PDF" from the preview iframe — that uses
the OS's PDF rendering (which embeds fonts) so the resulting PDF is
device-independent. No need for LaTeX / pandoc / libreoffice on the
server.

If the user ALSO has libreoffice installed, we provide an opt-in
``soffice_to_pdf`` helper for true server-side .pdf generation.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


# Conservative print-friendly CSS — uses common system fonts so the preview
# matches roughly what Word would render. The ``@page`` rule sets A4 margins
# so the user's ⌘P → "Save as PDF" gets the right page size.
_PREVIEW_CSS = """
<style>
  @page { size: A4; margin: 1.5cm 2cm; }
  body {
    font-family: "Source Han Sans CN", "PingFang SC", "Microsoft YaHei",
                 "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #2c2925;
    max-width: 21cm;
    margin: 0 auto;
    padding: 1cm 1.5cm;
    background: #ffffff;
  }
  h1, h2, h3 { font-weight: 600; margin: 1em 0 0.4em; }
  h1 { font-size: 1.6em; border-bottom: 1px solid #e8e1d2; padding-bottom: 0.2em; }
  h2 { font-size: 1.25em; color: #cc785c; }
  h3 { font-size: 1.05em; }
  p { margin: 0.4em 0; }
  ul, ol { margin: 0.3em 0; padding-left: 1.5em; }
  li { margin: 0.15em 0; }
  table { border-collapse: collapse; margin: 0.5em 0; }
  td, th { padding: 0.3em 0.6em; border: 1px solid #e8e1d2; }
  strong, b { font-weight: 600; }
  .preview-banner {
    background: #fbf0e8; color: #b85f44;
    padding: 8px 14px; border-radius: 6px; font-size: 0.85em;
    margin-bottom: 1em; border: 1px solid #e8c5a8;
  }
  .preview-banner strong { color: #cc785c; }
  @media print { .preview-banner { display: none; } }
</style>
"""


def docx_to_html(docx_path: Path | str) -> str:
    """Convert a .docx to a self-contained HTML string for browser preview.

    Returns full HTML page (including CSS). Caller serves it as
    ``Content-Type: text/html``.

    Raises FileNotFoundError if path missing. Other failures bubble up
    (import error if mammoth missing, mammoth's own errors).
    """
    docx_path = Path(docx_path)
    if not docx_path.exists():
        raise FileNotFoundError(f"docx not found: {docx_path}")

    import mammoth
    with docx_path.open("rb") as f:
        result = mammoth.convert_to_html(f)
    body_html = result.value

    warnings_block = ""
    if result.messages:
        warning_msgs = [m.message for m in result.messages if m.message]
        if warning_msgs:
            log.debug("mammoth conversion warnings: %s", warning_msgs[:3])

    return _PREVIEW_CSS + (
        '<div class="preview-banner">'
        '<strong>📄 OfferGuide 简历预览</strong> — '
        '想要 PDF? 浏览器 ⌘P / Ctrl-P → "保存为 PDF" '
        '(嵌入字体, 任何设备打开格式不变)。'
        '</div>'
    ) + body_html + warnings_block


def soffice_to_pdf(
    docx_path: Path | str,
    output_dir: Path | str,
    *,
    timeout_s: int = 60,
) -> Path | None:
    """Convert .docx → .pdf via libreoffice (if installed).

    Tries ``soffice``, ``libreoffice``, and the Mac LibreOffice.app
    locations in that order. Returns the output PDF path on success,
    ``None`` if libreoffice isn't found (caller should fall back to
    HTML preview + browser print-to-PDF).

    libreoffice produces server-rendered PDFs with embedded fonts, so
    the file is device-independent. Slower than HTML preview (~3-8s)
    and requires user to install libreoffice — hence opt-in.
    """
    docx_path = Path(docx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    soffice_bin = (
        shutil.which("soffice")
        or shutil.which("libreoffice")
        or _try_mac_libreoffice()
    )
    if not soffice_bin:
        log.info("libreoffice not found; HTML preview is the only option")
        return None

    expected_output = output_dir / (docx_path.stem + ".pdf")
    try:
        result = subprocess.run(
            [
                soffice_bin, "--headless", "--convert-to", "pdf",
                "--outdir", str(output_dir),
                str(docx_path),
            ],
            capture_output=True, text=True,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        log.warning("soffice timed out converting %s", docx_path)
        return None
    except Exception as e:
        log.warning("soffice subprocess failed: %s", e)
        return None

    if result.returncode != 0:
        log.warning(
            "soffice returned %d converting %s: %s",
            result.returncode, docx_path, result.stderr[:300],
        )
        return None
    if not expected_output.exists():
        log.warning("soffice ran but output PDF missing: %s", expected_output)
        return None
    return expected_output


def _try_mac_libreoffice() -> str | None:
    """Mac LibreOffice.app installs to /Applications. Try the canonical path."""
    candidates = [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/opt/homebrew/bin/soffice",  # apple silicon brew
        "/usr/local/bin/soffice",      # intel brew
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None
