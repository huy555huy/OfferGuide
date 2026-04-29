"""User-paste route — when Scout's automatic crawl can't reach a JD.

Two flavours:

- `from_url(url)`: dispatches to a known platform adapter if the URL pattern
  matches (today: nowcoder). Returns `RawJob` from that platform.
- `from_text(text)`: wraps free-form pasted text. The user paste boss-style
  bullet points into here when the browser-extension path isn't ready (W6).
"""

from __future__ import annotations

from . import nowcoder
from ._spec import RawJob

NAME = "manual"


def from_url(url: str, *, client: nowcoder.NowcoderClient | None = None) -> RawJob:
    """Best-effort fetch+parse for a user-pasted URL.

    Today only nowcoder URLs are auto-handled. For unknown hosts the user must
    fall through to `from_text(...)` after copy-pasting the JD body themselves.
    """
    if nowcoder.JD_URL_RE.match(url):
        own = client or nowcoder.NowcoderClient()
        try:
            return nowcoder.fetch_and_parse(own, url)
        finally:
            if client is None:
                own.close()
    raise NotImplementedError(
        f"No automatic adapter yet for URL host: {url}. "
        "Paste the JD text via `from_text(...)` instead."
    )


def from_text(
    text: str,
    *,
    title: str | None = None,
    company: str | None = None,
    location: str | None = None,
    url: str | None = None,
) -> RawJob:
    """Wrap a free-form pasted JD as a `RawJob` so Scout can ingest it like any other."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty JD text")
    return RawJob(
        source=NAME,
        title=title or _guess_title(cleaned),
        company=company,
        location=location,
        url=url,
        raw_text=cleaned,
    )


def _guess_title(text: str) -> str:
    """First non-empty line, capped, used only when the user didn't supply a title."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return "(untitled)"
