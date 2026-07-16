"""Local authenticated-browser bridge used by the two research agents.

The domain agents never talk to Chrome directly.  They enqueue a validated URL
for a registered local extension, wait for an explicit terminal response, and
then let :class:`SourceReader` persist the returned rendered document as
untrusted evidence.  Browser cookies, local storage, and request headers are
never part of this protocol.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..memory import Store
from .sources import (
    AuthenticatedBrowserPage,
    SourceScope,
    canonicalize_http_url,
    validate_public_source_url,
)

MAX_RENDERED_HTML_BYTES = 5_000_000
MAX_RENDERED_TEXT_BYTES = 5_000_000
DEFAULT_REQUEST_TIMEOUT_SECONDS = 75.0
DEFAULT_LEASE_SECONDS = 55.0
LIVE_CLIENT_SECONDS = 75.0

_CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_TERMINAL_STATUSES = frozenset({
    "succeeded",
    "login_required",
    "failed",
    "timed_out",
    "rejected",
})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS authenticated_browser_bridge_identity (
    singleton       INTEGER PRIMARY KEY CHECK (singleton = 1),
    bridge_id       TEXT NOT NULL UNIQUE,
    access_token    TEXT NOT NULL,
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS authenticated_browser_clients (
    client_id         TEXT PRIMARY KEY,
    bridge_id         TEXT NOT NULL,
    extension_version TEXT NOT NULL DEFAULT '',
    registered_at     REAL NOT NULL,
    last_seen_at      REAL NOT NULL,
    FOREIGN KEY (bridge_id) REFERENCES authenticated_browser_bridge_identity(bridge_id)
);

CREATE TABLE IF NOT EXISTS authenticated_browser_requests (
    request_id       TEXT PRIMARY KEY,
    run_id           TEXT NOT NULL,
    agent_name       TEXT NOT NULL,
    subject_kind     TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    subject_revision INTEGER NOT NULL CHECK (subject_revision >= 0),
    requested_url    TEXT NOT NULL,
    canonical_url    TEXT NOT NULL,
    purpose          TEXT NOT NULL DEFAULT '',
    title_hint       TEXT NOT NULL DEFAULT '',
    status           TEXT NOT NULL
                         CHECK (status IN (
                             'queued', 'claimed', 'succeeded', 'login_required',
                             'failed', 'timed_out', 'rejected'
                         )),
    claimed_by       TEXT REFERENCES authenticated_browser_clients(client_id),
    lease_digest     TEXT,
    lease_expires_at REAL,
    deadline_at      REAL NOT NULL,
    final_url        TEXT,
    title            TEXT NOT NULL DEFAULT '',
    rendered_html    BLOB,
    rendered_text    TEXT,
    error_code       TEXT,
    error_text       TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    CHECK (
        (status = 'succeeded'
            AND final_url IS NOT NULL
            AND rendered_html IS NOT NULL
            AND rendered_text IS NOT NULL
            AND length(trim(rendered_text)) > 0)
        OR status != 'succeeded'
    )
);
CREATE INDEX IF NOT EXISTS idx_authenticated_browser_request_queue
    ON authenticated_browser_requests(status, created_at);
CREATE INDEX IF NOT EXISTS idx_authenticated_browser_request_subject
    ON authenticated_browser_requests(subject_kind, subject_id, subject_revision, created_at);
"""


class BrowserBridgeError(RuntimeError):
    """Base error for deterministic browser-bridge protocol failures."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class BrowserBridgeAuthError(BrowserBridgeError):
    pass


class BrowserBridgeConflictError(BrowserBridgeError):
    pass


@dataclass(frozen=True, slots=True)
class BrowserBridgeIdentity:
    bridge_id: str
    access_token: str


@dataclass(frozen=True, slots=True)
class BrowserBridgeRequest:
    request_id: str
    run_id: str
    agent_name: str
    subject_kind: str
    subject_id: str
    subject_revision: int
    requested_url: str
    canonical_url: str
    purpose: str
    title_hint: str
    status: str
    claimed_by: str | None
    deadline_at: float
    final_url: str | None
    title: str
    rendered_html: bytes | None
    rendered_text: str | None
    error_code: str | None
    error_text: str | None
    created_at: float
    updated_at: float


@dataclass(frozen=True, slots=True)
class ClaimedBrowserRequest:
    request_id: str
    lease_token: str
    requested_url: str
    purpose: str
    title_hint: str
    deadline_at: float


Resolver = Callable[[str, int], Sequence[Any]]


class AuthenticatedBrowserBridgeStore:
    """SQLite-backed queue and authenticated extension/client identity."""

    def __init__(self, store: Store, *, resolver: Resolver | None = None) -> None:
        self.store = store
        self.resolver = resolver

    def init_schema(self) -> None:
        now = time.time()
        with self.store.connect() as conn:
            conn.executescript(_SCHEMA)
            row = conn.execute(
                "SELECT bridge_id FROM authenticated_browser_bridge_identity WHERE singleton = 1"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO authenticated_browser_bridge_identity("
                    "singleton, bridge_id, access_token, created_at"
                    ") VALUES (1, ?, ?, ?)",
                    (secrets.token_urlsafe(24), secrets.token_urlsafe(32), now),
                )

    def identity(self) -> BrowserBridgeIdentity:
        self.init_schema()
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT bridge_id, access_token FROM authenticated_browser_bridge_identity "
                "WHERE singleton = 1"
            ).fetchone()
        if row is None:  # pragma: no cover - schema transaction guarantees this
            raise BrowserBridgeError("bridge_identity_missing", "browser bridge identity is missing")
        return BrowserBridgeIdentity(bridge_id=str(row[0]), access_token=str(row[1]))

    def authenticate(self, access_token: str) -> None:
        expected = self.identity().access_token
        if not access_token or not hmac.compare_digest(expected, access_token):
            raise BrowserBridgeAuthError(
                "invalid_bridge_token", "browser bridge authentication failed"
            )

    def register_client(self, client_id: str, *, extension_version: str = "") -> None:
        client = _validated_client_id(client_id)
        version = str(extension_version or "").strip()[:100]
        identity = self.identity()
        now = time.time()
        with self.store.connect() as conn:
            conn.execute(
                "INSERT INTO authenticated_browser_clients("
                "client_id, bridge_id, extension_version, registered_at, last_seen_at"
                ") VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(client_id) DO UPDATE SET "
                "bridge_id = excluded.bridge_id, "
                "extension_version = excluded.extension_version, "
                "last_seen_at = excluded.last_seen_at",
                (client, identity.bridge_id, version, now, now),
            )

    def has_live_client(self, *, max_age_seconds: float = LIVE_CLIENT_SECONDS) -> bool:
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self.init_schema()
        cutoff = time.time() - max_age_seconds
        bridge_id = self.identity().bridge_id
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM authenticated_browser_clients "
                "WHERE bridge_id = ? AND last_seen_at >= ? LIMIT 1",
                (bridge_id, cutoff),
            ).fetchone()
        return row is not None

    def enqueue(
        self,
        url: str,
        *,
        scope: SourceScope,
        purpose: str = "",
        title_hint: str = "",
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> BrowserBridgeRequest:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        requested_url = str(url or "").strip()
        validate_public_source_url(requested_url, resolver=self.resolver)
        canonical_url = canonicalize_http_url(requested_url)
        clean_purpose = str(purpose or "").strip()
        clean_title_hint = str(title_hint or "").strip()
        request_id = _request_id(
            scope=scope,
            canonical_url=canonical_url,
            purpose=clean_purpose,
        )
        now = time.time()
        self.init_schema()
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "INSERT OR IGNORE INTO authenticated_browser_requests("
                "request_id, run_id, agent_name, subject_kind, subject_id, subject_revision, "
                "requested_url, canonical_url, purpose, title_hint, status, deadline_at, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)",
                (
                    request_id,
                    scope.run_id,
                    scope.agent_name,
                    scope.subject_kind,
                    str(scope.subject_id),
                    scope.subject_revision,
                    requested_url,
                    canonical_url,
                    clean_purpose,
                    clean_title_hint,
                    now + timeout_seconds,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        return _request_from_row(row)

    def get(self, request_id: str) -> BrowserBridgeRequest:
        self.init_schema()
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
        if row is None:
            raise BrowserBridgeError("request_not_found", "browser bridge request not found")
        return _request_from_row(row)

    def claim_next(
        self,
        client_id: str,
        *,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> ClaimedBrowserRequest | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        client = _validated_client_id(client_id)
        self.init_schema()
        now = time.time()
        lease_token = secrets.token_urlsafe(32)
        lease_digest = _secret_digest(lease_token)
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            registered = conn.execute(
                "SELECT 1 FROM authenticated_browser_clients WHERE client_id = ?",
                (client,),
            ).fetchone()
            if registered is None:
                raise BrowserBridgeAuthError(
                    "browser_client_not_registered", "browser bridge client is not registered"
                )
            conn.execute(
                "UPDATE authenticated_browser_clients SET last_seen_at = ? WHERE client_id = ?",
                (now, client),
            )
            conn.execute(
                "UPDATE authenticated_browser_requests SET "
                "status = 'timed_out', error_code = 'browser_timeout', "
                "error_text = 'browser bridge request expired before completion', updated_at = ? "
                "WHERE status IN ('queued', 'claimed') AND deadline_at <= ?",
                (now, now),
            )
            conn.execute(
                "UPDATE authenticated_browser_requests SET "
                "status = 'queued', claimed_by = NULL, lease_digest = NULL, "
                "lease_expires_at = NULL, updated_at = ? "
                "WHERE status = 'claimed' AND lease_expires_at <= ? AND deadline_at > ?",
                (now, now, now),
            )
            row = conn.execute(
                "SELECT * FROM authenticated_browser_requests "
                "WHERE status = 'queued' AND deadline_at > ? "
                "ORDER BY created_at ASC, request_id ASC LIMIT 1",
                (now,),
            ).fetchone()
            if row is None:
                return None
            updated = conn.execute(
                "UPDATE authenticated_browser_requests SET "
                "status = 'claimed', claimed_by = ?, lease_digest = ?, lease_expires_at = ?, "
                "updated_at = ? WHERE request_id = ? AND status = 'queued'",
                (client, lease_digest, now + lease_seconds, now, row["request_id"]),
            )
            if updated.rowcount != 1:  # pragma: no cover - guarded by BEGIN IMMEDIATE
                return None
        return ClaimedBrowserRequest(
            request_id=str(row["request_id"]),
            lease_token=lease_token,
            requested_url=str(row["requested_url"]),
            purpose=str(row["purpose"]),
            title_hint=str(row["title_hint"]),
            deadline_at=float(row["deadline_at"]),
        )

    def complete(
        self,
        request_id: str,
        *,
        client_id: str,
        lease_token: str,
        status: str,
        final_url: str | None = None,
        title: str = "",
        rendered_html: str | bytes | None = None,
        rendered_text: str | None = None,
        error_code: str | None = None,
        error_text: str | None = None,
    ) -> BrowserBridgeRequest:
        if status not in {"succeeded", "login_required", "failed", "rejected"}:
            raise ValueError("invalid browser bridge completion status")
        client = _validated_client_id(client_id)
        now = time.time()
        clean_final_url = str(final_url or "").strip() or None
        clean_title = str(title or "").strip()[:1000]
        clean_error_code = str(error_code or "").strip()[:100] or None
        clean_error_text = str(error_text or "").strip()[:2000] or None
        html_bytes: bytes | None = None
        text: str | None = None

        if status == "succeeded":
            if clean_final_url is None:
                raise BrowserBridgeError(
                    "missing_final_url", "successful browser response requires final_url"
                )
            validate_public_source_url(clean_final_url, resolver=self.resolver)
            if isinstance(rendered_html, str):
                html_bytes = rendered_html.encode("utf-8")
            elif rendered_html is not None:
                html_bytes = bytes(rendered_html)
            text = str(rendered_text or "")
            if not html_bytes or not text.strip():
                raise BrowserBridgeError(
                    "incomplete_rendered_page",
                    "successful browser response requires complete rendered HTML and readable text",
                )
            if len(html_bytes) > MAX_RENDERED_HTML_BYTES:
                raise BrowserBridgeError(
                    "rendered_html_too_large",
                    f"rendered HTML exceeds {MAX_RENDERED_HTML_BYTES} bytes",
                )
            if len(text.encode("utf-8")) > MAX_RENDERED_TEXT_BYTES:
                raise BrowserBridgeError(
                    "rendered_text_too_large",
                    f"rendered text exceeds {MAX_RENDERED_TEXT_BYTES} bytes",
                )
        elif not clean_error_code or not clean_error_text:
            raise BrowserBridgeError(
                "missing_browser_failure",
                "failed browser response requires an explicit error_code and error_text",
            )
        elif clean_final_url is not None:
            validate_public_source_url(clean_final_url, resolver=self.resolver)

        self.init_schema()
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if row is None:
                raise BrowserBridgeConflictError(
                    "request_not_found", "browser bridge request not found"
                )
            if str(row["status"]) != "claimed" or str(row["claimed_by"] or "") != client:
                raise BrowserBridgeConflictError(
                    "request_not_claimed", "browser bridge request is not claimed by this client"
                )
            expected_digest = str(row["lease_digest"] or "")
            if not expected_digest or not hmac.compare_digest(
                expected_digest, _secret_digest(str(lease_token or ""))
            ):
                raise BrowserBridgeAuthError(
                    "invalid_lease_token", "browser bridge lease authentication failed"
                )
            if float(row["deadline_at"]) <= now:
                conn.execute(
                    "UPDATE authenticated_browser_requests SET status = 'timed_out', "
                    "error_code = 'browser_timeout', "
                    "error_text = 'browser bridge request expired before completion', "
                    "updated_at = ? WHERE request_id = ?",
                    (now, str(request_id)),
                )
                conn.commit()
                raise BrowserBridgeConflictError(
                    "request_timed_out", "browser bridge request has timed out"
                )
            conn.execute(
                "UPDATE authenticated_browser_requests SET "
                "status = ?, final_url = ?, title = ?, rendered_html = ?, rendered_text = ?, "
                "error_code = ?, error_text = ?, lease_digest = NULL, lease_expires_at = NULL, "
                "updated_at = ? WHERE request_id = ?",
                (
                    status,
                    clean_final_url,
                    clean_title,
                    html_bytes,
                    text,
                    clean_error_code,
                    clean_error_text,
                    now,
                    str(request_id),
                ),
            )
            completed = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
        return _request_from_row(completed)

    def mark_timed_out(self, request_id: str) -> BrowserBridgeRequest:
        now = time.time()
        self.init_schema()
        with self.store.connect() as conn:
            conn.execute(
                "UPDATE authenticated_browser_requests SET status = 'timed_out', "
                "error_code = 'browser_timeout', "
                "error_text = 'browser bridge did not return a rendered page before timeout', "
                "updated_at = ? WHERE request_id = ? AND status IN ('queued', 'claimed')",
                (now, str(request_id)),
            )
        return self.get(request_id)

    def reject_claimed_response(
        self,
        request_id: str,
        *,
        client_id: str,
        lease_token: str,
        error_code: str,
        error_text: str,
    ) -> BrowserBridgeRequest:
        """Make an invalid extension payload a visible terminal failure.

        Validation can fail before ``complete`` reaches its update transaction
        (for example, a redirect resolving to a private address).  Leaving that
        request claimed would hide the real cause behind a later timeout.
        """

        client = _validated_client_id(client_id)
        code = str(error_code or "invalid_browser_response").strip()[:100]
        message = str(error_text or "invalid browser response").strip()[:2000]
        now = time.time()
        self.init_schema()
        with self.store.connect() as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
            if row is None:
                raise BrowserBridgeConflictError(
                    "request_not_found", "browser bridge request not found"
                )
            if str(row["status"]) != "claimed" or str(row["claimed_by"] or "") != client:
                raise BrowserBridgeConflictError(
                    "request_not_claimed", "browser bridge request is not claimed by this client"
                )
            if not hmac.compare_digest(
                str(row["lease_digest"] or ""),
                _secret_digest(str(lease_token or "")),
            ):
                raise BrowserBridgeAuthError(
                    "invalid_lease_token", "browser bridge lease authentication failed"
                )
            conn.execute(
                "UPDATE authenticated_browser_requests SET status = 'rejected', "
                "error_code = ?, error_text = ?, lease_digest = NULL, "
                "lease_expires_at = NULL, updated_at = ? WHERE request_id = ?",
                (code, message, now, str(request_id)),
            )
            rejected = conn.execute(
                "SELECT * FROM authenticated_browser_requests WHERE request_id = ?",
                (str(request_id),),
            ).fetchone()
        return _request_from_row(rejected)


class AuthenticatedBrowserBridgeClient:
    """Blocking domain-side client for the local browser extension queue."""

    def __init__(
        self,
        bridge_store: AuthenticatedBrowserBridgeStore,
        *,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        poll_interval_seconds: float = 0.1,
        live_client_seconds: float = LIVE_CLIENT_SECONDS,
    ) -> None:
        if request_timeout_seconds <= 0 or poll_interval_seconds <= 0:
            raise ValueError("browser bridge timeouts must be positive")
        self.bridge_store = bridge_store
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.live_client_seconds = live_client_seconds

    def fetch_rendered(
        self,
        url: str,
        *,
        scope: SourceScope,
        purpose: str = "",
        title_hint: str = "",
    ) -> AuthenticatedBrowserPage:
        requested_url = str(url or "").strip()
        if not self.bridge_store.has_live_client(max_age_seconds=self.live_client_seconds):
            return AuthenticatedBrowserPage(
                status="unavailable",
                requested_url=requested_url,
                final_url=None,
                title="",
                rendered_html=None,
                rendered_text=None,
                error_code="browser_bridge_unavailable",
                error_text=(
                    "no authenticated browser extension is connected; open the OfferGuide "
                    "extension in the signed-in browser and retry"
                ),
            )

        request = self.bridge_store.enqueue(
            requested_url,
            scope=scope,
            purpose=purpose,
            title_hint=title_hint,
            timeout_seconds=self.request_timeout_seconds,
        )
        deadline = time.monotonic() + self.request_timeout_seconds
        while request.status not in _TERMINAL_STATUSES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                request = self.bridge_store.mark_timed_out(request.request_id)
                break
            time.sleep(min(self.poll_interval_seconds, remaining))
            request = self.bridge_store.get(request.request_id)
        return _page_from_request(request)


def _request_id(*, scope: SourceScope, canonical_url: str, purpose: str) -> str:
    payload = "\x00".join((
        scope.run_id,
        scope.agent_name,
        scope.subject_kind,
        str(scope.subject_id),
        str(scope.subject_revision),
        canonical_url,
        purpose,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_client_id(client_id: str) -> str:
    value = str(client_id or "").strip()
    if not _CLIENT_ID_RE.fullmatch(value):
        raise BrowserBridgeAuthError(
            "invalid_client_id", "browser bridge client_id has an invalid format"
        )
    return value


def _request_from_row(row: sqlite3.Row) -> BrowserBridgeRequest:
    return BrowserBridgeRequest(
        request_id=str(row["request_id"]),
        run_id=str(row["run_id"]),
        agent_name=str(row["agent_name"]),
        subject_kind=str(row["subject_kind"]),
        subject_id=str(row["subject_id"]),
        subject_revision=int(row["subject_revision"]),
        requested_url=str(row["requested_url"]),
        canonical_url=str(row["canonical_url"]),
        purpose=str(row["purpose"]),
        title_hint=str(row["title_hint"]),
        status=str(row["status"]),
        claimed_by=str(row["claimed_by"]) if row["claimed_by"] is not None else None,
        deadline_at=float(row["deadline_at"]),
        final_url=str(row["final_url"]) if row["final_url"] is not None else None,
        title=str(row["title"] or ""),
        rendered_html=(bytes(row["rendered_html"]) if row["rendered_html"] is not None else None),
        rendered_text=(str(row["rendered_text"]) if row["rendered_text"] is not None else None),
        error_code=str(row["error_code"]) if row["error_code"] is not None else None,
        error_text=str(row["error_text"]) if row["error_text"] is not None else None,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


def _page_from_request(request: BrowserBridgeRequest) -> AuthenticatedBrowserPage:
    return AuthenticatedBrowserPage(
        status=request.status,
        requested_url=request.requested_url,
        final_url=request.final_url,
        title=request.title,
        rendered_html=request.rendered_html,
        rendered_text=request.rendered_text,
        error_code=request.error_code,
        error_text=request.error_text,
    )


__all__ = [
    "MAX_RENDERED_HTML_BYTES",
    "MAX_RENDERED_TEXT_BYTES",
    "AuthenticatedBrowserBridgeClient",
    "AuthenticatedBrowserBridgeStore",
    "BrowserBridgeAuthError",
    "BrowserBridgeConflictError",
    "BrowserBridgeError",
    "BrowserBridgeIdentity",
    "BrowserBridgeRequest",
    "ClaimedBrowserRequest",
]
