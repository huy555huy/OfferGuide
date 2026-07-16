"""Authenticated local API used only by the OfferGuide browser extension."""

from __future__ import annotations

import ipaddress
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from ..research_agents.browser_bridge import (
    DEFAULT_LEASE_SECONDS,
    MAX_RENDERED_HTML_BYTES,
    MAX_RENDERED_TEXT_BYTES,
    AuthenticatedBrowserBridgeStore,
    BrowserBridgeAuthError,
    BrowserBridgeConflictError,
    BrowserBridgeError,
)

PROTOCOL_VERSION = 1


class _BridgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RegisterBrowserClientPayload(_BridgePayload):
    client_id: str = Field(min_length=16, max_length=128)
    extension_version: str = Field(default="", max_length=100)


class ClaimBrowserRequestPayload(_BridgePayload):
    client_id: str = Field(min_length=16, max_length=128)


class CompleteBrowserRequestPayload(_BridgePayload):
    client_id: str = Field(min_length=16, max_length=128)
    lease_token: str = Field(min_length=20, max_length=256)
    status: Literal["succeeded", "login_required", "failed", "rejected"]
    final_url: str | None = Field(default=None, max_length=20_000)
    title: str = Field(default="", max_length=1_000)
    rendered_html: str | None = Field(default=None, max_length=MAX_RENDERED_HTML_BYTES)
    rendered_text: str | None = Field(default=None, max_length=MAX_RENDERED_TEXT_BYTES)
    error_code: str | None = Field(default=None, max_length=100)
    error_text: str | None = Field(default=None, max_length=2_000)


def build_browser_bridge_router(
    bridge_store: AuthenticatedBrowserBridgeStore,
) -> APIRouter:
    """Expose stable extension endpoints over the loopback-only local app."""

    router = APIRouter(prefix="/api/browser-bridge", tags=["browser-bridge"])

    @router.get("/config", response_class=JSONResponse)
    def browser_bridge_config(request: Request) -> JSONResponse:
        _require_loopback(request)
        identity = bridge_store.identity()
        response = JSONResponse({
            "protocol_version": PROTOCOL_VERSION,
            "bridge_id": identity.bridge_id,
            "access_token": identity.access_token,
            "max_rendered_html_bytes": MAX_RENDERED_HTML_BYTES,
            "max_rendered_text_bytes": MAX_RENDERED_TEXT_BYTES,
        })
        response.headers["Cache-Control"] = "no-store"
        return response

    @router.post("/clients/register", response_class=JSONResponse)
    def register_browser_client(
        payload: RegisterBrowserClientPayload,
        request: Request,
    ) -> JSONResponse:
        _require_loopback(request)
        _authenticate_request(request, bridge_store)
        try:
            bridge_store.register_client(
                payload.client_id,
                extension_version=payload.extension_version,
            )
        except BrowserBridgeAuthError as exc:
            raise HTTPException(
                status_code=401, detail={"code": exc.code, "error": str(exc)}
            ) from exc
        return JSONResponse({"ok": True, "protocol_version": PROTOCOL_VERSION})

    @router.post("/requests/claim")
    def claim_browser_request(
        payload: ClaimBrowserRequestPayload,
        request: Request,
    ) -> Response:
        _require_loopback(request)
        _authenticate_request(request, bridge_store)
        try:
            claimed = bridge_store.claim_next(
                payload.client_id,
                lease_seconds=DEFAULT_LEASE_SECONDS,
            )
        except BrowserBridgeAuthError as exc:
            raise HTTPException(
                status_code=401, detail={"code": exc.code, "error": str(exc)}
            ) from exc
        if claimed is None:
            return Response(status_code=204)
        return JSONResponse({
            "request_id": claimed.request_id,
            "lease_token": claimed.lease_token,
            "requested_url": claimed.requested_url,
            "purpose": claimed.purpose,
            "title_hint": claimed.title_hint,
            "deadline_at": claimed.deadline_at,
        })

    @router.post("/requests/{request_id}/complete", response_class=JSONResponse)
    def complete_browser_request(
        request_id: str,
        payload: CompleteBrowserRequestPayload,
        request: Request,
    ) -> JSONResponse:
        _require_loopback(request)
        _authenticate_request(request, bridge_store)
        try:
            completed = bridge_store.complete(
                request_id,
                client_id=payload.client_id,
                lease_token=payload.lease_token,
                status=payload.status,
                final_url=payload.final_url,
                title=payload.title,
                rendered_html=payload.rendered_html,
                rendered_text=payload.rendered_text,
                error_code=payload.error_code,
                error_text=payload.error_text,
            )
        except BrowserBridgeAuthError as exc:
            raise HTTPException(
                status_code=401, detail={"code": exc.code, "error": str(exc)}
            ) from exc
        except BrowserBridgeConflictError as exc:
            raise HTTPException(
                status_code=409, detail={"code": exc.code, "error": str(exc)}
            ) from exc
        except (BrowserBridgeError, ValueError) as exc:
            code = getattr(exc, "code", "invalid_browser_response")
            bridge_store.reject_claimed_response(
                request_id,
                client_id=payload.client_id,
                lease_token=payload.lease_token,
                error_code=str(code),
                error_text=str(exc),
            )
            raise HTTPException(
                status_code=422, detail={"code": code, "error": str(exc)}
            ) from exc
        return JSONResponse({"ok": True, "request_id": completed.request_id})

    return router


def _authenticate_request(
    request: Request,
    bridge_store: AuthenticatedBrowserBridgeStore,
) -> None:
    authorization = request.headers.get("authorization", "")
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail={"code": "missing_bridge_token", "error": "browser bridge token required"},
        )
    try:
        bridge_store.authenticate(token.strip())
    except BrowserBridgeAuthError as exc:
        raise HTTPException(
            status_code=401, detail={"code": exc.code, "error": str(exc)}
        ) from exc


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client is not None else ""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is None or not address.is_loopback:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "loopback_required",
                "error": "browser bridge endpoints are available only over loopback",
            },
        )


__all__ = ["PROTOCOL_VERSION", "build_browser_bridge_router"]
