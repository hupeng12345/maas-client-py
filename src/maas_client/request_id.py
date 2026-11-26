from __future__ import annotations

import secrets
import uuid
from collections.abc import Mapping
from dataclasses import replace

from .context import RequestContext, current_request_id

HEADER_REQUEST_ID = "X-Request-ID"
HEADER_API_REQUEST_ID = "X-Api-Request-Id"


def with_request_id(ctx: RequestContext | None, request_id: str) -> RequestContext:
    return replace(ctx or RequestContext(), request_id=request_id.strip())


def request_id_from_context(ctx: RequestContext | None) -> str:
    if ctx is not None and ctx.request_id:
        return ctx.request_id
    return current_request_id()


def request_id_from_header(headers: Mapping[str, str]) -> str:
    lowered = {key.lower(): value for key, value in headers.items()}
    return (lowered.get(HEADER_REQUEST_ID.lower(), "") or lowered.get(HEADER_API_REQUEST_ID.lower(), "")).strip()


def new_request_id() -> str:
    return secrets.token_hex(16)


def new_uuid() -> str:
    return str(uuid.uuid4())


def is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value.lower()
    except (ValueError, AttributeError):
        return False


def ensure_request_id(ctx: RequestContext | None) -> tuple[RequestContext, str]:
    base = ctx or RequestContext()
    request_id = request_id_from_context(base)
    if not request_id:
        request_id = new_request_id()
        base = replace(base, request_id=request_id)
    return base, request_id


def request_ids_for_asr(ctx: RequestContext | None) -> tuple[RequestContext, str, str]:
    base, generic = ensure_request_id(ctx)
    return base, generic if is_uuid(generic) else new_uuid(), generic


__all__ = [
    "HEADER_REQUEST_ID",
    "HEADER_API_REQUEST_ID",
    "with_request_id",
    "request_id_from_context",
    "request_id_from_header",
    "new_request_id",
    "new_uuid",
    "is_uuid",
    "ensure_request_id",
    "request_ids_for_asr",
]
