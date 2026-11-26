from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode

import httpx

from .context import RequestContext

_SENSITIVE_QUERY_KEYS = {"authorization", "api_key", "apikey", "access_token", "token"}


def safe_url(url: httpx.URL) -> str:
    query = parse_qsl(url.query.decode("ascii"), keep_blank_values=True)
    redacted = [(key, "***" if key.lower() in _SENSITIVE_QUERY_KEYS else value) for key, value in query]
    return str(url.copy_with(username="", password="", query=urlencode(redacted).encode("ascii")))


@dataclass(frozen=True, slots=True)
class RequestEvent:
    op: str
    method: str
    url: str
    request_id: str
    api_key_masked: str


@dataclass(frozen=True, slots=True)
class ResponseEvent:
    op: str
    method: str
    url: str
    status: int
    request_id: str
    duration: float
    err: Exception | None = None


@dataclass(frozen=True, slots=True)
class RetryEvent:
    op: str
    attempt: int
    delay: float
    reason: str


@dataclass(frozen=True, slots=True)
class PollEvent:
    op: str
    task_uuid: str
    status: str
    attempt: int


@dataclass(frozen=True, slots=True)
class Hooks:
    on_request: Callable[[RequestContext, RequestEvent], None] | None = None
    on_response: Callable[[RequestContext, ResponseEvent], None] | None = None
    on_retry: Callable[[RequestContext, RetryEvent], None] | None = None
    on_poll: Callable[[RequestContext, PollEvent], None] | None = None


@dataclass(frozen=True, slots=True)
class AsyncHooks:
    on_request: Callable[[RequestContext, RequestEvent], Awaitable[None]] | None = None
    on_response: Callable[[RequestContext, ResponseEvent], Awaitable[None]] | None = None
    on_retry: Callable[[RequestContext, RetryEvent], Awaitable[None]] | None = None
    on_poll: Callable[[RequestContext, PollEvent], Awaitable[None]] | None = None


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "****"
    if len(key) <= 16:
        return f"{key[:3]}****{key[-2:]}"
    return f"{key[:6]}****{key[-4:]}"


__all__ = [
    "AsyncHooks",
    "RequestEvent",
    "ResponseEvent",
    "RetryEvent",
    "PollEvent",
    "Hooks",
    "mask_key",
    "safe_url",
]
