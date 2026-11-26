from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from .hooks import Hooks

VERSION = "0.1.0"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_UPLOAD_SIZE = 100 * 1024 * 1024
DEFAULT_USER_AGENT = f"maas-client-py/{VERSION}"
DEFAULT_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.3
    max_delay: float = 3.0
    jitter: float = 0.2


@dataclass(frozen=True, slots=True)
class PollPolicy:
    initial_delay: float = 0.8
    max_delay: float = 5.0
    multiplier: float = 1.5
    jitter: float = 0.2
    timeout: float = 300.0
    query_timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class ResponseMeta:
    request_id: str = ""
    http_status: int = 0
    duration: float = 0.0

    def __str__(self) -> str:
        return f"request_id={self.request_id}" if self.request_id else ""


@dataclass(slots=True)
class Config:
    api_key: str
    base_url: str
    timeout: float = DEFAULT_TIMEOUT
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    poll: PollPolicy = field(default_factory=PollPolicy)
    max_upload_size: int = DEFAULT_MAX_UPLOAD_SIZE
    user_agent: str = DEFAULT_USER_AGENT
    keep_raw: bool = False
    logger: logging.Logger | None = None
    hooks: Hooks = field(default_factory=Hooks)
    http_client: httpx.Client | None = None
    stream_http_client: httpx.Client | None = None
    transport: httpx.BaseTransport | None = None
    follow_redirects: bool = False


def default_retry_policy() -> RetryPolicy:
    return RetryPolicy()


def default_poll_policy() -> PollPolicy:
    return PollPolicy()


__all__ = [
    "VERSION",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_UPLOAD_SIZE",
    "DEFAULT_USER_AGENT",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "RetryPolicy",
    "PollPolicy",
    "ResponseMeta",
    "Config",
    "default_retry_policy",
    "default_poll_policy",
]
