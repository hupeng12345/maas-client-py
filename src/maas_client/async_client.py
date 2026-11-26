from __future__ import annotations

import logging
import random
import time
from collections.abc import AsyncIterable, Iterable
from typing import BinaryIO
from urllib.parse import urlsplit

import httpx

from .config import (
    DEFAULT_MAX_UPLOAD_SIZE,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    AsyncConfig,
    PollPolicy,
    ResponseMeta,
    RetryPolicy,
)
from .context import (
    RequestCanceled,
    RequestContext,
    RequestDeadlineExceeded,
    _context_error,
    run_with_context,
)
from .errors import ConfigError, MaaSError, classify_transport_error, new_local_error, parse_retry_after
from .hooks import AsyncHooks, RequestEvent, ResponseEvent, RetryEvent, mask_key, safe_url
from .multipart import AsyncMultipartBody, MultipartBody
from .request_id import HEADER_REQUEST_ID, ensure_request_id, request_id_from_header

_IDEMPOTENT_METHODS = {"GET", "HEAD", "PUT", "DELETE", "OPTIONS", "TRACE"}


class AsyncClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        http_client: httpx.AsyncClient | None = None,
        stream_http_client: httpx.AsyncClient | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        retry: RetryPolicy | None = None,
        poll: PollPolicy | None = None,
        max_upload_size: int = DEFAULT_MAX_UPLOAD_SIZE,
        user_agent: str = DEFAULT_USER_AGENT,
        logger: logging.Logger | None = None,
        hooks: AsyncHooks | None = None,
        keep_raw: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        follow_redirects: bool = False,
    ) -> None:
        api_key = api_key.strip()
        base_url = base_url.strip()
        if not api_key:
            raise ConfigError("api_key", "must not be empty")
        if not base_url:
            raise ConfigError("base_url", "must not be empty; pass your deployment domain")
        try:
            parsed = urlsplit(base_url)
        except ValueError as exc:
            raise ConfigError("base_url", f"cannot parse: {exc}") from exc
        if parsed.scheme not in {"http", "https"}:
            raise ConfigError("base_url", "scheme must be http or https")
        if not parsed.hostname:
            raise ConfigError("base_url", "missing host")
        if timeout <= 0:
            raise ConfigError("timeout", "must be greater than 0")
        retry = retry or RetryPolicy()
        poll = poll or PollPolicy()
        if retry.max_attempts < 1:
            raise ConfigError("retry.max_attempts", "must be at least 1")
        if not 0 <= retry.jitter <= 1:
            raise ConfigError("retry.jitter", "must be within [0, 1]")
        if min(poll.initial_delay, poll.max_delay, poll.timeout, poll.query_timeout) <= 0:
            raise ConfigError("poll", "durations must be greater than 0")
        if poll.multiplier < 1:
            raise ConfigError("poll.multiplier", "must be >= 1")
        if max_upload_size < 0:
            raise ConfigError("max_upload_size", "must not be negative")
        if http_client is not None and transport is not None:
            raise ConfigError("transport", "mutually exclusive with http_client")

        self.config = AsyncConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            retry=retry,
            poll=poll,
            max_upload_size=max_upload_size,
            user_agent=user_agent,
            logger=logger,
            hooks=hooks or AsyncHooks(),
            keep_raw=keep_raw,
            http_client=http_client,
            stream_http_client=stream_http_client,
            transport=transport,
            follow_redirects=follow_redirects,
        )
        self._built_in = http_client is None
        if http_client is not None:
            self._unary = http_client
            self._stream = stream_http_client or http_client
        else:
            self._unary = httpx.AsyncClient(
                timeout=timeout,
                transport=transport,
                trust_env=False,
                follow_redirects=follow_redirects,
            )
            self._stream = stream_http_client or httpx.AsyncClient(
                timeout=None,
                transport=transport,
                trust_env=False,
                follow_redirects=follow_redirects,
            )

    @classmethod
    def must_new(cls, **kwargs: object) -> AsyncClient:
        return cls(**kwargs)  # type: ignore[arg-type]

    @property
    def base_url(self) -> str:
        return self.config.base_url

    @property
    def api_key(self) -> str:
        return self.config.api_key

    @property
    def poll_policy(self) -> PollPolicy:
        return self.config.poll

    @property
    def hooks(self) -> AsyncHooks:
        return self.config.hooks

    @property
    def max_upload_size(self) -> int:
        return self.config.max_upload_size

    @property
    def keep_raw(self) -> bool:
        return self.config.keep_raw

    @property
    def logger(self) -> logging.Logger | None:
        return self.config.logger

    def require_base_url(self, op: str) -> None:
        if not self.base_url:
            raise new_local_error(op, "no endpoint configured", "Pass base_url when building the client")

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return self.base_url + (path if path.startswith("/") else f"/{path}")

    def new_request(
        self,
        method: str,
        path: str,
        body: (
            bytes
            | BinaryIO
            | Iterable[bytes]
            | AsyncIterable[bytes]
            | MultipartBody
            | AsyncMultipartBody
            | None
        ) = None,
        *,
        ctx: RequestContext | None = None,
    ) -> httpx.Request:
        extensions: dict[str, object] = {}
        if ctx is not None and ctx.deadline is not None:
            remaining = ctx.remaining()
            if remaining is None:
                remaining = 0.0
            extensions["timeout"] = {"connect": remaining, "read": remaining, "write": remaining, "pool": remaining}
        content: object = body
        if isinstance(body, (MultipartBody, AsyncMultipartBody)):
            content = body.reader
        try:
            return httpx.Request(method, self.url(path), content=content, extensions=extensions)  # type: ignore[arg-type]
        except (TypeError, ValueError, httpx.InvalidURL) as exc:
            raise new_local_error(
                "http.NewRequest",
                "failed to build request",
                "Check the URL and request body",
            ) from exc

    @staticmethod
    def _replayable(request: httpx.Request) -> bool:
        try:
            _ = request.content
            return True
        except (httpx.RequestNotRead, httpx.StreamConsumed):
            return False

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        base = min(0.1 * (2 ** (attempt - 1)), 5.0)
        return max(0.0, random.uniform(base * 0.75, base * 1.25))

    async def do(
        self,
        request: httpx.Request,
        op: str,
        *,
        streaming: bool = False,
        ctx: RequestContext | None = None,
    ) -> tuple[httpx.Response | None, ResponseMeta, MaaSError | None]:
        if request is None:
            return None, ResponseMeta(), new_local_error(op, "nil request", "Pass a request built with new_request")
        ctx, request_id = ensure_request_id(ctx)
        if ctx.deadline is not None:
            remaining = ctx.remaining() or 0.0
            request.extensions["timeout"] = {
                "connect": remaining,
                "read": remaining,
                "write": remaining,
                "pool": remaining,
            }
        if not request.headers.get(HEADER_REQUEST_ID):
            request.headers[HEADER_REQUEST_ID] = request_id
        if not request.headers.get("Authorization"):
            request.headers["Authorization"] = f"Bearer {self.api_key}"
        if not request.headers.get("User-Agent"):
            request.headers["User-Agent"] = self.config.user_agent
        if not request.headers.get("Accept"):
            request.headers["Accept"] = "application/json"

        if self.hooks.on_request is not None:
            await self.hooks.on_request(
                ctx,
                RequestEvent(op, request.method, safe_url(request.url), request_id, mask_key(self.api_key)),
            )
        client = self._stream if streaming else self._unary
        attempts = self.config.retry.max_attempts if self._built_in and not streaming else 1
        can_retry = request.method.upper() in _IDEMPOTENT_METHODS and self._replayable(request)
        start = time.monotonic()
        response: httpx.Response | None = None
        error: MaaSError | None = None

        for attempt in range(1, attempts + 1):
            try:
                response = await run_with_context(ctx, lambda: client.send(request, stream=streaming))
            except RequestCanceled as exc:
                error = _context_error(op, ctx, exc)
                break
            except RequestDeadlineExceeded as exc:
                error = _context_error(op, ctx, exc)
                break
            except httpx.TimeoutException as exc:
                error = classify_transport_error(op, exc, request_id=request_id)
            except httpx.HTTPError as exc:
                error = classify_transport_error(op, exc, request_id=request_id)
            else:
                retry_status = response.status_code == 429 or response.status_code >= 500
                if can_retry and retry_status and attempt < attempts:
                    delay = parse_retry_after(response.headers.get("Retry-After")) or self._retry_delay(attempt)
                    status = response.status_code
                    await response.aclose()
                    if self.hooks.on_retry is not None:
                        await self.hooks.on_retry(ctx, RetryEvent(op, attempt + 1, delay, f"HTTP {status}"))
                    try:
                        await ctx.wait_async(delay)
                    except (RequestCanceled, RequestDeadlineExceeded) as exc:
                        error = _context_error(op, ctx, exc)
                        break
                    continue
                error = None
                break

            if can_retry and attempt < attempts:
                delay = self._retry_delay(attempt)
                if self.hooks.on_retry is not None:
                    await self.hooks.on_retry(ctx, RetryEvent(op, attempt + 1, delay, error.message))
                try:
                    await ctx.wait_async(delay)
                except (RequestCanceled, RequestDeadlineExceeded) as exc:
                    error = _context_error(op, ctx, exc)
                    break
                continue
            break

        elapsed = time.monotonic() - start
        if error is not None:
            error.request_id = request_id
            if self.hooks.on_response is not None:
                await self.hooks.on_response(
                    ctx,
                    ResponseEvent(op, request.method, safe_url(request.url), 0, request_id, elapsed, error),
                )
            if self.logger:
                self.logger.warning(
                    "maas request failed op=%s method=%s request_id=%s error=%s",
                    op,
                    request.method,
                    request_id,
                    error,
                )
            return None, ResponseMeta(request_id=request_id, duration=elapsed), error

        assert response is not None
        response_id = request_id_from_header(response.headers) or request_id
        if not request_id_from_header(response.headers):
            response.headers[HEADER_REQUEST_ID] = response_id
        meta = ResponseMeta(request_id=response_id, http_status=response.status_code, duration=elapsed)
        if self.hooks.on_response is not None:
            await self.hooks.on_response(
                ctx,
                ResponseEvent(op, request.method, safe_url(request.url), response.status_code, response_id, elapsed),
            )
        if self.logger:
            self.logger.info(
                "maas response op=%s method=%s status=%d request_id=%s duration=%.3fs",
                op,
                request.method,
                response_id,
                elapsed,
            )
        return response, meta, None

    async def close(self) -> None:
        if self._built_in:
            await self._unary.aclose()
            if self._stream is not self._unary:
                await self._stream.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()


__all__ = ["AsyncClient"]
