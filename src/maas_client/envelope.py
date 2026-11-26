from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from .config import DEFAULT_MAX_RESPONSE_BYTES, ResponseMeta
from .context import RequestCanceled, RequestContext, RequestDeadlineExceeded, _context_error, run_with_context
from .errors import (
    CODE_SUCCESS,
    MaaSError,
    new_api_error,
    new_protocol_error,
    new_transport_error,
    parse_retry_after,
)
from .request_id import request_id_from_header


@dataclass(frozen=True, slots=True)
class Envelope:
    code: str
    message: str
    data: bytes | None

    @property
    def value(self) -> Any:
        if self.data is None:
            return None
        return json.loads(self.data)

    def is_success(self) -> bool:
        return self.code == CODE_SUCCESS

    def error(self, op: str, response: httpx.Response) -> MaaSError:
        err = new_api_error(op, self.code, self.message, response.status_code)
        err.request_id = request_id_from_header(response.headers)
        err.retry_after = parse_retry_after(response.headers.get("Retry-After"))
        if self.data is not None:
            err.raw_body = self.data[:4096]
        return err


def _read_limited(response: httpx.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > limit:
            raise ValueError("response body exceeds the parse limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _extract_data(text: str) -> bytes | None:
    decoder = json.JSONDecoder()
    index = 0

    def skip_ws(position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    index = skip_ws(index)
    if index >= len(text) or text[index] != "{":
        return None
    index = skip_ws(index + 1)
    raw: bytes | None = None
    while index < len(text) and text[index] != "}":
        key, index = decoder.raw_decode(text, index)
        index = skip_ws(index)
        if index >= len(text) or text[index] != ":":
            break
        index = skip_ws(index + 1)
        start = index
        _value, index = decoder.raw_decode(text, index)
        if key == "data":
            raw = text[start:index].encode("utf-8")
        index = skip_ws(index)
        if index < len(text) and text[index] == ",":
            index = skip_ws(index + 1)
            continue
        break
    return raw


def decode_envelope(
    op: str,
    response: httpx.Response,
    *,
    meta: ResponseMeta | None = None,
    limit: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> Envelope:
    request_id = meta.request_id if meta else request_id_from_header(response.headers)
    try:
        body = _read_limited(response, limit)
    except ValueError as exc:
        err = new_protocol_error(op, "response body exceeds the parse limit", response.status_code, None)
        err.request_id = request_id
        err.hint = "Response too large to parse; contact the platform for very large documents"
        raise err from exc
    except httpx.HTTPError as exc:
        err = new_transport_error(op, exc, request_id=request_id)
        err.http_status = response.status_code
        raise err from exc
    return _decode_body(op, body, response.status_code, request_id)


def _decode_body(op: str, body: bytes, http_status: int, request_id: str) -> Envelope:
    if not body.strip():
        err = new_protocol_error(op, "empty response body", http_status, body)
        err.request_id = request_id
        raise err
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        err = new_protocol_error(op, "response is not a JSON object", http_status, body)
        err.request_id = request_id
        raise err from exc
    if not isinstance(parsed, dict):
        err = new_protocol_error(op, "response is not a JSON object", http_status, body)
        err.request_id = request_id
        raise err

    code = parsed.get("code")
    if not isinstance(code, str) or len(code) != 6 or any(char < "0" or char > "9" for char in code):
        err = new_protocol_error(op, "response code is missing or not a six-digit string", http_status, body)
        err.request_id = request_id
        raise err
    message = parsed.get("message")
    if message is None:
        message = ""
    if not isinstance(message, str):
        err = new_protocol_error(op, "response message is not a string", http_status, body)
        err.request_id = request_id
        raise err

    text = body.decode("utf-8")
    return Envelope(code=code, message=message, data=_extract_data(text))


async def _read_limited_async(response: httpx.Response, limit: int, ctx: RequestContext) -> bytes:
    async def read() -> bytes:
        chunks: list[bytes] = []
        total = 0
        async for chunk in response.aiter_bytes():
            total += len(chunk)
            if total > limit:
                raise ValueError("response body exceeds the parse limit")
            chunks.append(chunk)
        return b"".join(chunks)

    return await run_with_context(ctx, read)


async def decode_envelope_async(
    op: str,
    response: httpx.Response,
    *,
    meta: ResponseMeta | None = None,
    ctx: RequestContext | None = None,
    limit: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> Envelope:
    request_id = meta.request_id if meta else request_id_from_header(response.headers)
    context = ctx or RequestContext(request_id=request_id)
    try:
        body = await _read_limited_async(response, limit, context)
    except RequestCanceled as exc:
        err = _context_error(op, context, exc)
        err.request_id = request_id
        raise err from exc
    except RequestDeadlineExceeded as exc:
        err = _context_error(op, context, exc)
        err.request_id = request_id
        err.http_status = response.status_code
        raise err from exc
    except ValueError as exc:
        err = new_protocol_error(op, "response body exceeds the parse limit", response.status_code, None)
        err.request_id = request_id
        err.hint = "Response too large to parse; contact the platform for very large documents"
        raise err from exc
    except httpx.HTTPError as exc:
        err = new_transport_error(op, exc, request_id=request_id)
        err.http_status = response.status_code
        raise err from exc
    return _decode_body(op, body, response.status_code, request_id)


__all__ = ["Envelope", "decode_envelope", "decode_envelope_async"]
