from __future__ import annotations

import struct
import threading
import time
from collections import deque
from typing import Any, BinaryIO
from urllib.parse import urlsplit, urlunsplit

from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect

from ..client import Client
from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded
from ..errors import ErrorKind, MaaSError, new_local_error, new_protocol_error
from ..files import FileSource, file
from ..request_id import HEADER_API_REQUEST_ID, HEADER_REQUEST_ID, request_ids_for_asr
from .config import DEFAULT_ASR_PATH, ASRConfig, Config, chunk_samples, init_payload
from .frame import FLAG_POSITIVE_SEQ, MSG_TYPE_FULL_CLIENT, encode_frame
from .session import Result, Session

HANDSHAKE_TIMEOUT = 15.0


class ASRService:
    def __init__(self, client: Client) -> None:
        self.client = client

    def start(self, config: ASRConfig, *, ctx: RequestContext | None = None) -> Session:
        op = "asr.Start"
        ctx = ctx or RequestContext()
        try:
            ctx.raise_if_cancelled()
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            raise MaaSError("canceled while opening the session", kind=ErrorKind.TRANSPORT, op=op, cause=exc) from exc
        endpoint = self.endpoint_url(config)
        try:
            payload = config.init_payload()
        except (TypeError, ValueError) as exc:
            raise new_local_error(op, "cannot build session configuration", "check ASRConfig fields") from exc
        ctx, api_request_id, request_id = request_ids_for_asr(ctx)
        headers = self.handshake_headers(config, api_request_id, request_id)
        remaining = ctx.remaining()
        open_timeout = HANDSHAKE_TIMEOUT if remaining is None else max(0.001, min(HANDSHAKE_TIMEOUT, remaining))
        try:
            conn = connect(
                endpoint,
                additional_headers=headers,
                user_agent_header=None,
                subprotocols=None,
                open_timeout=open_timeout,
                close_timeout=1,
                proxy=None,
            )
        except InvalidStatus as exc:
            raise self._handshake_error(op, exc.response.status_code, exc, request_id) from exc
        except (OSError, TimeoutError) as exc:
            raise MaaSError("ASR handshake failed", kind=ErrorKind.TRANSPORT, op=op, request_id=request_id,
                            hint="check the URL, credentials and network", cause=exc) from exc
        meta = ResponseMeta(request_id=request_id, http_status=101)
        session = Session(conn, config, endpoint, meta)
        try:
            conn.send(encode_frame(MSG_TYPE_FULL_CLIENT, FLAG_POSITIVE_SEQ, 1, payload))
        except Exception as exc:
            conn.close()
            raise MaaSError("failed to send session configuration", kind=ErrorKind.TRANSPORT, op=op,
                            request_id=request_id, cause=exc) from exc
        session._start_reader()
        timeout = config.ready_timeout if config.ready_timeout > 0 else HANDSHAKE_TIMEOUT
        deadline = time.monotonic() + timeout
        while not session._ready.is_set():
            if session._done.is_set():
                result, error = session._outcome()
                session.close()
                if error is not None:
                    raise error
                raise new_protocol_error(
                    op, "connection closed before the session became ready", 101, result.text.encode()
                )
            try:
                ctx.raise_if_cancelled()
            except (RequestCanceled, RequestDeadlineExceeded) as exc:
                session.close()
                raise MaaSError("canceled while opening the session", kind=ErrorKind.TRANSPORT, op=op,
                                request_id=request_id, cause=exc) from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                session.close()
                raise MaaSError("timed out waiting for the session to become ready", kind=ErrorKind.TRANSPORT,
                                op=op, request_id=request_id, retryable=True,
                                hint="check the configuration and any Sec-WebSocket-Protocol header")
            if ctx.wait(min(0.05, remaining)):
                continue
        return session

    def endpoint_url(self, config: ASRConfig) -> str:
        if config.url.strip():
            raw, hint = config.url.strip(), "check Config.url"
            full = True
        else:
            raw = (config.host.strip() or self.client.base_url).rstrip("/")
            hint = "check Config.host or the client base_url"
            full = False
        try:
            parsed = urlsplit(raw)
            if parsed.scheme.lower() not in {"http", "https", "ws", "wss"}:
                raise new_local_error("asr.Start", "unsupported ASR URL scheme", "use http, https, ws or wss")
            if not parsed.hostname:
                raise ValueError("missing host")
            scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme.lower(), parsed.scheme.lower())
            path = parsed.path
            if not full:
                path = path.rstrip("/") + DEFAULT_ASR_PATH
            query, fragment = (parsed.query, parsed.fragment) if full else ("", "")
            return urlunsplit((scheme, parsed.netloc, path, query, fragment))
        except ValueError as exc:
            raise new_local_error("asr.Start", "invalid ASR URL", hint) from exc

    def handshake_headers(self, config: ASRConfig, api_request_id: str, request_id: str) -> dict[str, str]:
        headers = {
            HEADER_API_REQUEST_ID: api_request_id,
            HEADER_REQUEST_ID: request_id,
            "User-Agent": self.client.config.user_agent,
        }
        auth = config.auth
        if auth.resource_id.strip():
            key = auth.api_key or self.client.api_key
            if not key:
                raise MaaSError("missing API key for the direct ASR auth form", kind=ErrorKind.CONFIG, op="asr.Start",
                                hint="set Auth.api_key or the client API key")
            headers.update({"X-Api-Key": key, "X-Api-Resource-Id": auth.resource_id})
        elif auth.app_key.strip() and auth.access_key.strip():
            headers.update({"X-Api-App-Key": auth.app_key, "X-Api-Access-Key": auth.access_key})
        else:
            key = auth.bearer_key or self.client.api_key
            if not key:
                raise MaaSError("missing API key", kind=ErrorKind.CONFIG, op="asr.Start",
                                hint="pass api_key when building the client")
            headers["Authorization"] = f"Bearer {key}"
        reserved = {key.lower() for key in headers}
        for key, value in config.headers.items():
            if value and key.lower() not in reserved:
                headers[key] = value
        return headers

    @staticmethod
    def _handshake_error(op: str, status: int, cause: BaseException, request_id: str) -> MaaSError:
        messages = {
            400: (ErrorKind.API, "bad handshake request", "use a GET upgrade and remove duplicate headers"),
            401: (ErrorKind.API, "unauthorized", "the API key is missing, wrong or expired"),
            404: (ErrorKind.API, "endpoint not found", "the ASR path is wrong; check Config.url"),
            429: (ErrorKind.API, "rate limited", "retry later"),
        }
        kind, message, hint = messages.get(status, (ErrorKind.TRANSPORT, "ASR handshake failed",
                                                    "check the URL, credentials and network"))
        return MaaSError(message, kind=kind, http_status=status, op=op, request_id=request_id,
                         retryable=status == 429, hint=hint, cause=cause)

    def transcribe(
        self, source: FileSource | Any, config: ASRConfig, *, ctx: RequestContext | None = None
    ) -> tuple[Result, MaaSError | None]:
        op = "asr.Transcribe"
        source = file(source)
        if source is None:
            raise new_local_error(op, "no file provided", "pass a file path, bytes, reader or FileSource")
        try:
            reader = source.open()
        except OSError as exc:
            raise new_local_error(op, "cannot open audio file", "check the path") from exc
        with reader:
            try:
                pcm = _strip_wav_header(reader)
            except (OSError, ValueError) as exc:
                raise new_local_error(op, "cannot parse WAV header", "check the file, or pass raw PCM instead") from exc
            if not config.audio.format:
                config.audio.format = "pcm"
            session = self.start(config, ctx=ctx)
            with session:
                consumer = threading.Thread(target=lambda: deque(session.events(), maxlen=0),
                                            name="maas-asr-consumer", daemon=True)
                consumer.start()
                chunk = session.chunk_size() * 2
                try:
                    while True:
                        if ctx is not None:
                            ctx.raise_if_cancelled()
                        data = pcm.read(chunk)
                        if not data:
                            break
                        session.write(data)
                except RequestCanceled as exc:
                    result, _ = session._outcome()
                    return result, session._context_failure(exc)
                except RequestDeadlineExceeded as exc:
                    result, _ = session._outcome()
                    return result, session._context_failure(exc)
                except OSError as exc:
                    result, _ = session._outcome()
                    return result, MaaSError("cannot read audio file", kind=ErrorKind.LOCAL, op=op, cause=exc)
                except MaaSError as exc:
                    result, _ = session._outcome()
                    return result, exc
                return session.finish(ctx=ctx)


def _read_exact(reader: BinaryIO, size: int) -> bytes:
    data = reader.read(size)
    if len(data) != size:
        raise ValueError("truncated WAV header")
    return data


class _HeadReader:
    def __init__(self, head: bytes, reader: BinaryIO) -> None:
        self.head = head
        self.reader = reader

    def read(self, size: int = -1) -> bytes:
        if not self.head:
            return self.reader.read(size)
        if size < 0:
            head, self.head = self.head, b""
            return head + self.reader.read()
        head, self.head = self.head[:size], self.head[size:]
        return head


def _strip_wav_header(reader: BinaryIO) -> BinaryIO | _HeadReader:
    head = reader.read(12)
    if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return _HeadReader(head, reader)
    while True:
        chunk_header = _read_exact(reader, 8)
        chunk_id = chunk_header[:4]
        size = struct.unpack("<I", chunk_header[4:])[0]
        if chunk_id == b"data":
            return reader
        remaining = size
        while remaining:
            skipped = reader.read(min(remaining, 65536))
            if not skipped:
                raise ValueError(f"truncated WAV chunk {chunk_id!r}")
            remaining -= len(skipped)
        if size % 2:
            if not reader.read(1):
                raise ValueError("truncated WAV chunk padding")


Service = ASRService

__all__ = ["ASRService", "Service", "Result", "Session", "DEFAULT_ASR_PATH", "Config", "init_payload", "chunk_samples"]
