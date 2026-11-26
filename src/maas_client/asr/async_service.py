from __future__ import annotations

import asyncio
from functools import partial
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from ..async_client import AsyncClient
from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded, run_with_context
from ..errors import ErrorKind, MaaSError, new_local_error, new_protocol_error
from ..files import FileSource, file
from ..request_id import HEADER_API_REQUEST_ID, HEADER_REQUEST_ID, request_ids_for_asr
from .async_session import AsyncSession
from .config import DEFAULT_ASR_PATH, ASRConfig
from .frame import FLAG_POSITIVE_SEQ, MSG_TYPE_FULL_CLIENT, encode_frame
from .service import _strip_wav_header
from .session import Result

HANDSHAKE_TIMEOUT = 15.0


class AsyncASRService:
    def __init__(self, client: AsyncClient) -> None:
        self.client = client

    async def start(self, config: ASRConfig, *, ctx: RequestContext | None = None) -> AsyncSession:
        op = "asr.Start"
        ctx = ctx or RequestContext()
        try:
            ctx.raise_if_cancelled()
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            raise MaaSError(
                "canceled while opening the session", kind=ErrorKind.TRANSPORT, op=op, cause=exc
            ) from exc
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
            conn: ClientConnection = await run_with_context(
                ctx,
                lambda: connect(
                    endpoint,
                    additional_headers=headers,
                    user_agent_header=None,
                    subprotocols=None,
                    open_timeout=open_timeout,
                    close_timeout=1,
                    proxy=None,
                ),
            )
        except InvalidStatus as exc:
            raise self._handshake_error(op, exc.response.status_code, exc, request_id) from exc
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            raise MaaSError(
                "canceled while opening the session",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                cause=exc,
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise MaaSError(
                "ASR handshake failed",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                hint="check the URL, credentials and network",
                cause=exc,
            ) from exc

        meta = ResponseMeta(request_id=request_id, http_status=101)
        session = AsyncSession(conn, config, endpoint, meta)
        session._start_reader()
        try:
            frame = encode_frame(MSG_TYPE_FULL_CLIENT, FLAG_POSITIVE_SEQ, 1, payload)
            await run_with_context(ctx, lambda: conn.send(frame))
        except asyncio.CancelledError:
            await session.close()
            raise
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            await session.close()
            raise MaaSError(
                "canceled while opening the session",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                cause=exc,
            ) from exc
        except Exception as exc:
            await session.close()
            raise MaaSError(
                "failed to send session configuration",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                cause=exc,
            ) from exc

        timeout = config.ready_timeout if config.ready_timeout > 0 else HANDSHAKE_TIMEOUT
        try:
            early_outcome = await run_with_context(ctx, lambda: session._wait_ready(timeout))
        except asyncio.CancelledError:
            await session.close()
            raise
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            await session.close()
            raise MaaSError(
                "canceled while opening the session",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                cause=exc,
            ) from exc
        except TimeoutError as exc:
            await session.close()
            raise MaaSError(
                "timed out waiting for the session to become ready",
                kind=ErrorKind.TRANSPORT,
                op=op,
                request_id=request_id,
                retryable=True,
                hint="check the configuration and any Sec-WebSocket-Protocol header",
                cause=exc,
            ) from exc
        if early_outcome is not None:
            result, error = early_outcome
            await session.close()
            if error is not None:
                raise error
            raise new_protocol_error(
                op, "connection closed before the session became ready", 101, result.text.encode()
            )
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
                raise MaaSError(
                    "missing API key for the direct ASR auth form",
                    kind=ErrorKind.CONFIG,
                    op="asr.Start",
                    hint="set Auth.api_key or the client API key",
                )
            headers.update({"X-Api-Key": key, "X-Api-Resource-Id": auth.resource_id})
        elif auth.app_key.strip() and auth.access_key.strip():
            headers.update({"X-Api-App-Key": auth.app_key, "X-Api-Access-Key": auth.access_key})
        else:
            key = auth.bearer_key or self.client.api_key
            if not key:
                raise MaaSError(
                    "missing API key",
                    kind=ErrorKind.CONFIG,
                    op="asr.Start",
                    hint="pass api_key when building the client",
                )
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
        kind, message, hint = messages.get(
            status,
            (ErrorKind.TRANSPORT, "ASR handshake failed", "check the URL, credentials and network"),
        )
        return MaaSError(
            message,
            kind=kind,
            http_status=status,
            op=op,
            request_id=request_id,
            retryable=status == 429,
            hint=hint,
            cause=cause,
        )

    async def transcribe(
        self, source: FileSource | Any, config: ASRConfig, *, ctx: RequestContext | None = None
    ) -> tuple[Result, MaaSError | None]:
        op = "asr.Transcribe"
        source = file(source)
        if source is None:
            raise new_local_error(op, "no file provided", "pass a file path, bytes, reader or FileSource")
        try:
            reader = await asyncio.to_thread(source.open)
        except OSError as exc:
            raise new_local_error(op, "cannot open audio file", "check the path") from exc
        try:
            try:
                pcm = await asyncio.to_thread(_strip_wav_header, reader)
            except (OSError, ValueError) as exc:
                raise new_local_error(
                    op, "cannot parse WAV header", "check the file, or pass raw PCM instead"
                ) from exc
            if not config.audio.format:
                config.audio.format = "pcm"
            session = await self.start(config, ctx=ctx)
            async with session:
                consumer = asyncio.create_task(_drain_events(session), name="maas-asr-consumer")
                try:
                    chunk = session.chunk_size() * 2
                    while True:
                        if ctx is not None:
                            ctx.raise_if_cancelled()
                        data = await asyncio.to_thread(pcm.read, chunk)
                        if not data:
                            break
                        if ctx is None:
                            await session.write(data)
                        else:
                            await run_with_context(ctx, partial(session.write, data))
                    return await session.finish(ctx=ctx)
                except asyncio.CancelledError:
                    await session.close()
                    raise
                except (RequestCanceled, RequestDeadlineExceeded) as exc:
                    result, _ = session._outcome()
                    return result, session._context_failure(exc)
                except OSError as exc:
                    result, _ = session._outcome()
                    return result, MaaSError("cannot read audio file", kind=ErrorKind.LOCAL, op=op, cause=exc)
                except MaaSError as exc:
                    result, _ = session._outcome()
                    return result, exc
                finally:
                    if not consumer.done():
                        consumer.cancel()
                    await asyncio.gather(consumer, return_exceptions=True)
        finally:
            await asyncio.to_thread(reader.close)


async def _drain_events(session: AsyncSession) -> None:
    async for _ in session.events():
        pass


AsyncService = AsyncASRService

__all__ = ["AsyncASRService", "AsyncService"]
