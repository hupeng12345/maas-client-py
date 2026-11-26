from __future__ import annotations

import asyncio
import json
import math
import zlib
from collections.abc import AsyncIterator
from typing import cast

from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded, run_with_context
from ..errors import CODE_HOTWORD_TABLE_NOT_PERMITTED, ErrorKind, MaaSError, new_local_error
from .config import ASRConfig
from .frame import (
    FLAG_NEGATIVE_SEQ,
    FLAG_POSITIVE_SEQ,
    MSG_TYPE_AUDIO_ONLY,
    MSG_TYPE_ERROR,
    MSG_TYPE_FULL_SERVER,
    ASRFrame,
    decode_frame,
    encode_frame,
)
from .session import Event, EventKind, Result, Utterance

_STREAM_END = object()


async def _close_connection(conn: ClientConnection) -> None:
    aclose = getattr(conn, "aclose", None)
    if aclose is not None:
        await aclose()
    else:
        # websockets 14/15 name the asynchronous close method close().
        await conn.close()


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


class AsyncSession:
    """A streaming ASR session backed by an asyncio websocket and reader task."""

    def __init__(self, conn: ClientConnection, config: ASRConfig, endpoint: str, meta: ResponseMeta) -> None:
        self._conn = conn
        self._chunk_size = config.chunk_samples()
        self._endpoint = endpoint
        self._meta = meta
        self._events: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=64)
        self._ready = asyncio.Event()
        self._done = asyncio.Event()
        self._write_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._sequence = 1
        self._result = Result(meta=meta)
        self._run_error: MaaSError | None = None
        self._reader: asyncio.Task[None] | None = None

    def _start_reader(self) -> None:
        self._reader = asyncio.create_task(self._read_loop(), name="maas-asr-reader")

    async def events(self) -> AsyncIterator[Event]:
        try:
            while True:
                if self._done.is_set() and self._events.empty():
                    return
                item = await self._events.get()
                if item is _STREAM_END:
                    return
                assert isinstance(item, Event)
                yield item
        except asyncio.CancelledError:
            await self.close()
            raise

    def __aiter__(self) -> AsyncIterator[Event]:
        return self.events()

    async def write(self, pcm: bytes | bytearray | memoryview) -> int:
        data = bytes(pcm)
        if not data:
            return 0
        try:
            async with self._write_lock:
                if self._closed or self._done.is_set():
                    raise new_local_error("asr.Write", "session is closed", "open a new session")
                sequence = self._sequence + 1
                frame = encode_frame(MSG_TYPE_AUDIO_ONLY, FLAG_POSITIVE_SEQ, sequence, data)
                await self._conn.send(frame)
                self._sequence = sequence
        except asyncio.CancelledError:
            await self.close()
            raise
        except ConnectionClosed as exc:
            raise MaaSError(
                "failed to send audio",
                kind=ErrorKind.TRANSPORT,
                op="asr.Write",
                request_id=self._meta.request_id,
                cause=exc,
            ) from exc
        return len(data)

    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_samples(self) -> int:
        return self._chunk_size

    def request_id(self) -> str:
        return self._meta.request_id

    async def finish(self, *, ctx: RequestContext | None = None) -> tuple[Result, MaaSError | None]:
        ctx = ctx or RequestContext(request_id=self._meta.request_id)
        if self._closed or self._done.is_set():
            return self._outcome()
        try:
            async with self._write_lock:
                if self._closed or self._done.is_set():
                    return self._outcome()
                sequence = self._sequence + 1
                self._sequence = sequence
                frame = encode_frame(MSG_TYPE_AUDIO_ONLY, FLAG_NEGATIVE_SEQ, -sequence, b"")
                try:
                    await run_with_context(ctx, lambda: self._conn.send(frame))
                except ConnectionClosed as exc:
                    error = MaaSError(
                        "failed to send the finish frame",
                        kind=ErrorKind.TRANSPORT,
                        op="asr.Finish",
                        request_id=self._meta.request_id,
                        cause=exc,
                    )
                    self._set_error(error)
                    return self._outcome()
            await run_with_context(ctx, self._done.wait)
        except asyncio.CancelledError:
            await self.close()
            raise
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            outcome, run_error = self._outcome()
            error = MaaSError(
                "timed out waiting for the terminal state",
                kind=ErrorKind.TRANSPORT,
                op="asr.Finish",
                request_id=ctx.request_id,
                retryable=True,
                cause=exc,
            )
            return outcome, run_error or error
        return self._outcome()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._done.set()
            reader = self._reader
            if reader is not None and reader is not asyncio.current_task() and not reader.done():
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            try:
                await _close_connection(self._conn)
            except (ConnectionClosed, OSError):
                pass
            self._notify_stream_end()

    async def __aenter__(self) -> AsyncSession:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    def _context_failure(self, exc: RequestCanceled | RequestDeadlineExceeded) -> MaaSError:
        return MaaSError(
            "ASR request context ended",
            kind=ErrorKind.TRANSPORT,
            op="asr.Transcribe",
            request_id=self._meta.request_id,
            retryable=True,
            cause=exc,
        )

    def _outcome(self) -> tuple[Result, MaaSError | None]:
        return Result(
            meta=self._result.meta,
            status=self._result.status,
            text=self._result.text,
            utterances=list(self._result.utterances),
            duration=self._result.duration,
            log_id=self._result.log_id,
            succeeded=self._result.succeeded,
        ), self._run_error

    def _set_error(self, error: MaaSError) -> None:
        if self._run_error is None:
            self._run_error = error

    def _notify_stream_end(self) -> None:
        try:
            self._events.put_nowait(_STREAM_END)
        except asyncio.QueueFull:
            # Once the queued events are drained, events() observes _done and returns.
            pass

    async def _emit(self, event: Event) -> bool:
        while not self._done.is_set():
            try:
                await self._events.put(event)
                return True
            except asyncio.CancelledError:
                raise
        return False

    async def _wait_ready(self, timeout: float) -> tuple[Result, MaaSError | None] | None:
        ready_task = asyncio.create_task(self._ready.wait())
        done_task = asyncio.create_task(self._done.wait())
        try:
            done, _ = await asyncio.wait({ready_task, done_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if self._ready.is_set():
                return None
            if done_task in done:
                return self._outcome()
            raise TimeoutError
        finally:
            for task in (ready_task, done_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(ready_task, done_task, return_exceptions=True)

    async def _read_loop(self) -> None:
        try:
            while True:
                try:
                    message = await self._conn.recv()
                except ConnectionClosed:
                    if not self._closed:
                        self._set_error(
                            MaaSError(
                                "ASR connection closed before the terminal state",
                                kind=ErrorKind.TRANSPORT,
                                op="asr.read",
                                request_id=self._meta.request_id,
                            )
                        )
                    return
                if not isinstance(message, bytes):
                    continue
                try:
                    frame = decode_frame(message)
                    terminal, error = await self._handle_frame(frame)
                except (ValueError, TypeError, AttributeError, OverflowError, zlib.error) as exc:
                    error = MaaSError(
                        "malformed ASR frame",
                        kind=ErrorKind.PROTOCOL,
                        op="asr.read",
                        request_id=self._meta.request_id,
                        cause=exc,
                    )
                    terminal = True
                if error is not None:
                    self._set_error(error)
                    await self._emit(Event(EventKind.ERROR, is_final=True, err=error))
                    return
                if terminal:
                    return
        finally:
            self._done.set()
            self._notify_stream_end()

    async def _handle_frame(self, frame: ASRFrame) -> tuple[bool, MaaSError | None]:
        if frame.type == MSG_TYPE_ERROR:
            return True, self._api_error(frame)
        if frame.type != MSG_TYPE_FULL_SERVER:
            return False, None
        try:
            wire_value = json.loads(frame.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot decode ASR payload: {exc}") from exc
        wire = _object(wire_value, "ASR payload")

        body_value = wire.get("result")
        body = {} if body_value is None else _object(body_value, "result")
        additions_value = wire.get("additions")
        additions = {} if additions_value is None else _object(additions_value, "additions")
        duration = 0.0
        payload_msg_value = wire.get("payload_msg")
        if payload_msg_value is not None:
            payload_msg = _object(payload_msg_value, "payload_msg")
            wrapped_result = payload_msg.get("result")
            if wrapped_result is not None:
                body = _object(wrapped_result, "payload_msg.result")
            audio_info_value = payload_msg.get("audio_info")
            if audio_info_value is not None:
                audio_info = _object(audio_info_value, "payload_msg.audio_info")
                raw_duration = audio_info.get("duration")
                if raw_duration is not None:
                    if isinstance(raw_duration, bool) or not isinstance(raw_duration, (int, float)):
                        raise ValueError("audio duration must be a number")
                    duration = float(raw_duration) / 1000
                    if not math.isfinite(duration):
                        raise ValueError("audio duration must be finite")

        body_additions = body.get("additions")
        if body_additions is not None:
            additions = _object(body_additions, "result.additions")
        parret_value = additions.get("parret")
        parret = {} if parret_value is None else _object(parret_value, "additions.parret")
        kind = _text(parret.get("kind", ""), "additions.parret.kind")

        raw_utterances = body.get("utterances", [])
        if not isinstance(raw_utterances, list):
            raise ValueError("result.utterances must be an array")
        utterances = [
            Utterance(
                text=_text(item.get("text", ""), "utterance.text"),
                definite=_boolean(item.get("definite", False), "utterance.definite"),
                start_ms=_integer(item.get("start_time", 0), "utterance.start_time"),
                end_ms=_integer(item.get("end_time", 0), "utterance.end_time"),
            )
            for item in raw_utterances
            if isinstance(item, dict)
        ]
        raw_final = wire.get("is_last_package", False)
        if raw_final is None:
            raw_final = False
        is_final = _boolean(raw_final, "is_last_package") or frame.flags == FLAG_NEGATIVE_SEQ or kind == "terminal"
        event = Event(
            EventKind.SNAPSHOT,
            text=_text(body.get("text", ""), "result.text"),
            utterances=utterances,
            is_final=is_final,
            log_id=_text(additions.get("log_id", ""), "additions.log_id"),
            duration=duration,
            raw=frame.payload,
        )
        terminal = False
        if kind == "ready":
            event.kind = EventKind.READY
            self._ready.set()
        elif kind == "unit_settled":
            event.kind = EventKind.UTTERANCE
        elif kind == "terminal" or is_final:
            terminal = True
            event.kind = EventKind.TERMINAL
            session_value = parret.get("session")
            session = {} if session_value is None else _object(session_value, "additions.parret.session")
            event.status = _text(session.get("status", ""), "session.status")
            if _integer(wire.get("code", 0), "code") != 0:
                event.status = "failed"
            elif not event.status:
                event.status = "succeeded"
            if not self._ready.is_set():
                return True, MaaSError(
                    "session ended before it became ready",
                    kind=ErrorKind.API,
                    op="asr.Start",
                    request_id=self._meta.request_id,
                )
        self._record_event(event)
        await self._emit(event)
        return terminal, None

    def _record_event(self, event: Event) -> None:
        if event.kind in {EventKind.TERMINAL, EventKind.ERROR} or event.text or event.kind == EventKind.READY:
            if event.text:
                self._result.text = event.text
            if event.utterances:
                self._result.utterances = list(event.utterances)
            if event.log_id:
                self._result.log_id = event.log_id
            if event.duration:
                self._result.duration = event.duration
            if event.kind == EventKind.TERMINAL:
                self._result.status = event.status
                self._result.succeeded = event.status == "succeeded"

    def _api_error(self, frame: ASRFrame) -> MaaSError:
        hint = "inspect the error frame payload"
        try:
            payload = json.loads(frame.payload)
            if isinstance(payload, dict):
                parret = payload.get("parret")
                reason = str(parret.get("reason", "")) if isinstance(parret, dict) else ""
                hints = {
                    "invalid_request": "check the frame structure and request fields",
                    "unsupported_option": "check format, language and option values",
                    "configuration_rejected": "check hotword configuration",
                    "invalid_sequence": "sequence numbers went backwards or skipped",
                    "unusable_audio": "check that samples are 16 kHz/16-bit/mono",
                    "input_idle_timeout": "send audio continuously",
                    "capacity_exhausted": "a frame exceeded 16000 samples or the concurrency quota is exhausted",
                    "input_after_end": "audio was sent after the finish frame",
                }
                if reason:
                    hint = f"{reason}: {hints.get(reason, 'the server rejected the session')}"
                elif payload.get("code") == CODE_HOTWORD_TABLE_NOT_PERMITTED:
                    hint = "a hotword table is not authorised; check forbidden_table_ids"
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
        return MaaSError(
            "ASR session error",
            kind=ErrorKind.API,
            code=str(frame.code),
            hint=hint,
            op="asr.session",
            request_id=self._meta.request_id,
            raw_body=frame.payload[:1024],
        )


__all__ = ["AsyncSession"]
