from __future__ import annotations

import json
import queue
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, cast

from websockets.exceptions import ConnectionClosed
from websockets.sync.connection import Connection

from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded
from ..errors import CODE_HOTWORD_TABLE_NOT_PERMITTED, ErrorKind, MaaSError, new_local_error
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


class EventKind(StrEnum):
    READY = "ready"
    SNAPSHOT = "snapshot"
    UTTERANCE = "utterance"
    TERMINAL = "terminal"
    ERROR = "error"


@dataclass(slots=True)
class Utterance:
    text: str = ""
    definite: bool = False
    start_ms: int = 0
    end_ms: int = 0


@dataclass(slots=True)
class Event:
    kind: EventKind
    text: str = ""
    utterances: list[Utterance] = field(default_factory=list)
    is_final: bool = False
    status: str = ""
    log_id: str = ""
    duration: float = 0.0
    err: MaaSError | None = None
    raw: bytes = b""


@dataclass(slots=True)
class Result:
    meta: ResponseMeta = field(default_factory=ResponseMeta)
    status: str = ""
    text: str = ""
    utterances: list[Utterance] = field(default_factory=list)
    duration: float = 0.0
    log_id: str = ""
    succeeded: bool = False


_STREAM_END = object()


def _wait_done(done: threading.Event, ctx: RequestContext, op: str, message: str) -> MaaSError | None:
    while not done.is_set():
        try:
            ctx.raise_if_cancelled()
        except (RequestCanceled, RequestDeadlineExceeded) as exc:
            return MaaSError(message, kind=ErrorKind.TRANSPORT, op=op, request_id=ctx.request_id,
                             retryable=True, cause=exc)
        if not ctx.wait(0.05):
            continue
    return None


class Session:
    """A streaming ASR session. Events must be drained to avoid backpressure."""

    def __init__(self, conn: Connection, config: Any, endpoint: str, meta: ResponseMeta) -> None:
        self._conn = conn
        self._chunk_size = cast(int, config.chunk_samples())
        self._endpoint = endpoint
        self._meta = meta
        self._events: queue.Queue[Event | object] = queue.Queue(maxsize=64)
        self._ready = threading.Event()
        self._done = threading.Event()
        self._stop = threading.Event()
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._sequence = 1
        self._result = Result(meta=meta)
        self._run_error: MaaSError | None = None
        self._reader = threading.Thread(target=self._read_loop, name="maas-asr-reader", daemon=True)

    def _start_reader(self) -> None:
        self._reader.start()

    def events(self) -> Iterator[Event]:
        while True:
            item = self._events.get()
            if item is _STREAM_END:
                return
            assert isinstance(item, Event)
            yield item

    def __iter__(self) -> Iterator[Event]:
        return self.events()

    def write(self, pcm: bytes | bytearray | memoryview) -> int:
        data = bytes(pcm)
        if not data:
            return 0
        with self._write_lock:
            if self._closed or self._done.is_set():
                raise new_local_error("asr.Write", "session is closed", "open a new session")
            sequence = self._sequence + 1
            try:
                frame = encode_frame(MSG_TYPE_AUDIO_ONLY, FLAG_POSITIVE_SEQ, sequence, data)
                self._conn.send(frame)
            except ConnectionClosed as exc:
                raise MaaSError("failed to send audio", kind=ErrorKind.TRANSPORT, op="asr.Write",
                                request_id=self._meta.request_id, cause=exc) from exc
            self._sequence = sequence
        return len(data)

    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_samples(self) -> int:
        return self._chunk_size

    def request_id(self) -> str:
        return self._meta.request_id

    def finish(self, *, ctx: RequestContext | None = None) -> tuple[Result, MaaSError | None]:
        ctx = ctx or RequestContext(request_id=self._meta.request_id)
        with self._state_lock:
            already_closed = self._closed
        if already_closed or self._done.is_set():
            return self._outcome()
        with self._write_lock:
            if self._closed or self._done.is_set():
                return self._outcome()
            sequence = self._sequence + 1
            self._sequence = sequence
            try:
                self._conn.send(encode_frame(MSG_TYPE_AUDIO_ONLY, FLAG_NEGATIVE_SEQ, -sequence, b""))
            except ConnectionClosed as exc:
                error = MaaSError("failed to send the finish frame", kind=ErrorKind.TRANSPORT, op="asr.Finish",
                                  request_id=self._meta.request_id, cause=exc)
                self._set_error(error)
                return self._outcome()
        wait_error = _wait_done(self._done, ctx, "asr.Finish", "timed out waiting for the terminal state")
        if wait_error is not None:
            outcome, run_error = self._outcome()
            return outcome, run_error or wait_error
        return self._outcome()

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
        try:
            self._conn.close()
        except (ConnectionClosed, OSError):
            pass

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _context_failure(self, exc: RequestCanceled | RequestDeadlineExceeded) -> MaaSError:
        return MaaSError("ASR request context ended", kind=ErrorKind.TRANSPORT, op="asr.Transcribe",
                         request_id=self._meta.request_id, retryable=True, cause=exc)

    def _outcome(self) -> tuple[Result, MaaSError | None]:
        with self._state_lock:
            # Return a snapshot so callers cannot mutate aggregation state.
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
        with self._state_lock:
            if self._run_error is None:
                self._run_error = error

    def _emit(self, event: Event) -> bool:
        with self._state_lock:
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
        while not self._stop.is_set():
            try:
                self._events.put(event, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _read_loop(self) -> None:
        try:
            while True:
                try:
                    message = self._conn.recv()
                except ConnectionClosed:
                    if not self._closed:
                        self._set_error(MaaSError("ASR connection closed before the terminal state",
                                                  kind=ErrorKind.TRANSPORT, op="asr.read",
                                                  request_id=self._meta.request_id))
                    return
                if not isinstance(message, bytes):
                    continue
                try:
                    frame = decode_frame(message)
                    terminal, error = self._handle_frame(frame)
                except (ValueError, TypeError, AttributeError) as exc:
                    error = MaaSError("malformed ASR frame", kind=ErrorKind.PROTOCOL, op="asr.read",
                                      request_id=self._meta.request_id, cause=exc)
                    terminal = True
                if error is not None:
                    self._set_error(error)
                    self._emit(Event(EventKind.ERROR, is_final=True, err=error))
                    return
                if terminal:
                    return
        finally:
            self._done.set()
            self._stop.set()
            while True:
                try:
                    self._events.put(_STREAM_END, timeout=0.05)
                    break
                except queue.Full:
                    continue

    def _handle_frame(self, frame: ASRFrame) -> tuple[bool, MaaSError | None]:
        if frame.type == MSG_TYPE_ERROR:
            return True, self._api_error(frame)
        if frame.type != MSG_TYPE_FULL_SERVER:
            return False, None
        try:
            wire = json.loads(frame.payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot decode ASR payload: {exc}") from exc
        if not isinstance(wire, dict):
            raise ValueError("cannot decode ASR payload: expected an object")
        body = wire.get("result")
        additions = wire.get("additions")
        payload_msg = wire.get("payload_msg")
        duration = 0.0
        if isinstance(payload_msg, dict):
            if isinstance(payload_msg.get("result"), dict):
                body = payload_msg["result"]
            audio_info = payload_msg.get("audio_info")
            if isinstance(audio_info, dict):
                duration = float(audio_info.get("duration") or 0) / 1000
        if not isinstance(body, dict):
            body = {}
        if isinstance(body.get("additions"), dict):
            additions = body["additions"]
        parret = additions.get("parret") if isinstance(additions, dict) else None
        if not isinstance(parret, dict):
            parret = {}
        kind = parret.get("kind", "")
        utterances = [
            Utterance(str(item.get("text", "")), bool(item.get("definite", False)),
                      int(item.get("start_time", 0)), int(item.get("end_time", 0)))
            for item in body.get("utterances", []) if isinstance(item, dict)
        ]
        is_final = bool(wire.get("is_last_package")) or frame.flags == FLAG_NEGATIVE_SEQ or kind == "terminal"
        event = Event(
            EventKind.SNAPSHOT,
            text=str(body.get("text", "")),
            utterances=utterances,
            is_final=is_final,
            log_id=str(additions.get("log_id", "")) if isinstance(additions, dict) else "",
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
            session = parret.get("session")
            event.status = str(session.get("status", "")) if isinstance(session, dict) else ""
            if int(wire.get("code") or 0) != 0:
                event.status = "failed"
            elif not event.status:
                event.status = "succeeded"
            if not self._ready.is_set():
                return True, MaaSError("session ended before it became ready", kind=ErrorKind.API,
                                       op="asr.Start", request_id=self._meta.request_id)
        self._emit(event)
        return terminal, None

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
        return MaaSError("ASR session error", kind=ErrorKind.API, code=str(frame.code), hint=hint,
                         op="asr.session", request_id=self._meta.request_id, raw_body=frame.payload[:1024])


__all__ = ["EventKind", "Utterance", "Event", "Result", "Session"]
