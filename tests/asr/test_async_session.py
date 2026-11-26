from __future__ import annotations

import asyncio
import gzip
import io
import json
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import pytest
from websockets.exceptions import ConnectionClosed

from maas_client.asr import AsyncASRService, AsyncService, AsyncSession, Auth, Config, EventKind, Segmentation
from maas_client.asr.frame import (
    FLAG_NEGATIVE_SEQ,
    FLAG_POSITIVE_SEQ,
    MSG_TYPE_AUDIO_ONLY,
    MSG_TYPE_FULL_CLIENT,
    MSG_TYPE_FULL_SERVER,
    ASRFrame,
    decode_frame,
    encode_frame,
)
from maas_client.async_client import AsyncClient
from maas_client.context import RequestCanceled, RequestContext
from maas_client.errors import ErrorKind, MaaSError


@dataclass
class Recording:
    headers: dict[str, str] = field(default_factory=dict)
    path: str = ""
    frames: list[ASRFrame] = field(default_factory=list)
    received: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    closed: threading.Event = field(default_factory=threading.Event)


def make_client(base_url: str = "http://127.0.0.1:1") -> AsyncClient:
    return AsyncClient(api_key="sk_test", base_url=base_url, user_agent="test-agent")


def endpoint(server: Any, suffix: str = "") -> str:
    return f"http://127.0.0.1:{server.socket.getsockname()[1]}{suffix}"


def send_json(conn: Any, payload: object, *, flags: int = FLAG_POSITIVE_SEQ, sequence: int = 1) -> None:
    conn.send(encode_frame(MSG_TYPE_FULL_SERVER, flags, sequence, json.dumps(payload).encode()))


def ready(conn: Any) -> None:
    send_json(conn, {"result": {"text": ""}, "additions": {"parret": {"kind": "ready"}}})


def terminal(conn: Any, text: str = "hello", *, code: int = 0, wrapped: bool = False) -> None:
    body = {
        "text": text,
        "utterances": [{"text": text, "definite": True, "start_time": 0, "end_time": 500}],
        "additions": {"log_id": "log-1", "parret": {"kind": "terminal", "session": {"status": "succeeded"}}},
    }
    payload: dict[str, object] = {"payload_msg": {"audio_info": {"duration": 500}, "result": body}, "code": code}
    if not wrapped:
        payload = {"result": body, "code": code}
    send_json(conn, payload, flags=FLAG_NEGATIVE_SEQ, sequence=2)


def record_handler(
    recording: Recording, reply: Callable[[Any, ASRFrame], bool | None]
) -> Callable[[Any], None]:
    def handler(conn: Any) -> None:
        recording.headers = {key.lower(): value for key, value in conn.request.headers.items()}
        recording.path = conn.request.path
        try:
            while True:
                message = conn.recv()
                if not isinstance(message, bytes):
                    continue
                frame = decode_frame(message)
                recording.frames.append(frame)
                recording.received.set()
                if frame.type == MSG_TYPE_AUDIO_ONLY and frame.flags == FLAG_NEGATIVE_SEQ:
                    recording.finished.set()
                if reply(conn, frame):
                    return
        except ConnectionClosed:
            pass
        finally:
            recording.closed.set()
            conn.close()

    return handler


def test_framing_events_headers_and_endpoint(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn, wrapped=True)
            return True
        return False

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            ctx = RequestContext.timeout(3, request_id="generic-request")
            config = Config(
                host=endpoint(server, "/prefix?q=1"),
                headers={"X-Custom": "yes", "Authorization": "bad", "x-request-id": "bad", "X-Empty": ""},
            )
            async with (await AsyncASRService(client).start(config, ctx=ctx)) as session:
                assert isinstance(session, AsyncSession)
                assert session.chunk_size() == session.chunk_samples == 3200
                assert session.request_id() == "generic-request"
                assert await session.write(b"") == 0
                assert await session.write(b"\x01\x02") == 2
                result, error = await session.finish(ctx=ctx)
                assert error is None
                assert result.succeeded and result.status == "succeeded"
                assert result.text == "hello"
                assert result.log_id == "log-1"
                assert result.duration == 0.5
                assert result.meta.http_status == 101
                assert result.utterances[0].end_ms == 500
                events = [event async for event in session.__aiter__()]
                assert [event.kind for event in events] == [EventKind.READY, EventKind.TERMINAL]
                assert events[-1].raw

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)
    assert recording.path == "/prefix/-/volcengine/api/v3/sauc/bigmodel_async"
    UUID(recording.headers["x-api-request-id"])
    assert recording.headers["x-request-id"] == "generic-request"
    assert recording.headers["user-agent"] == "test-agent"
    assert recording.headers["authorization"] == "Bearer sk_test"
    assert recording.headers["x-custom"] == "yes"
    assert "x-empty" not in recording.headers
    assert recording.headers.get("sec-websocket-protocol", "") == ""
    assert [frame.sequence for frame in recording.frames] == [1, 2, -3]
    assert recording.frames[0].type == MSG_TYPE_FULL_CLIENT
    assert json.loads(recording.frames[0].payload)["request"]["model_name"] == "bigmodel"
    assert recording.frames[1].payload == b"\x01\x02"
    assert recording.frames[2].payload == b""


@pytest.mark.parametrize("auth_kind", ["bearer", "direct", "legacy"])
def test_auth_precedence_and_full_url_override(server_factory: Any, auth_kind: str) -> None:
    recording = Recording()
    auth = Auth()
    if auth_kind == "bearer":
        auth = Auth(bearer_key="override")
    elif auth_kind == "direct":
        auth = Auth(resource_id="res", api_key="direct-key")
    else:
        auth = Auth(app_key="app", access_key="access")

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            service = AsyncService(client)
            assert service.endpoint_url(Config(url="https://example.com/exact?q=1#frag")) == (
                "wss://example.com/exact?q=1#frag"
            )
            session = await service.start(Config(url=endpoint(server, "/exact?q=1"), auth=auth))
            await session.close()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        return None

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)
    assert recording.path == "/exact?q=1"
    if auth_kind == "bearer":
        assert recording.headers["authorization"] == "Bearer override"
    elif auth_kind == "direct":
        assert recording.headers["x-api-key"] == "direct-key"
        assert recording.headers["x-api-resource-id"] == "res"
        assert "authorization" not in recording.headers
    else:
        assert recording.headers["x-api-app-key"] == "app"
        assert recording.headers["x-api-access-key"] == "access"


def test_unknown_frames_snapshots_and_partial_terminal(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_POSITIVE_SEQ:
            conn.send(encode_frame(0xA, FLAG_POSITIVE_SEQ, 2, b"ignored"))
            send_json(
                conn,
                {
                    "result": {"text": "stale"},
                    "payload_msg": {
                        "audio_info": {"duration": 250},
                        "result": {"text": "snapshot", "additions": {"log_id": "inner"}},
                    },
                },
            )
            send_json(conn, {"result": {"text": "settled"}, "additions": {"parret": {"kind": "unit_settled"}}})
        else:
            send_json(conn, {"code": 0, "is_last_package": True, "result": {"text": "fallback"}})
            return True
        return False

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                await session.write(b"audio")
                result, error = await session.finish(ctx=RequestContext.timeout(3))
                assert error is None
                assert result.succeeded and result.text == "fallback"
                assert result.duration == 0.25 and result.log_id == "inner"
                events = [event async for event in session.events()]
                assert [event.kind for event in events] == [
                    EventKind.READY,
                    EventKind.SNAPSHOT,
                    EventKind.UTTERANCE,
                    EventKind.TERMINAL,
                ]
                assert events[1].text == "snapshot"
                assert events[2].text == "settled"

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_error_frame_and_terminal_failure(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            payload = b'{"parret":{"reason":"unusable_audio"}}'
            body = gzip.compress(payload, mtime=0)
            conn.send(struct.pack(">BBBBiII", 0x11, 0xF1, 0x11, 0, 1, 45000002, len(body)) + body)
            return True
        return False

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                result, error = await session.finish(ctx=RequestContext.timeout(3))
                assert not result.succeeded
                assert error is not None and error.kind == ErrorKind.API
                assert error.code == "45000002" and "unusable_audio" in error.hint
                events = [event async for event in session.events()]
                assert events[-1].kind == EventKind.ERROR
                assert events[-1].err is error

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)

    failed = Recording()

    def failed_reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn, code=400001)
            return True
        return False

    async def failed_scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                result, error = await session.finish(ctx=RequestContext.timeout(3))
                assert error is None
                assert result.status == "failed" and not result.succeeded

    with server_factory(record_handler(failed, failed_reply)) as server:
        asyncio.run(failed_scenario(server))
    assert failed.closed.wait(2)


def malformed_gzip() -> bytes:
    body = bytearray(gzip.compress(b'{"result":{}}', mtime=0))
    body[10] ^= 0xFF
    return struct.pack(">BBBBii", 0x11, 0x91, 0x11, 0, 1, len(body)) + body


@pytest.mark.parametrize(
    "malformed",
    [
        b"short",
        malformed_gzip(),
        encode_frame(MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 1, b"not-json"),
        encode_frame(MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 1, b'{"result":{"text":1}}'),
        encode_frame(MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 1, b'{"result":{"utterances":{}}}'),
        encode_frame(MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 1, b'{"result":{"utterances":[{"definite":"false"}]}}'),
    ],
    ids=["short", "gzip", "json", "text", "utterances", "definite"],
)
def test_malformed_frames_are_protocol_errors(server_factory: Any, malformed: bytes) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        else:
            conn.send(malformed)
            return True
        return False

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                await session.write(b"audio")
                result, error = await session.finish(ctx=RequestContext.timeout(3))
                assert not result.succeeded
                assert error is not None and error.kind == ErrorKind.PROTOCOL
                assert error.message == "malformed ASR frame"
                assert [event.kind async for event in session.events()] == [EventKind.READY, EventKind.ERROR]

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_dropped_connection_preserves_partial_result(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            send_json(conn, {"result": {"text": "partial"}})
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            return True
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                result, error = await session.finish(ctx=RequestContext.timeout(3))
                assert result.text == "partial" and not result.succeeded
                assert error is not None and error.kind == ErrorKind.TRANSPORT

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_ready_timeout_context_cancellation_and_finish_timeout(server_factory: Any) -> None:
    recording = Recording()

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            service = AsyncASRService(client)
            with pytest.raises(MaaSError, match="timed out"):
                await service.start(Config(host=endpoint(server), ready_timeout=0.05), ctx=RequestContext.timeout(2))
            ctx = RequestContext.timeout(2)
            task = asyncio.create_task(service.start(Config(host=endpoint(server)), ctx=ctx))
            assert await asyncio.to_thread(recording.received.wait, 2)
            ctx.cancel()
            with pytest.raises(MaaSError, match="canceled"):
                await asyncio.wait_for(task, 2)

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            send_json(conn, {"result": {"text": "partial"}})
        return None

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)

    partial = Recording()

    def partial_reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            send_json(conn, {"result": {"text": "partial"}})
        return None

    async def partial_scenario(server: Any) -> None:
        async with make_client() as client:
            async with (await AsyncASRService(client).start(Config(host=endpoint(server)))) as session:
                result, error = await session.finish(ctx=RequestContext.timeout(0.1))
                assert result.text == "partial" and not result.succeeded
                assert error is not None and "timed out" in error.message

    with server_factory(record_handler(partial, partial_reply)) as server:
        asyncio.run(partial_scenario(server))
    assert partial.closed.wait(2)


def test_native_cancellation_closes_websocket(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            session = await AsyncASRService(client).start(Config(host=endpoint(server)))
            task = asyncio.create_task(session.finish())
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert await asyncio.to_thread(recording.closed.wait, 2)
            assert session._reader is not None and session._reader.done()
            await session.close()

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_backpressure_drain_and_close_do_not_deadlock(server_factory: Any) -> None:
    release = threading.Event()
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_POSITIVE_SEQ:
            for index in range(70):
                send_json(conn, {"result": {"text": str(index)}})
            assert release.wait(3)
        else:
            terminal(conn)
            return True
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            session = await AsyncASRService(client).start(Config(host=endpoint(server)))
            await session.write(b"audio")
            finish_task = asyncio.create_task(session.finish(ctx=RequestContext.timeout(3)))
            await asyncio.sleep(0.1)
            assert not finish_task.done()

            async def collect() -> list[Any]:
                return [event async for event in session.events()]

            consumer = asyncio.create_task(collect())
            release.set()
            events = await asyncio.wait_for(consumer, 3)
            result, error = await asyncio.wait_for(finish_task, 3)
            assert error is None and result.succeeded
            assert len([event for event in events if event.kind == EventKind.SNAPSHOT]) == 70
            await session.close()

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)

    blocked = Recording()

    def blocked_reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            for index in range(70):
                send_json(conn, {"result": {"text": str(index)}})
        return None

    async def blocked_scenario(server: Any) -> None:
        async with make_client() as client:
            session = await AsyncASRService(client).start(Config(host=endpoint(server)))
            await asyncio.sleep(0.1)
            assert session._events.full()
            await asyncio.wait_for(session.close(), 2)
            assert session._reader is not None and session._reader.done()
            events = await asyncio.wait_for(collect_events(session), 2)
            assert len(events) == 64

    with server_factory(record_handler(blocked, blocked_reply)) as server:
        asyncio.run(blocked_scenario(server))
    assert blocked.closed.wait(2)


async def collect_events(session: AsyncSession) -> list[Any]:
    return [event async for event in session.events()]


def wav_with_odd_junk(pcm: bytes) -> bytes:
    junk = b"JUNK" + (3).to_bytes(4, "little") + b"abc" + b"\0"
    fmt = b"fmt " + (16).to_bytes(4, "little") + struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    data = b"data" + len(pcm).to_bytes(4, "little") + pcm
    body = b"WAVE" + junk + fmt + data
    return b"RIFF" + len(body).to_bytes(4, "little") + body


class TrackingReader(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_threads: list[int] = []
        self.close_threads: list[int] = []

    def read(self, size: int | None = -1) -> bytes:
        self.read_threads.append(threading.get_ident())
        return super().read(size)

    def close(self) -> None:
        self.close_threads.append(threading.get_ident())
        super().close()


class TrackingSource:
    def __init__(self, data: bytes) -> None:
        self.reader = TrackingReader(data)
        self.open_threads: list[int] = []

    def name(self) -> str:
        return "audio.wav"

    def size(self) -> int:
        return len(self.reader.getvalue())

    def open(self) -> TrackingReader:
        self.open_threads.append(threading.get_ident())
        return self.reader

    def replayable(self) -> bool:
        return True


@pytest.mark.parametrize("wav", [False, True])
def test_transcribe_wav_raw_and_thread_offload(server_factory: Any, wav: bool) -> None:
    recording = Recording()
    pcm = bytes(range(20))
    tracked = TrackingSource(wav_with_odd_junk(pcm) if wav else pcm)

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn, "wav" if wav else "raw")
            return True
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            config = Config(host=endpoint(server), seg=Segmentation(chunk_samples=4))
            result, error = await AsyncASRService(client).transcribe(
                tracked, config, ctx=RequestContext.timeout(3)
            )
            assert error is None and result.succeeded

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)
    audio = [
        frame.payload
        for frame in recording.frames
        if frame.type == MSG_TYPE_AUDIO_ONLY and frame.flags == FLAG_POSITIVE_SEQ
    ]
    expected = [pcm[:8], pcm[8:16], pcm[16:]] if wav else [pcm[:8], pcm[8:12], pcm[12:20]]
    assert audio == expected
    assert tracked.open_threads and all(thread != threading.get_ident() for thread in tracked.open_threads)
    assert tracked.reader.read_threads
    assert all(thread != threading.get_ident() for thread in tracked.reader.read_threads)
    assert tracked.reader.close_threads
    assert all(thread != threading.get_ident() for thread in tracked.reader.close_threads)


def test_transcribe_accepts_bytes_and_returns_context_error(server_factory: Any) -> None:
    recording = Recording()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            send_json(conn, {"result": {"text": "partial"}})
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            result, error = await AsyncASRService(client).transcribe(
                b"raw", Config(host=endpoint(server)), ctx=RequestContext.timeout(0.1)
            )
            assert result.text == "partial" and not result.succeeded
            assert error is not None and error.kind == ErrorKind.TRANSPORT

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)
    assert recording.frames[1].payload == b"raw"


def test_transcribe_context_cancellation_interrupts_audio_send(
    server_factory: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = Recording()
    started = asyncio.Event()

    async def blocked_write(session: AsyncSession, pcm: bytes) -> int:
        started.set()
        await asyncio.Event().wait()
        return len(pcm)

    monkeypatch.setattr(AsyncSession, "write", blocked_write)

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            ctx = RequestContext(request_id="send-id")
            task = asyncio.create_task(
                AsyncASRService(client).transcribe(b"audio", Config(host=endpoint(server)), ctx=ctx)
            )
            await started.wait()
            ctx.cancel()
            result, error = await asyncio.wait_for(task, 2)
            assert not result.succeeded
            assert error is not None and error.kind == ErrorKind.TRANSPORT
            assert isinstance(error.cause, RequestCanceled)
            assert error.request_id == "send-id"

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


class BlockingReader(io.BytesIO):
    def __init__(self) -> None:
        super().__init__(b"audio")
        self.started = threading.Event()
        self.release = threading.Event()
        self.read_count = 0

    def read(self, size: int | None = -1) -> bytes:
        self.read_count += 1
        if self.read_count > 1:
            self.started.set()
            self.release.wait(2)
        return b"" if self.closed else super().read(size)


class BlockingSource:
    def __init__(self) -> None:
        self.reader = BlockingReader()

    def name(self) -> str:
        return "audio.pcm"

    def size(self) -> int:
        return 5

    def open(self) -> BlockingReader:
        return self.reader

    def replayable(self) -> bool:
        return True


def test_transcribe_native_cancellation_closes_websocket(server_factory: Any) -> None:
    recording = Recording()
    source = BlockingSource()

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        return None

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            task = asyncio.create_task(
                AsyncASRService(client).transcribe(source, Config(host=endpoint(server)))
            )
            assert await asyncio.to_thread(source.reader.started.wait, 2)
            task.cancel()
            try:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 2)
                assert await asyncio.to_thread(recording.closed.wait, 2)
            finally:
                source.reader.release.set()

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_terminal_before_ready_is_api_error(server_factory: Any) -> None:
    recording = Recording()

    async def scenario(server: Any) -> None:
        async with make_client() as client:
            with pytest.raises(MaaSError, match="before it became ready") as caught:
                await AsyncASRService(client).start(Config(host=endpoint(server)), ctx=RequestContext.timeout(2))
            assert caught.value.kind == ErrorKind.API

    def reply(conn: Any, frame: ASRFrame) -> bool | None:
        if frame.type == MSG_TYPE_FULL_CLIENT:
            terminal(conn)
            return True
        return None

    with server_factory(record_handler(recording, reply)) as server:
        asyncio.run(scenario(server))
    assert recording.closed.wait(2)


def test_pre_cancelled_context_is_maas_error() -> None:
    async def scenario() -> None:
        async with make_client() as client:
            ctx = RequestContext.timeout(2)
            ctx.cancel()
            with pytest.raises(MaaSError, match="canceled") as caught:
                await AsyncASRService(client).start(Config(), ctx=ctx)
            assert caught.value.kind == ErrorKind.TRANSPORT
            assert not isinstance(caught.value, asyncio.CancelledError)

    asyncio.run(scenario())
