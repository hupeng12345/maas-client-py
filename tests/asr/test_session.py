from __future__ import annotations

import gzip
import io
import json
import struct
import threading
import time
from dataclasses import dataclass, field
from uuid import UUID

import pytest
from websockets.exceptions import ConnectionClosed

from maas_client.asr import Auth, Config, EventKind, Segmentation
from maas_client.asr.frame import (
    FLAG_NEGATIVE_SEQ,
    FLAG_POSITIVE_SEQ,
    MSG_TYPE_AUDIO_ONLY,
    MSG_TYPE_FULL_CLIENT,
    MSG_TYPE_FULL_SERVER,
    decode_frame,
    encode_frame,
)
from maas_client.asr.service import ASRService
from maas_client.client import Client
from maas_client.context import RequestContext
from maas_client.errors import ErrorKind
from maas_client.files import file


@dataclass
class Recording:
    headers: dict = field(default_factory=dict)
    path: str = ""
    protocol: str = ""
    frames: list = field(default_factory=list)
    finished: threading.Event = field(default_factory=threading.Event)


def send_json(conn, payload, *, flags=FLAG_POSITIVE_SEQ, sequence=1):
    conn.send(encode_frame(MSG_TYPE_FULL_SERVER, flags, sequence, json.dumps(payload).encode()))


def ready(conn):
    send_json(conn, {"result": {"text": ""}, "additions": {"parret": {"kind": "ready"}}})


def terminal(conn, text="hello", *, code=0, wrapped=False, status="succeeded"):
    body = {
        "text": text,
        "utterances": [{"text": text, "definite": True, "start_time": 0, "end_time": 500}],
        "additions": {"log_id": "log-1", "parret": {"kind": "terminal", "session": {"status": status}}},
    }
    payload = {"payload_msg": {"audio_info": {"duration": 500}, "result": body}, "code": code}
    if not wrapped:
        payload = {"result": body, "code": code}
    send_json(conn, payload, flags=FLAG_NEGATIVE_SEQ, sequence=2)


def make_client(base_url="http://127.0.0.1:1"):
    return Client(api_key="sk_test", base_url=base_url, user_agent="test-agent")


def record_handler(recording, reply):
    def handler(conn):
        recording.headers = {key.lower(): value for key, value in conn.request.headers.items()}
        recording.path = conn.request.path
        recording.protocol = recording.headers.get("sec-websocket-protocol", "")
        try:
            while True:
                message = conn.recv()
                if not isinstance(message, bytes):
                    continue
                frame = decode_frame(message)
                recording.frames.append(frame)
                if frame.type == MSG_TYPE_AUDIO_ONLY and frame.flags == FLAG_NEGATIVE_SEQ:
                    recording.finished.set()
                if reply(conn, frame):
                    return
        finally:
            conn.close()
    return handler


def test_session_happy_path_and_headers(server_factory):
    recording = Recording()

    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn)
            return True
        return False

    with server_factory(record_handler(recording, reply)) as server:
        ctx = RequestContext.timeout(3, request_id="generic-request")
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}/prefix?q=1"
        with ASRService(make_client()).start(Config(host=endpoint), ctx=ctx) as session:
            assert session.chunk_size() == session.chunk_samples == 3200
            assert session.request_id() == "generic-request"
            UUID(recording.headers["x-api-request-id"])
            assert recording.headers["x-request-id"] == "generic-request"
            assert recording.headers["user-agent"] == "test-agent"
            assert recording.headers["authorization"] == "Bearer sk_test"
            assert recording.path == "/prefix/-/volcengine/api/v3/sauc/bigmodel_async"
            assert recording.headers.get("sec-websocket-protocol", "") == ""
            assert session.write(b"") == 0
            assert session.write(b"\x01\x02") == 2
            result, error = session.finish(ctx=ctx)
            assert error is None
            assert result.succeeded and result.status == "succeeded"
            assert result.text == "hello"
            assert result.log_id == "log-1"
            assert result.duration == 0
            assert result.meta.http_status == 101
            assert result.utterances[0].end_ms == 500
            events = list(session.events())
            assert [event.kind for event in events] == [EventKind.READY, EventKind.TERMINAL]
            assert events[-1].raw
        assert recording.finished.wait(2)
        assert [frame.sequence for frame in recording.frames] == [1, 2, -3]
        assert recording.frames[0].type == MSG_TYPE_FULL_CLIENT
        assert recording.frames[1].payload == b"\x01\x02"
        assert recording.frames[2].payload == b""


def test_snapshot_wrapped_result_and_additions_override(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_POSITIVE_SEQ:
            conn.send(encode_frame(0xA, FLAG_POSITIVE_SEQ, 2, b"ignored unknown type"))
            send_json(conn, {"result": {"text": "stale", "additions": {"log_id": "top"}},
                             "payload_msg": {"audio_info": {"duration": 250},
                                             "result": {"text": "snapshot", "additions": {"log_id": "inner"}}}})
            send_json(conn, {"result": {"text": "settled"}, "additions": {"parret": {"kind": "unit_settled"}}})
        else:
            terminal(conn, "final", wrapped=True)
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            session.write(b"audio")
            result, error = session.finish(ctx=RequestContext.timeout(3))
            assert error is None
            assert result.text == "final"
            assert result.log_id == "log-1"
            assert result.duration == 0.5
            events = list(session.events())
            assert [event.kind for event in events] == [
                EventKind.READY, EventKind.SNAPSHOT, EventKind.UTTERANCE, EventKind.TERMINAL
            ]
            assert events[1].text == "snapshot"
            assert events[1].log_id == "inner"
            assert events[2].text == "settled"


@pytest.mark.parametrize("auth_kind", ["bearer", "direct", "legacy"])
def test_auth_forms(server_factory, auth_kind):
    recording = Recording()
    with server_factory(record_handler(recording, lambda conn, frame: ready(conn) is None)) as server:
        auth = Auth()
        if auth_kind == "bearer":
            auth = Auth(bearer_key="override")
        elif auth_kind == "direct":
            auth = Auth(resource_id="res")
        else:
            auth = Auth(app_key="app", access_key="access")
        session = ASRService(make_client()).start(
            Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}", auth=auth)
        )
        session.close()
        headers = recording.headers
        if auth_kind == "bearer":
            assert headers["authorization"] == "Bearer override"
        elif auth_kind == "direct":
            assert headers["x-api-key"] == "sk_test"
            assert headers["x-api-resource-id"] == "res"
            assert "authorization" not in headers
        else:
            assert headers["x-api-app-key"] == "app"
            assert headers["x-api-access-key"] == "access"


def test_error_frame(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            payload = b'{"parret":{"reason":"unusable_audio"}}'
            body = gzip.compress(payload, mtime=0)
            conn.send(struct.pack(">BBBBiII", 0x11, 0xF1, 0x11, 0, 1, 45000002, len(body)) + body)
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        with pytest.raises(Exception) as caught:
            ASRService(make_client()).start(Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}"),
                                           ctx=RequestContext.timeout(3))
        assert caught.value.kind == ErrorKind.API
        assert caught.value.code == "45000002"
        assert "unusable_audio" in caught.value.hint


def test_dropped_connection_preserves_partial_result(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            send_json(conn, {"result": {"text": "partial"}})
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            result, error = session.finish(ctx=RequestContext.timeout(3))
            assert result.text == "partial"
            assert not result.succeeded
            assert error is not None and error.kind == ErrorKind.TRANSPORT


def test_terminal_failure_code(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn, code=400001)
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            result, error = session.finish(ctx=RequestContext.timeout(3))
            assert error is None
            assert result.status == "failed"
            assert not result.succeeded


def wav_with_odd_junk(pcm: bytes) -> bytes:
    junk = b"JUNK" + (3).to_bytes(4, "little") + b"abc" + b"\0"
    fmt = b"fmt " + (16).to_bytes(4, "little") + struct.pack(
        "<HHIIHH", 1, 1, 16000, 32000, 2, 16
    )
    data = b"data" + len(pcm).to_bytes(4, "little") + pcm
    body = b"WAVE" + junk + fmt + data
    return b"RIFF" + len(body).to_bytes(4, "little") + body


def test_transcribe_strips_wav_and_chunks(server_factory):
    recording = Recording()

    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn, "wav")
            return True
        return False

    pcm = bytes(range(20))
    source = file(io.BytesIO(wav_with_odd_junk(pcm)))
    with server_factory(record_handler(recording, reply)) as server:
        config = Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}", seg=Segmentation(chunk_samples=4))
        result, error = ASRService(make_client()).transcribe(source, config, ctx=RequestContext.timeout(3))
        assert error is None and result.succeeded
        assert recording.finished.wait(2)
        audio = [frame.payload for frame in recording.frames if frame.type == MSG_TYPE_AUDIO_ONLY
                 and frame.flags == FLAG_POSITIVE_SEQ]
        assert audio == [pcm[:8], pcm[8:16], pcm[16:]]
        assert json.loads(recording.frames[0].payload)["audio"]["format"] == "pcm"


def test_transcribe_raw_pcm(server_factory):
    recording = Recording()

    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            terminal(conn)
            return True
        return False

    with server_factory(record_handler(recording, reply)) as server:
        ASRService(make_client()).transcribe(b"raw", Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}"),
                                            ctx=RequestContext.timeout(3))
        assert recording.finished.wait(2)
        assert recording.frames[1].payload == b"raw"


def test_finish_timeout_keeps_partial_result(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
            send_json(conn, {"result": {"text": "partial"}})
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            result, error = session.finish(ctx=RequestContext.timeout(0.1))
            assert result.text == "partial"
            assert not result.succeeded
            assert error is not None and "timed out" in error.message


def test_is_last_package_fallback(server_factory):
    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_NEGATIVE_SEQ:
            send_json(conn, {"code": 0, "is_last_package": True, "result": {"text": "fallback"}})
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            result, error = session.finish(ctx=RequestContext.timeout(3))
            assert error is None
            assert result.succeeded and result.text == "fallback"


def test_event_queue_backpressure(server_factory):
    release = threading.Event()

    def reply(conn, frame):
        if frame.type == MSG_TYPE_FULL_CLIENT:
            ready(conn)
        elif frame.flags == FLAG_POSITIVE_SEQ:
            for index in range(70):
                send_json(conn, {"result": {"text": str(index)}})
            assert release.wait(3)
        else:
            terminal(conn)
            return True
        return False

    with server_factory(record_handler(Recording(), reply)) as server:
        endpoint = f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        with ASRService(make_client()).start(Config(host=endpoint)) as session:
            session.write(b"audio")
            finished = threading.Event()
            outcome = []

            def finish():
                outcome.append(session.finish(ctx=RequestContext.timeout(3)))
                finished.set()

            worker = threading.Thread(target=finish)
            worker.start()
            try:
                assert not finished.wait(0.1)
                release.set()
                events = list(session.events())
                assert len([event for event in events if event.kind == EventKind.SNAPSHOT]) == 70
                assert finished.wait(3)
                assert outcome[0][1] is None
            finally:
                release.set()
                worker.join(timeout=3)


def test_ready_timeout_and_cancelled_context(server_factory):
    def handler(conn):
        try:
            while conn.recv():
                pass
        except ConnectionClosed:
            pass

    with server_factory(handler) as server:
        service = ASRService(make_client())
        started = time.monotonic()
        with pytest.raises(Exception, match="timed out"):
            service.start(Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}", ready_timeout=0.05),
                          ctx=RequestContext.timeout(2))
        assert time.monotonic() - started < 2
        ctx = RequestContext.timeout(2)
        ctx.cancel()
        with pytest.raises(Exception, match="canceled"):
            service.start(Config(host=f"http://127.0.0.1:{server.socket.getsockname()[1]}"), ctx=ctx)
