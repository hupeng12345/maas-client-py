from __future__ import annotations

import io
import json
import struct
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from maas_client.client import Client
from maas_client.context import RequestContext
from maas_client.errors import ErrorKind, MaaSError
from maas_client.tts import (
    DEFAULT_TTS_PATH,
    PATH_SPEECH,
    Audio,
    Service,
    SpeechRequest,
    TTSService,
    with_bgm,
    with_endpoint,
    with_instructions,
    with_language,
    with_model,
    with_pitch,
    with_sample,
    with_speed,
    with_stream,
    with_voice,
)

PCM = b"\x00\x01\x02\x03\x04\x05\x06\x07"


class TrackedStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False
        self.iterated = False
        self.read_calls = 0

    def __iter__(self) -> Iterator[bytes]:
        self.iterated = True
        yield from self.chunks

    def read(self, size: int = -1) -> bytes:
        self.read_calls += 1
        return b"".join(self.chunks)

    def close(self) -> None:
        self.closed = True


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> Client:
    return Client(api_key="sk_test", base_url="https://example.test", transport=httpx.MockTransport(handler))


def audio_response(
    body: bytes = PCM,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
) -> tuple[httpx.Response, TrackedStream]:
    stream = TrackedStream([body])
    all_headers = {"Content-Type": "audio/pcm", **(headers or {})}
    return httpx.Response(status, headers=all_headers, stream=stream), stream


def valid_request(**changes: object) -> SpeechRequest:
    values: dict[str, object] = {"model": "m", "input": "hello", "voice": "v"}
    values.update(changes)
    return SpeechRequest(**values)  # type: ignore[arg-type]


def test_constants_and_alias() -> None:
    assert PATH_SPEECH == DEFAULT_TTS_PATH == "/tts/v1/speech"
    assert Service is TTSService


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model": ""}, "model is required"),
        ({"model": "  "}, "model is required"),
        ({"model": None}, "model is required"),
        ({"input": ""}, "input is required"),
        ({"input": "\n"}, "input is required"),
        ({"voice": ""}, "voice is required"),
        ({"voice": "  "}, "voice is required"),
        ({"response_format": "mp3"}, 'response_format must be "pcm"'),
    ],
)
def test_speech_validation(changes: dict[str, object], message: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return audio_response()[0]

    with make_client(handler) as client:
        with pytest.raises(MaaSError, match=message) as caught:
            TTSService(client).speech(valid_request(**changes))
    assert caught.value.kind is ErrorKind.LOCAL
    assert caught.value.op == "tts.Speech"
    assert calls == 0


def test_speech_calls_require_base_url() -> None:
    with make_client(lambda request: audio_response()[0]) as client:
        client.config.base_url = ""
        with pytest.raises(MaaSError, match="no endpoint configured"):
            TTSService(client).speech(valid_request())


def test_request_to_dict_omission_and_pointer_like_values() -> None:
    request = valid_request(
        language="",
        response_format="",
        stream=False,
        sample=0,
        speed=0.0,
        loud_ratio=0.0,
        pitch=0,
        endpoint="/custom",
        seed=0,
        instructions="",
        bgm_id="",
        bgm_vol=0.0,
        use_8k_enhancer=False,
    )
    assert request.to_dict() == {
        "model": "m",
        "input": "hello",
        "voice": "v",
        "stream": False,
        "seed": 0,
        "instructions": "",
        "bgm_id": "",
        "bgm_vol": 0.0,
        "use_8k_enhancer": False,
    }


def test_speech_posts_json_and_say_applies_all_options() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(
            method=request.method,
            path=request.url.path,
            content_type=request.headers["Content-Type"],
            accept=request.headers["Accept"],
            body=json.loads(request.content),
        )
        return audio_response()[0]

    with make_client(handler) as client:
        with TTSService(client).say(
            "hello",
            with_model("m1"),
            with_voice("v1"),
            with_language("zh-CN"),
            with_sample(16000),
            with_speed(1.5),
            with_pitch(-2),
            with_stream(False),
            with_bgm("rain"),
            with_endpoint("/custom/v2/speech"),
            with_instructions("cheerful"),
            None,
        ) as audio:
            assert audio.to_bytes() == PCM

    assert captured["method"] == "POST"
    assert captured["path"] == "/custom/v2/speech"
    assert captured["content_type"] == "application/json"
    assert captured["accept"] == "audio/pcm, application/json"
    assert captured["body"] == {
        "model": "m1",
        "input": "hello",
        "voice": "v1",
        "response_format": "pcm",
        "language": "zh-CN",
        "sample": 16000,
        "speed": 1.5,
        "pitch": -2,
        "stream": False,
        "bgm_id": "rain",
        "instructions": "cheerful",
    }


def test_speech_serializes_optional_nonzero_fields() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return audio_response()[0]

    request = valid_request(
        response_format="pcm",
        stream=True,
        sample=24000,
        speed=1.25,
        loud_ratio=0.75,
        pitch=3,
        seed=42,
        instructions="soft",
        bgm_id="rain",
        bgm_vol=0.5,
        use_8k_enhancer=True,
    )
    with make_client(handler) as client, TTSService(client).speech(request) as audio:
        audio.read()
    assert bodies == [request.to_dict()]
    assert "endpoint" not in bodies[0]


@pytest.mark.parametrize("endpoint", [None, "", "  "])
def test_default_endpoint(endpoint: str | None) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return audio_response()[0]

    with make_client(handler) as client, TTSService(client).speech(valid_request(endpoint=endpoint)) as audio:
        audio.read()
    assert paths == [PATH_SPEECH]


def test_non_finite_numbers_are_local_encoding_errors() -> None:
    with make_client(lambda request: audio_response()[0]) as client:
        with pytest.raises(MaaSError, match="cannot encode request") as caught:
            TTSService(client).speech(valid_request(speed=float("nan")))
    assert caught.value.kind is ErrorKind.LOCAL
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("status", [200, 201, 206, 400, 422, 503])
def test_audio_content_type_succeeds_regardless_of_status(status: int) -> None:
    response, stream = audio_response(status=status, headers={"Content-Type": "Audio/PCM; charset=binary"})
    with make_client(lambda request: response) as client:
        audio = TTSService(client).say("hi", with_model("m"), with_voice("v"))
        assert audio.content_type == "audio/pcm"
        assert audio.meta.http_status == status
        assert not stream.closed
        audio.close()
        assert stream.closed


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({}, (8000, 1, "pcm_s16le")),
        (
            {"X-Sample-Rate": " 24000 ", "X-Audio-Channels": "2", "X-Audio-Format": " pcm_s16le "},
            (24000, 2, "pcm_s16le"),
        ),
        (
            {"X-Sample-Rate": "abc", "X-Audio-Channels": "-2", "X-Audio-Format": "  "},
            (8000, 1, "pcm_s16le"),
        ),
        ({"X-Sample-Rate": "0", "X-Audio-Channels": "0"}, (8000, 1, "pcm_s16le")),
        ({"X-Sample-Rate": "1_000", "X-Audio-Channels": "1.5"}, (8000, 1, "pcm_s16le")),
        ({"X-Sample-Rate": "999999999999999999999999", "X-Audio-Channels": "2"}, (8000, 2, "pcm_s16le")),
    ],
)
def test_audio_metadata(
    headers: dict[str, str], expected: tuple[int, int, str]
) -> None:
    response, _ = audio_response(headers=headers)
    with make_client(lambda request: response) as client:
        with TTSService(client).speech(valid_request()) as audio:
            assert (audio.sample_rate, audio.channels, audio.format) == expected


@pytest.mark.parametrize(
    ("status", "body", "kind", "code"),
    [
        (422, b'{"code":"700000","message":"unsupported voice","data":null}', ErrorKind.API, "700000"),
        (402, b'{"code":"402001","message":"insufficient balance","data":null}', ErrorKind.API, "402001"),
        (200, b'{"code":200}', ErrorKind.PROTOCOL, ""),
        (200, b"not-json", ErrorKind.PROTOCOL, ""),
        (200, b"", ErrorKind.PROTOCOL, ""),
    ],
)
def test_non_audio_envelope_failures_are_closed(
    status: int, body: bytes, kind: ErrorKind, code: str
) -> None:
    stream = TrackedStream([body])
    response = httpx.Response(
        status,
        headers={"Content-Type": "application/json", "X-Request-ID": "server-id"},
        stream=stream,
    )
    with make_client(lambda request: response) as client:
        with pytest.raises(MaaSError) as caught:
            TTSService(client).speech(valid_request())
    assert caught.value.kind is kind
    assert caught.value.code == code
    assert caught.value.request_id == "server-id"
    assert caught.value.http_status == status
    assert stream.closed


def test_success_envelope_without_audio_is_protocol_error() -> None:
    stream = TrackedStream([b'{"code":"000000","message":"ok","data":{"url":"x"}}'])
    response = httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream)
    with make_client(lambda request: response) as client:
        with pytest.raises(MaaSError, match="expected audio") as caught:
            TTSService(client).speech(valid_request())
    assert caught.value.kind is ErrorKind.PROTOCOL
    assert caught.value.raw_body == b'{"url":"x"}'
    assert "check model and voice" in caught.value.hint
    assert stream.closed


def test_context_is_passed_to_streaming_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["timeout"] = request.extensions["timeout"]
        captured["request_id"] = request.headers["X-Request-ID"]
        return audio_response()[0]

    ctx = RequestContext.timeout(10, request_id="ctx-id")
    with make_client(handler) as client:
        with TTSService(client).speech(valid_request(), ctx=ctx) as audio:
            assert audio.meta.request_id == "ctx-id"
            audio.read()
    assert captured["request_id"] == "ctx-id"
    timeout = captured["timeout"]
    assert isinstance(timeout, dict)
    assert 0 < timeout["connect"] <= 10


def test_cancelled_context_does_not_send_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("request must not be sent")

    ctx = RequestContext(request_id="cancel-me")
    ctx.cancel()
    with make_client(handler) as client:
        with pytest.raises(MaaSError, match="request canceled") as caught:
            TTSService(client).speech(valid_request(), ctx=ctx)
    assert caught.value.kind is ErrorKind.TRANSPORT


def test_transport_failure_is_raised() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with make_client(handler) as client:
        with pytest.raises(MaaSError, match="request failed") as caught:
            TTSService(client).speech(valid_request())
    assert caught.value.kind is ErrorKind.TRANSPORT
    assert isinstance(caught.value.cause, httpx.ConnectError)


def test_audio_read_bytes_iteration_and_close() -> None:
    stream = TrackedStream([b"\x00\x01", b"\x02\x03\x04"])
    response = httpx.Response(200, headers={"Content-Type": "audio/pcm"}, stream=stream)
    audio = Audio(response)
    assert not stream.iterated
    assert audio.read(0) == b""
    assert audio.read(1) == b"\x00"
    assert audio.read(2) == b"\x01\x02"
    assert list(audio) == [b"\x03\x04"]
    assert audio.read() == b""
    assert audio.to_bytes() == b""
    audio.close()
    audio.close()
    assert stream.closed
    assert response.is_closed


def test_audio_bytes_reads_remaining_stream_and_context_closes() -> None:
    response, stream = audio_response()
    with Audio(response) as audio:
        assert audio.to_bytes() == PCM
    assert stream.closed


def test_write_wav_uses_unknown_lengths_and_streams() -> None:
    output = io.BytesIO()
    response, stream = audio_response()
    audio = Audio(response, sample_rate=16000, channels=2)
    written = audio.write_wav(output)
    data = output.getvalue()
    assert written == len(data) == 44 + len(PCM)
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"
    assert data[36:40] == b"data"
    assert struct.unpack_from("<I", data, 4)[0] == 36
    assert struct.unpack_from("<I", data, 40)[0] == 0
    assert struct.unpack_from("<H", data, 22)[0] == 2
    assert struct.unpack_from("<I", data, 24)[0] == 16000
    assert struct.unpack_from("<I", data, 28)[0] == 64000
    assert struct.unpack_from("<H", data, 32)[0] == 4
    assert struct.unpack_from("<H", data, 34)[0] == 16
    assert data[44:] == PCM
    assert stream.closed
    audio.close()


def test_write_wav_supports_partial_writers_and_default_metadata() -> None:
    class ShortWriter(io.BytesIO):
        def write(self, data: bytes) -> int:
            return super().write(data[:1])

    response, _ = audio_response()
    audio = Audio(response, sample_rate=0, channels=-1)
    output = ShortWriter()
    assert audio.write_wav(output) == 44 + len(PCM)
    assert struct.unpack_from("<I", output.getvalue(), 24)[0] == 8000
    assert struct.unpack_from("<H", output.getvalue(), 22)[0] == 1
    audio.close()


def test_write_wav_file_has_valid_patched_lengths(tmp_path: Path) -> None:
    path = tmp_path / "speech.wav"
    response, _ = audio_response()
    audio = Audio(response, sample_rate=24000, channels=1)
    audio.write_wav_file(path)
    data = path.read_bytes()
    assert len(data) == 44 + len(PCM)
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    assert struct.unpack_from("<I", data, 40)[0] == len(PCM)
    with wave.open(str(path), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 24000
        assert reader.getnframes() == len(PCM) // 2
        assert reader.readframes(reader.getnframes()) == PCM
    audio.close()
