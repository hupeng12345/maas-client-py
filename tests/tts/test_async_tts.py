from __future__ import annotations

import asyncio
import io
import json
import struct
import threading
import wave
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Any

import httpx
import pytest

from maas_client.async_client import AsyncClient
from maas_client.context import RequestContext
from maas_client.errors import ErrorKind, MaaSError
from maas_client.tts import (
    PATH_SPEECH,
    AsyncAudio,
    AsyncService,
    AsyncTTSService,
    SpeechRequest,
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


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, gated: bool = False) -> None:
        self.chunks = chunks
        self.gated = gated
        self.closed = False
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not gated:
            self.release.set()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self.release.wait()
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def make_client(handler: Callable[[httpx.Request], Awaitable[httpx.Response]]) -> AsyncClient:
    return AsyncClient(api_key="sk_test", base_url="https://example.test", transport=httpx.MockTransport(handler))


def audio_response(
    body: bytes = PCM,
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    gated: bool = False,
) -> tuple[httpx.Response, TrackedStream]:
    stream = TrackedStream([body], gated=gated)
    all_headers = {"Content-Type": "audio/pcm", **(headers or {})}
    return httpx.Response(status, headers=all_headers, stream=stream), stream


def valid_request(**changes: object) -> SpeechRequest:
    values: dict[str, object] = {"model": "m", "input": "hello", "voice": "v"}
    values.update(changes)
    return SpeechRequest(**values)  # type: ignore[arg-type]


def test_alias_and_async_validation() -> None:
    assert AsyncService is AsyncTTSService
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return audio_response()[0]

    async def main() -> None:
        async with make_client(handler) as client:
            with pytest.raises(MaaSError, match="model is required") as caught:
                await AsyncTTSService(client).speech(valid_request(model=" "))
            assert caught.value.kind is ErrorKind.LOCAL
            assert caught.value.op == "tts.Speech"
        assert calls == 0

    run(main())


def test_say_options_payload_and_omission() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(
            method=request.method,
            path=request.url.path,
            content_type=request.headers["Content-Type"],
            accept=request.headers["Accept"],
            body=json.loads(request.content),
        )
        return audio_response()[0]

    async def main() -> None:
        async with make_client(handler) as client:
            async with await AsyncTTSService(client).say(
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
                assert await audio.to_bytes() == PCM

    run(main())
    assert captured == {
        "method": "POST",
        "path": "/custom/v2/speech",
        "content_type": "application/json",
        "accept": "audio/pcm, application/json",
        "body": {
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
        },
    }
    omitted = valid_request(
        language="",
        response_format="",
        sample=0,
        speed=0.0,
        loud_ratio=0.0,
        pitch=0,
        seed=0,
        instructions="",
        bgm_id="",
        bgm_vol=0.0,
        use_8k_enhancer=False,
        stream=False,
    ).to_dict()
    assert omitted == {
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


@pytest.mark.parametrize("status", [200, 201, 206, 400, 422, 503])
def test_audio_content_type_succeeds_regardless_of_status(status: int) -> None:
    async def main() -> None:
        response, stream = audio_response(status=status, headers={"Content-Type": "Audio/PCM; charset=binary"})

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        async with make_client(handler) as client:
            audio = await AsyncTTSService(client).speech(valid_request())
            assert isinstance(audio, AsyncAudio)
            assert audio.content_type == "audio/pcm"
            assert audio.meta.http_status == status
            assert not stream.closed
            await audio.close()
            await audio.close()
            assert stream.closed

    run(main())


def test_audio_metadata_and_context_request_id() -> None:
    captured: dict[str, object] = {}

    async def main() -> None:
        response, _ = audio_response(
            headers={"X-Sample-Rate": " 24000 ", "X-Audio-Channels": "2", "X-Audio-Format": " pcm_s16le "}
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["request_id"] = request.headers["X-Request-ID"]
            return response

        ctx = RequestContext.timeout(10, request_id="ctx-id")
        async with make_client(handler) as client:
            async with await AsyncTTSService(client).speech(valid_request(), ctx=ctx) as audio:
                assert (audio.sample_rate, audio.channels, audio.format) == (24000, 2, "pcm_s16le")
                assert audio.meta.request_id == "ctx-id"
                assert await audio.read(1) == PCM[:1]
        assert captured["request_id"] == "ctx-id"

    run(main())


@pytest.mark.parametrize(
    ("status", "body", "kind", "code"),
    [
        (422, b'{"code":"700000","message":"unsupported voice","data":null}', ErrorKind.API, "700000"),
        (200, b'{"code":200}', ErrorKind.PROTOCOL, ""),
        (200, b"not-json", ErrorKind.PROTOCOL, ""),
        (200, b"", ErrorKind.PROTOCOL, ""),
    ],
)
def test_non_audio_strict_errors_are_closed(status: int, body: bytes, kind: ErrorKind, code: str) -> None:
    async def main() -> None:
        stream = TrackedStream([body])
        response = httpx.Response(
            status,
            headers={"Content-Type": "application/json", "X-Request-ID": "server-id"},
            stream=stream,
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        async with make_client(handler) as client:
            with pytest.raises(MaaSError) as caught:
                await AsyncTTSService(client).speech(valid_request())
        assert caught.value.kind is kind
        assert caught.value.code == code
        assert caught.value.request_id == "server-id"
        assert caught.value.http_status == status
        assert stream.closed

    run(main())


def test_success_envelope_without_audio_is_protocol_error() -> None:
    async def main() -> None:
        body = b'{"code":"000000","message":"ok","data":{"url":"x"}}'
        stream = TrackedStream([body])
        response = httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream)

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        async with make_client(handler) as client:
            with pytest.raises(MaaSError, match="expected audio") as caught:
                await AsyncTTSService(client).speech(valid_request())
        assert caught.value.kind is ErrorKind.PROTOCOL
        assert caught.value.raw_body == b'{"url":"x"}'
        assert "check model and voice" in caught.value.hint
        assert stream.closed

    run(main())


def test_partial_reads_async_iteration_and_to_bytes() -> None:
    async def main() -> None:
        stream = TrackedStream([b"\x00\x01", b"\x02\x03\x04", b"\x05\x06\x07"])
        response = httpx.Response(200, headers={"Content-Type": "audio/pcm"}, stream=stream)
        audio = AsyncAudio(response)
        assert await audio.read(0) == b""
        assert await audio.read(1) == b"\x00"
        assert await audio.read(2) == b"\x01\x02"
        assert [chunk async for chunk in audio] == [b"\x03\x04", b"\x05\x06\x07"]
        assert await audio.read() == b""
        assert await audio.to_bytes() == b""
        await audio.close()

        response, stream = audio_response()
        async with AsyncAudio(response) as complete:
            assert await complete.to_bytes() == PCM
        assert stream.closed

    run(main())


def test_write_wav_offloads_and_supports_partial_writers() -> None:
    class ShortWriter(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.threads: list[int] = []

        def write(self, data: bytes | bytearray | memoryview) -> int:
            self.threads.append(threading.get_ident())
            return super().write(data[:1])

    async def main() -> None:
        response, _ = audio_response()
        audio = AsyncAudio(response, sample_rate=16000, channels=2)
        output = ShortWriter()
        written = await audio.write_wav(output)
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
        assert output.threads and all(thread != threading.get_ident() for thread in output.threads)
        await audio.close()

    run(main())


def test_write_wav_file_patches_lengths(tmp_path: Path) -> None:
    async def main() -> None:
        path = tmp_path / "speech.wav"
        response, _ = audio_response()
        audio = AsyncAudio(response, sample_rate=24000, channels=1)
        await audio.write_wav_file(path)
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
        await audio.close()

    run(main())


def test_context_cancellation_interrupts_audio_reads() -> None:
    async def main() -> None:
        response, stream = audio_response(gated=True)

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        ctx = RequestContext(request_id="read-id")
        async with make_client(handler) as client:
            audio = await AsyncTTSService(client).speech(valid_request(), ctx=ctx)
            task = asyncio.create_task(audio.read())
            await stream.started.wait()
            ctx.cancel()
            with pytest.raises(MaaSError, match="request canceled") as caught:
                await task
            assert caught.value.kind is ErrorKind.TRANSPORT
            assert caught.value.request_id == "read-id"
            assert stream.closed

    run(main())


def test_context_deadline_interrupts_audio_reads() -> None:
    async def main() -> None:
        response, stream = audio_response(gated=True)

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        async with make_client(handler) as client:
            audio = await AsyncTTSService(client).speech(valid_request(), ctx=RequestContext.timeout(0.01))
            with pytest.raises(MaaSError, match="request timed out") as caught:
                await asyncio.wait_for(audio.read(), timeout=1)
            assert caught.value.kind is ErrorKind.TRANSPORT
            assert stream.closed

    run(main())


def test_cancelled_context_does_not_send_request() -> None:
    async def main() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("request must not be sent")
            return audio_response()[0]

        ctx = RequestContext(request_id="cancel-me")
        ctx.cancel()
        async with make_client(handler) as client:
            with pytest.raises(MaaSError, match="request canceled") as caught:
                await AsyncTTSService(client).speech(valid_request(), ctx=ctx)
            assert caught.value.kind is ErrorKind.TRANSPORT
            assert caught.value.request_id == "cancel-me"

    run(main())


def test_context_cancellation_interrupts_non_audio_decode_and_closes() -> None:
    async def main() -> None:
        stream = TrackedStream([b'{"code":"000000"}'], gated=True)
        response = httpx.Response(200, headers={"Content-Type": "application/json"}, stream=stream)

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        ctx = RequestContext(request_id="decode-id")
        async with make_client(handler) as client:
            task = asyncio.create_task(AsyncTTSService(client).speech(valid_request(), ctx=ctx))
            await stream.started.wait()
            ctx.cancel()
            with pytest.raises(MaaSError, match="request canceled") as caught:
                await task
            assert caught.value.kind is ErrorKind.TRANSPORT
            assert caught.value.request_id == "decode-id"
            assert stream.closed

    run(main())


def test_native_task_cancellation_propagates_from_request() -> None:
    async def main() -> None:
        started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            started.set()
            await asyncio.Event().wait()
            return audio_response()[0]

        async with make_client(handler) as client:
            task = asyncio.create_task(AsyncTTSService(client).speech(valid_request()))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    run(main())


def test_native_task_cancellation_propagates_from_audio_read() -> None:
    async def main() -> None:
        response, stream = audio_response(gated=True)

        async def handler(request: httpx.Request) -> httpx.Response:
            return response

        async with make_client(handler) as client:
            audio = await AsyncTTSService(client).speech(valid_request())
            task = asyncio.create_task(audio.read())
            await stream.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed

    run(main())


def test_non_finite_numbers_are_local_encoding_errors() -> None:
    async def main() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return audio_response()[0]

        async with make_client(handler) as client:
            with pytest.raises(MaaSError, match="cannot encode request") as caught:
                await AsyncTTSService(client).speech(valid_request(speed=float("nan")))
            assert caught.value.kind is ErrorKind.LOCAL
            assert isinstance(caught.value.__cause__, ValueError)

    run(main())


def test_custom_endpoint_whitespace_uses_default() -> None:
    paths: list[str] = []

    async def main() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return audio_response()[0]

        async with make_client(handler) as client:
            audio = await AsyncTTSService(client).speech(valid_request(endpoint="  "))
            await audio.close()

    run(main())
    assert paths == [PATH_SPEECH]
