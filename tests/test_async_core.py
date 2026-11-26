from __future__ import annotations

import asyncio
import io
import threading
from collections.abc import AsyncIterator
from typing import BinaryIO

import httpx
import pytest

from maas_client import RequestCanceled, RequestContext, RequestDeadlineExceeded, RetryPolicy
from maas_client.async_client import AsyncClient
from maas_client.context import run_with_context
from maas_client.envelope import decode_envelope_async
from maas_client.errors import ErrorKind
from maas_client.hooks import AsyncHooks
from maas_client.multipart import new_async_multipart


def response(body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test/vision/v1/result")
    return httpx.Response(status, content=body, headers=headers, request=request)


def test_async_public_exports() -> None:
    from maas_client import (
        AsyncClient,
        AsyncConfig,
        AsyncHooks,
        AsyncMultipartBody,
        decode_envelope_async,
        new_async_multipart,
        run_with_context,
    )

    assert AsyncClient.__name__ == "AsyncClient"
    assert AsyncConfig.__name__ == "AsyncConfig"
    assert AsyncHooks.__name__ == "AsyncHooks"
    assert AsyncMultipartBody.__name__ == "AsyncMultipartBody"
    assert callable(decode_envelope_async)
    assert callable(new_async_multipart)
    assert callable(run_with_context)


def test_async_config_requires_explicit_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "secret")
    monkeypatch.setenv("MAAS_BASE_URL", "https://example.test")
    with pytest.raises(TypeError):
        AsyncClient()  # type: ignore[call-arg]
    with pytest.raises(Exception, match="api_key"):
        AsyncClient(api_key=" ", base_url="https://example.test")
    with pytest.raises(Exception, match="base_url"):
        AsyncClient(api_key="key", base_url="ftp://example.test")


def test_async_client_headers_metadata_and_hooks() -> None:
    async def scenario() -> None:
        calls: list[httpx.Request] = []
        requests: list[str] = []
        responses: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"code": "000000", "data": {}}, headers={"X-Request-ID": "server-id"})

        async def on_request(_ctx: RequestContext, event: object) -> None:
            requests.append(event.op)  # type: ignore[attr-defined]

        async def on_response(_ctx: RequestContext, event: object) -> None:
            responses.append(event.request_id)  # type: ignore[attr-defined]

        async with AsyncClient(
            api_key=" sk_test ",
            base_url=" https://example.test/gateway/ ",
            transport=httpx.MockTransport(handler),
            hooks=AsyncHooks(on_request=on_request, on_response=on_response),
        ) as client:
            ctx = RequestContext(request_id="local-id")
            request = client.new_request("GET", "/vision/v1/result", ctx=ctx)
            request.headers["Accept"] = "application/custom"
            resp, meta, error = await client.do(request, "test", ctx=ctx)
            assert resp is not None
            await resp.aclose()
            assert error is None
            assert calls[0].headers["Authorization"] == "Bearer sk_test"
            assert calls[0].headers["X-Request-ID"] == "local-id"
            assert calls[0].headers["User-Agent"].startswith("maas-client-py/")
            assert calls[0].headers["Accept"] == "application/custom"
            assert str(calls[0].url) == "https://example.test/gateway/vision/v1/result"
            assert meta.request_id == "server-id"
            assert meta.http_status == 200
            assert requests == ["test"]
            assert responses == ["server-id"]
            assert client._unary.trust_env is False

    asyncio.run(scenario())


def test_async_retry_only_replays_idempotent_requests() -> None:
    async def scenario() -> None:
        get_calls = 0
        post_calls = 0
        retries: list[tuple[int, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal get_calls, post_calls
            if request.method == "GET":
                get_calls += 1
                if get_calls < 3:
                    return httpx.Response(503, json={"code": "503001"})
                return httpx.Response(200, json={"code": "000000"})
            post_calls += 1
            return httpx.Response(503, json={"code": "503001"})

        async def on_retry(_ctx: RequestContext, event: object) -> None:
            retries.append((event.attempt, event.reason))  # type: ignore[attr-defined]

        async with AsyncClient(
            api_key="key",
            base_url="https://example.test",
            retry=RetryPolicy(max_attempts=3),
            transport=httpx.MockTransport(handler),
            hooks=AsyncHooks(on_retry=on_retry),
        ) as client:
            resp, _, error = await client.do(client.new_request("GET", "/x"), "get")
            assert resp is not None
            await resp.aclose()
            assert error is None and resp.status_code == 200
            resp, _, error = await client.do(client.new_request("POST", "/x", b"{}"), "post")
            assert resp is not None
            await resp.aclose()
            assert error is None and resp.status_code == 503
            assert get_calls == 3
            assert post_calls == 1
            assert retries == [(2, "HTTP 503"), (3, "HTTP 503")]

    asyncio.run(scenario())


def test_context_cancellation_during_retry_delay_stops_retries() -> None:
    async def scenario() -> None:
        calls = 0
        ctx = RequestContext(request_id="retry-id")

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503, json={"code": "503001"})

        async def on_retry(context: RequestContext, event: object) -> None:
            assert context is ctx
            context.cancel()

        async with AsyncClient(
            api_key="key",
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
            hooks=AsyncHooks(on_retry=on_retry),
        ) as client:
            response, _, error = await client.do(client.new_request("GET", "/x"), "get", ctx=ctx)
            assert response is None
            assert error is not None and isinstance(error.cause, RequestCanceled)
            assert error.request_id == "retry-id"
            assert calls == 1

    asyncio.run(scenario())


def test_custom_async_http_client_disables_builtin_retry_and_is_not_closed() -> None:
    async def scenario() -> None:
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(503, json={"code": "503001"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = AsyncClient(api_key="key", base_url="https://example.test", http_client=http)
            resp, _, error = await client.do(client.new_request("GET", "/x"), "get")
            assert resp is not None
            await resp.aclose()
            assert error is None
            await client.close()
            assert calls == 1
            assert http.is_closed is False

    asyncio.run(scenario())


def test_async_client_normalizes_context_cancellation() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("unreachable")

        async with AsyncClient(
            api_key="key",
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            ctx = RequestContext(request_id="cancel-me")
            task = asyncio.create_task(client.do(client.new_request("GET", "/x"), "get", ctx=ctx))
            await started.wait()
            ctx.cancel()
            response, meta, error = await task
            assert response is None
            assert error is not None and error.kind == ErrorKind.TRANSPORT
            assert isinstance(error.cause, RequestCanceled)
            assert error.request_id == meta.request_id == "cancel-me"
            await stopped.wait()

    asyncio.run(scenario())


def test_async_client_normalizes_context_deadline() -> None:
    async def scenario() -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async with AsyncClient(
            api_key="key",
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            _, _, error = await client.do(
                client.new_request("GET", "/x"),
                "get",
                ctx=RequestContext.timeout(0.02),
            )
            assert error is not None and error.kind == ErrorKind.TRANSPORT
            assert isinstance(error.cause, RequestDeadlineExceeded)

    asyncio.run(scenario())


def test_native_task_cancellation_propagates() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async with AsyncClient(
            api_key="key",
            base_url="https://example.test",
            transport=httpx.MockTransport(handler),
        ) as client:
            task = asyncio.create_task(client.do(client.new_request("GET", "/x"), "get"))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"[]",
        b'{"code":0,"data":{}}',
        b'{"code":"00000","data":{}}',
        b'{"code":"000000","message":1,"data":{}}',
        b"not-json",
    ],
)
def test_async_envelope_rejects_non_standard_payloads(payload: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(Exception) as caught:
            await decode_envelope_async("test", response(payload))
        assert caught.value.kind == ErrorKind.PROTOCOL

    asyncio.run(scenario())


def test_async_envelope_preserves_raw_data_and_enforces_limit() -> None:
    async def scenario() -> None:
        env = await decode_envelope_async("test", response(b'{"code":"000000","message":null,"data":{"a": 1}}'))
        assert env.is_success()
        assert env.message == ""
        assert env.data == b'{"a": 1}'
        assert env.value == {"a": 1}
        empty = await decode_envelope_async("test", response(b'{"code":"000000"}'))
        assert empty.data is None
        with pytest.raises(Exception) as caught:
            await decode_envelope_async("test", response(b'{"code":"000000","data":{}}'), limit=10)
        assert caught.value.kind == ErrorKind.PROTOCOL
        assert "parse limit" in caught.value.message

    asyncio.run(scenario())


def test_async_envelope_maps_stream_transport_errors() -> None:
    async def scenario() -> None:
        class BrokenStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                raise httpx.ReadError("connection lost")
                yield b"never"

        resp = httpx.Response(
            200,
            stream=BrokenStream(),
            headers={"X-Request-ID": "server-id"},
            request=httpx.Request("GET", "https://example.test"),
        )
        with pytest.raises(Exception) as caught:
            await decode_envelope_async("test", resp)
        assert caught.value.kind == ErrorKind.TRANSPORT
        assert caught.value.http_status == 200
        assert caught.value.request_id == "server-id"
        assert isinstance(caught.value.cause, httpx.ReadError)
        await resp.aclose()

    asyncio.run(scenario())


def test_async_envelope_normalizes_context_cancellation() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await asyncio.Event().wait()
                yield b"never"

        resp = httpx.Response(200, stream=BlockingStream(), request=httpx.Request("GET", "https://example.test"))
        ctx = RequestContext()
        task = asyncio.create_task(decode_envelope_async("test", resp, ctx=ctx))
        await started.wait()
        ctx.cancel()
        with pytest.raises(Exception) as caught:
            await task
        assert caught.value.kind == ErrorKind.TRANSPORT
        assert isinstance(caught.value.cause, RequestCanceled)
        await resp.aclose()

    asyncio.run(scenario())


def test_async_envelope_native_cancellation_propagates() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        class BlockingStream(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await asyncio.Event().wait()
                yield b"never"

        resp = httpx.Response(200, stream=BlockingStream(), request=httpx.Request("GET", "https://example.test"))
        task = asyncio.create_task(decode_envelope_async("test", resp))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await resp.aclose()

    asyncio.run(scenario())


def test_run_with_context_and_async_wait_cancel_promptly() -> None:
    async def scenario() -> None:
        started = asyncio.Event()

        async def blocked() -> None:
            started.set()
            await asyncio.Event().wait()

        ctx = RequestContext()
        task = asyncio.create_task(run_with_context(ctx, blocked))
        await started.wait()
        ctx.cancel()
        with pytest.raises(RequestCanceled):
            await task

        waiting_ctx = RequestContext()
        task = asyncio.create_task(waiting_ctx.wait_async(10))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        expired = RequestContext.timeout(0.01)
        with pytest.raises(RequestDeadlineExceeded):
            await expired.wait_async(1)

    asyncio.run(scenario())


def test_async_multipart_is_deterministic_and_offloads_file_io() -> None:
    main_thread = threading.get_ident()
    threads: list[int] = []

    class Source:
        def name(self) -> str:
            return 'a"b.pdf'

        def size(self) -> int:
            return 5

        def open(self) -> BinaryIO:
            threads.append(threading.get_ident())
            return io.BytesIO(b"hello")

        def replayable(self) -> bool:
            return True

    async def scenario() -> None:
        body = new_async_multipart(Source(), {"z": 1, "a": True, "skip": None, "file": "shadow"})  # type: ignore[arg-type]
        chunks = [chunk async for chunk in body.iter_bytes()]
        raw = b"".join(chunks)
        assert body.replayable
        assert body.content_length == len(raw)
        assert b'name="a"\r\n\r\ntrue\r\n' in raw
        assert b'name="z"\r\n\r\n1\r\n' in raw
        assert b'name="skip"' not in raw
        assert b'filename="a\\"b.pdf"' in raw
        assert raw.endswith(b"--\r\n")
        assert b"hello\r\n--" in raw
        assert threads and all(thread != main_thread for thread in threads)

    asyncio.run(scenario())


def test_async_multipart_rejects_missing_file() -> None:
    async def scenario() -> None:
        with pytest.raises(Exception, match="no file"):
            new_async_multipart(None, {})

    asyncio.run(scenario())
