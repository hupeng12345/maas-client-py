from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
import pytest

from maas_client.async_client import AsyncClient
from maas_client.config import PollPolicy, ResponseMeta
from maas_client.context import RequestCanceled, RequestContext, RequestDeadlineExceeded
from maas_client.errors import ErrorKind, MaaSError
from maas_client.files import file_from_bytes
from maas_client.hooks import AsyncHooks
from maas_client.ocr import (
    PATH_BANK_STATEMENT,
    PATH_FINANCIAL_RECEIPT,
    PATH_IMAGE_BLUR,
    PATH_MEDICAL_RECEIPT,
    PATH_OCR_GENERAL,
    PATH_OCR_PADDLE,
    PATH_OCR_PAGE_DIRECTION,
    PATH_OCR_SEAL,
    PATH_PARSE_ARABIC,
    PATH_PARSE_GENERAL,
    PATH_PARSE_HUNYUAN,
    PATH_PARSE_MULTILANG,
    AsyncOCRService,
    AsyncService,
    AsyncTask,
    CardKind,
    OCRService,
    TaskError,
    TaskStatus,
    VisionResult,
    card_kinds,
    enable_toc,
    with_async_submit,
    with_endpoint,
    with_fields,
    with_image_analysis,
    with_language,
    with_max_poll_interval,
    with_page_details,
    with_poll_callback,
    with_poll_interval,
    with_wait_options,
    with_wait_timeout,
)

BASE_URL = "https://example.test"
SOURCE = file_from_bytes("scan.pdf", b"%PDF-1.4\ncontents")
Handler = Callable[[httpx.Request], httpx.Response] | Callable[[httpx.Request], Any]


def envelope(data: object, *, code: str = "000000", message: str = "ok", request_id: str = "req-1") -> httpx.Response:
    return httpx.Response(
        200,
        json={"code": code, "message": message, "data": data},
        headers={"X-Request-ID": request_id},
    )


def submission(uuid: str = "task-1", status: str = "PENDING") -> dict[str, object]:
    return {"task_uuid": uuid, "task_status": status}


def vision(status: str = "SUCCESS", **extra: object) -> dict[str, object]:
    return {
        "task_uuid": "task-1",
        "task_status": status,
        "start_time": 100,
        "end_time": 103,
        "markdown_content": "# title",
        "pages": [
            {
                "page_index": 1,
                "page_markdown": "# title",
                "layout_result": [
                    {
                        "type": "text",
                        "page_idx": 1,
                        "bbox": [0, 0.5, 1, 1],
                        "content": "hello",
                        "metadata": {"text_level": 2},
                    }
                ],
                "ie_result": [
                    {
                        "recognized": True,
                        "category": "ID",
                        "bbox": [0, 0, 1, 1],
                        "angle": 90,
                        "fields": {"name": "Alice"},
                    }
                ],
            }
        ],
        **extra,
    }


@asynccontextmanager
async def service_for(handler: Handler, **kwargs: Any) -> AsyncIterator[AsyncOCRService]:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with AsyncClient(
            api_key="sk_test",
            base_url=BASE_URL,
            http_client=http,
            retry=kwargs.pop("retry", None),
            poll=kwargs.pop("poll", PollPolicy(initial_delay=0.001, max_delay=0.01, jitter=0, timeout=1)),
            **kwargs,
        ) as client:
            yield AsyncOCRService(client)


def multipart(request: httpx.Request) -> tuple[dict[str, str], bytes]:
    body = request.read()
    assert request.method == "POST"
    assert request.headers["Content-Type"].startswith("multipart/form-data; boundary=")
    fields = {
        match.group(1).decode(): match.group(2).decode()
        for match in re.finditer(rb'Content-Disposition: form-data; name="([^"]+)"\r\n\r\n(.*?)\r\n--', body, re.S)
    }
    file_match = re.search(rb'name="file"; filename="[^"]+"\r\n.*?\r\n\r\n(.*?)\r\n--', body, re.S)
    assert file_match is not None
    fields["file"] = file_match.group(1).decode()
    return fields, body


def test_aliases_multipart_options_toc_raw_and_immediate_result() -> None:
    async def scenario() -> None:
        captured: dict[str, str] = {}
        responses: list[httpx.Response] = []
        toc = {
            "DocName": "guide",
            "Structure": [
                {
                    "Title": "one",
                    "NodeID": "n1",
                    "Text": "body",
                    "StartIndex": 1,
                    "EndIndex": 2,
                    "Children": [{"Title": "child", "NodeID": "n2"}],
                }
            ],
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/vision/v1/card/idcard/sync"
            assert request.headers["Authorization"] == "Bearer sk_test"
            assert int(request.headers["Content-Length"]) == len(request.read())
            captured.update(multipart(request)[0])
            response = envelope(vision(toc=toc))
            responses.append(response)
            return response

        assert AsyncService is AsyncOCRService
        async with service_for(handler, keep_raw=True) as service:
            result = await service.recognize(
                CardKind.ID_CARD,
                SOURCE,
                enable_toc(True),
                with_language("ja"),
                with_image_analysis(False),
                with_page_details(True),
                with_fields({"language": "de", "custom": 7, "ignored": None}),
            )

        assert captured["file"] == "%PDF-1.4\ncontents"
        assert captured["enable_toc"] == "true"
        assert captured["image_analysis"] == "false"
        assert captured["page_details"] == "true"
        assert captured["language"] == "de"
        assert captured["custom"] == "7"
        assert "ignored" not in captured
        assert result.succeeded() and result.duration == timedelta(seconds=3)
        assert result.markdown_content == "# title"
        assert result.meta.request_id == "req-1" and result.meta.http_status == 200
        assert result.raw is not None and json.loads(result.raw)["markdown_content"] == "# title"
        assert result.pages[0].layout[0].bbox == [0, 0.5, 1, 1]
        assert result.pages[0].ie_result[0].fields == {"name": "Alice"}
        assert result.toc is not None and result.toc.doc_name == "guide"
        node = result.toc.structure[0]
        assert (node.title, node.node_id, node.text, node.start_index, node.end_index) == (
            "one",
            "n1",
            "body",
            1,
            2,
        )
        assert node.children[0].node_id == "n2"
        assert responses[0].is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("parse_document", PATH_PARSE_GENERAL),
        ("parse_document_multilang", PATH_PARSE_MULTILANG),
        ("parse_document_arabic", PATH_PARSE_ARABIC),
        ("parse_document_hunyuan", PATH_PARSE_HUNYUAN),
        ("recognize_bank_statement", PATH_BANK_STATEMENT),
        ("recognize_medical_receipt", PATH_MEDICAL_RECEIPT),
        ("recognize_financial_receipt", PATH_FINANCIAL_RECEIPT),
        ("ocr_text", PATH_OCR_GENERAL),
        ("ocr_text_paddle", PATH_OCR_PADDLE),
        ("recognize_seal", PATH_OCR_SEAL),
        ("detect_page_direction", PATH_OCR_PAGE_DIRECTION),
        ("detect_image_blur", PATH_IMAGE_BLUR),
    ],
)
def test_async_endpoint_mappings(method: str, path: str) -> None:
    async def scenario() -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return envelope(submission() if request.method == "POST" else vision())

        async with service_for(handler) as service:
            result = await getattr(service, method)(SOURCE)
        assert result.succeeded()
        assert paths == [path, "/vision/v1/result"]

    asyncio.run(scenario())


def test_submit_returns_task_without_polling_and_preserves_meta() -> None:
    async def scenario() -> None:
        paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return envelope(submission(status="QUEUED"), request_id="submit-id")

        async with service_for(handler) as service:
            task = await service.submit_document(SOURCE)
            assert isinstance(task, AsyncTask)
            assert task.uuid == "task-1" and task.status == "QUEUED"
            assert task.meta == ResponseMeta(request_id="submit-id", http_status=200, duration=task.meta.duration)
        assert paths == [PATH_PARSE_GENERAL]

    asyncio.run(scenario())


def test_all_card_kinds_sync_async_and_endpoint_override() -> None:
    async def scenario() -> None:
        assert len(card_kinds()) == 13
        for kind in card_kinds():
            for asynchronous in (False, True):
                paths: list[str] = []

                def handler(request: httpx.Request, paths: list[str] = paths) -> httpx.Response:
                    paths.append(request.url.path)
                    if request.method == "GET" or request.url.path.endswith("/sync"):
                        return envelope(vision())
                    return envelope(submission())

                options = (with_async_submit(),) if asynchronous else ()
                async with service_for(handler) as service:
                    result = await service.recognize(kind, SOURCE, *options)
                suffix = "async" if asynchronous else "sync"
                expected = [f"/vision/v1/card/{kind.value}/{suffix}"]
                if asynchronous:
                    expected.append("/vision/v1/result")
                assert paths == expected and result.succeeded()

        paths: list[str] = []

        def override_handler(request: httpx.Request) -> httpx.Response:
            paths.append(request.url.path)
            return envelope(submission() if request.url.path == "/custom/ocr" else vision())

        async with service_for(override_handler) as service:
            await service.recognize(CardKind.ID_CARD, SOURCE, with_endpoint("/custom/ocr"))
            task = await service.recognize_async(CardKind.PASSPORT, SOURCE, with_endpoint("/custom/ocr"))
            assert task.uuid == "task-1"
        assert paths == ["/custom/ocr", "/custom/ocr"]

    asyncio.run(scenario())


def test_polling_unknown_status_backoff_callbacks_and_async_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        statuses = iter(["PROCESSING", "QUEUED", "", "PENDING", "SUCCESS"])
        waits: list[float] = []
        callbacks: list[tuple[TaskStatus, int]] = []
        events: list[tuple[str, int]] = []
        original_wait = RequestContext.wait_async

        async def fake_wait(ctx: RequestContext, seconds: float) -> bool:
            waits.append(seconds)
            return await original_wait(ctx, 0)

        monkeypatch.setattr(RequestContext, "wait_async", fake_wait)
        monkeypatch.setattr("maas_client.ocr.task.random.uniform", lambda low, high: 1.0)

        def handler(request: httpx.Request) -> httpx.Response:
            return envelope(submission() if request.method == "POST" else vision(next(statuses)))

        def callback(ctx: RequestContext, status: TaskStatus, attempt: int) -> None:
            assert isinstance(ctx, RequestContext)
            callbacks.append((status, attempt))

        async def on_poll(ctx: RequestContext, event: Any) -> None:
            events.append((event.status, event.attempt))

        hooks = AsyncHooks(on_poll=on_poll)
        async with service_for(
            handler,
            hooks=hooks,
            poll=PollPolicy(initial_delay=0.001, max_delay=0.003, multiplier=2, jitter=0.5, timeout=1),
        ) as service:
            result = await service.parse_document(
                SOURCE,
                with_wait_options(
                    with_poll_interval(0.01),
                    with_max_poll_interval(0.03),
                    with_poll_callback(callback),
                ),
            )
        assert result.succeeded()
        assert waits == [0.015, 0.03, 0.045, 0.045]
        assert callbacks == [
            (TaskStatus.PROCESSING, 1),
            (TaskStatus("QUEUED"), 2),
            (TaskStatus(""), 3),
            (TaskStatus.PENDING, 4),
            (TaskStatus.SUCCESS, 5),
        ]
        assert events == [(status.value, attempt) for status, attempt in callbacks]

    asyncio.run(scenario())


@pytest.mark.parametrize("status", ["FAILED", "CANCELED"])
def test_terminal_failure_retains_result(status: str) -> None:
    async def scenario() -> None:
        responses: list[httpx.Response] = []

        def handler(request: httpx.Request) -> httpx.Response:
            response = envelope(submission() if request.method == "POST" else vision(status))
            responses.append(response)
            return response

        async with service_for(handler) as service:
            with pytest.raises(TaskError) as caught:
                await service.parse_document(SOURCE)
        error = caught.value
        assert error.kind is ErrorKind.TASK and error.task_uuid == "task-1"
        assert error.request_id == "req-1" and error.http_status == 200
        assert error.result is not None and error.result.task_status.value == status
        assert error.result.markdown_content == "# title"
        assert all(response.is_closed for response in responses)

    asyncio.run(scenario())


def test_timeout_retains_latest_result_and_initial_result() -> None:
    async def scenario() -> None:
        polls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST":
                return envelope(submission())
            polls += 1
            return envelope(vision("PROCESSING", markdown_content=f"poll-{polls}"))

        async with service_for(handler) as service:
            task = await service.submit_document(SOURCE)
            with pytest.raises(TaskError) as caught:
                await task.wait(with_poll_interval(0.05), with_wait_timeout(0.001))
            assert polls == 1
            assert caught.value.retryable and caught.value.kind is ErrorKind.TASK
            assert caught.value.result is not None
            assert caught.value.result.markdown_content == "poll-1"

            context = RequestContext.timeout(0)
            with pytest.raises(TaskError) as initial:
                await task.wait(ctx=context)
            assert initial.value.result is not None
            assert initial.value.result.task_status is TaskStatus.PENDING

    asyncio.run(scenario())


def test_context_cancellation_retains_latest_result_and_stops_polling() -> None:
    async def scenario() -> None:
        polls = 0
        context = RequestContext(request_id="caller")

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST":
                return envelope(submission())
            polls += 1
            return envelope(vision("PROCESSING", markdown_content="latest"))

        def cancel_after_poll(ctx: RequestContext, status: TaskStatus, attempt: int) -> None:
            assert ctx is context and attempt == 1
            ctx.cancel()

        async with service_for(handler) as service:
            with pytest.raises(TaskError) as caught:
                await service.parse_document(
                    SOURCE,
                    with_wait_options(with_poll_callback(cancel_after_poll), with_poll_interval(10)),
                    ctx=context,
                )
        assert polls == 1
        assert caught.value.kind is ErrorKind.TASK and not caught.value.retryable
        assert isinstance(caught.value.cause, Exception)
        assert caught.value.result is not None and caught.value.result.markdown_content == "latest"

    asyncio.run(scenario())


@pytest.mark.parametrize("deadline", [False, True])
def test_context_errors_from_async_poll_hooks_become_task_errors(deadline: bool) -> None:
    async def scenario() -> None:
        responses: list[httpx.Response] = []

        def handler(request: httpx.Request) -> httpx.Response:
            response = envelope(submission() if request.method == "POST" else vision("PROCESSING"))
            responses.append(response)
            return response

        async def on_poll(ctx: RequestContext, event: Any) -> None:
            if deadline:
                raise RequestDeadlineExceeded("poll deadline")
            raise RequestCanceled("poll canceled")

        async with service_for(handler, hooks=AsyncHooks(on_poll=on_poll)) as service:
            with pytest.raises(TaskError) as caught:
                await service.parse_document(SOURCE)
        assert caught.value.result is not None
        assert caught.value.result.task_status is TaskStatus.PROCESSING
        assert caught.value.retryable is deadline
        assert responses[-1].is_closed

    asyncio.run(scenario())


def test_native_cancellation_propagates_from_submission() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        blocker = asyncio.Event()
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            started.set()
            await blocker.wait()
            return envelope(submission())

        async with service_for(handler) as service:
            pending = asyncio.create_task(service.submit_document(SOURCE))
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert calls == 1

    asyncio.run(scenario())


def test_native_cancellation_propagates_from_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        waiting = asyncio.Event()
        blocker = asyncio.Event()
        responses: list[httpx.Response] = []
        polls = 0

        async def wait_forever(ctx: RequestContext, seconds: float) -> bool:
            waiting.set()
            await blocker.wait()
            return True

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal polls
            if request.method == "POST":
                return envelope(submission())
            polls += 1
            response = envelope(vision("PROCESSING"))
            responses.append(response)
            return response

        monkeypatch.setattr(RequestContext, "wait_async", wait_forever)
        async with service_for(handler) as service:
            pending = asyncio.create_task(service.parse_document(SOURCE))
            await waiting.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert polls == 1 and responses[0].is_closed

    asyncio.run(scenario())


def test_native_cancellation_during_response_read_closes_response() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        blocker = asyncio.Event()

        class SlowStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                started.set()
                await blocker.wait()
                yield b"{}"

            async def aclose(self) -> None:
                self.closed = True

        stream = SlowStream()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=stream)

        async with service_for(handler) as service:
            pending = asyncio.create_task(service.submit_document(SOURCE))
            await started.wait()
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert stream.closed

    asyncio.run(scenario())


def test_submission_is_never_retried() -> None:
    async def scenario() -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return envelope(None, code="500001", message="unavailable")

        async with service_for(handler) as service:
            with pytest.raises(MaaSError) as caught:
                await service.submit_document(SOURCE)
        assert calls == 1
        assert caught.value.kind is ErrorKind.API and caught.value.retryable

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("body", "code", "kind"),
    [
        (b'{"code":200,"message":"ok","data":{}}', "", ErrorKind.PROTOCOL),
        (b'{"message":"ok","data":{}}', "", ErrorKind.PROTOCOL),
        (b"not-json", "", ErrorKind.PROTOCOL),
        (b'{"code":"401001","message":"missing api key","data":null}', "401001", ErrorKind.API),
    ],
)
def test_strict_envelopes_cleanup_and_request_ids(body: bytes, code: str, kind: ErrorKind) -> None:
    async def scenario() -> None:
        responses: list[httpx.Response] = []

        def handler(request: httpx.Request) -> httpx.Response:
            response = httpx.Response(200, content=body, headers={"X-Request-ID": "failure-id"})
            responses.append(response)
            return response

        async with service_for(handler) as service:
            with pytest.raises(MaaSError) as caught:
                await service.parse_document(SOURCE)
        assert caught.value.kind is kind and caught.value.code == code
        assert caught.value.request_id == "failure-id"
        assert responses[0].is_closed

    asyncio.run(scenario())


@pytest.mark.parametrize("data", [{"task_status": "PENDING"}, [], {"task_uuid": 12}])
def test_invalid_submission_payload_is_protocol_error(data: object) -> None:
    async def scenario() -> None:
        responses: list[httpx.Response] = []

        def handler(request: httpx.Request) -> httpx.Response:
            response = envelope(data)
            responses.append(response)
            return response

        async with service_for(handler) as service:
            with pytest.raises(MaaSError) as caught:
                await service.submit_document(SOURCE)
        assert caught.value.kind is ErrorKind.PROTOCOL
        assert caught.value.request_id == "req-1"
        assert responses[0].is_closed

    asyncio.run(scenario())


def test_query_escaping_business_and_protocol_errors_cleanup() -> None:
    async def scenario() -> None:
        uuid = "task /?&=+"
        responses: list[httpx.Response] = []
        mode = "success"

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                return envelope(submission())
            assert request.method == "GET"
            assert request.url.path == "/vision/v1/result"
            assert request.url.query == b"task_uuid=task+%2F%3F%26%3D%2B"
            if mode == "business":
                response = envelope(None, code="640402", message="task not found")
            elif mode == "protocol":
                response = envelope([])
            else:
                response = envelope(vision())
            responses.append(response)
            return response

        async with service_for(handler) as service:
            task = await service.recognize_async(CardKind.ID_CARD, SOURCE)
            task.uuid = uuid
            assert (await task.get()).succeeded()
            mode = "business"
            with pytest.raises(MaaSError) as business:
                await task.get()
            assert business.value.kind is ErrorKind.API and business.value.task_uuid == uuid
            mode = "protocol"
            with pytest.raises(MaaSError) as protocol:
                await task.get()
            assert protocol.value.kind is ErrorKind.PROTOCOL and protocol.value.task_uuid == uuid
        assert all(response.is_closed for response in responses)

    asyncio.run(scenario())


def test_local_validation_upload_limit_and_unknown_card_never_send() -> None:
    async def scenario() -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return envelope(vision())

        async with service_for(handler, max_upload_size=4) as service:
            with pytest.raises(MaaSError, match="no file provided") as missing:
                await service.parse_document(None)
            assert missing.value.kind is ErrorKind.LOCAL
            with pytest.raises(MaaSError, match="local size limit") as oversized:
                await service.parse_document(SOURCE)
            assert oversized.value.kind is ErrorKind.LOCAL
            with pytest.raises(MaaSError, match="unknown card kind") as unknown:
                await service.recognize("nope", SOURCE)  # type: ignore[arg-type]
            assert unknown.value.kind is ErrorKind.LOCAL
        assert calls == 0

    asyncio.run(scenario())


def test_precanceled_context_is_maas_error_and_never_sends() -> None:
    async def scenario() -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return envelope(submission())

        context = RequestContext()
        context.cancel()
        async with service_for(handler) as service:
            with pytest.raises(MaaSError) as caught:
                await service.submit_document(SOURCE, ctx=context)
        assert calls == 0
        assert caught.value.kind is ErrorKind.TRANSPORT
        assert isinstance(caught.value.cause, Exception)

    asyncio.run(scenario())


def test_response_deadline_becomes_maas_error_and_closes_response() -> None:
    async def scenario() -> None:
        class SlowStream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self) -> AsyncIterator[bytes]:
                await asyncio.Event().wait()
                yield b"{}"

            async def aclose(self) -> None:
                self.closed = True

        stream = SlowStream()
        context = RequestContext.timeout(0.01, request_id="deadline-id")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, stream=stream, headers={"X-Request-ID": "response-id"})

        async with service_for(handler) as service:
            with pytest.raises(MaaSError) as caught:
                await service.submit_document(SOURCE, ctx=context)
        assert caught.value.kind is ErrorKind.TRANSPORT
        assert caught.value.retryable
        assert stream.closed

    asyncio.run(scenario())


def test_unattached_and_empty_async_tasks_are_local_errors() -> None:
    async def scenario() -> None:
        with pytest.raises(MaaSError, match="task uuid is empty") as empty:
            await AsyncTask("", TaskStatus.PENDING, ResponseMeta()).wait()
        assert empty.value.kind is ErrorKind.LOCAL
        with pytest.raises(MaaSError, match="not attached") as unattached:
            await AsyncTask("t", TaskStatus.PENDING, ResponseMeta()).get()
        assert unattached.value.kind is ErrorKind.LOCAL
        assert not OCRService.__name__.startswith("Async")
        assert VisionResult(task_status="SUCCESS").succeeded()

    asyncio.run(scenario())
