from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any

import httpx
import pytest

from maas_client.client import Client
from maas_client.config import PollPolicy, ResponseMeta
from maas_client.context import RequestContext
from maas_client.errors import ErrorKind, MaaSError
from maas_client.files import file_from_bytes
from maas_client.hooks import Hooks
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
    CardKind,
    OCRService,
    Service,
    Task,
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


def envelope(data: object, *, code: str = "000000", message: str = "ok", request_id: str = "req-1") -> httpx.Response:
    return httpx.Response(
        200,
        json={"code": code, "message": message, "data": data},
        headers={"X-Request-ID": request_id},
    )


def submission(uuid: str = "task-1", status: str = "PENDING") -> dict[str, object]:
    return {"task_uuid": uuid, "task_status": status, "request_id": "submit-id"}


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


@pytest.fixture
def make_client() -> Iterator[Callable[..., Client]]:
    clients: list[Client] = []

    def make(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> Client:
        client = Client(
            api_key="sk_test",
            base_url=BASE_URL,
            transport=httpx.MockTransport(handler),
            retry=kwargs.pop("retry", None),
            poll=kwargs.pop("poll", PollPolicy(initial_delay=0.001, max_delay=0.01, jitter=0, timeout=1)),
            **kwargs,
        )
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.close()


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


def test_service_alias_and_full_result_decoding(make_client: Callable[..., Client]) -> None:
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/vision/v1/card/idcard/sync"
        response = envelope(vision())
        responses.append(response)
        return response

    service = OCRService(make_client(handler))
    assert Service is OCRService
    result = service.recognize(CardKind.ID_CARD, SOURCE)

    assert result.succeeded()
    assert result.task_status is TaskStatus.SUCCESS
    assert result.duration == timedelta(seconds=3)
    assert result.markdown_content == "# title"
    assert result.meta.request_id == "req-1"
    assert result.meta.http_status == 200
    assert result.raw is None
    page = result.pages[0]
    assert (page.index, page.markdown) == (1, "# title")
    block = page.layout[0]
    assert (block.type, block.page_idx, block.bbox, block.content) == ("text", 1, [0, 0.5, 1, 1], "hello")
    assert block.metadata == {"text_level": 2}
    entity = page.ie_result[0]
    assert entity.recognized and entity.category == "ID" and entity.angle == 90
    assert entity.fields == {"name": "Alice"}
    assert responses[0].is_closed


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
def test_async_endpoint_mappings(make_client: Callable[..., Client], method: str, path: str) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.method == "POST":
            return envelope(submission())
        assert request.url.params["task_uuid"] == "task-1"
        return envelope(vision())

    result = getattr(OCRService(make_client(handler)), method)(SOURCE)
    assert result.succeeded()
    assert paths == [path, "/vision/v1/result"]


def test_submit_document_returns_immediately_and_preserves_meta(make_client: Callable[..., Client]) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return envelope(submission(status="QUEUED"), request_id="submit-id")

    task = OCRService(make_client(handler)).submit_document(SOURCE)
    assert isinstance(task, Task)
    assert task.uuid == "task-1"
    assert task.status == "QUEUED"
    assert task.meta == ResponseMeta(request_id="submit-id", http_status=200, duration=task.meta.duration)
    assert paths == [PATH_PARSE_GENERAL]


def test_all_card_kinds_sync_and_async(make_client: Callable[..., Client]) -> None:
    kinds = card_kinds()
    assert len(kinds) == 13
    assert kinds[0] is CardKind.GENERIC and kinds[-1] is CardKind.MARRIAGE
    kinds[0] = CardKind.MARRIAGE
    assert card_kinds()[0] is CardKind.GENERIC

    for kind in card_kinds():
        for asynchronous in (False, True):
            paths: list[str] = []

            def handler(request: httpx.Request, paths: list[str] = paths) -> httpx.Response:
                paths.append(request.url.path)
                if request.url.path.endswith("/async"):
                    return envelope(submission())
                if request.method == "GET":
                    return envelope(vision())
                return envelope(vision())

            options = (with_async_submit(),) if asynchronous else ()
            result = OCRService(make_client(handler)).recognize(kind, SOURCE, *options)
            suffix = "async" if asynchronous else "sync"
            expected = [f"/vision/v1/card/{kind.value}/{suffix}"]
            if asynchronous:
                expected.append("/vision/v1/result")
            assert paths == expected
            assert result.succeeded()


def test_recognize_async_returns_task_without_polling(make_client: Callable[..., Client]) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return envelope(submission())

    task = OCRService(make_client(handler)).recognize_async(CardKind.PASSPORT, SOURCE)
    assert task.uuid == "task-1"
    assert paths == ["/vision/v1/card/passport/async"]


def test_polling_is_immediate_and_unknown_nonempty_statuses_continue(make_client: Callable[..., Client]) -> None:
    statuses = iter(["PROCESSING", "QUEUED", "", "PENDING", "SUCCESS"])
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(f"{request.method} {request.url.path}")
        if request.method == "POST":
            return envelope(submission())
        return envelope(vision(next(statuses)))

    result = OCRService(make_client(handler)).parse_document(SOURCE)
    assert result.succeeded()
    assert paths == ["POST " + PATH_PARSE_GENERAL] + ["GET /vision/v1/result"] * 5


@pytest.mark.parametrize("status", ["FAILED", "CANCELED"])
def test_terminal_failure_returns_result_on_task_error(make_client: Callable[..., Client], status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return envelope(submission() if request.method == "POST" else vision(status))

    with pytest.raises(TaskError) as caught:
        OCRService(make_client(handler)).parse_document(SOURCE)
    error = caught.value
    assert isinstance(error, MaaSError)
    assert error.kind is ErrorKind.TASK
    assert error.task_uuid == "task-1"
    assert error.request_id == "req-1"
    assert error.http_status == 200
    assert error.result is not None
    assert error.result.task_status.value == status
    assert error.result.markdown_content == "# title"


def test_timeout_returns_latest_result_and_retryable_task_error(make_client: Callable[..., Client]) -> None:
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            return envelope(submission())
        polls += 1
        return envelope(vision("PROCESSING", markdown_content=f"poll-{polls}"))

    with pytest.raises(TaskError) as caught:
        OCRService(make_client(handler)).parse_document(
            SOURCE, with_wait_options(with_poll_interval(0.05), with_wait_timeout(0.001))
        )
    assert polls == 1
    assert caught.value.kind is ErrorKind.TASK
    assert caught.value.retryable
    assert caught.value.task_uuid == "task-1"
    assert caught.value.result is not None
    assert caught.value.result.markdown_content == "poll-1"


def test_cancellation_stops_polling_without_canceling_platform_task(make_client: Callable[..., Client]) -> None:
    polls = 0
    context = RequestContext(request_id="caller")

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.method == "POST":
            return envelope(submission())
        polls += 1
        return envelope(vision("PROCESSING"))

    def cancel_after_poll(ctx: RequestContext, status: TaskStatus, attempt: int) -> None:
        assert ctx is context and attempt == 1
        ctx.cancel()

    with pytest.raises(MaaSError) as caught:
        OCRService(make_client(handler)).parse_document(
            SOURCE,
            with_wait_options(with_poll_callback(cancel_after_poll), with_poll_interval(10)),
            ctx=context,
        )
    assert polls == 1
    assert caught.value.kind is ErrorKind.TASK
    assert caught.value.task_uuid == "task-1"
    assert not caught.value.retryable
    assert isinstance(caught.value.cause, Exception)


def test_canceled_before_submission_is_not_a_task_error(make_client: Callable[..., Client]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return envelope(submission())

    context = RequestContext()
    context.cancel()
    with pytest.raises(MaaSError) as caught:
        OCRService(make_client(handler)).parse_document(SOURCE, ctx=context)
    assert calls == 0
    assert caught.value.kind is ErrorKind.TRANSPORT
    assert not caught.value.retryable


def test_poll_callback_hooks_and_capped_exponential_backoff(
    make_client: Callable[..., Client], monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = iter(["PROCESSING", "PROCESSING", "PROCESSING", "SUCCESS"])
    waits: list[float] = []
    callbacks: list[tuple[TaskStatus, int]] = []
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(RequestContext, "wait", lambda self, seconds: waits.append(seconds) or True)

    def handler(request: httpx.Request) -> httpx.Response:
        return envelope(submission() if request.method == "POST" else vision(next(statuses)))

    def callback(ctx: RequestContext, status: TaskStatus, attempt: int) -> None:
        assert isinstance(ctx, RequestContext)
        callbacks.append((status, attempt))

    hooks = Hooks(on_poll=lambda ctx, event: events.append((event.status, event.attempt)))
    result = OCRService(
        make_client(
            handler,
            hooks=hooks,
            poll=PollPolicy(initial_delay=0.001, max_delay=0.03, multiplier=2, jitter=0, timeout=1),
        )
    ).parse_document(
        SOURCE,
        with_wait_options(
            with_poll_interval(0.01),
            with_max_poll_interval(0.03),
            with_poll_callback(callback),
        ),
    )
    assert result.succeeded()
    assert waits == [0.01, 0.02, 0.03]
    assert callbacks == [
        (TaskStatus.PROCESSING, 1),
        (TaskStatus.PROCESSING, 2),
        (TaskStatus.PROCESSING, 3),
        (TaskStatus.SUCCESS, 4),
    ]
    assert events == [("PROCESSING", 1), ("PROCESSING", 2), ("PROCESSING", 3), ("SUCCESS", 4)]


def test_task_get_escapes_uuid_and_closes_response(make_client: Callable[..., Client]) -> None:
    uuid = "task /?&=+"
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return envelope(submission())
        assert request.method == "GET"
        assert request.url.path == "/vision/v1/result"
        assert request.url.query == b"task_uuid=task+%2F%3F%26%3D%2B"
        response = envelope(vision())
        responses.append(response)
        return response

    service = OCRService(make_client(handler))
    task = service.recognize_async(CardKind.ID_CARD, SOURCE, with_endpoint("/submit"))
    task.uuid = uuid
    result = task.get()
    assert result.succeeded()
    assert responses[0].is_closed


@pytest.mark.parametrize(
    "toc",
    [
        {
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
        },
        {
            "doc_name": "guide",
            "structure": [
                {
                    "title": "one",
                    "node_id": "n1",
                    "text": "body",
                    "start_index": 1,
                    "end_index": 2,
                    "children": [{"title": "child", "node_id": "n2"}],
                }
            ],
        },
        {
            "DOCNAME": "guide",
            "STRUCTURE": [
                {
                    "TITLE": "one",
                    "NODEID": "n1",
                    "TEXT": "body",
                    "STARTINDEX": 1,
                    "ENDINDEX": 2,
                    "CHILDREN": [{"TITLE": "child", "NODEID": "n2"}],
                }
            ],
        },
    ],
)
def test_toc_accepts_go_snake_case_and_case_insensitive_names(
    make_client: Callable[..., Client], toc: dict[str, object]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return envelope(vision(toc=toc))

    result = OCRService(make_client(handler)).recognize(CardKind.ID_CARD, SOURCE)
    assert result.toc is not None
    assert result.toc.doc_name == "guide"
    node = result.toc.structure[0]
    assert (node.title, node.node_id, node.text, node.start_index, node.end_index) == ("one", "n1", "body", 1, 2)
    assert node.children[0].title == "child"
    assert node.children[0].node_id == "n2"


def test_multipart_options_and_raw_retention(make_client: Callable[..., Client]) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            captured.update(multipart(request)[0])
            return envelope(submission())
        return envelope(vision())

    result = OCRService(make_client(handler, keep_raw=True)).parse_document(
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
    assert result.raw is not None
    assert json.loads(result.raw)["markdown_content"] == "# title"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_endpoint_override_and_empty_override(make_client: Callable[..., Client], asynchronous: bool) -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path == "/custom/ocr" or request.url.path.endswith("/async"):
            return envelope(submission())
        if request.method == "GET":
            return envelope(vision())
        return envelope(vision())

    service = OCRService(make_client(handler))
    options = (with_async_submit(),) if asynchronous else ()
    service.recognize(CardKind.ID_CARD, SOURCE, *options, with_endpoint("/custom/ocr"))
    service.recognize(CardKind.ID_CARD, SOURCE, *options, with_endpoint(""))
    expected = ["/custom/ocr"]
    default_path = f"/vision/v1/card/idcard/{'async' if asynchronous else 'sync'}"
    if asynchronous:
        expected.append("/vision/v1/result")
    expected.append(default_path)
    if asynchronous:
        expected.append("/vision/v1/result")
    assert paths == expected


@pytest.mark.parametrize(
    ("body", "code", "kind"),
    [
        (b'{"code":200,"message":"ok","data":{}}', "", ErrorKind.PROTOCOL),
        (b'{"message":"ok","data":{}}', "", ErrorKind.PROTOCOL),
        (b"not-json", "", ErrorKind.PROTOCOL),
        (b'{"code":"401001","message":"missing api key","data":null}', "401001", ErrorKind.API),
        (b'{"code":"402001","message":"insufficient balance","data":null}', "402001", ErrorKind.API),
        (b'{"code":"413001","message":"too large","data":null}', "413001", ErrorKind.API),
    ],
)
def test_strict_envelopes_and_business_failures(
    make_client: Callable[..., Client], body: bytes, code: str, kind: ErrorKind
) -> None:
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(200, content=body, headers={"X-Request-ID": "failure-id"})
        responses.append(response)
        return response

    with pytest.raises(MaaSError) as caught:
        OCRService(make_client(handler)).parse_document(SOURCE)
    assert caught.value.kind is kind
    assert caught.value.code == code
    assert caught.value.request_id == "failure-id"
    assert responses[0].is_closed


@pytest.mark.parametrize("data", [{"task_status": "PENDING"}, [], {"task_uuid": 12}])
def test_invalid_submission_payload_is_protocol_error(make_client: Callable[..., Client], data: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return envelope(data)

    with pytest.raises(MaaSError) as caught:
        OCRService(make_client(handler)).submit_document(SOURCE)
    assert caught.value.kind is ErrorKind.PROTOCOL
    assert caught.value.request_id == "req-1"


def test_query_business_failure_carries_task_uuid(make_client: Callable[..., Client]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return envelope(submission())
        return envelope(None, code="640402", message="task not found")

    with pytest.raises(MaaSError) as caught:
        OCRService(make_client(handler)).parse_document(SOURCE)
    assert caught.value.kind is ErrorKind.API
    assert caught.value.task_uuid == "task-1"


def test_malformed_query_result_carries_task_uuid(make_client: Callable[..., Client]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return envelope(submission() if request.method == "POST" else [])

    service = OCRService(make_client(handler))
    task = service.submit_document(SOURCE)
    with pytest.raises(MaaSError) as caught:
        task.get()
    assert caught.value.kind is ErrorKind.PROTOCOL
    assert caught.value.task_uuid == "task-1"


def test_local_validation_and_unknown_card_kind_never_send(make_client: Callable[..., Client]) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return envelope(vision())

    service = OCRService(make_client(handler, max_upload_size=4))
    with pytest.raises(MaaSError, match="no file provided") as missing:
        service.parse_document(None)
    assert missing.value.kind is ErrorKind.LOCAL
    with pytest.raises(MaaSError, match="local size limit") as oversized:
        service.parse_document(SOURCE)
    assert oversized.value.kind is ErrorKind.LOCAL
    with pytest.raises(MaaSError, match="unknown card kind") as unknown:
        service.recognize("nope", SOURCE)  # type: ignore[arg-type]
    assert unknown.value.kind is ErrorKind.LOCAL
    assert calls == 0


def test_task_status_and_duration_edge_cases() -> None:
    unknown = TaskStatus("FUTURE_STATE")
    assert isinstance(unknown, str)
    assert unknown.value == "FUTURE_STATE"
    assert str(unknown) == "FUTURE_STATE"
    assert not unknown.terminal and not unknown.ok
    assert TaskStatus.SUCCESS.terminal and TaskStatus.SUCCESS.ok
    assert TaskStatus.FAILED.terminal and TaskStatus.CANCELED.terminal
    assert not TaskStatus("").terminal
    assert VisionResult(task_status="SUCCESS").succeeded()
    assert VisionResult(start_time=5, end_time=4).duration == timedelta(0)
    assert VisionResult(start_time=5, end_time=0).duration == timedelta(0)


def test_unattached_or_empty_task_is_a_local_error() -> None:
    task = Task("", TaskStatus.PENDING, ResponseMeta())
    with pytest.raises(MaaSError, match="task uuid is empty") as empty:
        task.wait()
    assert empty.value.kind is ErrorKind.LOCAL
    with pytest.raises(MaaSError, match="not attached") as unattached:
        Task("t", TaskStatus.PENDING, ResponseMeta()).get()
    assert unattached.value.kind is ErrorKind.LOCAL
