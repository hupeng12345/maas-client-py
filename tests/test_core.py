from __future__ import annotations

import io
import logging

import httpx
import pytest

from maas_client import (
    Client,
    ConfigError,
    ErrorKind,
    Hooks,
    RequestContext,
    RetryPolicy,
    decode_envelope,
    file_from_bytes,
    new_multipart,
    with_request_id,
)


def response(body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test/vision/v1/result")
    return httpx.Response(status, content=body, headers=headers, request=request)


def test_config_requires_explicit_values_and_ignores_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAAS_API_KEY", "secret")
    monkeypatch.setenv("MAAS_BASE_URL", "https://example.test")
    with pytest.raises(TypeError):
        Client()  # type: ignore[call-arg]
    with pytest.raises(ConfigError, match="api_key"):
        Client(api_key=" ", base_url="https://example.test")
    with pytest.raises(ConfigError, match="base_url"):
        Client(api_key="key", base_url="ftp://example.test")


def test_client_normalizes_base_url_and_injects_headers() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"code": "000000", "data": {}}, headers={"X-Request-ID": "server-id"})

    client = Client(
        api_key=" sk_test ",
        base_url=" https://example.test/gateway/ ",
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    ctx = RequestContext(request_id="local-id")
    req = client.new_request("GET", "/vision/v1/result", ctx=ctx)
    req.headers["Accept"] = "application/custom"
    resp, meta, err = client.do(req, "test", ctx=ctx)
    assert err is None
    assert resp is not None
    assert calls[0].headers["Authorization"] == "Bearer sk_test"
    assert calls[0].headers["X-Request-ID"] == "local-id"
    assert calls[0].headers["User-Agent"].startswith("maas-client-py/")
    assert calls[0].headers["Accept"] == "application/custom"
    assert str(calls[0].url) == "https://example.test/gateway/vision/v1/result"
    assert meta.request_id == "server-id"
    assert meta.http_status == 200
    client.close()


def test_retry_only_replays_idempotent_requests() -> None:
    get_calls = 0
    post_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal get_calls, post_calls
        if request.method == "GET":
            get_calls += 1
            if get_calls < 3:
                return httpx.Response(503, json={"code": "503001"})
            return httpx.Response(200, json={"code": "000000"})
        post_calls += 1
        return httpx.Response(503, json={"code": "503001"})

    client = Client(
        api_key="key",
        base_url="https://example.test",
        retry=RetryPolicy(max_attempts=3),
        transport=httpx.MockTransport(handler),
    )
    resp, _, err = client.do(client.new_request("GET", "/x"), "get")
    assert err is None and resp is not None and resp.status_code == 200
    assert get_calls == 3
    resp, _, err = client.do(client.new_request("POST", "/x", b"{}"), "post")
    assert err is None and resp is not None and resp.status_code == 503
    assert post_calls == 1
    client.close()


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
def test_envelope_rejects_non_standard_payloads(payload: bytes) -> None:
    with pytest.raises(Exception) as caught:
        decode_envelope("test", response(payload))
    assert isinstance(caught.value, Exception)
    assert caught.value.kind == ErrorKind.PROTOCOL


def test_envelope_accepts_null_data_and_preserves_raw_data() -> None:
    env = decode_envelope("test", response(b'{"code":"000000","message":null,"data":{"a": 1}}'))
    assert env.is_success()
    assert env.message == ""
    assert env.data == b'{"a": 1}'
    assert env.value == {"a": 1}
    empty = decode_envelope("test", response(b'{"code":"000000"}'))
    assert empty.data is None


def test_file_and_multipart_are_lazy_and_deterministic() -> None:
    source = file_from_bytes('a"b.pdf', b"hello")
    body = new_multipart(source, {"z": 1, "a": True, "skip": None, "file": "shadow"})
    raw = b"".join(body.iter_bytes())
    assert body.replayable
    assert body.content_length == len(raw)
    assert b'name="a"\r\n\r\ntrue\r\n' in raw
    assert b'name="z"\r\n\r\n1\r\n' in raw
    assert b'name="skip"' not in raw
    assert b'filename="a\\"b.pdf"' in raw
    assert raw.endswith(b"--\r\n")
    assert b"hello\r\n--" in raw


def test_one_shot_reader_cannot_be_reopened() -> None:
    from maas_client import file_from_reader
    from maas_client.files import FileConsumedError

    source = file_from_reader("x.bin", io.BytesIO(b"abc"))
    assert source.open().read() == b"abc"
    with pytest.raises(FileConsumedError):
        source.open()


def test_hooks_redact_url_credentials(caplog: pytest.LogCaptureFixture) -> None:
    request_events = []
    response_events = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"code": "000000"})

    logger = logging.getLogger("maas-redaction-test")
    with Client(
        api_key="sk_secret",
        base_url="https://example.test",
        transport=httpx.MockTransport(handler),
        logger=logger,
        hooks=Hooks(on_request=lambda ctx, event: request_events.append(event),
                   on_response=lambda ctx, event: response_events.append(event)),
    ) as client:
        with caplog.at_level(logging.INFO, logger=logger.name):
            response, _, error = client.do(
                client.new_request(
                    "GET",
                    "https://user:password@example.test/x?api_key=query-secret&access_token=token-secret&task_uuid=abc",
                ),
                "test",
            )
        assert response is not None
        response.close()
        assert error is None

    observed = [request_events[0].url, response_events[0].url, caplog.text]
    for text in observed:
        assert "password" not in text
        assert "query-secret" not in text
        assert "token-secret" not in text
    for text in observed[:2]:
        assert "%2A%2A%2A" in text
        assert "task_uuid=abc" in text


def test_with_request_id_keeps_deadline() -> None:
    original = RequestContext.timeout(10, request_id="old")
    changed = with_request_id(original, " new ")
    assert changed.request_id == "new"
    assert changed.deadline == original.deadline
