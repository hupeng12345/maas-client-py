from __future__ import annotations

from datetime import UTC, datetime

from maas_client.errors import (
    CODE_BACKEND_TIMEOUT,
    CODE_INVALID_API_KEY,
    CODE_RATE_LIMITED,
    ErrorKind,
    is_auth,
    is_rate_limited,
    is_retryable,
    is_timeout,
    new_api_error,
    parse_retry_after,
)


def test_business_error_classification() -> None:
    auth = new_api_error("ocr", CODE_INVALID_API_KEY, "bad key", 401)
    assert is_auth(auth)
    assert not is_retryable(auth)
    limited = new_api_error("ocr", CODE_RATE_LIMITED, "slow down", 429)
    assert is_rate_limited(limited)
    assert is_retryable(limited)
    timeout = new_api_error("tts", CODE_BACKEND_TIMEOUT, "timeout", 504)
    assert is_timeout(timeout)
    assert timeout.kind == ErrorKind.API


def test_unknown_6xx_errors_are_not_retryable() -> None:
    assert not new_api_error("ocr", "640999", "bad", 500).retryable
    assert new_api_error("ocr", "599999", "server", 500).retryable


def test_retry_after_seconds_and_http_date() -> None:
    assert parse_retry_after("12") == 12.0
    assert parse_retry_after("-1") == 0.0
    now = datetime(2015, 10, 21, 7, 27, 59, tzinfo=UTC)
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", now=now) == 1.0
    assert parse_retry_after("invalid") == 0.0


def test_error_string_contains_actionable_fields() -> None:
    err = new_api_error("ocr.ParseDocument", CODE_INVALID_API_KEY, "expired", 401)
    err.request_id = "req-1"
    text = str(err)
    assert "ocr.ParseDocument" in text
    assert "401002" in text
    assert "req-1" in text
    assert "hint:" in text
