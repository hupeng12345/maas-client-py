from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from enum import StrEnum


class ErrorKind(StrEnum):
    UNKNOWN = "unknown"
    CONFIG = "config"
    LOCAL = "local"
    TRANSPORT = "transport"
    PROTOCOL = "protocol"
    API = "api"
    TASK = "task"


CODE_SUCCESS = "000000"
CODE_BAD_REQUEST = "400001"
CODE_UNKNOWN_PROVIDER = "400002"
CODE_UNKNOWN_CAPABILITY = "400003"
CODE_CAPABILITY_NOT_SUPPORTED = "400004"
CODE_MISSING_API_KEY = "401001"
CODE_INVALID_API_KEY = "401002"
CODE_MISSING_CONTROL_AUTH = "401003"
CODE_INVALID_CONTROL_AUTH = "401004"
CODE_INSUFFICIENT_BALANCE = "402001"
CODE_ACCESS_DENIED = "403001"
CODE_IP_NOT_ALLOWED = "403002"
CODE_HOTWORD_TABLE_NOT_PERMITTED = "403003"
CODE_NO_ROUTE = "404001"
CODE_MODEL_NOT_FOUND = "404002"
CODE_RESOURCE_NOT_FOUND = "404003"
CODE_BODY_TOO_LARGE = "413001"
CODE_FILE_TOO_LARGE = "413002"
CODE_SYNC_FILE_TOO_LARGE = "640003"
CODE_RATE_LIMITED = "429001"
CODE_INTERNAL_ERROR = "500001"
CODE_BACKEND_UNAVAILABLE = "502001"
CODE_OBJECT_STORAGE_UNAVAILABLE = "502002"
CODE_CIRCUIT_OPEN = "503001"
CODE_SERVER_OVERLOADED = "503002"
CODE_RESOURCE_NOT_READY = "503003"
CODE_BACKEND_TIMEOUT = "504001"
CODE_UNSUPPORTED_FILE_TYPE = "640001"
CODE_TASK_NOT_FOUND = "640402"
CODE_REQUEST_VALIDATION = "642203"
CODE_TTS_UNAVAILABLE = "700000"


_CODE_SPECS: dict[str, tuple[bool, str]] = {
    CODE_BAD_REQUEST: (False, "Check the request parameters and file format"),
    CODE_UNKNOWN_PROVIDER: (False, "Provider is not registered; ask the platform to confirm the model"),
    CODE_UNKNOWN_CAPABILITY: (False, "Capability not recognised; verify the request path"),
    CODE_CAPABILITY_NOT_SUPPORTED: (False, "This provider does not support the capability"),
    CODE_MISSING_API_KEY: (False, "No API key sent: pass api_key when building the client"),
    CODE_INVALID_API_KEY: (False, "API key is invalid, disabled or expired; check its status in the console"),
    CODE_MISSING_CONTROL_AUTH: (False, "Missing user session credential"),
    CODE_INVALID_CONTROL_AUTH: (False, "User session is invalid or expired; sign in again"),
    CODE_INSUFFICIENT_BALANCE: (False, "Insufficient balance; top up or confirm postpaid credit in the console"),
    CODE_ACCESS_DENIED: (False, "The key has no access to this resource; ask the platform to enable it"),
    CODE_IP_NOT_ALLOWED: (False, "Source IP rejected by policy; ask the platform to allowlist it"),
    CODE_HOTWORD_TABLE_NOT_PERMITTED: (False, "Hotword table is not authorised or disabled"),
    CODE_NO_ROUTE: (False, "No route matches this path and method; verify base_url and the endpoint path"),
    CODE_MODEL_NOT_FOUND: (False, "Model is not registered or the alias did not match"),
    CODE_RESOURCE_NOT_FOUND: (False, "No registered resource matched; confirm the model is enabled"),
    CODE_BODY_TOO_LARGE: (False, "Request body too large; shrink the file or split the work"),
    CODE_FILE_TOO_LARGE: (False, "File exceeds the per-file limit"),
    CODE_SYNC_FILE_TOO_LARGE: (False, "Sync recognition images are capped at 10 MiB; use the async endpoint"),
    CODE_RATE_LIMITED: (True, "Rate limited; back off according to Retry-After and retry"),
    CODE_INTERNAL_ERROR: (True, "Internal platform error; retry later"),
    CODE_BACKEND_UNAVAILABLE: (True, "Upstream unavailable; retry later"),
    CODE_OBJECT_STORAGE_UNAVAILABLE: (True, "Object storage unavailable; retry later"),
    CODE_CIRCUIT_OPEN: (True, "Upstream circuit breaker is open; retry later"),
    CODE_SERVER_OVERLOADED: (True, "Gateway concurrency is saturated; retry later"),
    CODE_RESOURCE_NOT_READY: (True, "Resource catalogue is not ready; retry later"),
    CODE_BACKEND_TIMEOUT: (True, "Upstream timed out; retry or send a smaller file"),
    CODE_UNSUPPORTED_FILE_TYPE: (False, "Unsupported file type; check the supported formats"),
    CODE_TASK_NOT_FOUND: (False, "task_uuid does not exist or has expired; submit the task again"),
    CODE_REQUEST_VALIDATION: (False, "Validation failed; check form fields and values"),
    CODE_TTS_UNAVAILABLE: (False, "TTS model, voice or parameters are unavailable"),
}

_AUTH_CODES = {
    CODE_MISSING_API_KEY,
    CODE_INVALID_API_KEY,
    CODE_MISSING_CONTROL_AUTH,
    CODE_INVALID_CONTROL_AUTH,
}
_FORBIDDEN_CODES = {CODE_ACCESS_DENIED, CODE_IP_NOT_ALLOWED}
_NOT_FOUND_CODES = {CODE_NO_ROUTE, CODE_MODEL_NOT_FOUND, CODE_RESOURCE_NOT_FOUND, CODE_TASK_NOT_FOUND}
_TOO_LARGE_CODES = {CODE_BODY_TOO_LARGE, CODE_FILE_TOO_LARGE, CODE_SYNC_FILE_TOO_LARGE}


class MaaSError(Exception):
    def __init__(
        self,
        message: str = "",
        *,
        kind: ErrorKind = ErrorKind.UNKNOWN,
        code: str = "",
        hint: str = "",
        http_status: int = 0,
        request_id: str = "",
        task_uuid: str = "",
        retry_after: float = 0.0,
        retryable: bool = False,
        op: str = "",
        raw_body: bytes | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.code = code
        self.hint = hint
        self.http_status = http_status
        self.request_id = request_id
        self.task_uuid = task_uuid
        self.retry_after = retry_after
        self.retryable = retryable
        self.op = op
        self.raw_body = raw_body
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        text = "maas"
        if self.op:
            text += f": {self.op}"
        if self.code:
            text += f": [{self.code}]"
            if self.message:
                text += f" {self.message}"
        elif self.message:
            text += f": {self.message}"
        if self.http_status:
            text += f" (HTTP {self.http_status}"
            if self.request_id:
                text += f", request_id={self.request_id}"
            text += ")"
        elif self.request_id:
            text += f" (request_id={self.request_id})"
        if self.task_uuid:
            text += f" (task={self.task_uuid})"
        if self.hint:
            text += f"\n  hint: {self.hint}"
        if self.cause is not None:
            text += f"\n  cause: {self.cause}"
        return text


Error = MaaSError


class ConfigError(MaaSError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"{field}: {reason}", kind=ErrorKind.CONFIG, op="config")
        self.field = field
        self.reason = reason


def _clip(value: bytes | None, size: int) -> bytes | None:
    if value is None:
        return None
    return bytes(value[:size])


def new_api_error(op: str, code: str, message: str, http_status: int) -> MaaSError:
    retryable, hint = _CODE_SPECS.get(code, (False, "Request failed; check the parameters and retry"))
    if code not in _CODE_SPECS:
        if code.startswith(("6", "7")):
            hint = "Business processing failed; check the request or contact the platform"
        elif http_status >= 500:
            retryable = True
            hint = "Server-side error; retry later"
    return MaaSError(
        message,
        kind=ErrorKind.API,
        code=code,
        hint=hint,
        http_status=http_status,
        retryable=retryable,
        op=op,
    )


def new_protocol_error(op: str, reason: str, http_status: int, raw: bytes | None) -> MaaSError:
    return MaaSError(
        reason,
        kind=ErrorKind.PROTOCOL,
        hint="Response is not a standard gateway envelope: confirm base_url and gateway version",
        http_status=http_status,
        op=op,
        raw_body=_clip(raw, 4096),
    )


def new_local_error(op: str, reason: str, hint: str = "") -> MaaSError:
    return MaaSError(reason, kind=ErrorKind.LOCAL, hint=hint, op=op)


def new_transport_error(op: str, cause: BaseException, *, request_id: str = "") -> MaaSError:
    return MaaSError(
        "request failed",
        kind=ErrorKind.TRANSPORT,
        hint="Check network, proxy and timeout settings",
        retryable=True,
        op=op,
        request_id=request_id,
        cause=cause,
    )


def classify_transport_error(op: str, cause: BaseException, *, request_id: str = "") -> MaaSError:
    if isinstance(cause, (TimeoutError,)):
        return MaaSError(
            "request timed out",
            kind=ErrorKind.TRANSPORT,
            hint="Raise timeout, or use the async endpoint for large files",
            retryable=True,
            op=op,
            request_id=request_id,
            cause=cause,
        )
    return new_transport_error(op, cause, request_id=request_id)


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float:
    if not value or not value.strip():
        return 0.0
    value = value.strip()
    try:
        return float(int(value)) if int(value) >= 0 else 0.0
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        current = now or datetime.now(UTC)
        return max(0.0, (parsed - current).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def is_auth(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code in _AUTH_CODES


def is_forbidden(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code in _FORBIDDEN_CODES


def is_insufficient(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code == CODE_INSUFFICIENT_BALANCE


def is_rate_limited(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code == CODE_RATE_LIMITED


def is_not_found(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code in _NOT_FOUND_CODES


def is_too_large(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code in _TOO_LARGE_CODES


def is_unsupported(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.code == CODE_UNSUPPORTED_FILE_TYPE


def is_timeout(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and (
        err.code == CODE_BACKEND_TIMEOUT or isinstance(err.cause, TimeoutError)
    )


def is_task_failed(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.kind == ErrorKind.TASK


def is_non_standard_response(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.kind == ErrorKind.PROTOCOL


def is_retryable(err: BaseException | None) -> bool:
    return isinstance(err, MaaSError) and err.retryable


def retry_after(err: BaseException | None) -> float:
    return err.retry_after if isinstance(err, MaaSError) else 0.0


def as_config_error(err: BaseException | None) -> ConfigError | None:
    return err if isinstance(err, ConfigError) else None


__all__ = [
    name
    for name in globals()
    if name.startswith(
        (
            "CODE_",
            "Error",
            "MaaSError",
            "ConfigError",
            "new_",
            "classify_",
            "is_",
            "retry_after",
            "as_config_error",
            "parse_retry_after",
        )
    )
]
