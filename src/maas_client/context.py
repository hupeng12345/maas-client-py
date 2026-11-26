from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

_current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("maas_request_id", default="")


class RequestCanceled(Exception):
    pass


class RequestDeadlineExceeded(TimeoutError):
    pass


@dataclass(slots=True)
class RequestContext:
    request_id: str = ""
    deadline: float | None = None
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)

    @classmethod
    def timeout(cls, seconds: float, *, request_id: str = "") -> RequestContext:
        return cls(request_id=request_id, deadline=time.monotonic() + seconds)

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def remaining(self) -> float | None:
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.monotonic())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RequestCanceled("request canceled")
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            raise RequestDeadlineExceeded("request deadline exceeded")

    def wait(self, seconds: float) -> bool:
        self.raise_if_cancelled()
        remaining = self.remaining()
        if remaining is not None:
            seconds = min(seconds, remaining)
        return not self._cancelled.wait(max(0.0, seconds))

    def child(self, *, timeout: float | None = None, request_id: str | None = None) -> RequestContext:
        deadline = self.deadline
        if timeout is not None:
            timeout_deadline = time.monotonic() + timeout
            deadline = timeout_deadline if deadline is None else min(deadline, timeout_deadline)
        return replace(
            self,
            request_id=self.request_id if request_id is None else request_id,
            deadline=deadline,
        )


@contextmanager
def request_id_scope(request_id: str) -> Iterator[None]:
    token = _current_request_id.set(request_id.strip())
    try:
        yield
    finally:
        _current_request_id.reset(token)


def current_request_id() -> str:
    return _current_request_id.get()


__all__ = [
    "RequestContext",
    "RequestCanceled",
    "RequestDeadlineExceeded",
    "request_id_scope",
    "current_request_id",
]
