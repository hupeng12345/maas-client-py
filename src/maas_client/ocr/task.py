from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded
from ..errors import ErrorKind, MaaSError, new_local_error
from ..hooks import PollEvent
from .options import WaitOption, _WaitConfig
from .types import TaskStatus, VisionResult

if TYPE_CHECKING:
    from .service import OCRService

_ACTION_RETRY_INPUT = "submit a new task after fixing the input"
_ACTION_KEEP_TASK = "keep the task_uuid and query it later"


class TaskError(MaaSError):
    """A task outcome error that retains the latest platform result."""

    def __init__(self, message: str, *, result: VisionResult | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.result = result


@dataclass(slots=True)
class Task:
    uuid: str
    status: TaskStatus
    meta: ResponseMeta
    _service: OCRService | None = field(default=None, repr=False, compare=False)
    _op: str = field(default="ocr.Task", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, TaskStatus):
            self.status = TaskStatus(self.status)

    def get(self, *, ctx: RequestContext | None = None) -> VisionResult:
        service = self._require_service("ocr.Task.Get")
        return service._query_vision(ctx or RequestContext(), "ocr.Task.Get", self.uuid)

    def wait(self, *options: WaitOption, ctx: RequestContext | None = None) -> VisionResult:
        op = "ocr.Task.Wait"
        if not self.uuid:
            raise new_local_error(op, "task uuid is empty", "pass the task returned by the submission")
        service = self._require_service(op)
        context = ctx or RequestContext()
        config = _WaitConfig(service._client.poll_policy)
        for option in options:
            if option is not None:
                option(config)

        deadline = time.monotonic() + config.policy.timeout
        if context.deadline is not None:
            deadline = min(deadline, context.deadline)
        delay = config.policy.initial_delay
        attempt = 0
        latest: VisionResult | None = None

        while True:
            if context.cancelled:
                raise self._canceled_error(op, RequestCanceled("request canceled"))
            if time.monotonic() >= deadline:
                raise self._timeout_error(op, latest)

            attempt += 1
            query_context = context.child(timeout=config.policy.query_timeout)
            try:
                result = service._query_vision(query_context, op, self.uuid)
            except MaaSError as exc:
                exc.task_uuid = self.uuid
                raise
            latest = result

            if config.on_status is not None:
                config.on_status(context, result.task_status, attempt)
            hook = service._client.hooks.on_poll
            if hook is not None:
                hook(
                    context,
                    PollEvent(
                        op=op,
                        task_uuid=self.uuid,
                        status=result.task_status.value,
                        attempt=attempt,
                    ),
                )

            if result.task_status is TaskStatus.SUCCESS:
                return result
            if result.task_status in {TaskStatus.FAILED, TaskStatus.CANCELED}:
                raise TaskError(
                    "task did not succeed",
                    result=result,
                    kind=ErrorKind.TASK,
                    hint=f"inspect the result for details; {_ACTION_RETRY_INPUT}",
                    http_status=result.meta.http_status,
                    request_id=result.meta.request_id,
                    task_uuid=self.uuid,
                    op=op,
                )

            if context.cancelled:
                raise self._canceled_error(op, RequestCanceled("request canceled"))
            if time.monotonic() + delay >= deadline:
                raise self._timeout_error(op, result)
            try:
                completed = context.wait(_jitter(delay, config.policy.jitter))
            except RequestCanceled as exc:
                raise self._canceled_error(op, exc) from exc
            except RequestDeadlineExceeded as exc:
                raise self._timeout_error(op, result) from exc
            if not completed:
                if context.cancelled:
                    raise self._canceled_error(op, RequestCanceled("request canceled"))
                raise self._timeout_error(op, result)

            delay = min(delay * config.policy.multiplier, config.policy.max_delay)

    def _require_service(self, op: str) -> OCRService:
        if self._service is None:
            raise new_local_error(op, "task is not attached to a service", "use a task returned by a submission")
        return self._service

    def _canceled_error(self, op: str, cause: BaseException) -> MaaSError:
        return MaaSError(
            "stopped waiting for the task",
            kind=ErrorKind.TASK,
            hint="the context was canceled; the platform keeps processing the task",
            task_uuid=self.uuid,
            op=op,
            cause=cause,
        )

    def _timeout_error(self, op: str, latest: VisionResult | None) -> TaskError:
        result = latest or VisionResult(meta=self.meta, task_uuid=self.uuid, task_status=self.status)
        return TaskError(
            "timed out waiting for the task",
            result=result,
            kind=ErrorKind.TASK,
            hint=f"raise with_wait_timeout, or {_ACTION_KEEP_TASK}",
            http_status=result.meta.http_status,
            request_id=result.meta.request_id,
            task_uuid=self.uuid,
            op=op,
            retryable=True,
        )


def _jitter(delay: float, ratio: float) -> float:
    if ratio <= 0:
        return delay
    return max(0.0, delay + (random.uniform(-1.0, 1.0) * delay * ratio))


__all__ = ["Task", "TaskError"]
