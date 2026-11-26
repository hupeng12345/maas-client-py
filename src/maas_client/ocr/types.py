from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import Any

from ..config import ResponseMeta


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

    @classmethod
    def _missing_(cls, value: object) -> TaskStatus | None:
        if isinstance(value, str):
            status = str.__new__(cls, value)
            status._name_ = value
            status._value_ = value
            return status
        return None

    @property
    def terminal(self) -> bool:
        # Unknown nonempty states must remain pollable for forward compatibility.
        return self in {TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELED}

    @property
    def ok(self) -> bool:
        return self is TaskStatus.SUCCESS


@dataclass(slots=True)
class VisionResult:
    meta: ResponseMeta = field(default_factory=ResponseMeta)
    task_uuid: str = ""
    task_status: TaskStatus = TaskStatus.PENDING
    start_time: int = 0
    end_time: int = 0
    markdown_content: str = ""
    pages: list[Page] = field(default_factory=list)
    toc: TOC | None = None
    raw: bytes | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task_status, TaskStatus):
            self.task_status = TaskStatus(self.task_status)

    def succeeded(self) -> bool:
        return self.task_status.ok

    @property
    def duration(self) -> timedelta:
        if not self.start_time or not self.end_time or self.end_time < self.start_time:
            return timedelta(0)
        return timedelta(seconds=self.end_time - self.start_time)


@dataclass(slots=True)
class Page:
    index: int = 0
    markdown: str = ""
    layout: list[LayoutBlock] = field(default_factory=list)
    ie_result: list[IEResult] = field(default_factory=list)


@dataclass(slots=True)
class LayoutBlock:
    type: str = ""
    page_idx: int = 0
    bbox: list[float] | None = None
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IEResult:
    recognized: bool = False
    category: str = ""
    bbox: list[float] | None = None
    angle: int = 0
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TOC:
    doc_name: str = ""
    structure: list[TOCNode] = field(default_factory=list)


@dataclass(slots=True)
class TOCNode:
    title: str = ""
    node_id: str = ""
    text: str = ""
    start_index: int = 0
    end_index: int = 0
    children: list[TOCNode] = field(default_factory=list)


__all__ = ["IEResult", "LayoutBlock", "Page", "TOC", "TOCNode", "TaskStatus", "VisionResult"]
