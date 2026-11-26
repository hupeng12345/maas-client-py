from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

from ..config import PollPolicy
from ..context import RequestContext
from .types import TaskStatus


@dataclass(slots=True)
class _RequestOptions:
    fields: dict[str, object] = field(default_factory=dict)
    async_submit: bool = False
    wait: list[WaitOption] = field(default_factory=list)
    endpoint: str = ""

    def set(self, name: str, value: object) -> None:
        self.fields[name] = value

    def path_or(self, fallback: str) -> str:
        return self.endpoint or fallback


RequestOption = Callable[[_RequestOptions], None]
ParseOption = RequestOption
RecognizeOption = RequestOption
OCROption = RequestOption


@dataclass(slots=True)
class _WaitConfig:
    policy: PollPolicy
    on_status: Callable[[RequestContext, TaskStatus, int], None] | None = None


WaitOption = Callable[[_WaitConfig], None]
PollCallback = Callable[[RequestContext, TaskStatus, int], None]


def enable_toc(enabled: bool) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        options.set("enable_toc", enabled)

    return apply


def with_language(language: str) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        options.set("language", language)

    return apply


def with_image_analysis(enabled: bool) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        options.set("image_analysis", enabled)

    return apply


def with_page_details(enabled: bool) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        options.set("page_details", enabled)

    return apply


def with_fields(fields: Mapping[str, object]) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        for key, value in fields.items():
            options.set(key, value)

    return apply


def with_async_submit() -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        options.async_submit = True

    return apply


def with_endpoint(path: str) -> RequestOption:
    def apply(options: _RequestOptions) -> None:
        if path:
            options.endpoint = path

    return apply


def with_wait_options(*options: WaitOption) -> RequestOption:
    def apply(request: _RequestOptions) -> None:
        request.wait.extend(options)

    return apply


def with_poll_interval(seconds: float) -> WaitOption:
    def apply(config: _WaitConfig) -> None:
        if seconds > 0:
            config.policy = replace(config.policy, initial_delay=seconds)

    return apply


def with_max_poll_interval(seconds: float) -> WaitOption:
    def apply(config: _WaitConfig) -> None:
        if seconds > 0:
            config.policy = replace(config.policy, max_delay=seconds)

    return apply


def with_wait_timeout(seconds: float) -> WaitOption:
    def apply(config: _WaitConfig) -> None:
        if seconds > 0:
            config.policy = replace(config.policy, timeout=seconds)

    return apply


def with_poll_callback(callback: PollCallback | None) -> WaitOption:
    def apply(config: _WaitConfig) -> None:
        config.on_status = callback

    return apply


def resolve_options(options: tuple[RequestOption, ...]) -> _RequestOptions:
    resolved = _RequestOptions()
    for option in options:
        if option is not None:
            option(resolved)
    return resolved


__all__ = [
    "OCROption",
    "ParseOption",
    "PollCallback",
    "RecognizeOption",
    "RequestOption",
    "WaitOption",
    "enable_toc",
    "with_async_submit",
    "with_endpoint",
    "with_fields",
    "with_image_analysis",
    "with_language",
    "with_max_poll_interval",
    "with_page_details",
    "with_poll_callback",
    "with_poll_interval",
    "with_wait_options",
    "with_wait_timeout",
]
