from __future__ import annotations

import secrets
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum

from .errors import new_local_error
from .files import FileSource


@dataclass(frozen=True, slots=True)
class MultipartBody:
    reader: Iterable[bytes]
    content_type: str
    content_length: int
    replayable: bool

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self.reader


def _escape_header(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


def _render_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, float):
        return format(value, "f").rstrip("0").rstrip(".") if value else "0"
    return str(value)


def new_multipart(file: FileSource | None, fields: Mapping[str, object]) -> MultipartBody:
    if file is None:
        raise new_local_error("multipart", "no file provided", "Pass file(...) to specify the upload")
    if not file.name():
        raise new_local_error("multipart", "file name is empty", "Provide a non-empty file name")

    boundary = secrets.token_hex(30)
    field_parts: list[bytes] = []
    for key in sorted(key for key, value in fields.items() if value is not None and key != "file"):
        name = _escape_header(key)
        value = _render_value(fields[key])
        field_parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )

    filename = _escape_header(file.name())
    file_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode()
    closing = f"\r\n--{boundary}--\r\n".encode()
    file_size = file.size()
    content_length = (
        sum(map(len, field_parts)) + len(file_header) + len(closing) + file_size if file_size >= 0 else -1
    )

    def chunks() -> Iterator[bytes]:
        yield from field_parts
        yield file_header
        source = file.open()
        try:
            while chunk := source.read(1024 * 1024):
                yield chunk
        finally:
            source.close()
        yield closing

    return MultipartBody(
        reader=chunks(),
        content_type=f"multipart/form-data; boundary={boundary}",
        content_length=content_length,
        replayable=file.replayable(),
    )


__all__ = ["MultipartBody", "new_multipart"]
