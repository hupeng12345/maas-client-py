from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast, runtime_checkable


class FileConsumedError(IOError):
    pass


@runtime_checkable
class FileSource(Protocol):
    def name(self) -> str: ...

    def size(self) -> int: ...

    def open(self) -> BinaryIO: ...

    def replayable(self) -> bool: ...


class _PathSource:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def name(self) -> str:
        return self.path.name

    def size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return -1

    def open(self) -> BinaryIO:
        return self.path.open("rb")

    def replayable(self) -> bool:
        return True


class _BytesSource:
    def __init__(self, name: str, data: bytes) -> None:
        self._name = name or "upload.bin"
        self.data = data

    def name(self) -> str:
        return self._name

    def size(self) -> int:
        return len(self.data)

    def open(self) -> BinaryIO:
        return io.BytesIO(self.data)

    def replayable(self) -> bool:
        return True


class _NonClosingReader(io.RawIOBase):
    def __init__(self, reader: BinaryIO) -> None:
        self.reader = reader

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return self.reader.seekable()

    def readinto(self, buffer: Any) -> int:
        data = self.reader.read(len(buffer))
        if not data:
            return 0
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self.reader.seek(offset, whence)

    def tell(self) -> int:
        return self.reader.tell()

    def close(self) -> None:
        pass


class _ReaderSource:
    def __init__(self, name: str, reader: BinaryIO) -> None:
        self._name = name or "upload.bin"
        self.reader = reader
        self._consumed = False
        self._lock = threading.Lock()

    def name(self) -> str:
        return self._name

    def size(self) -> int:
        return -1

    def open(self) -> BinaryIO:
        with self._lock:
            if self._consumed:
                raise FileConsumedError("maas: file reader already consumed")
            self._consumed = True
        return io.BufferedReader(_NonClosingReader(self.reader))

    def replayable(self) -> bool:
        return False


class _SeekerSource:
    def __init__(self, name: str, reader: BinaryIO) -> None:
        self._name = name or "upload.bin"
        self.reader = reader

    def name(self) -> str:
        return self._name

    def size(self) -> int:
        try:
            current = self.reader.tell()
            self.reader.seek(0, io.SEEK_END)
            size = self.reader.tell()
            self.reader.seek(current, io.SEEK_SET)
            return size
        except (OSError, ValueError, io.UnsupportedOperation):
            return -1

    def open(self) -> BinaryIO:
        self.reader.seek(0, io.SEEK_SET)
        return io.BufferedReader(_NonClosingReader(self.reader))

    def replayable(self) -> bool:
        return True


def file(value: object) -> FileSource | None:
    if value is None:
        return None
    if isinstance(value, FileSource):
        return value
    if isinstance(value, (str, os.PathLike)):
        return file_from_path(value)
    if isinstance(value, bytes):
        return file_from_bytes("upload.bin", value)
    if hasattr(value, "seek") and hasattr(value, "tell") and hasattr(value, "seekable") and value.seekable():
        return file_from_reader_seeker("upload.bin", cast(BinaryIO, value))
    if hasattr(value, "read"):
        return file_from_reader("upload.bin", cast(BinaryIO, value))
    return None


def file_from_path(path: str | os.PathLike[str]) -> FileSource:
    return _PathSource(path)


def file_from_bytes(name: str, data: bytes) -> FileSource:
    return _BytesSource(name, data)


def file_from_reader(name: str, reader: BinaryIO) -> FileSource:
    return _ReaderSource(name, reader)


def file_from_reader_seeker(name: str, reader: BinaryIO) -> FileSource:
    return _SeekerSource(name, reader)


__all__ = [
    "FileSource",
    "FileConsumedError",
    "file",
    "file_from_path",
    "file_from_bytes",
    "file_from_reader",
    "file_from_reader_seeker",
]
