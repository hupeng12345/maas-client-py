from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from os import PathLike
from typing import BinaryIO

import httpx

from ..config import ResponseMeta

WAV_HEADER_SIZE = 44


def _wav_header(sample_rate: int, channels: int, data_size: int = 0) -> bytes:
    if sample_rate <= 0:
        sample_rate = 8000
    if channels <= 0:
        channels = 1
    block_align = channels * 2
    byte_rate = sample_rate * block_align
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_size,
    )


def _write_all(writer: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    view = memoryview(data)
    while view:
        written = writer.write(view)
        if written is None or written <= 0:
            raise OSError("short write")
        view = view[written:]


@dataclass(slots=True)
class Audio:
    response: httpx.Response
    sample_rate: int = 8000
    channels: int = 1
    format: str = "pcm_s16le"
    content_type: str = ""
    meta: ResponseMeta = field(default_factory=ResponseMeta)
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _chunks: Iterator[bytes] = field(init=False, repr=False)
    _exhausted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._chunks = self.response.iter_bytes()

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            data = bytes(self._buffer) + b"".join(self._chunks)
            self._buffer.clear()
            return data
        while len(self._buffer) < size and not self._exhausted:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                self._exhausted = True
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    def close(self) -> None:
        self.response.close()

    def write_wav(self, writer: BinaryIO) -> int:
        _write_all(writer, _wav_header(self.sample_rate, self.channels))
        written = WAV_HEADER_SIZE
        for chunk in self:
            _write_all(writer, chunk)
            written += len(chunk)
        return written

    def write_wav_file(self, path: str | PathLike[str]) -> None:
        with open(path, "wb") as output:
            header = bytearray(_wav_header(self.sample_rate, self.channels))
            _write_all(output, header)
            data_size = 0
            for chunk in self:
                _write_all(output, chunk)
                data_size += len(chunk)
                if data_size > 0xFFFFFFFF - 36:
                    raise ValueError("audio is too large for a PCM WAV file")
            output.flush()
            struct.pack_into("<I", header, 4, 36 + data_size)
            struct.pack_into("<I", header, 40, data_size)
            output.seek(0)
            _write_all(output, header)

    def to_bytes(self) -> bytes:
        return self.read()

    def __iter__(self) -> Iterator[bytes]:
        if self._buffer:
            yield bytes(self._buffer)
            self._buffer.clear()
        yield from self._chunks

    def __enter__(self) -> Audio:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


__all__ = ["Audio", "WAV_HEADER_SIZE"]
