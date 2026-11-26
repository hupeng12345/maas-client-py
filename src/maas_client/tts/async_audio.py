from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from os import PathLike
from typing import BinaryIO

import httpx

from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded, _context_error, run_with_context
from .audio import WAV_HEADER_SIZE, _wav_header, _write_all


@dataclass(slots=True)
class AsyncAudio:
    response: httpx.Response
    sample_rate: int = 8000
    channels: int = 1
    format: str = "pcm_s16le"
    content_type: str = ""
    meta: ResponseMeta = field(default_factory=ResponseMeta)
    ctx: RequestContext | None = None
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _chunks: AsyncIterator[bytes] = field(init=False, repr=False)
    _exhausted: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._chunks = self.response.aiter_bytes()

    async def _next_chunk(self) -> bytes:
        try:
            if self.ctx is None:
                return await self._chunks.__anext__()
            return await run_with_context(self.ctx, self._chunks.__anext__)
        except StopAsyncIteration:
            self._exhausted = True
            raise
        except BaseException as exc:
            try:
                await self.response.aclose()
            except Exception:
                pass
            if self.ctx is not None and isinstance(exc, (RequestCanceled, RequestDeadlineExceeded)):
                raise _context_error("tts.Speech", self.ctx, exc) from exc
            raise

    async def _iter_chunks(self) -> AsyncIterator[bytes]:
        while not self._exhausted:
            try:
                chunk = await self._next_chunk()
            except StopAsyncIteration:
                return
            yield chunk

    async def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if size < 0:
            chunks = [bytes(self._buffer)]
            self._buffer.clear()
            async for chunk in self._iter_chunks():
                chunks.append(chunk)
            return b"".join(chunks)
        while len(self._buffer) < size and not self._exhausted:
            try:
                self._buffer.extend(await self._next_chunk())
            except StopAsyncIteration:
                break
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def to_bytes(self) -> bytes:
        return await self.read()

    async def close(self) -> None:
        await self.response.aclose()

    async def write_wav(self, writer: BinaryIO) -> int:
        await asyncio.to_thread(_write_all, writer, _wav_header(self.sample_rate, self.channels))
        written = WAV_HEADER_SIZE
        async for chunk in self:
            await asyncio.to_thread(_write_all, writer, chunk)
            written += len(chunk)
        return written

    async def write_wav_file(self, path: str | PathLike[str]) -> None:
        output: BinaryIO = await asyncio.to_thread(lambda: open(path, "wb"))
        try:
            header = bytearray(_wav_header(self.sample_rate, self.channels))
            await asyncio.to_thread(_write_all, output, header)
            data_size = 0
            async for chunk in self:
                await asyncio.to_thread(_write_all, output, chunk)
                data_size += len(chunk)
                if data_size > 0xFFFFFFFF - 36:
                    raise ValueError("audio is too large for a PCM WAV file")
            await asyncio.to_thread(output.flush)
            struct.pack_into("<I", header, 4, 36 + data_size)
            struct.pack_into("<I", header, 40, data_size)
            await asyncio.to_thread(output.seek, 0)
            await asyncio.to_thread(_write_all, output, header)
        finally:
            await asyncio.to_thread(output.close)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._buffer:
            data = bytes(self._buffer)
            self._buffer.clear()
            yield data
        async for chunk in self._iter_chunks():
            yield chunk

    async def __aenter__(self) -> AsyncAudio:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()


__all__ = ["AsyncAudio"]
