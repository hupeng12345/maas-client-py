from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass

FRAME_VERSION = 0x11
FRAME_SERIALIZATION_COMPRESSION = 0x11
MSG_TYPE_FULL_CLIENT = 0x1
MSG_TYPE_AUDIO_ONLY = 0x2
MSG_TYPE_FULL_SERVER = 0x9
MSG_TYPE_ERROR = 0xF
FLAG_POSITIVE_SEQ = 0x1
FLAG_LAST_NO_SEQ = 0x2
FLAG_NEGATIVE_SEQ = 0x3
HEADER_LEN = 12
ERROR_HEADER_LEN = 16


@dataclass(frozen=True, slots=True)
class ASRFrame:
    type: int
    flags: int
    sequence: int
    code: int = 0
    payload: bytes = b""


def encode_frame(message_type: int, flags: int, sequence: int, payload: bytes | bytearray | memoryview) -> bytes:
    """Encode a protocol frame, including a gzip member for empty payloads."""
    if not 0 <= message_type <= 0xF or not 0 <= flags <= 0xF:
        raise ValueError("asr: message type and flags must fit in a nibble")
    if not -(2**31) <= sequence < 2**31:
        raise ValueError("asr: sequence must fit in a signed 32-bit integer")
    body = gzip.compress(bytes(payload), mtime=0)
    return struct.pack(">BBBBii", FRAME_VERSION, (message_type << 4) | flags, FRAME_SERIALIZATION_COMPRESSION, 0,
                      sequence, len(body)) + body


def decode_frame(data: bytes | bytearray | memoryview) -> ASRFrame:
    """Decode and validate a normal or error server frame."""
    raw = bytes(data)
    if len(raw) < HEADER_LEN:
        raise ValueError(f"asr: frame too short: {len(raw)} bytes")
    if raw[0] & 0xF0 != FRAME_VERSION & 0xF0:
        raise ValueError(f"asr: unsupported protocol version 0x{raw[0]:02x}")
    message_type, flags = raw[1] >> 4, raw[1] & 0x0F
    sequence = struct.unpack_from(">i", raw, 4)[0]
    offset = HEADER_LEN
    code = 0
    if message_type == MSG_TYPE_ERROR:
        if len(raw) < ERROR_HEADER_LEN:
            raise ValueError(f"asr: error frame too short: {len(raw)} bytes")
        code = struct.unpack_from(">I", raw, 8)[0]
        offset = ERROR_HEADER_LEN
    size = struct.unpack_from(">I", raw, offset - 4)[0]
    if size > len(raw) - offset:
        raise ValueError(f"asr: payload size {size} exceeds frame length {len(raw)}")
    body = raw[offset:offset + size]
    if body and raw[2] & 0x0F == 1:
        try:
            payload = gzip.decompress(body)
        except (OSError, EOFError) as exc:
            raise ValueError(f"asr: invalid gzip payload: {exc}") from exc
    else:
        payload = body
    return ASRFrame(message_type, flags, sequence, code, payload)


__all__ = [
    "ASRFrame", "encode_frame", "decode_frame", "FRAME_VERSION", "FRAME_SERIALIZATION_COMPRESSION",
    "MSG_TYPE_FULL_CLIENT", "MSG_TYPE_AUDIO_ONLY", "MSG_TYPE_FULL_SERVER", "MSG_TYPE_ERROR",
    "FLAG_POSITIVE_SEQ", "FLAG_LAST_NO_SEQ", "FLAG_NEGATIVE_SEQ", "HEADER_LEN", "ERROR_HEADER_LEN",
]
