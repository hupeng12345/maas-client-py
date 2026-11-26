import gzip
import struct

import pytest

from maas_client.asr import ASRFrame, decode_frame, encode_frame
from maas_client.asr.frame import (
    ERROR_HEADER_LEN,
    FLAG_NEGATIVE_SEQ,
    FLAG_POSITIVE_SEQ,
    MSG_TYPE_AUDIO_ONLY,
    MSG_TYPE_FULL_CLIENT,
    MSG_TYPE_FULL_SERVER,
)


def error_frame(code: int, payload: bytes, *, plain: bool = False) -> bytes:
    body = payload if plain else gzip.compress(payload, mtime=0)
    return struct.pack(">BBBBiiI", 0x11, 0xF1, 0x11 if not plain else 0x10, 0, 3, code, len(body)) + body


@pytest.mark.parametrize(
    ("message_type", "flags", "sequence", "payload"),
    [
        (MSG_TYPE_FULL_CLIENT, FLAG_POSITIVE_SEQ, 1, b'{"audio":{}}'),
        (MSG_TYPE_AUDIO_ONLY, FLAG_POSITIVE_SEQ, 2, b"\x01\x02\x03\x04"),
        (MSG_TYPE_AUDIO_ONLY, FLAG_NEGATIVE_SEQ, -7, b""),
        (MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 2**31 - 1, b"{}"),
    ],
)
def test_roundtrip(message_type, flags, sequence, payload):
    raw = encode_frame(message_type, flags, sequence, payload)
    assert raw[:4] == bytes((0x11, message_type << 4 | flags, 0x11, 0))
    assert struct.unpack(">i", raw[4:8])[0] == sequence
    assert decode_frame(raw) == ASRFrame(message_type, flags, sequence, 0, payload)
    assert gzip.decompress(raw[12:]) == payload


def test_empty_payload_is_compressed():
    raw = encode_frame(MSG_TYPE_AUDIO_ONLY, FLAG_NEGATIVE_SEQ, -2, b"")
    assert len(raw) > 12
    assert decode_frame(raw).payload == b""


def test_error_and_plain_frames():
    payload = b'{"parret":{"reason":"unusable_audio"}}'
    frame = decode_frame(error_frame(45000002, payload))
    assert frame.code == 45000002
    assert frame.payload == payload
    plain = bytearray(error_frame(45000003, b"plain", plain=True))
    assert len(plain) == ERROR_HEADER_LEN + 5
    assert decode_frame(plain).payload == b"plain"


@pytest.mark.parametrize("raw", [b"", b"\x11\x91", bytes(12), error_frame(1, b"x")[:15]])
def test_rejects_short_frames(raw):
    with pytest.raises(ValueError):
        decode_frame(raw)


def test_rejects_version_length_and_gzip():
    valid = encode_frame(MSG_TYPE_FULL_SERVER, FLAG_POSITIVE_SEQ, 1, b"{}")
    bad_version = bytearray(valid)
    bad_version[0] = 0x21
    bad_length = bytearray(valid)
    bad_length[8:12] = struct.pack(">I", 1 << 20)
    bad_gzip = bytearray(valid)
    bad_gzip[12:] = b"not gzip"
    for raw in (bad_version, bad_length, bad_gzip):
        with pytest.raises(ValueError):
            decode_frame(raw)


def test_encode_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_frame(16, 1, 1, b"")
    with pytest.raises(ValueError):
        encode_frame(1, 1, 2**31, b"")
