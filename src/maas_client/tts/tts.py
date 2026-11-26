from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message

import httpx

from ..client import Client
from ..config import ResponseMeta
from ..context import RequestContext
from ..envelope import decode_envelope
from ..errors import MaaSError, new_local_error, new_protocol_error
from .audio import Audio

PATH_SPEECH = "/tts/v1/speech"
DEFAULT_TTS_PATH = PATH_SPEECH


@dataclass(slots=True)
class SpeechRequest:
    model: str = ""
    input: str = ""
    voice: str = ""
    language: str | None = None
    response_format: str | None = None
    stream: bool | None = None
    sample: int | None = None
    speed: float | None = None
    loud_ratio: float | None = None
    pitch: int | None = None
    endpoint: str | None = None
    seed: int | None = None
    instructions: str | None = None
    bgm_id: str | None = None
    bgm_vol: float | None = None
    use_8k_enhancer: bool | None = None

    def to_dict(self) -> dict[str, object]:
        body: dict[str, object] = {
            "model": self.model,
            "input": self.input,
            "voice": self.voice,
        }
        for name in ("language", "response_format"):
            value = getattr(self, name)
            if value:
                body[name] = value
        for name in ("stream", "seed", "instructions", "bgm_id", "bgm_vol", "use_8k_enhancer"):
            value = getattr(self, name)
            if value is not None:
                body[name] = value
        for name in ("sample", "speed", "loud_ratio", "pitch"):
            value = getattr(self, name)
            if value:
                body[name] = value
        return body


SpeechOption = Callable[[SpeechRequest], None]


def with_model(model: str) -> SpeechOption:
    return lambda request: setattr(request, "model", model)


def with_voice(voice: str) -> SpeechOption:
    return lambda request: setattr(request, "voice", voice)


def with_language(language: str) -> SpeechOption:
    return lambda request: setattr(request, "language", language)


def with_sample(sample: int) -> SpeechOption:
    return lambda request: setattr(request, "sample", sample)


def with_speed(speed: float) -> SpeechOption:
    return lambda request: setattr(request, "speed", speed)


def with_pitch(pitch: int) -> SpeechOption:
    return lambda request: setattr(request, "pitch", pitch)


def with_stream(stream: bool) -> SpeechOption:
    return lambda request: setattr(request, "stream", stream)


def with_bgm(bgm_id: str) -> SpeechOption:
    return lambda request: setattr(request, "bgm_id", bgm_id)


def with_endpoint(endpoint: str) -> SpeechOption:
    return lambda request: setattr(request, "endpoint", endpoint)


def with_instructions(instructions: str) -> SpeechOption:
    return lambda request: setattr(request, "instructions", instructions)


def _validate(request: SpeechRequest) -> None:
    if not isinstance(request.model, str) or not request.model.strip():
        raise new_local_error("tts.Speech", "model is required", "pass the TTS model name assigned to your account")
    if not isinstance(request.input, str) or not request.input.strip():
        raise new_local_error("tts.Speech", "input is required", "pass the text to synthesise")
    if not isinstance(request.voice, str) or not request.voice.strip():
        raise new_local_error("tts.Speech", "voice is required", "pass a voice_id supported by the model")
    if request.response_format not in (None, "", "pcm"):
        raise new_local_error("tts.Speech", 'response_format must be "pcm"', "the platform only returns raw PCM today")


def _drain_and_close(response: httpx.Response) -> None:
    try:
        if not response.is_stream_consumed:
            total = 0
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                total += len(chunk)
                if total >= 64 * 1024:
                    break
    finally:
        response.close()


def _audio_failure(response: httpx.Response, meta: ResponseMeta) -> MaaSError:
    try:
        envelope = decode_envelope("tts.Speech", response, meta=meta)
    except MaaSError as exc:
        exc.request_id = meta.request_id
        raise
    finally:
        _drain_and_close(response)
    if not envelope.is_success():
        failure = envelope.error("tts.Speech", response)
        failure.request_id = meta.request_id
        return failure
    failure = new_protocol_error(
        "tts.Speech",
        "expected audio, got " + response.headers.get("Content-Type", ""),
        response.status_code,
        envelope.data,
    )
    failure.request_id = meta.request_id
    failure.hint = "the endpoint returned a JSON envelope instead of audio; check model and voice"
    return failure


def _media_type(value: str) -> str:
    message = Message()
    message["Content-Type"] = value
    return message.get_content_type().lower()


def _header_int(value: str, fallback: int) -> int:
    value = value.strip()
    if not re.fullmatch(r"[+-]?[0-9]+", value):
        return fallback
    try:
        number = int(value)
    except ValueError:
        return fallback
    return number if 0 < number <= sys.maxsize else fallback


class TTSService:
    def __init__(self, client: Client) -> None:
        self.client = client

    def speech(self, request: SpeechRequest, *, ctx: RequestContext | None = None) -> Audio:
        self.client.require_base_url("tts.Speech")
        _validate(request)
        try:
            payload = json.dumps(request.to_dict(), allow_nan=False).encode("utf-8")
        except (TypeError, ValueError, OverflowError) as error:
            raise new_local_error("tts.Speech", "cannot encode request", "check the request fields") from error

        path = request.endpoint if request.endpoint and request.endpoint.strip() else PATH_SPEECH
        http_request = self.client.new_request("POST", path, payload, ctx=ctx)
        http_request.headers["Content-Type"] = "application/json"
        http_request.headers["Accept"] = "audio/pcm, application/json"
        response, meta, failure = self.client.do(http_request, "tts.Speech", streaming=True, ctx=ctx)
        if failure is not None:
            if response is not None:
                response.close()
            raise failure
        assert response is not None

        content_type = _media_type(response.headers.get("Content-Type", ""))
        if not content_type.startswith("audio/"):
            raise _audio_failure(response, meta)
        return Audio(
            response=response,
            sample_rate=_header_int(response.headers.get("X-Sample-Rate", ""), 8000),
            channels=_header_int(response.headers.get("X-Audio-Channels", ""), 1),
            format=response.headers.get("X-Audio-Format", "").strip() or "pcm_s16le",
            content_type=content_type,
            meta=meta,
        )

    def say(self, text: str, *options: SpeechOption | None, ctx: RequestContext | None = None) -> Audio:
        request = SpeechRequest(input=text, response_format="pcm")
        for option in options:
            if option is not None:
                option(request)
        return self.speech(request, ctx=ctx)


Service = TTSService


__all__ = [
    "DEFAULT_TTS_PATH",
    "PATH_SPEECH",
    "Audio",
    "Service",
    "SpeechOption",
    "SpeechRequest",
    "TTSService",
    "with_bgm",
    "with_endpoint",
    "with_instructions",
    "with_language",
    "with_model",
    "with_pitch",
    "with_sample",
    "with_speed",
    "with_stream",
    "with_voice",
]
