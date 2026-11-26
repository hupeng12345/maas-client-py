from __future__ import annotations

import json

import httpx

from ..async_client import AsyncClient
from ..config import ResponseMeta
from ..context import RequestContext
from ..envelope import decode_envelope_async
from ..errors import MaaSError, new_local_error, new_protocol_error
from .async_audio import AsyncAudio
from .tts import (
    DEFAULT_TTS_PATH,
    PATH_SPEECH,
    SpeechOption,
    SpeechRequest,
    _header_int,
    _media_type,
    _validate,
)


async def _audio_failure(
    response: httpx.Response,
    meta: ResponseMeta,
    ctx: RequestContext | None,
) -> MaaSError:
    try:
        envelope = await decode_envelope_async("tts.Speech", response, meta=meta, ctx=ctx)
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
    finally:
        await response.aclose()


class AsyncTTSService:
    def __init__(self, client: AsyncClient) -> None:
        self.client = client

    async def speech(self, request: SpeechRequest, *, ctx: RequestContext | None = None) -> AsyncAudio:
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
        response, meta, failure = await self.client.do(http_request, "tts.Speech", streaming=True, ctx=ctx)
        if failure is not None:
            if response is not None:
                await response.aclose()
            raise failure
        assert response is not None

        content_type = _media_type(response.headers.get("Content-Type", ""))
        if not content_type.startswith("audio/"):
            raise await _audio_failure(response, meta, ctx)
        return AsyncAudio(
            response=response,
            sample_rate=_header_int(response.headers.get("X-Sample-Rate", ""), 8000),
            channels=_header_int(response.headers.get("X-Audio-Channels", ""), 1),
            format=response.headers.get("X-Audio-Format", "").strip() or "pcm_s16le",
            content_type=content_type,
            meta=meta,
            ctx=ctx,
        )

    async def say(
        self,
        text: str,
        *options: SpeechOption | None,
        ctx: RequestContext | None = None,
    ) -> AsyncAudio:
        request = SpeechRequest(input=text, response_format="pcm")
        for option in options:
            if option is not None:
                option(request)
        return await self.speech(request, ctx=ctx)


AsyncService = AsyncTTSService

__all__ = [
    "AsyncAudio",
    "AsyncService",
    "AsyncTTSService",
    "DEFAULT_TTS_PATH",
    "PATH_SPEECH",
    "SpeechOption",
    "SpeechRequest",
]
