from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

DEFAULT_ASR_PATH = "/-/volcengine/api/v3/sauc/bigmodel_async"
DEFAULT_CHUNK_SAMPLES = 3200
MAX_CHUNK_SAMPLES = 16000


@dataclass(slots=True)
class Auth:
    bearer_key: str = ""
    api_key: str = ""
    resource_id: str = ""
    app_key: str = ""
    access_key: str = ""


@dataclass(slots=True)
class Audio:
    format: str = ""
    codec: str = ""
    rate: int = 0
    bits: int = 0
    channel: int = 0
    language: str = ""


@dataclass(slots=True)
class Corpus:
    boosting_table_ids: list[str] = field(default_factory=list)
    boosting_table_id: str = ""
    correct_table_id: str = ""
    context: str = ""


@dataclass(slots=True)
class ASRRequest:
    model_name: str = ""
    enable_nonstream: bool | None = None
    enable_itn: bool | None = None
    enable_punc: bool | None = None
    enable_ddc: bool | None = None
    enable_speaker_info: bool | None = None
    show_utterances: bool | None = None
    show_speech_rate: bool | None = None
    show_volume: bool | None = None
    enable_emotion_detection: bool | None = None
    enable_gender_detection: bool | None = None
    enable_age_detection: bool | None = None
    enable_accelerate_text: bool | None = None
    enable_poi_fc: bool | None = None
    enable_music_fc: bool | None = None
    result_type: str = ""
    output_zh_variant: str = ""
    accelerate_score: int | None = None
    vad_segment_duration: int | None = None
    end_window_size: int | None = None
    force_to_speech_time: int | None = None
    sensitive_words_filter: str = ""
    corpus: Corpus | None = None
    parret: dict[str, Any] | None = None


Request = ASRRequest


@dataclass(slots=True)
class Segmentation:
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES


@dataclass(slots=True)
class ASRConfig:
    url: str = ""
    host: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    auth: Auth = field(default_factory=Auth)
    audio: Audio = field(default_factory=Audio)
    request: ASRRequest = field(default_factory=ASRRequest)
    user: dict[str, Any] = field(default_factory=dict)
    seg: Segmentation = field(default_factory=Segmentation)
    ready_timeout: float = 15.0

    def init_payload(self) -> bytes:
        return init_payload(self)

    def chunk_samples(self) -> int:
        return chunk_samples(self)


Config = ASRConfig


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def init_payload(config: ASRConfig) -> bytes:
    audio = asdict(config.audio)
    defaults = {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1}
    for key, value in defaults.items():
        if key in {"rate", "bits", "channel"}:
            if audio[key] <= 0:
                audio[key] = value
        elif not _present(audio[key]):
            audio[key] = value
    audio = {
        key: value.strip() if isinstance(value, str) else value
        for key, value in audio.items()
        if _present(value.strip() if isinstance(value, str) else value)
    }

    request = asdict(config.request)
    request = {key: value for key, value in request.items() if _present(value)}
    request.setdefault("model_name", "bigmodel")
    request.setdefault("enable_nonstream", True)
    corpus = request.get("corpus")
    if isinstance(corpus, dict):
        ids = corpus.pop("boosting_table_ids", [])
        if ids:
            corpus["boosting_table_id"] = ids
        request["corpus"] = {key: value for key, value in corpus.items() if _present(value)}

    payload = {"audio": audio, "request": request}
    if config.user:
        payload["user"] = config.user
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def chunk_samples(config: ASRConfig) -> int:
    if 0 < config.seg.chunk_samples <= MAX_CHUNK_SAMPLES:
        return config.seg.chunk_samples
    return DEFAULT_CHUNK_SAMPLES


__all__ = [
    "DEFAULT_ASR_PATH", "DEFAULT_CHUNK_SAMPLES", "MAX_CHUNK_SAMPLES", "Auth", "Audio", "Corpus", "ASRRequest",
    "Request", "Segmentation", "ASRConfig", "Config", "init_payload", "chunk_samples",
]
