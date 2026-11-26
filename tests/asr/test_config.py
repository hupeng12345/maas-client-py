import json

import pytest

from maas_client.asr import ASRConfig, Audio, Auth, Config, Corpus, Request, Segmentation
from maas_client.asr.service import ASRService
from maas_client.client import Client


def client(base_url="https://example.com", api_key="sk_test"):
    return Client(api_key=api_key, base_url=base_url)


def test_payload_defaults_and_explicit_false():
    payload = json.loads(Config().init_payload())
    assert payload["audio"] == {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1}
    assert payload["request"] == {"model_name": "bigmodel", "enable_nonstream": True}
    explicit = json.loads(Config(request=Request(enable_itn=False, enable_punc=False)).init_payload())
    assert explicit["request"]["enable_itn"] is False
    assert explicit["request"]["enable_punc"] is False


def test_payload_overrides_user_corpus_and_parret():
    config = ASRConfig(
        audio=Audio(format="wav", rate=8000, channel=2, language="ja-JP"),
        request=Request(
            model_name="custom",
            end_window_size=500,
            corpus=Corpus(boosting_table_ids=["t1", "t2"], correct_table_id="c1", context='{"x":1}'),
            parret={"transcript_representations": ["ja_normal"]},
        ),
        user={"uid": "u1"},
    )
    payload = json.loads(config.init_payload())
    assert payload["audio"] == {"format": "wav", "codec": "raw", "rate": 8000, "bits": 16, "channel": 2,
                                "language": "ja-JP"}
    assert payload["request"]["model_name"] == "custom"
    assert payload["request"]["end_window_size"] == 500
    assert payload["request"]["enable_nonstream"] is True
    assert payload["request"]["corpus"] == {
        "boosting_table_id": ["t1", "t2"], "correct_table_id": "c1", "context": '{"x":1}'
    }
    assert payload["request"]["parret"] == {"transcript_representations": ["ja_normal"]}
    assert payload["user"] == {"uid": "u1"}


def test_singular_corpus_id():
    request = Request(corpus=Corpus(boosting_table_id="table"))
    assert json.loads(Config(request=request).init_payload())["request"]["corpus"] == {"boosting_table_id": "table"}


@pytest.mark.parametrize(("value", "expected"), [(0, 3200), (-1, 3200), (16000, 16000), (16001, 3200), (800, 800)])
def test_chunk_samples(value, expected):
    assert Config(seg=Segmentation(chunk_samples=value)).chunk_samples() == expected


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (Config(), "wss://example.com/-/volcengine/api/v3/sauc/bigmodel_async"),
        (Config(host="http://other.test"), "ws://other.test/-/volcengine/api/v3/sauc/bigmodel_async"),
        (Config(host="https://other.test/gateway/?q=1#frag"),
         "wss://other.test/gateway/-/volcengine/api/v3/sauc/bigmodel_async"),
        (Config(host="https://other.test", url="https://direct.test/x?q=1#frag"), "wss://direct.test/x?q=1#frag"),
    ],
)
def test_endpoints(config, expected):
    assert ASRService(client()).endpoint_url(config) == expected


def test_bad_endpoint():
    with pytest.raises(Exception, match="unsupported ASR URL scheme"):
        ASRService(client()).endpoint_url(Config(host="ftp://other.test"))


def test_auth_precedence_and_sdk_headers():
    service = ASRService(client())
    config = Config(
        auth=Auth(resource_id="res", api_key="direct", app_key="app", access_key="access", bearer_key="bearer"),
        headers={
            "authorization": "bad", "x-api-request-id": "bad", "User-Agent": "bad", "X-Custom": "yes", "Empty": ""
        },
    )
    headers = service.handshake_headers(config, "uuid", "generic")
    assert headers["X-Api-Key"] == "direct"
    assert headers["X-Api-Resource-Id"] == "res"
    assert "Authorization" not in headers
    assert headers["X-Custom"] == "yes"
    assert "Empty" not in headers
    assert headers["User-Agent"] == "maas-client-py/0.2.0"
    assert headers["X-Request-ID"] == "generic"
    assert headers["X-Api-Request-Id"] == "uuid"
