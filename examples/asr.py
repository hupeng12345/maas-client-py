from __future__ import annotations

import argparse

from maas_client import Client, RequestContext, asr


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a WAV or raw PCM file with MaaS ASR")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("file")
    args = parser.parse_args()

    config = asr.ASRConfig(
        request=asr.ASRRequest(enable_nonstream=True, show_utterances=True),
    )
    with Client(api_key=args.api_key, base_url=args.base_url) as client:
        result, error = asr.ASRService(client).transcribe(
            args.file,
            config,
            ctx=RequestContext.timeout(300),
        )
        if error is not None:
            raise error
        print(result.text)
        print(f"request_id={result.meta.request_id}")


if __name__ == "__main__":
    main()
