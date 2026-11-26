from __future__ import annotations

import argparse
import asyncio

from maas_client import AsyncClient, RequestContext, asr


async def run(api_key: str, base_url: str, path: str) -> None:
    config = asr.ASRConfig(
        request=asr.ASRRequest(enable_nonstream=True, show_utterances=True),
    )
    async with AsyncClient(api_key=api_key, base_url=base_url) as client:
        result, error = await asr.AsyncASRService(client).transcribe(
            path,
            config,
            ctx=RequestContext.timeout(300),
        )
        if error is not None:
            raise error
        print(result.text)
        print(f"request_id={result.meta.request_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a WAV or raw PCM file asynchronously with MaaS ASR")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("file")
    args = parser.parse_args()
    asyncio.run(run(args.api_key, args.base_url, args.file))


if __name__ == "__main__":
    main()
