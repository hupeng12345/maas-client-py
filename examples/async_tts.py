from __future__ import annotations

import argparse
import asyncio

from maas_client import AsyncClient, RequestContext, tts


async def run(api_key: str, base_url: str, output: str, text: str, model: str, voice: str) -> None:
    async with AsyncClient(api_key=api_key, base_url=base_url) as client:
        audio = await tts.AsyncTTSService(client).say(
            text,
            tts.with_model(model),
            tts.with_voice(voice),
            tts.with_sample(24000),
            ctx=RequestContext.timeout(120),
        )
        async with audio:
            await audio.write_wav_file(output)
            print(f"request_id={audio.meta.request_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize PCM asynchronously and write a WAV file")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("output")
    parser.add_argument("--text", default="欢迎使用 MaaS")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.api_key, args.base_url, args.output, args.text, args.model, args.voice))


if __name__ == "__main__":
    main()
