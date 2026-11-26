from __future__ import annotations

import argparse

from maas_client import Client, RequestContext, tts


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize PCM audio and write a WAV file")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("output")
    parser.add_argument("--text", default="欢迎使用 MaaS")
    parser.add_argument("--model", required=True)
    parser.add_argument("--voice", required=True)
    args = parser.parse_args()

    with Client(api_key=args.api_key, base_url=args.base_url) as client:
        with tts.TTSService(client).say(
            args.text,
            tts.with_model(args.model),
            tts.with_voice(args.voice),
            tts.with_sample(24000),
            ctx=RequestContext.timeout(120),
        ) as audio:
            audio.write_wav_file(args.output)
            print(f"request_id={audio.meta.request_id}")


if __name__ == "__main__":
    main()
