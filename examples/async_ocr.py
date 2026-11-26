from __future__ import annotations

import argparse
import asyncio

from maas_client import AsyncClient, RequestContext, ocr


async def run(api_key: str, base_url: str, path: str) -> None:
    async with AsyncClient(api_key=api_key, base_url=base_url) as client:
        result = await ocr.AsyncOCRService(client).parse_document(
            path,
            ocr.enable_toc(True),
            ctx=RequestContext.timeout(300),
        )
        print(result.markdown_content)
        print(f"request_id={result.meta.request_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a document asynchronously with MaaS OCR")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("file")
    args = parser.parse_args()
    asyncio.run(run(args.api_key, args.base_url, args.file))


if __name__ == "__main__":
    main()
