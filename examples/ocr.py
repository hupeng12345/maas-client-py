from __future__ import annotations

import argparse

from maas_client import Client, RequestContext, ocr


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a document with MaaS OCR")
    parser.add_argument("api_key")
    parser.add_argument("base_url")
    parser.add_argument("file")
    args = parser.parse_args()

    with Client(api_key=args.api_key, base_url=args.base_url) as client:
        result = ocr.OCRService(client).parse_document(
            args.file,
            ocr.enable_toc(True),
            ctx=RequestContext.timeout(300),
        )
        print(result.markdown_content)
        print(f"request_id={result.meta.request_id}")


if __name__ == "__main__":
    main()
