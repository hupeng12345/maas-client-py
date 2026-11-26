from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.parse import urlencode

import httpx

from ..async_client import AsyncClient
from ..config import ResponseMeta
from ..context import RequestCanceled, RequestContext, RequestDeadlineExceeded, _context_error
from ..envelope import Envelope, decode_envelope_async
from ..errors import ErrorKind, MaaSError, new_local_error, new_protocol_error
from ..files import FileSource
from ..multipart import new_async_multipart
from .async_task import AsyncTask
from .endpoints import (
    PATH_BANK_STATEMENT,
    PATH_FINANCIAL_RECEIPT,
    PATH_IMAGE_BLUR,
    PATH_MEDICAL_RECEIPT,
    PATH_OCR_GENERAL,
    PATH_OCR_PADDLE,
    PATH_OCR_PAGE_DIRECTION,
    PATH_OCR_SEAL,
    PATH_PARSE_ARABIC,
    PATH_PARSE_GENERAL,
    PATH_PARSE_HUNYUAN,
    PATH_PARSE_MULTILANG,
    PATH_RESULT,
    CardKind,
    card_kinds,
)
from .options import OCROption, ParseOption, RecognizeOption, RequestOption, resolve_options
from .service import _decode_page, _decode_toc, _integer, _object_payload, _sequence, _string
from .types import TaskStatus, VisionResult


class AsyncOCRService:
    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    async def parse_document(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.ParseDocument", PATH_PARSE_GENERAL, file, options, ctx=ctx)

    async def parse_document_multilang(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.ParseDocumentMultilang", PATH_PARSE_MULTILANG, file, options, ctx=ctx)

    async def parse_document_arabic(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.ParseDocumentArabic", PATH_PARSE_ARABIC, file, options, ctx=ctx)

    async def parse_document_hunyuan(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.ParseDocumentHunyuan", PATH_PARSE_HUNYUAN, file, options, ctx=ctx)

    async def submit_document(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> AsyncTask:
        resolved = resolve_options(options)
        return await self._submit_vision(
            ctx or RequestContext(), "ocr.SubmitDocument", resolved.path_or(PATH_PARSE_GENERAL), file, resolved.fields
        )

    async def recognize(
        self,
        kind: CardKind | str,
        file: FileSource | None,
        *options: RecognizeOption,
        ctx: RequestContext | None = None,
    ) -> VisionResult:
        card_kind = self._card_kind(kind, "ocr.Recognize")
        resolved = resolve_options(options)
        context = ctx or RequestContext()
        if resolved.async_submit:
            return await self._run_async("ocr.Recognize", card_kind.path(sync=False), file, options, ctx=context)
        return await self._invoke_vision(
            context, "ocr.Recognize", resolved.path_or(card_kind.path()), file, resolved.fields
        )

    async def recognize_async(
        self,
        kind: CardKind | str,
        file: FileSource | None,
        *options: RecognizeOption,
        ctx: RequestContext | None = None,
    ) -> AsyncTask:
        card_kind = self._card_kind(kind, "ocr.RecognizeAsync")
        resolved = resolve_options(options)
        return await self._submit_vision(
            ctx or RequestContext(),
            "ocr.RecognizeAsync",
            resolved.path_or(card_kind.path(sync=False)),
            file,
            resolved.fields,
        )

    async def recognize_bank_statement(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.RecognizeBankStatement", PATH_BANK_STATEMENT, file, options, ctx=ctx)

    async def recognize_medical_receipt(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.RecognizeMedicalReceipt", PATH_MEDICAL_RECEIPT, file, options, ctx=ctx)

    async def recognize_financial_receipt(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.RecognizeFinancialReceipt", PATH_FINANCIAL_RECEIPT, file, options, ctx=ctx)

    async def ocr_text(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.OCRText", PATH_OCR_GENERAL, file, options, ctx=ctx)

    async def ocr_text_paddle(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.OCRTextPaddle", PATH_OCR_PADDLE, file, options, ctx=ctx)

    async def recognize_seal(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.RecognizeSeal", PATH_OCR_SEAL, file, options, ctx=ctx)

    async def detect_page_direction(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.DetectPageDirection", PATH_OCR_PAGE_DIRECTION, file, options, ctx=ctx)

    async def detect_image_blur(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return await self._run_async("ocr.DetectImageBlur", PATH_IMAGE_BLUR, file, options, ctx=ctx)

    @staticmethod
    def _card_kind(kind: CardKind | str, op: str) -> CardKind:
        try:
            card_kind = kind if isinstance(kind, CardKind) else CardKind(kind)
        except ValueError as exc:
            raise new_local_error(op, f"unknown card kind: {kind}", "use one of ocr.card_kinds()") from exc
        if card_kind not in card_kinds():
            raise new_local_error(op, f"unknown card kind: {kind}", "use one of ocr.card_kinds()")
        return card_kind

    async def _run_async(
        self,
        op: str,
        path: str,
        file: FileSource | None,
        options: tuple[RequestOption, ...],
        *,
        ctx: RequestContext | None,
    ) -> VisionResult:
        resolved = resolve_options(options)
        task = await self._submit_vision(ctx or RequestContext(), op, resolved.path_or(path), file, resolved.fields)
        return await task.wait(*resolved.wait, ctx=ctx)

    async def _submit_vision(
        self,
        ctx: RequestContext,
        op: str,
        path: str,
        file: FileSource | None,
        fields: Mapping[str, object],
    ) -> AsyncTask:
        env, meta = await self._call_vision(ctx, op, path, file, fields)
        try:
            data = _object_payload(env.data)
            task_uuid = _string(data, "task_uuid")
            status = _string(data, "task_status")
            task_status = TaskStatus(status)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            error = new_protocol_error(op, "task submission payload is not an object", meta.http_status, env.data)
            error.request_id = meta.request_id
            raise error from exc
        if not task_uuid:
            error = new_protocol_error(op, "task submission did not return task_uuid", meta.http_status, env.data)
            error.request_id = meta.request_id
            raise error
        return AsyncTask(task_uuid, task_status, meta, _service=self, _op=op)

    async def _invoke_vision(
        self,
        ctx: RequestContext,
        op: str,
        path: str,
        file: FileSource | None,
        fields: Mapping[str, object],
    ) -> VisionResult:
        env, meta = await self._call_vision(ctx, op, path, file, fields)
        return self._decode_vision_result(op, env, meta)

    async def _call_vision(
        self,
        ctx: RequestContext,
        op: str,
        path: str,
        file: FileSource | None,
        fields: Mapping[str, object],
    ) -> tuple[Envelope, ResponseMeta]:
        self._client.require_base_url(op)
        if file is None:
            raise new_local_error(op, "no file provided", "pass maas.file(...) to specify the upload")
        maximum = int(self._client.max_upload_size)
        if maximum > 0 and file.size() > maximum:
            raise MaaSError(
                "file exceeds the local size limit",
                kind=ErrorKind.LOCAL,
                hint="raise the client max_upload_size or split the input",
                op=op,
            )

        body = new_async_multipart(file, fields)
        response: httpx.Response | None = None
        try:
            try:
                request = self._client.new_request("POST", path, body, ctx=ctx)
                request.headers["Content-Type"] = body.content_type
                if body.content_length > 0:
                    request.headers["Content-Length"] = str(body.content_length)
                response, meta, error = await self._client.do(request, op, ctx=ctx)
                if error is not None:
                    raise error
                assert response is not None
                try:
                    env = await decode_envelope_async(op, response, meta=meta, ctx=ctx)
                    if not env.is_success():
                        error = env.error(op, response)
                        if not error.request_id:
                            error.request_id = meta.request_id
                        raise error
                    return env, meta
                except MaaSError as exc:
                    if not exc.request_id:
                        exc.request_id = meta.request_id
                    raise
            except (RequestCanceled, RequestDeadlineExceeded) as exc:
                raise _context_error(op, ctx, exc) from exc
        finally:
            await self._close_after_call(response)

    async def _query_vision(self, ctx: RequestContext, op: str, task_uuid: str) -> VisionResult:
        self._client.require_base_url(op)
        path = f"{PATH_RESULT}?{urlencode({'task_uuid': task_uuid})}"
        response: httpx.Response | None = None
        try:
            try:
                request = self._client.new_request("GET", path, ctx=ctx)
                response, meta, error = await self._client.do(request, op, ctx=ctx)
                if error is not None:
                    error.task_uuid = task_uuid
                    raise error
                assert response is not None
                try:
                    env = await decode_envelope_async(op, response, meta=meta, ctx=ctx)
                except MaaSError as exc:
                    exc.task_uuid = task_uuid
                    if not exc.request_id:
                        exc.request_id = meta.request_id
                    raise
                if not env.is_success():
                    error = env.error(op, response)
                    error.task_uuid = task_uuid
                    if not error.request_id:
                        error.request_id = meta.request_id
                    raise error
                try:
                    return self._decode_vision_result(op, env, meta)
                except MaaSError as exc:
                    exc.task_uuid = task_uuid
                    raise
            except (RequestCanceled, RequestDeadlineExceeded) as exc:
                error = _context_error(op, ctx, exc)
                error.task_uuid = task_uuid
                raise error from exc
        finally:
            await self._close_after_call(response)

    def _decode_vision_result(self, op: str, env: Envelope, meta: ResponseMeta) -> VisionResult:
        try:
            data = _object_payload(env.data)
            return VisionResult(
                meta=meta,
                task_uuid=_string(data, "task_uuid"),
                task_status=TaskStatus(_string(data, "task_status")),
                start_time=_integer(data, "start_time"),
                end_time=_integer(data, "end_time"),
                markdown_content=_string(data, "markdown_content"),
                pages=[_decode_page(page) for page in _sequence(data.get("pages"))],
                toc=_decode_toc(data.get("toc")),
                raw=env.data if bool(self._client.keep_raw) else None,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            error = new_protocol_error(op, "result payload is not an object", meta.http_status, env.data)
            error.request_id = meta.request_id
            raise error from exc

    @staticmethod
    async def _close_after_call(response: httpx.Response | None) -> None:
        if response is not None:
            await response.aclose()


AsyncService = AsyncOCRService

__all__ = ["AsyncOCRService", "AsyncService"]
