from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode

import httpx

from ..client import Client
from ..config import ResponseMeta
from ..context import RequestContext
from ..envelope import Envelope, decode_envelope
from ..errors import ErrorKind, MaaSError, new_local_error, new_protocol_error
from ..files import FileSource
from ..multipart import new_multipart
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
from .task import Task
from .types import TOC, IEResult, LayoutBlock, Page, TaskStatus, TOCNode, VisionResult


def _string(data: Mapping[str, Any], key: str, default: str = "") -> str:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _integer(data: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = data.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _boolean(data: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("value must be an object")
    return value


def _sequence(value: object) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("value must be an array")
    return value


def _bbox(value: object) -> list[float] | None:
    if value is None:
        return None
    values = _sequence(value)
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        raise ValueError("bbox must contain numbers")
    return [float(item) for item in values]


def _toc_value(data: Mapping[str, Any], *names: str) -> Any:
    normalized = {key.casefold(): value for key, value in data.items()}
    for name in names:
        key = name.casefold()
        if key in normalized:
            return normalized[key]
    return None


def _toc_string(data: Mapping[str, Any], *names: str) -> str:
    value = _toc_value(data, *names)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("TOC string field has the wrong type")
    return value


def _decode_toc(value: object) -> TOC | None:
    if value is None:
        return None
    data = _mapping(value)
    return TOC(
        doc_name=_toc_string(data, "doc_name", "docname"),
        structure=[_decode_toc_node(_mapping(node)) for node in _sequence(_toc_value(data, "structure"))],
    )


def _decode_toc_node(data: Mapping[str, Any]) -> TOCNode:
    # Accept both Go field names (NodeID, StartIndex) and snake_case wire names.
    normalized = {key.casefold(): value for key, value in data.items()}

    def string(*names: str, default: str = "") -> str:
        value = _toc_value(data, *names)
        if value is None:
            return default
        if not isinstance(value, str):
            raise ValueError("TOC string field has the wrong type")
        return value

    def integer(*names: str) -> int:
        value = _toc_value(data, *names)
        if value is None:
            return 0
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("TOC index field has the wrong type")
        return int(value)

    return TOCNode(
        title=string("title"),
        node_id=string("node_id", "nodeid"),
        text=string("text"),
        start_index=integer("start_index", "startindex"),
        end_index=integer("end_index", "endindex"),
        children=[_decode_toc_node(_mapping(node)) for node in _sequence(normalized.get("children"))],
    )


def _decode_page(value: object) -> Page:
    data = _mapping(value)
    return Page(
        index=_integer(data, "page_index"),
        markdown=_string(data, "page_markdown"),
        layout=[_decode_layout(block) for block in _sequence(data.get("layout_result"))],
        ie_result=[_decode_ie(item) for item in _sequence(data.get("ie_result"))],
    )


def _decode_layout(value: object) -> LayoutBlock:
    data = _mapping(value)
    return LayoutBlock(
        type=_string(data, "type"),
        page_idx=_integer(data, "page_idx"),
        bbox=_bbox(data.get("bbox")),
        content=_string(data, "content"),
        metadata=_mapping(data.get("metadata")),
    )


def _decode_ie(value: object) -> IEResult:
    data = _mapping(value)
    return IEResult(
        recognized=_boolean(data, "recognized"),
        category=_string(data, "category"),
        bbox=_bbox(data.get("bbox")),
        angle=_integer(data, "angle"),
        fields=_mapping(data.get("fields")),
    )


def _object_payload(data: bytes | None) -> dict[str, Any]:
    if data is None:
        return {}
    value = json.loads(data)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("payload is not an object")
    return value


class OCRService:
    def __init__(self, client: Client) -> None:
        self._client = client

    def parse_document(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.ParseDocument", PATH_PARSE_GENERAL, file, options, ctx=ctx)

    def parse_document_multilang(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.ParseDocumentMultilang", PATH_PARSE_MULTILANG, file, options, ctx=ctx)

    def parse_document_arabic(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.ParseDocumentArabic", PATH_PARSE_ARABIC, file, options, ctx=ctx)

    def parse_document_hunyuan(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.ParseDocumentHunyuan", PATH_PARSE_HUNYUAN, file, options, ctx=ctx)

    def submit_document(
        self, file: FileSource | None, *options: ParseOption, ctx: RequestContext | None = None
    ) -> Task:
        resolved = resolve_options(options)
        return self._submit_vision(
            ctx or RequestContext(), "ocr.SubmitDocument", resolved.path_or(PATH_PARSE_GENERAL), file, resolved.fields
        )

    def recognize(
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
            return self._run_async("ocr.Recognize", card_kind.path(sync=False), file, options, ctx=context)
        return self._invoke_vision(context, "ocr.Recognize", resolved.path_or(card_kind.path()), file, resolved.fields)

    def recognize_async(
        self,
        kind: CardKind | str,
        file: FileSource | None,
        *options: RecognizeOption,
        ctx: RequestContext | None = None,
    ) -> Task:
        card_kind = self._card_kind(kind, "ocr.RecognizeAsync")
        resolved = resolve_options(options)
        return self._submit_vision(
            ctx or RequestContext(),
            "ocr.RecognizeAsync",
            resolved.path_or(card_kind.path(sync=False)),
            file,
            resolved.fields,
        )

    def recognize_bank_statement(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.RecognizeBankStatement", PATH_BANK_STATEMENT, file, options, ctx=ctx)

    def recognize_medical_receipt(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.RecognizeMedicalReceipt", PATH_MEDICAL_RECEIPT, file, options, ctx=ctx)

    def recognize_financial_receipt(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.RecognizeFinancialReceipt", PATH_FINANCIAL_RECEIPT, file, options, ctx=ctx)

    def ocr_text(self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None) -> VisionResult:
        return self._run_async("ocr.OCRText", PATH_OCR_GENERAL, file, options, ctx=ctx)

    def ocr_text_paddle(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.OCRTextPaddle", PATH_OCR_PADDLE, file, options, ctx=ctx)

    def recognize_seal(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.RecognizeSeal", PATH_OCR_SEAL, file, options, ctx=ctx)

    def detect_page_direction(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.DetectPageDirection", PATH_OCR_PAGE_DIRECTION, file, options, ctx=ctx)

    def detect_image_blur(
        self, file: FileSource | None, *options: OCROption, ctx: RequestContext | None = None
    ) -> VisionResult:
        return self._run_async("ocr.DetectImageBlur", PATH_IMAGE_BLUR, file, options, ctx=ctx)

    @staticmethod
    def _card_kind(kind: CardKind | str, op: str) -> CardKind:
        try:
            card_kind = kind if isinstance(kind, CardKind) else CardKind(kind)
        except ValueError as exc:
            raise new_local_error(op, f"unknown card kind: {kind}", "use one of ocr.card_kinds()") from exc
        if card_kind not in card_kinds():
            raise new_local_error(op, f"unknown card kind: {kind}", "use one of ocr.card_kinds()")
        return card_kind

    def _run_async(
        self,
        op: str,
        path: str,
        file: FileSource | None,
        options: tuple[RequestOption, ...],
        *,
        ctx: RequestContext | None,
    ) -> VisionResult:
        resolved = resolve_options(options)
        task = self._submit_vision(ctx or RequestContext(), op, resolved.path_or(path), file, resolved.fields)
        return task.wait(*resolved.wait, ctx=ctx)

    def _submit_vision(
        self,
        ctx: RequestContext,
        op: str,
        path: str,
        file: FileSource | None,
        fields: Mapping[str, object],
    ) -> Task:
        env, meta = self._call_vision(ctx, op, path, file, fields)
        try:
            data = _object_payload(env.data)
            task_uuid = _string(data, "task_uuid")
            status = TaskStatus(_string(data, "task_status"))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            error = new_protocol_error(op, "task submission payload is not an object", meta.http_status, env.data)
            error.request_id = meta.request_id
            raise error from exc
        if not task_uuid:
            error = new_protocol_error(op, "task submission did not return task_uuid", meta.http_status, env.data)
            error.request_id = meta.request_id
            raise error
        return Task(task_uuid, status, meta, _service=self, _op=op)

    def _invoke_vision(
        self,
        ctx: RequestContext,
        op: str,
        path: str,
        file: FileSource | None,
        fields: Mapping[str, object],
    ) -> VisionResult:
        env, meta = self._call_vision(ctx, op, path, file, fields)
        return self._decode_vision_result(op, env, meta)

    def _call_vision(
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

        body = new_multipart(file, fields)
        response: httpx.Response | None = None
        try:
            request = self._client.new_request("POST", path, body, ctx=ctx)
            request.headers["Content-Type"] = body.content_type
            if body.content_length > 0:
                request.headers["Content-Length"] = str(body.content_length)
            response, meta, error = self._client.do(request, op, ctx=ctx)
            if error is not None:
                raise error
            assert response is not None
            try:
                env = decode_envelope(op, response, meta=meta)
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
        finally:
            self._close_after_call(response)

    def _query_vision(self, ctx: RequestContext, op: str, task_uuid: str) -> VisionResult:
        self._client.require_base_url(op)
        path = f"{PATH_RESULT}?{urlencode({'task_uuid': task_uuid})}"
        request = self._client.new_request("GET", path, ctx=ctx)
        response, meta, error = self._client.do(request, op, ctx=ctx)
        if error is not None:
            error.task_uuid = task_uuid
            raise error
        assert response is not None
        try:
            try:
                env = decode_envelope(op, response, meta=meta)
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
        finally:
            response.close()

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

    def _close_after_call(self, response: httpx.Response | None) -> None:
        if response is not None:
            response.close()


Service = OCRService


__all__ = ["OCRService", "Service"]
