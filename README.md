# maas-client-py

MaaS 平台的 Python SDK，支持 OCR、TTS 和实时 ASR。要求 **Python 3.12+**，API 与 `maas-client-go` 的平台行为保持一致。

## 安装

```bash
pip install maas-python-sdk
```

发布包名为 `maas-python-sdk`，Python 导入名为 `maas_client`：

```python
from maas_client import Client, ocr, tts, asr
```

本地开发：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
ruff check src tests
mypy src/maas_client
python -m build
```

## 快速开始

```python
from maas_client import Client, RequestContext, ocr

with Client(
    api_key="your-api-key",
    base_url="https://maas.example.com",  # 替换为实际平台地址
) as client:
    result = ocr.OCRService(client).parse_document(
        "contract.pdf",
        ocr.enable_toc(True),
        ctx=RequestContext.timeout(120, request_id="order-123"),
    )
    print(result.markdown_content)
    print(result.meta.request_id)
```

`api_key` 和 `base_url` 必须显式传入。SDK 不读取 `MAAS_API_KEY`、`MAAS_BASE_URL` 或其他环境变量；应用应自行管理配置和密钥。

## 原生异步 API

同步与异步 API 并存。异步版本直接使用 `asyncio`、`httpx.AsyncClient` 和 `websockets`，不是把同步方法放进 `asyncio.to_thread`；只有文件打开、读取和音频落盘等阻塞 I/O 会在线程中执行。

```python
import asyncio

from maas_client import AsyncClient, RequestContext, asr, ocr, tts


async def main() -> None:
    async with AsyncClient(api_key="your-api-key", base_url="https://maas.example.com") as client:
        result = await ocr.AsyncOCRService(client).parse_document(
            "contract.pdf",
            ocr.enable_toc(True),
            ctx=RequestContext.timeout(120),
        )
        print(result.markdown_content)

        audio = await tts.AsyncTTSService(client).say(
            "欢迎使用 MaaS",
            tts.with_model("your-model"),
            tts.with_voice("your-voice-id"),
            tts.with_sample(24000),
        )
        async with audio:
            await audio.write_wav_file("output.wav")

        async with await asr.AsyncASRService(client).start(asr.ASRConfig()) as session:
            pcm_chunk = b"\x00\x00" * 1600
            await session.write(pcm_chunk)
            transcript, error = await session.finish(ctx=RequestContext.timeout(30))
            if error is not None:
                raise error
            print(transcript.text)


asyncio.run(main())
```

- `AsyncClient` 配合 `async with` 使用，关闭方法为 `await client.close()`。
- 异步 OCR 服务为 `ocr.AsyncOCRService`，异步任务提供 `await task.get()` 和 `await task.wait()`。
- 异步 TTS 服务为 `tts.AsyncTTSService`，`await service.speech()` / `await service.say()` 返回 `AsyncAudio`。
- `AsyncAudio` 支持 `await read()`、`await to_bytes()`、`async for`、`await write_wav()` 和 `await write_wav_file()`。
- 异步 ASR 服务为 `asr.AsyncASRService`，实时连接返回 `AsyncSession`；写入、结束和关闭均使用 `await`。
- `AsyncHooks` 的回调是可 await 的函数；轮询与重试等待不会阻塞事件循环。
- 调用方取消当前 `asyncio.Task` 时，`asyncio.CancelledError` 会继续向外传播；`RequestContext.cancel()` 或截止时间触发时，统一转换为 `MaaSError`。

## OCR

文档解析会自动提交任务并轮询结果；证件识别默认走同步接口，也可显式使用异步接口。

```python
from maas_client import Client, ocr

with Client(api_key="your-api-key", base_url="https://maas.example.com") as client:
    service = ocr.OCRService(client)

    result = service.parse_document("contract.pdf", ocr.enable_toc(True))
    for page in result.pages:
        print(page.index, page.markdown)

    card = service.recognize(ocr.CardIDCard, "id.png")
    print(card.markdown_content)

    task = service.submit_document("large.pdf", ocr.with_wait_timeout(10))
    try:
        result = task.wait()
    except ocr.TaskError as error:
        print(error.task_uuid, error.result)
```

异步任务失败时抛出 `TaskError`，可通过 `error.result` 获取平台返回的最后结果，通过 `error.task_uuid` 保留任务用于后续查询。未知的非终态任务状态会继续轮询。

## TTS

TTS 返回二进制音频流。任何 `audio/*` Content-Type 都视为成功；非音频响应必须符合统一 JSON 网关协议。

```python
from maas_client import Client, tts

with Client(api_key="your-api-key", base_url="https://maas.example.com") as client:
    with tts.TTSService(client).say(
        "欢迎使用 MaaS",
        tts.with_model("your-model"),
        tts.with_voice("your-voice-id"),
        tts.with_sample(24000),
        tts.with_stream(True),
    ) as audio:
        audio.write_wav_file("output.wav")
```

- `Audio.read()` 读取剩余字节，`Audio.to_bytes()` 一次读取全部。
- `Audio` 可直接迭代，用于边接收边写入。
- `write_wav_file()` 会流式写入并回填正确的 RIFF/data 长度。
- 必须关闭音频响应；`with Audio(...)` 或 `audio.close()` 都会释放连接。
- 当前平台只支持 `pcm` 响应格式。

## ASR

### 文件转写

```python
from maas_client import Client, RequestContext, asr

config = asr.ASRConfig(
    request=asr.ASRRequest(enable_nonstream=True, show_utterances=True),
)

with Client(api_key="your-api-key", base_url="https://maas.example.com") as client:
    result, error = asr.ASRService(client).transcribe(
        "audio.wav", config, ctx=RequestContext.timeout(120)
    )
    if error is not None:
        raise error
    print(result.text)
```

`transcribe()` 接受 WAV 或原始 PCM。WAV 会跳过 RIFF 头，默认发送 16 kHz、16-bit、单声道 PCM；其他音频需先由应用转换。

### 实时会话

```python
from maas_client import Client, RequestContext, asr

with Client(api_key="your-api-key", base_url="https://maas.example.com") as client:
    with asr.ASRService(client).start(asr.ASRConfig()) as session:
        session.write(pcm_chunk_1)
        session.write(pcm_chunk_2)
        result, error = session.finish(ctx=RequestContext.timeout(30))
        if error is not None:
            raise error
        print(result.text)
```

实时会话必须及时消费 `session.events()`，否则事件队列满时会影响读取线程。示例：

```python
for event in session.events():
    if event.kind is asr.EventKind.TERMINAL:
        print(event.text)
```

认证优先级为 `Auth.resource_id`、`Auth.app_key + Auth.access_key`、Bearer Token。ASR 需要 UUID 形式的 `X-Api-Request-Id`；传入非 UUID 请求 ID 时，SDK 会为协议头生成 UUID，同时保留原 ID 到 `X-Request-ID`。

## 错误处理

所有 SDK 失败统一为 `MaaSError`，包含错误分类、业务码、提示、HTTP 状态、请求 ID、任务 UUID、原始响应和底层异常。

```python
from maas_client import MaaSError, is_auth, is_rate_limited, is_task_failed, retry_after

try:
    result = service.parse_document("contract.pdf")
except MaaSError as error:
    if is_auth(error):
        print("API key 无效或过期")
    elif is_rate_limited(error):
        print(f"请等待 {retry_after(error):.1f} 秒")
    elif is_task_failed(error):
        print("任务失败", error.task_uuid)
    else:
        print(error, "request_id=", error.request_id)
```

错误分类为 `config`、`local`、`transport`、`protocol`、`api`、`task` 和 `unknown`。

## 请求 ID、超时与取消

```python
from maas_client import RequestContext, request_id_scope

ctx = RequestContext.timeout(30, request_id="order-123")
result = service.parse_document("contract.pdf", ctx=ctx)

ctx = RequestContext.timeout(30)
ctx.cancel()  # 停止尚未开始的请求或 OCR 轮询；平台已受理任务仍会继续处理

with request_id_scope("batch-2026-11-26"):
    result = service.parse_document("contract.pdf")
```

`RequestContext.child()` 可创建更短期限的子上下文，并与父上下文共享取消状态。

## 配置行为

- 普通请求默认超时 60 秒；TTS 流式响应没有整体读取超时，取消或关闭由调用方控制。
- 仅 GET、HEAD、PUT、DELETE、OPTIONS、TRACE 等幂等方法自动重试；OCR 提交和 TTS 合成不会自动重试。
- 默认最多尝试 3 次，指数退避并带抖动，遵守 `Retry-After`。
- 传入自定义 `httpx.Client` 或 `httpx.AsyncClient` 时，由调用方负责其超时、重试和生命周期，SDK 不会再叠加内置重试。
- 默认上传上限 100 MiB；设为 `max_upload_size=0` 可关闭。
- 默认不信任环境代理，网关 JSON 响应最大 64 MiB。
- 默认静默；可通过标准库 `logging.Logger`、同步 `Hooks` 或异步 `AsyncHooks` 观测请求、响应、重试和轮询。

## 自定义 HTTP 客户端

```python
import httpx
from maas_client import Client

with httpx.Client(timeout=httpx.Timeout(30), trust_env=False) as http:
    with Client(
        api_key="your-api-key",
        base_url="https://maas.example.com",
        http_client=http,
    ) as client:
        ...
```

也可通过 `transport=httpx.BaseTransport` 扩展内置客户端，或传入独立的 `stream_http_client` 给 TTS 使用。

异步客户端支持 `httpx.AsyncClient`、`httpx.AsyncBaseTransport` 和独立的 `stream_http_client`：

```python
import asyncio

import httpx

from maas_client import AsyncClient


async def main() -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30), trust_env=False) as http:
        async with AsyncClient(
            api_key="your-api-key",
            base_url="https://maas.example.com",
            http_client=http,
        ) as client:
            ...


asyncio.run(main())
```

## 与 Go SDK 的差异

- 使用标准库 `logging`，不引入 Go 的 `logit` 包。
- 配置由 Python 关键字参数和 dataclass 提供，不提供函数式 Option 客户端构造器。
- HTTP 扩展使用 `httpx.Client` / `httpx.BaseTransport`，不暴露 Go 的 `tripper` 中间件链。
- 额外提供原生异步客户端和异步能力服务，与同步 API 共享请求模型、响应模型、错误类型和平台协议。
- TTS WAV 文件写入会修正 RIFF/data 长度；ASR WAV 解析正确处理奇数长度 chunk 的填充字节。
- `Hooks.on_retry` 会在实际重试时触发。

## 示例

```bash
python examples/ocr.py YOUR_API_KEY https://maas.example.com contract.pdf
python examples/tts.py YOUR_API_KEY https://maas.example.com output.wav
python examples/asr.py YOUR_API_KEY https://maas.example.com audio.wav
python examples/async_ocr.py YOUR_API_KEY https://maas.example.com contract.pdf
python examples/async_tts.py YOUR_API_KEY https://maas.example.com output.wav --model YOUR_MODEL --voice YOUR_VOICE
python examples/async_asr.py YOUR_API_KEY https://maas.example.com audio.wav
```

项目采用 MIT 许可证。
