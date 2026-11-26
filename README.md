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
- 传入自定义 `httpx.Client` 时，由调用方负责其超时、重试和生命周期，SDK 不会再叠加内置重试。
- 默认上传上限 100 MiB；设为 `max_upload_size=0` 可关闭。
- 默认不信任环境代理，网关 JSON 响应最大 64 MiB。
- 默认静默；可通过标准库 `logging.Logger` 或 `Hooks` 观测请求、响应、重试和轮询。

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

## 与 Go SDK 的差异

- 使用标准库 `logging`，不引入 Go 的 `logit` 包。
- 配置由 Python 关键字参数和 dataclass 提供，不提供函数式 Option 客户端构造器。
- HTTP 扩展使用 `httpx.Client` / `httpx.BaseTransport`，不暴露 Go 的 `tripper` 中间件链。
- TTS WAV 文件写入会修正 RIFF/data 长度；ASR WAV 解析正确处理奇数长度 chunk 的填充字节。
- `Hooks.on_retry` 会在实际重试时触发。

## 示例

```bash
python examples/ocr.py YOUR_API_KEY https://maas.example.com contract.pdf
python examples/tts.py YOUR_API_KEY https://maas.example.com output.wav
python examples/asr.py YOUR_API_KEY https://maas.example.com audio.wav
```

项目采用 MIT 许可证。
