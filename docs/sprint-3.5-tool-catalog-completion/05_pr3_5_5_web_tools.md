# PR 3.5.5 — `WebFetchTool` + `WebSearchTool`

> **角色**：Aether 进入"能查互联网"的能力门槛。
> 没有它，模型只能回答训练数据里的内容，无法查最新文档 / 第三方 API / GitHub 问题。

## 一、目标

1. 实现 `WebFetchTool`：抓取 URL → 转 markdown → 应用模型 prompt 提取答案。
2. 实现 `WebSearchTool`：搜索引擎查询 → 返回 top N 结果（标题 / URL / 摘要）。
3. 共享 SSRF 防护与 URL 白名单基础设施。
4. 网页抓取走 PR 3.5.1 的 spill（大网页 → 磁盘 → preview）。

## 二、为什么要做

### 2.1 真实任务覆盖

模型在 Aether 中需要：
* 查 `react@19.0.0` 的新 API → 必须搜文档
* 看某个 GitHub issue 上的讨论 → 必须 fetch
* 比对两个 npm 包的 README → 必须 fetch + 处理

claude-code 实际数据：会话中 **WebFetch + WebSearch 占比工具调用约 8-12%**（不算 BashTool 的 git 操作）。

### 2.2 与 BashTool curl 的对比

模型可以用 `shell: curl https://example.com` 替代，但：
* 拿到的是 raw HTML（含 script / style / 噪声），耗 token
* 不知道何时设 User-Agent / cookie
* 没有 SSRF 防护（curl 可访问 `127.0.0.1:8080` 内部服务）

专用的 WebFetchTool 在工具内部做 HTML → markdown 转换 + SSRF 防护，输出干净。

## 三、设计

### 3.1 共享基础设施 — `runtime/web_safety.py`

新文件，处理 URL 校验：

```python
"""Web tool safety helpers.

Sprint 3.5 / PR 3.5.5.  Shared by WebFetchTool, WebSearchTool, and (later)
WebBrowserTool.

Provides:
* ``is_url_safe(url)`` — SSRF guard. Rejects loopback, private IPs, file:,
  ftp:, gopher:, etc.
* ``preapproved_hosts`` — fast-path for high-confidence sources
  (docs.python.org, github.com, etc.). Bypasses any future per-domain
  permission prompts.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Preapproved (read-only safe) hosts — no permission prompt needed.
PREAPPROVED_HOSTS: frozenset[str] = frozenset({
    "docs.python.org", "docs.rs", "doc.rust-lang.org",
    "developer.mozilla.org", "github.com", "raw.githubusercontent.com",
    "stackoverflow.com", "pypi.org", "npmjs.com",
    "registry.npmjs.org", "crates.io", "go.dev", "pkg.go.dev",
})


def is_url_safe(url: str) -> tuple[bool, str]:
    """Return (safe, reason). reason is empty when safe."""
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return False, f"invalid URL: {exc}"

    if parsed.scheme not in {"http", "https"}:
        return False, f"unsupported scheme: {parsed.scheme!r} (only http/https allowed)"

    host = parsed.hostname
    if not host:
        return False, "URL has no hostname"

    # 解析 IP 防 DNS rebind
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS lookup failed: {exc}"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast:
            return False, f"refusing to access internal IP: {addr}"

    return True, ""


def is_preapproved(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        return host.lower() in PREAPPROVED_HOSTS
    except ValueError:
        return False
```

### 3.2 `WebFetchTool`

#### 3.2.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "url": {"type": "string", "format": "uri"},
        "prompt": {
            "type": "string",
            "description": (
                "What to extract from the page. The tool will fetch the URL, "
                "convert HTML to markdown, then return the markdown plus your "
                "prompt — your subsequent reasoning runs over the result."
            ),
        },
    },
    "required": ["url", "prompt"],
}
```

#### 3.2.2 算法

```python
class WebFetchTool(ToolExecutor):
    NAME = "web_fetch"
    MAX_RESULT_CHARS = 80_000
    HTTP_TIMEOUT = 30
    MAX_DOWNLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB

    def execute(self, call, context):
        url = call.arguments["url"]
        prompt = call.arguments["prompt"]

        ok, reason = is_url_safe(url)
        if not ok:
            return ToolResult(content=f"refused: {reason}", is_error=True)

        try:
            response = self._http_get(url)
        except Exception as exc:
            return ToolResult(content=f"fetch failed: {exc}", is_error=True)

        # 大文件防御
        body = response.content[: self.MAX_DOWNLOAD_BYTES]
        markdown = self._html_to_markdown(body, content_type=response.headers.get("content-type", ""))

        # 应用 prompt（v1：直接拼接，让模型自己读；v2：考虑用 utility model 总结）
        full_output = (
            f"# Fetched {url}\n"
            f"# Status: {response.status_code}\n"
            f"# Bytes: {len(body)}\n\n"
            f"## Prompt\n{prompt}\n\n"
            f"## Content (markdown)\n{markdown}"
        )
        content = self._maybe_spill(full_output, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)

    def _html_to_markdown(self, html_bytes: bytes, *, content_type: str) -> str:
        # 用 markdownify (BeautifulSoup-based) 做转换
        from markdownify import markdownify
        try:
            text = html_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = html_bytes.decode("latin-1", errors="replace")
        return markdownify(text, heading_style="ATX")
```

#### 3.2.3 HTTP 客户端

* 用 `httpx`（已经是 Aether 依赖；OpenAI 客户端用同库）
* `timeout=30s`，`follow_redirects=True`，`max_redirects=5`
* User-Agent: `"Aether/0.1 (+https://github.com/...)"`

### 3.3 `WebSearchTool`

#### 3.3.1 选型

claude-code 用 Anthropic 的内部 search endpoint。
我们没有这个，需要用第三方 API：

| 选项 | 优势 | 劣势 |
|---|---|---|
| **Brave Search API** | 无 Google captcha；JSON API；免费层 2k/月 | 需要 API key |
| **DuckDuckGo HTML** | 无需 key | 反爬严格，定期失效 |
| **Tavily Search** | 专为 LLM 设计；返回干净文本 | 需要 API key；需付费 |
| **SearXNG self-host** | 完全可控 | 用户需 host |

**v1 选 Brave Search**（最稳定 + 有免费层）。
配置项：`web_search_api_key: str | None = None`（从环境变量 `BRAVE_API_KEY`）。
未配置时工具返回明确错误 + 配置说明，不 crash。

#### 3.3.2 输入 schema

```python
{
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "max_results": {
            "type": "integer",
            "default": 10,
            "minimum": 1,
            "maximum": 20,
        },
    },
    "required": ["query"],
}
```

#### 3.3.3 实现

```python
class WebSearchTool(ToolExecutor):
    NAME = "web_search"
    MAX_RESULT_CHARS = 30_000

    def execute(self, call, context):
        config = context.metadata.get("_engine_config")
        api_key = (
            getattr(config, "web_search_api_key", None)
            or os.environ.get("BRAVE_API_KEY")
        )
        if not api_key:
            return ToolResult(
                content=(
                    "WebSearch unavailable: no API key configured. "
                    "Set BRAVE_API_KEY env var or EngineConfig.web_search_api_key."
                ),
                is_error=True,
            )
        query = call.arguments["query"]
        max_results = int(call.arguments.get("max_results", 10))
        try:
            results = self._brave_search(query, api_key=api_key, max_results=max_results)
        except Exception as exc:
            return ToolResult(content=f"search failed: {exc}", is_error=True)

        # 格式化为 markdown
        body = self._format_results(query, results)
        content = self._maybe_spill(body, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)

    def _format_results(self, query, results):
        lines = [f"# Web search: {query}\n", f"Found {len(results)} results:\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. **{r['title']}**")
            lines.append(f"   {r['url']}")
            lines.append(f"   {r['snippet']}\n")
        return "\n".join(lines)
```

### 3.4 EngineConfig 新字段

```python
# Sprint 3.5 / PR 3.5.5 — web tools.
web_fetch_enabled: bool = True
web_search_enabled: bool = True
web_search_api_key: str | None = None  # Brave Search API key
web_fetch_max_download_bytes: int = 5 * 1024 * 1024  # 5 MiB
web_fetch_timeout_seconds: int = 30
```

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/runtime/web_safety.py` | **新文件** | ~80 |
| `backend/harness/aether/tools/builtins/web_fetch.py` | **新文件** | ~200 |
| `backend/harness/aether/tools/builtins/web_search.py` | **新文件** | ~150 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~6 |
| `backend/harness/aether/config/schema.py` | 修改 | ~30 |
| `backend/harness/aether/tests/test_web_safety.py` | **新文件** | ~150 |
| `backend/harness/aether/tests/test_web_fetch_tool.py` | **新文件** | ~250 |
| `backend/harness/aether/tests/test_web_search_tool.py` | **新文件** | ~200 |
| `pyproject.toml` | 修改 | 加 `markdownify` 依赖 | ~3 |

## 五、测试用例

### 5.1 测试组 A：URL 安全（`test_web_safety.py`）

| ID | URL | 预期 |
|---|---|---|
| **T-A1** | `http://localhost/` | unsafe (loopback) |
| **T-A2** | `http://127.0.0.1/` | unsafe |
| **T-A3** | `http://192.168.1.1/` | unsafe (private) |
| **T-A4** | `http://10.0.0.1/` | unsafe (private) |
| **T-A5** | `http://169.254.169.254/` | unsafe (link-local; AWS metadata) |
| **T-A6** | `file:///etc/passwd` | unsafe (scheme) |
| **T-A7** | `https://github.com/` | safe |
| **T-A8** | `not-a-url` | unsafe (parse) |
| **T-A9** | DNS rebind 域名（mock socket.getaddrinfo 返回 127.0.0.1） | unsafe |

### 5.2 测试组 B：WebFetchTool

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | mock httpx 返回 200 + 简单 HTML | content 含 markdown + Status 200 |
| **T-B2** | 不安全 URL | `is_error=True`；不发起请求 |
| **T-B3** | 大网页 (mock 200KB body) | spill 触发；preview ≤ 80k |
| **T-B4** | timeout（mock httpx.TimeoutException） | `is_error=True` |
| **T-B5** | 5xx 响应 | content 含 status；不报 error（让模型决策） |
| **T-B6** | content-type 非 HTML（如 JSON） | 仍能工作；markdown 转换 graceful |
| **T-B7** | 跟随 5 次 redirect | 成功 |
| **T-B8** | redirect 到不安全 URL | （v1 不防御 redirect 后；记入 v2 follow-up） |

### 5.3 测试组 C：WebSearchTool

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | 无 API key | `is_error=True`；提示配置 |
| **T-C2** | mock Brave 返回 5 条结果 | content 含 5 条；格式正确 |
| **T-C3** | mock Brave 返回 0 结果 | content 含 "0 results" |
| **T-C4** | mock Brave 返回 4xx | `is_error=True` |
| **T-C5** | max_results=1 | 只返回 1 条 |
| **T-C6** | max_results=100（超阈值） | clip 到 20 |

### 5.4 测试组 D：spill 集成

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | WebFetch 返回 200KB markdown | spill；`tier1_spilled_count += 1` |
| **T-D2** | WebSearch 50 条结果 + 长 snippet | spill 触发 |

## 六、验收门

* [ ] 30+ case 全绿
* [ ] 真实跑：`web_fetch https://docs.python.org/3/library/asyncio.html` 提取 task API → 输出干净
* [ ] 真实跑：`web_search "python asyncio task api"` 返回 10 条
* [ ] SSRF 验证：尝试 fetch `http://localhost:8000/` 被拒

## 七、回滚开关

* `web_fetch_enabled=False` / `web_search_enabled=False` 单工具关闭
* 完全 revert：删 3 个新文件 + config + 注册

## 八、实施顺序（建议 2.5 天）

| 步骤 | 时长 |
|---|---|
| 1. `runtime/web_safety.py` + 测试 | 3h |
| 2. `web_fetch.py` + httpx 集成 | 4h |
| 3. WebFetch 测试 | 3h |
| 4. `web_search.py` + Brave 集成 | 3h |
| 5. WebSearch 测试 | 2h |
| 6. config 字段 + smoke 验证 | 2h |
| 7. 回归 + docs | 1h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| Brave API 配额耗尽 | 工具返回明确错误；不 crash |
| markdownify 依赖增加包大小 | 单库 ~200KB，可接受；alternative 是 BeautifulSoup 自写 |
| HTTP redirect 到内部 IP | v1 不防；v2 加 redirect-walk-time SSRF 检查 |
| 长 snippet 触发频繁 spill | 阈值预设 30k，约 60-100 条结果才触发 |
| 网页编码非 utf-8 | latin-1 fallback；markdownify 容错 |
| Cookie / login wall | 工具不处理；返回页面给模型，模型决策 |

## 十、与后续 PR 的接合

* **PR 3.5.10 WebBrowserTool**：复用 `web_safety` + 同样阈值 80k；浏览器适合 JS-heavy 站点
* **未来**：WebFetch 可加 utility model 后处理（"summarize this page wrt prompt"），目前 v1 让主模型自己读
