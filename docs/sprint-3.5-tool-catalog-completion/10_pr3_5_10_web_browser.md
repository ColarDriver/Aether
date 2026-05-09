# PR 3.5.10 — `WebBrowserTool` + Playwright 集成

> **角色**：处理 JavaScript-rendered 页面，是 WebFetchTool 的"重型版本"。
> WebFetch 拿到 raw HTML（pre-render），WebBrowser 等 JS 跑完拿到 final DOM。

## 一、目标

1. 实现 `WebBrowserTool`，用 Playwright headless Chromium 渲染页面。
2. 支持 4 种操作：`fetch`（拿渲染后 HTML→markdown）、`screenshot`（截图）、
   `click`（点击 + 取后续 DOM）、`type`（填表单）。
3. 复用 PR 3.5.5 的 SSRF 防护 + 80k 阈值 + spill 路径。
4. 浏览器进程**按需启动**（cold start ~2s），idle 30s 后自动 shutdown 节约资源。

## 二、为什么要做

### 2.1 WebFetch 的局限

| 场景 | WebFetch | WebBrowser |
|---|---|---|
| 静态文档（docs.python.org） | ✓ | 不必要 |
| Google 搜索结果（JS-rendered） | 拿到 noscript fallback | 拿到完整结果 |
| GitHub 文件（JS lazy-load） | 拿到部分 | 拿到完整 |
| Notion 页面 | 几乎空 | 完整 |
| 现代 SPA（React/Vue 等） | 拿到 `<div id="root"></div>` | 拿到渲染后 DOM |

### 2.2 取舍：复杂度 vs. 价值

* 引入 Playwright 是 ~250MB 依赖（含 Chromium binary）
* 性能：cold start 1-2s，每页 1-3s
* 但**真实工作流**很多场景需要 — 用户经常需要查 SPA 里的内容

**v1 决策**：实现，但默认禁用 + 显式 opt-in（`web_browser_enabled=False`）。
用户主动开启表示愿意付出依赖代价。

## 三、设计

### 3.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["fetch", "screenshot", "click", "type"],
        },
        "url": {"type": "string", "format": "uri"},
        "selector": {"type": "string", "description": "CSS selector for click/type"},
        "text": {"type": "string", "description": "Text for type operation"},
        "wait_for": {
            "type": "string",
            "enum": ["load", "domcontentloaded", "networkidle"],
            "default": "domcontentloaded",
        },
        "wait_seconds": {"type": "number", "default": 0, "description": "Extra sleep after page load"},
    },
    "required": ["operation", "url"],
}
```

### 3.2 BrowserManager（singleton with idle shutdown）

```python
# runtime/browser_manager.py - 新文件
class BrowserManager:
    def __init__(self, *, idle_timeout_seconds: int = 30):
        self.idle_timeout = idle_timeout_seconds
        self._playwright = None
        self._browser = None
        self._last_use = 0.0
        self._lock = threading.Lock()
        self._idle_thread: Optional[threading.Thread] = None

    def get_browser(self):
        from playwright.sync_api import sync_playwright
        with self._lock:
            if self._browser is None or not self._browser.is_connected():
                self._playwright = sync_playwright().start()
                self._browser = self._playwright.chromium.launch(headless=True)
                self._start_idle_watcher()
            self._last_use = time.monotonic()
            return self._browser

    def _start_idle_watcher(self):
        def watch():
            while True:
                time.sleep(self.idle_timeout)
                with self._lock:
                    if self._browser is None:
                        return
                    if time.monotonic() - self._last_use > self.idle_timeout:
                        self._shutdown_unlocked()
                        return
        self._idle_thread = threading.Thread(target=watch, daemon=True)
        self._idle_thread.start()

    def shutdown(self):
        with self._lock:
            self._shutdown_unlocked()

    def _shutdown_unlocked(self):
        if self._browser:
            try: self._browser.close()
            except Exception: pass
            self._browser = None
        if self._playwright:
            try: self._playwright.stop()
            except Exception: pass
            self._playwright = None
```

### 3.3 WebBrowserTool

```python
class WebBrowserTool(ToolExecutor):
    NAME = "web_browser"
    MAX_RESULT_CHARS = 80_000
    NAVIGATION_TIMEOUT = 30  # seconds

    def __init__(self, browser_manager: BrowserManager):
        self.manager = browser_manager

    def execute(self, call, context):
        config = context.metadata.get("_engine_config")
        if not getattr(config, "web_browser_enabled", False):
            return ToolResult(
                content=(
                    "WebBrowserTool is disabled. Enable via "
                    "EngineConfig.web_browser_enabled=True (requires Playwright "
                    "and Chromium installed)."
                ),
                is_error=True,
            )

        op = call.arguments.get("operation", "")
        url = call.arguments.get("url", "")

        from aether.runtime.web_safety import is_url_safe
        ok, reason = is_url_safe(url)
        if not ok:
            return ToolResult(content=f"refused: {reason}", is_error=True)

        try:
            from playwright.sync_api import TimeoutError as PWTimeoutError
        except ImportError:
            return ToolResult(
                content=(
                    "Playwright not installed. Install via "
                    "`pip install playwright && playwright install chromium`."
                ),
                is_error=True,
            )

        try:
            browser = self.manager.get_browser()
            page = browser.new_page()
            page.set_default_navigation_timeout(self.NAVIGATION_TIMEOUT * 1000)

            try:
                page.goto(url, wait_until=call.arguments.get("wait_for", "domcontentloaded"))
                wait_extra = float(call.arguments.get("wait_seconds", 0))
                if wait_extra > 0:
                    page.wait_for_timeout(wait_extra * 1000)

                if op == "fetch":
                    body = self._do_fetch(page, url)
                elif op == "click":
                    body = self._do_click(page, url, selector=call.arguments["selector"])
                elif op == "type":
                    body = self._do_type(page, url,
                                          selector=call.arguments["selector"],
                                          text=call.arguments["text"])
                elif op == "screenshot":
                    body = self._do_screenshot(page, url, context=context, call=call)
                else:
                    return ToolResult(content=f"unknown operation: {op}", is_error=True)
            finally:
                page.close()
        except PWTimeoutError:
            return ToolResult(content=f"navigation timeout: {url}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"browser error: {exc}", is_error=True)

        content = self._maybe_spill(body, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)

    def _do_fetch(self, page, url) -> str:
        from markdownify import markdownify
        html = page.content()
        markdown = markdownify(html, heading_style="ATX")
        return f"# Browser-fetched {url}\n\n## Content (markdown)\n{markdown}"

    def _do_click(self, page, url, *, selector) -> str:
        page.click(selector)
        page.wait_for_load_state("domcontentloaded")
        from markdownify import markdownify
        html = page.content()
        return (
            f"# Browser clicked '{selector}' on {url}\n"
            f"## Resulting page (markdown)\n"
            f"{markdownify(html, heading_style='ATX')}"
        )

    def _do_type(self, page, url, *, selector, text) -> str:
        page.fill(selector, text)
        return f"# Filled '{selector}' on {url} with {len(text)} characters"

    def _do_screenshot(self, page, url, *, context, call) -> str:
        from aether.runtime.tool_result_storage import resolve_spill_dir
        config = context.metadata.get("_engine_config")
        spill_dir = resolve_spill_dir(
            session_id=context.session_id,
            config_dir=getattr(config, "tool_result_spill_dir", None),
        )
        out_path = spill_dir / f"{call.id}.png"
        page.screenshot(path=str(out_path), full_page=True)
        return (
            f"# Screenshot saved\n"
            f"- url: {url}\n"
            f"- path: {out_path}\n"
            f"- size: {out_path.stat().st_size} bytes\n"
            f"(Use the standard image-handling workflow to view this screenshot.)"
        )
```

### 3.4 EngineConfig 新字段

```python
# Sprint 3.5 / PR 3.5.10
web_browser_enabled: bool = False  # Default OFF — heavy dependency
web_browser_idle_timeout_seconds: int = 30
web_browser_navigation_timeout_seconds: int = 30
```

### 3.5 依赖管理

* `pyproject.toml`：把 `playwright` 放到 `[project.optional-dependencies]` 的 `web-browser` extra
  ```toml
  [project.optional-dependencies]
  web-browser = ["playwright>=1.40.0"]
  ```
* 工具检测 `import playwright` 失败时给出清晰安装提示
* CI：Playwright extra **不**默认装；专门一个 job 装 + 跑 web_browser tests

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/runtime/browser_manager.py` | **新文件** | ~150 |
| `backend/harness/aether/tools/builtins/web_browser.py` | **新文件** | ~250 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~5 |
| `backend/harness/aether/cli/main.py` | 修改 | atexit 关浏览器 | ~5 |
| `backend/harness/aether/config/schema.py` | 修改 | ~15 |
| `backend/harness/aether/pyproject.toml` | 修改 | optional-dep | ~5 |
| `backend/harness/aether/tests/test_browser_manager.py` | **新文件**（mock） | ~150 |
| `backend/harness/aether/tests/test_web_browser_tool.py` | **新文件**（mock + 真实） | ~250 |

## 五、测试用例

### 5.1 测试组 A：BrowserManager（mock playwright）

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 第一次 get_browser | 启动 chromium |
| **T-A2** | 第二次 get_browser | 复用 browser |
| **T-A3** | idle 后自动 shutdown | timer 到期清理 |
| **T-A4** | shutdown 后再 get | 重新启动 |
| **T-A5** | shutdown 异常 graceful | 不 raise |

### 5.2 测试组 B：WebBrowserTool（mock）

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | web_browser_enabled=False | `is_error=True`；提示开启 |
| **T-B2** | playwright 未安装 | `is_error=True`；含安装命令 |
| **T-B3** | unsafe URL | `is_error=True`；不启动浏览器 |
| **T-B4** | fetch 操作 | content 含 markdown |
| **T-B5** | navigation timeout | `is_error=True` |
| **T-B6** | unknown operation | `is_error=True` |
| **T-B7** | screenshot 写到 spill_dir | 文件存在；content 含路径 |

### 5.3 测试组 C：真实集成（pytest skip if not installed）

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | fetch example.com | 拿到含 "Example Domain" 的 markdown |
| **T-C2** | screenshot example.com | PNG 文件 > 1KB |

## 六、验收门

* [ ] 13+ unit case 全绿
* [ ] 真实跑（已安装 Playwright）：`web_browser(operation=fetch, url=https://www.google.com)` 拿到 SPA 内容（与 WebFetch 对比有 5x+ 内容）
* [ ] idle 30s 后浏览器进程消失（`ps aux | grep chromium`）

## 七、回滚开关

* `web_browser_enabled=False`（默认就是 False） → 工具完全不可用
* 完全 revert：删 3 个新文件 + config + pyproject 修改

## 八、实施顺序（建议 3 天）

| 步骤 | 时长 |
|---|---|
| 1. browser_manager.py + idle watcher | 4h |
| 2. browser_manager 测试（mock） | 3h |
| 3. web_browser.py 工具实现 | 5h |
| 4. web_browser 测试（mock） | 3h |
| 5. 真实集成（手动） | 2h |
| 6. CI 配置（optional dep job） | 2h |
| 7. 文档 | 1h |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| Playwright 安装失败（缺 chromium binary） | 工具检测时给出 `playwright install chromium` 提示 |
| 浏览器进程泄漏 | atexit + idle watcher 双保险 |
| 模型恶意 click 触发不可逆操作（如付款按钮） | v1 不做 — 用户开启 web_browser 自承风险；v2 加 click confirmation |
| HTTPS 证书错误 / cookie 弹窗 | playwright 默认行为；记入 docs |
| 多 tool call 并发触发 | 当前 single-threaded REPL；并发场景 v2 加 lock |
| 测试在 CI 跑 chromium 太重 | 默认 skip；专用 job 触发 |
| Playwright 包大小（250MB） | optional dep；用户选择 |

## 十、与后续 PR 的接合

* **CLI footer**：浏览器活跃时显示 "🌐 browser running"
* **PR 3.5.5 WebFetch**：模型有两个选项；description 引导先试 WebFetch，失败再 WebBrowser
* **未来 v2**：headed mode（用户能看见浏览器）；session cookie 持久化
