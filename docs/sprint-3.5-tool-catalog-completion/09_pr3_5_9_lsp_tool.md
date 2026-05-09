# PR 3.5.9 — `LSPTool` + LSP server 集成

> **角色**：让 LLM 用编辑器级别的语义查询代码 — go-to-definition、find-references、hover、
> document-symbols 等。在 grep / glob 之上提供"语义化"的代码导航。

## 一、目标

1. 集成 LSP client（pylsp / typescript-language-server / rust-analyzer 等）。
2. 实现 `LSPTool`，支持 9 个 LSP operation（与 claude-code 一致）。
3. 输出结构化结果（位置 + symbol info + hover docs），spill 大结果到磁盘。
4. **degrade gracefully**：LSP server 不可用时返回明确错误而非 crash，
   引导模型 fallback 到 grep。

## 二、为什么要做

### 2.1 grep 的不足

模型查"`Foo.bar` 方法被调用了哪些地方"时，grep `\.bar\(` 会：
* 误中其他 class 的同名方法
* 漏掉 import 改名后的 alias
* 看不见动态调用

LSP `findReferences` 用编译器的语义信息精确返回。

### 2.2 实际工作流

| 任务 | grep | LSP |
|---|---|---|
| 找 `class Foo` 的定义 | `grep -nr "class Foo"` 可能多文件多次匹配 | `goToDefinition` 准确 |
| 找 `bar()` 的所有调用 | regex 误中 | `findReferences` 准确 |
| 看 `complex_func` 的 docstring | 必须 read_file 上下文 | `hover` 直接返回 |
| 看一个文件的所有顶层符号 | grep `^def\|^class` | `documentSymbol` 含层级 |

### 2.3 取舍：复杂度 vs. 价值

LSP 集成是**最复杂**的 PR（4 天）。但它给模型的能力提升大，
尤其大型代码库（>10k LOC），grep 误差率显著。

## 三、设计

### 3.1 选 LSP client 库

Python 生态：
* `pylsp-jsonrpc` — JSON-RPC over stdin/stdout 的低层 client（已熟）
* `python-lsp-jsonrpc` — 同上，pylsp 维护
* **自写**轻量 LSP client（直接 spawn server，pipe stdin/stdout，按 JSON-RPC 协议读写）

**v1 选自写**（依赖最少；语义清晰；只需实现 9 个 op 就够）。
~300 行代码可覆盖。

### 3.2 LSP server 解析

按文件后缀匹配 server binary：

```python
# runtime/lsp_servers.py - 新文件
LANGUAGE_SERVERS: dict[str, list[str]] = {
    "python": ["pylsp", "pyright-langserver", "--stdio"],
    "typescript": ["typescript-language-server", "--stdio"],
    "javascript": ["typescript-language-server", "--stdio"],
    "rust": ["rust-analyzer"],
    "go": ["gopls"],
    # …
}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript", ".tsx": "typescript",
    ".js": "javascript", ".jsx": "javascript",
    ".rs": "rust", ".go": "go",
}


def resolve_server_for(file_path: Path) -> Optional[list[str]]:
    """Return the server invocation cmd for this file, or None."""
    lang = EXT_TO_LANG.get(file_path.suffix)
    if lang is None:
        return None
    return LANGUAGE_SERVERS.get(lang)
```

### 3.3 LSP client 主结构

```python
# runtime/lsp_client.py - 新文件
class LSPClient:
    """Minimal LSP client over stdio. Per-language singleton."""

    def __init__(self, *, command: list[str], project_root: Path):
        self.command = command
        self.project_root = project_root
        self._process: Optional[subprocess.Popen] = None
        self._next_id = 1
        self._initialized = False
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._process is not None:
            return
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(self.project_root),
        )
        self._initialize()

    def _initialize(self) -> None:
        params = {
            "processId": os.getpid(),
            "rootUri": pathlib.Path(self.project_root).as_uri(),
            "capabilities": {},  # accept all
        }
        result = self.request("initialize", params)
        self.notify("initialized", {})
        self._initialized = True

    def request(self, method: str, params: dict, *, timeout: float = 10.0) -> dict:
        ...  # send JSON-RPC; await response with matching id

    def notify(self, method: str, params: dict) -> None:
        ...  # send JSON-RPC notification (no id)

    def shutdown(self) -> None:
        if self._process:
            try:
                self.request("shutdown", None, timeout=2)
                self.notify("exit", None)
            finally:
                self._process.terminate()
                self._process = None
```

### 3.4 LSPManager（per-language singleton）

```python
class LSPManager:
    def __init__(self, *, project_root: Path):
        self.project_root = project_root
        self._clients: dict[str, LSPClient] = {}  # language → client
        self._failed: set[str] = set()  # don't retry

    def get_client_for(self, file_path: Path) -> Optional[LSPClient]:
        cmd = resolve_server_for(file_path)
        if cmd is None:
            return None
        lang = EXT_TO_LANG[file_path.suffix]
        if lang in self._failed:
            return None
        if lang not in self._clients:
            client = LSPClient(command=cmd, project_root=self.project_root)
            try:
                client.start()
            except FileNotFoundError:
                self._failed.add(lang)
                return None
            except Exception:
                self._failed.add(lang)
                return None
            self._clients[lang] = client
        return self._clients[lang]

    def shutdown_all(self) -> None:
        for c in self._clients.values():
            c.shutdown()
```

### 3.5 LSPTool

```python
class LSPTool(ToolExecutor):
    NAME = "lsp"
    MAX_RESULT_CHARS = 40_000

    OPERATIONS = {
        "goToDefinition", "findReferences", "hover",
        "documentSymbol", "workspaceSymbol",
        "goToImplementation", "prepareCallHierarchy",
        "incomingCalls", "outgoingCalls",
    }

    def __init__(self, lsp_manager: LSPManager):
        self.manager = lsp_manager

    def execute(self, call, context):
        op = call.arguments.get("operation", "")
        if op not in self.OPERATIONS:
            return ToolResult(
                content=f"unknown LSP operation: {op}; supported: {sorted(self.OPERATIONS)}",
                is_error=True,
            )

        # workspaceSymbol 不需要 file_path
        if op != "workspaceSymbol":
            file_path = self._resolve_path(call.arguments.get("filePath", ""))
            if not file_path or not file_path.exists():
                return ToolResult(content=f"file not found: {file_path}", is_error=True)
            client = self.manager.get_client_for(file_path)
            if client is None:
                return ToolResult(
                    content=(
                        f"no LSP server available for {file_path.suffix} files. "
                        f"Install one of: {LANGUAGE_SERVERS.get(EXT_TO_LANG.get(file_path.suffix, ''), 'unknown')}. "
                        f"Use grep / read_file as a fallback."
                    ),
                    is_error=True,
                )
            line = int(call.arguments.get("line", 1))
            character = int(call.arguments.get("character", 1))
            params = self._build_params(op, file_path=file_path, line=line, character=character)
            method = self._op_to_method(op)
        else:
            client = self._any_client()
            if client is None:
                return ToolResult(content="no LSP server initialized; cannot do workspaceSymbol", is_error=True)
            params = {"query": call.arguments.get("query", "")}
            method = "workspace/symbol"

        try:
            result = client.request(method, params, timeout=15)
        except TimeoutError:
            return ToolResult(content=f"LSP {op} timed out (15s)", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"LSP {op} failed: {exc}", is_error=True)

        body = self._format_result(op, result, file_path=file_path if op != "workspaceSymbol" else None)
        content = self._maybe_spill(body, call=call, context=context, extension="md")
        return ToolResult(call_id=call.id, content=content, is_error=False)

    def _format_result(self, op, result, *, file_path):
        # 把 LSP JSON 结果 → markdown：每个 location 一行 "<path>:<line>:<col>" + 上下文
        ...
```

### 3.6 输入 schema

```python
{
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["goToDefinition", "findReferences", "hover", "documentSymbol",
                     "workspaceSymbol", "goToImplementation", "prepareCallHierarchy",
                     "incomingCalls", "outgoingCalls"],
        },
        "filePath": {"type": "string"},
        "line": {"type": "integer", "minimum": 1, "description": "1-based"},
        "character": {"type": "integer", "minimum": 1, "description": "1-based"},
        "query": {"type": "string", "description": "Required for workspaceSymbol"},
    },
    "required": ["operation"],
}
```

### 3.7 EngineConfig 新字段

```python
# Sprint 3.5 / PR 3.5.9
lsp_tool_enabled: bool = True
lsp_request_timeout_seconds: int = 15
lsp_initialization_timeout_seconds: int = 10
```

### 3.8 进程生命周期

* LSP server 在第一次 LSPTool 调用时启动，留驻进程
* 主进程退出时通过 `atexit.register(manager.shutdown_all)` 清理
* 在 `cli/main.py` 的 cleanup 路径也调用一次

## 四、文件改动清单

| 文件 | 类型 | 行数 |
|---|---|---|
| `backend/harness/aether/runtime/lsp_servers.py` | **新文件** | ~80 |
| `backend/harness/aether/runtime/lsp_client.py` | **新文件** | ~300 |
| `backend/harness/aether/runtime/lsp_manager.py` | **新文件** | ~100 |
| `backend/harness/aether/tools/builtins/lsp.py` | **新文件** | ~250 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | ~5 |
| `backend/harness/aether/cli/main.py` | 修改 | atexit hook | ~10 |
| `backend/harness/aether/config/schema.py` | 修改 | ~20 |
| `backend/harness/aether/tests/test_lsp_client.py` | **新文件** | mock subprocess | ~250 |
| `backend/harness/aether/tests/test_lsp_manager.py` | **新文件** | ~150 |
| `backend/harness/aether/tests/test_lsp_tool.py` | **新文件** | ~300 |

## 五、测试用例

### 5.1 测试组 A：LSPClient（mock subprocess）

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | start + initialize | 发出 initialize JSON-RPC；接收 capabilities |
| **T-A2** | request 配 response | 拿到 result |
| **T-A3** | request 超时 | TimeoutError |
| **T-A4** | shutdown 清进程 | process.terminate 调用 |
| **T-A5** | server stderr 非空（warning） | client 不报错（只 stdout 是 protocol） |

### 5.2 测试组 B：LSPManager

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | get_client_for .py 文件 | 返回 pylsp client |
| **T-B2** | get_client_for .xyz 未知后缀 | 返回 None |
| **T-B3** | server binary 不存在（FileNotFoundError） | 返回 None；记入 _failed；不重试 |
| **T-B4** | 同 lang 二次调用 | 复用同 client（singleton） |

### 5.3 测试组 C：LSPTool

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | unknown operation | `is_error=True` |
| **T-C2** | filePath 不存在 | `is_error=True` |
| **T-C3** | LSP 不支持的语言 | `is_error=True`；含安装提示 |
| **T-C4** | request 超时 | `is_error=True` |
| **T-C5** | hover 返回结果 | content 含 markdown |
| **T-C6** | findReferences 100+ 结果 | 触发 spill |
| **T-C7** | workspaceSymbol 无 client 初始化 | `is_error=True` |

### 5.4 测试组 D：集成（需要真实 pylsp）

可跳过 if pylsp 未安装；用 pytest skip。

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | goToDefinition 真实 .py 函数 | location 正确 |
| **T-D2** | findReferences 真实使用 | 至少返回 1 个 reference |

## 六、验收门

* [ ] 25+ unit case 全绿
* [ ] 真实跑（已安装 pylsp）：`lsp(operation=goToDefinition, filePath=..., line=10, character=5)` 正确
* [ ] LSP server 不可用时模型收到清晰错误，不 crash

## 七、回滚开关

* `lsp_tool_enabled=False` → 不注册
* `LSPManager` 启动失败时自然 fallback 到 grep（模型决策）

## 八、实施顺序（建议 4 天）

| 步骤 | 时长 |
|---|---|
| 1. lsp_client.py（JSON-RPC 协议） | 1d |
| 2. lsp_manager.py + lsp_servers.py | 0.5d |
| 3. lsp.py 工具实现（9 个 op） | 1d |
| 4. formatters（result → markdown） | 0.5d |
| 5. 测试（unit + 集成） | 1d |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| pylsp 不存在 → graceful skip | 已有 _failed 集合 |
| LSP server 卡死（initialize 不返回） | initialization_timeout_seconds=10 |
| 大型项目 LSP 启动慢（rust-analyzer 索引可能 30s） | 第一次调用会等；后续复用 client |
| LSP 协议版本差异（v3.16 vs v3.17） | v1 用最小公共子集（9 op 都是基础协议） |
| 多线程 / 并发请求 | _lock 串行化 |
| LSP server crash 后 client 状态不一致 | 检测 process.poll() != None；清理后下次调用重启 |
| 模型滥用 LSP 导致大量 server 进程 | per-language 单例；最多 N 个 server（语言种类有限） |

## 十、与后续 PR 的接合

* CLI footer 显示活跃 LSP servers（"🔌 pylsp running"）
* 长任务的 token cost：LSP server 是单独进程，不计入 LLM token；可大幅降低"语义查询"的 token 成本
* 与 PR 3.5.10 WebBrowser 一样，LSP 也是"重型外部依赖"工具，需要 docs 引导用户安装
