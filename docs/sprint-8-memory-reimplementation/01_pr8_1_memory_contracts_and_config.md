# PR 8.1 - Memory Contracts + Config

## 目标

建立 Memory 子系统的稳定契约，替换当前过薄的
`MemoryProvider.build_context(session_id) -> str`。

这个 PR 只做数据结构、配置和空实现，不接入 run loop，不实现文件 store。

## 当前问题

当前接口的问题：

| 问题 | 影响 |
|---|---|
| 只返回字符串 | 无法表达 provenance、scope、token estimate |
| 只接受 `session_id` | 无法按 user message、路径、任务状态检索 |
| 没有预算 | 容易无限拼上下文 |
| 没有错误语义 | provider 失败难以可观测降级 |
| 没有写入契约 | 读写策略会散落在工具和 hook 中 |

## 新增/修改文件

| 文件 | 内容 |
|---|---|
| `backend/harness/aether/memory/contracts.py` | Memory contracts 和 provider protocol |
| `backend/harness/aether/memory/null.py` | `NullMemoryProvider` |
| `backend/harness/aether/memory/budget.py` | token budget 和基础 packing helper |
| `backend/harness/aether/memory/render.py` | `MemoryBundle` transient context renderer |
| `backend/harness/aether/agents/memory/*` | 旧路径兼容 re-export |
| `backend/harness/aether/config/schema.py` | Memory 配置项 |
| `backend/harness/aether/tests/memory/test_*.py` | contract/budget/render 单测 |

## 核心枚举

```python
class MemoryMode(str, Enum):
    OFF = "off"
    TASK = "task"
    PROJECT = "project"
    PERSONAL_ASSISTANT = "personal_assistant"


class MemoryScope(str, Enum):
    SESSION = "session"
    TASK = "task"
    PROJECT = "project"
    USER = "user"


class MemoryKind(str, Enum):
    TASK_STATE = "task_state"
    DECISION = "decision"
    CONSTRAINT = "constraint"
    PROJECT_FACT = "project_fact"
    USER_PREFERENCE = "user_preference"
    REFERENCE = "reference"
    WARNING = "warning"
```

## 数据模型

```python
@dataclass(slots=True, frozen=True)
class MemoryQuery:
    session_id: str
    task_id: str | None
    user_message: str
    recent_messages: list[dict[str, Any]]
    working_directory: str | None = None
    active_files: tuple[str, ...] = ()
    mode: MemoryMode = MemoryMode.PROJECT
    token_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass(slots=True, frozen=True)
class MemoryBlock:
    id: str
    scope: MemoryScope
    kind: MemoryKind
    text: str
    source: str
    token_estimate: int
    relevance: float = 0.0
    confidence: str = "medium"
    created_at: str | None = None
    updated_at: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
@dataclass(slots=True, frozen=True)
class MemoryBundle:
    blocks: tuple[MemoryBlock, ...] = ()
    token_estimate: int = 0
    skipped_reason: str | None = None
    latency_ms: float = 0.0
    provider_errors: tuple[str, ...] = ()
```

## Provider Protocol

```python
class MemoryProvider(Protocol):
    def retrieve(self, query: MemoryQuery) -> MemoryBundle:
        ...

    def observe_turn(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        ...

    def before_compaction(
        self,
        *,
        session_id: str,
        task_id: str | None,
        messages: list[dict[str, Any]],
        metadata: dict[str, Any],
    ) -> None:
        ...
```

语义：

- `retrieve()` 在 LLM call 前执行，必须 bounded latency。
- `observe_turn()` 在 turn 结束后执行，失败不得影响 result。
- `before_compaction()` 只允许更新 task/session summary，不允许把 retrieved memory 写入 transcript。

## 配置项

新增到 `EngineConfig`：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `memory_enabled` | `bool` | `True` | 总开关 |
| `memory_mode` | `str` | `"project"` | 默认 task + project |
| `memory_token_budget_pct` | `float` | `0.08` | context window 比例 |
| `memory_token_budget_max` | `int` | `2500` | 全局硬上限 |
| `memory_block_token_max` | `int` | `500` | 单 block 上限 |
| `memory_retrieval_timeout_ms` | `int` | `200` | 检索等待上限 |
| `memory_project_store_enabled` | `bool` | `True` | 项目 store 开关 |
| `memory_user_profile_enabled` | `bool` | `False` | user scope 开关 |
| `memory_auto_write_enabled` | `bool` | `False` | 自动写入开关 |
| `memory_llm_rerank_enabled` | `bool` | `False` | LLM rerank 开关 |
| `memory_debug_log_content` | `bool` | `False` | debug 内容日志 |

## 模式语义

| mode | session/task | project | user |
|---|---:|---:|---:|
| `off` | 关 | 关 | 关 |
| `task` | 开 | 关 | 关 |
| `project` | 开 | 开 | 关 |
| `personal_assistant` | 开 | 开 | 开 |

即使 `memory_user_profile_enabled=True`，只有
`memory_mode="personal_assistant"` 时 user scope 才能参与默认召回。

## 不变量

- `MemoryBlock.text` 必须是已裁剪文本，不能包含超大原始文件。
- `token_estimate` 缺失时视为不可注入。
- `source` 必须可审计，例如 `task:<task_id>` 或 `.aether/memory/topics/architecture.md#decision-3`。
- provider 返回异常时，engine 包装成 empty `MemoryBundle`。
- `NullMemoryProvider.retrieve()` 永远返回 skipped bundle。

## 测试

新增测试：

```python
def test_null_provider_returns_empty_bundle(): ...
def test_memory_mode_project_excludes_user_scope(): ...
def test_memory_mode_personal_allows_user_scope_when_enabled(): ...
def test_memory_block_requires_source_and_token_estimate(): ...
def test_budget_config_defaults_are_conservative(): ...
def test_invalid_memory_mode_falls_back_to_off_or_project(): ...
```

## 验收门

- 当前所有现有测试不需要真实 memory provider。
- `MemoryProvider` 不 import CLI、provider SDK 或 compaction 实现。
- 配置默认不会启用 personal memory。
- type contracts 能支持后续 PR 无破坏扩展。
