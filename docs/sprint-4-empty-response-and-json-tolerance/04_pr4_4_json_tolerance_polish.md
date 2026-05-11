# PR 4.4 — JSON 错误结构化升级

> **角色**：把 PR 1.3 已经实现的"silent retry → inject role=tool error"两段路径中**注入消息的内容**
> 从裸 `JSONDecodeError.message` 升级成结构化、模型友好的格式。**不改任何控制流**，只换 string 模板。
>
> 借鉴 [`open-claude-code/src/utils/toolErrors.ts:formatZodValidationError`](../../tmp/claude-code-references)。
> PR 4.1 已经实现了 `runtime/tool_error_format.py:format_invalid_tool_args_error`（JSON 语法错版）；
> 本 PR 把它接入 + 加上"参数 schema 错误"的同款分类（虽然 Aether 当前还没有强制 schema validator，
> 但通过 tool descriptor 已能拿到字段信息做轻量判断）。
>
> **来自 Codex 反馈的 2 处修订**：
>
> | 反馈 | 修订 |
> |---|---|
> | **#7** 原文档示例用 `file_path` 作为 read_file 的参数名，但 Aether 真实 schema 是 `path`（`backend/harness/aether/tools/builtins/read_file.py:62-85`，required=`["path"]`）。 | 所有 read_file 相关示例 / 测试 case / 验收断言改用 `path`。**不**修改 read_file schema（read_file 是已上线工具，schema 改名是 breaking change）。 |
> | **#8** 原文档使用 `ToolRegistry.get_descriptor(name)` 与 `list_names()`，但当前 `ToolRegistry` 只有 `get` / `has` / `list_descriptors` / `dispatch`（`backend/harness/aether/tools/registry.py:57-93`）。 | PR 4.4 以**只读公共 API** 的形式新增这两个方法（纯包装，零行为变更）。详见 § 2.5。 |

## 一、目标

1. **接入现有的 `format_invalid_tool_args_error`**：把 `agent.py::_validate_tool_call_arguments`
   注入消息中的 `f"Error: Invalid JSON arguments. {err_msg}. ..."` 替换为
   `format_invalid_tool_args_error(...).text`。
2. **新增 schema 错误格式化**：扩展 `runtime/tool_error_format.py` 加 `format_schema_error()`，
   分类为 `missing` / `unexpected` / `type_mismatch`，对齐 claude-code。
3. **在 invalid-tool-name 路径里也用结构化错误**：当 PR 2.3 `prepare_tool_calls` 修不出名字时，
   inject 的合成错误消息也用结构化格式。
4. **可观测性**：每个 inject 走"category"字段（json_syntax / schema_missing / schema_type_mismatch / unknown_tool）
   写入 `metadata["tool_errors"]`。
5. **零外部 schema 依赖**：本 PR 不引入 pydantic / Zod 等运行时 schema validator；
   schema 错误判定基于 tool descriptor 自带的 `parameters.required` / `parameters.properties` 信息
   做轻量比对。

## 二、现状分析

### 2.1 PR 1.3 当前的 inject 内容

`backend/harness/aether/agents/core/agent.py:2425-2453`：

```python
invalid_lookup = {name: err for name, err in invalid_json_args}
for c in response.tool_calls:
    if c.name in invalid_lookup:
        err_msg = invalid_lookup[c.name]
        injection.append(
            {
                "role": "tool",
                "name": c.name,
                "tool_call_id": c.id,
                "content": (
                    f"Error: Invalid JSON arguments. {err_msg}. "
                    "For tools with no required parameters, use an empty object: {}. "
                    "Please retry with valid JSON."
                ),
                "is_error": True,
                "metadata": {"_invalid_json_recovery": True},
            }
        )
    else:
        injection.append(
            {
                "role": "tool",
                "name": c.name,
                "tool_call_id": c.id,
                "content": "Skipped: another tool call in this response had invalid JSON.",
                "is_error": True,
                "metadata": {"_invalid_json_recovery": True},
            }
        )
```

`err_msg = str(JSONDecodeError)` 长这样：`"Expecting ',' delimiter: line 1 column 23 (char 22)"`。
信息太裸——模型知道哪一行出错但不知道**具体位置上下文**（出错前后是什么）。

### 2.2 PR 4.1 已经准备好的工具

`backend/harness/aether/runtime/tool_error_format.py`（PR 4.1 已实现）：

```python
@dataclass(frozen=True)
class FormattedToolError:
    text: str
    category: str          # "json_syntax"
    line: Optional[int]
    column: Optional[int]

def format_invalid_tool_args_error(tool_name, exc, raw_args) -> FormattedToolError:
    """对 JSONDecodeError 做友好格式化（缺位置上下文 + Hint）。"""
```

PR 4.4 的工作就是 **(a)** 接入 + **(b)** 扩展同一文件支持 schema 错误。

### 2.3 claude-code 的 schema 错误格式化

```typescript
// formatZodValidationError 把 ZodError.issues 分三类：
const missingParams = issues.filter(err =>
    err.code === 'invalid_type' && err.message.includes('received undefined')
).map(err => formatValidationPath(err.path))

const unexpectedParams = issues.filter(err => err.code === 'unrecognized_keys')
    .flatMap(err => err.keys)

const typeMismatchParams = issues.filter(err =>
    err.code === 'invalid_type' && !err.message.includes('received undefined')
).map(err => ({ param, expected, received }))

// 然后拼出：
// "<tool_name> failed due to the following N issues:
//  The required parameter `<p>` is missing
//  An unexpected parameter `<p>` was provided
//  The parameter `<p>` type is expected as `<E>` but provided as `<R>`"
```

我们没有 Zod，但有 tool descriptor 里的 `parameters` JSON Schema 片段，可以做同样的事。

### 2.4 Aether 的 ToolDescriptor 结构

`backend/harness/aether/tools/registry.py` 的 ToolDescriptor 大致是：

```python
@dataclass
class ToolDescriptor:
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema dict
    # parameters["required"]: list[str]
    # parameters["properties"]: dict[str, dict]   # name -> {type: ..., ...}
```

我们能基于 `parameters["required"]` 与 `parameters["properties"]` 做：
- **missing**: 一旦 parsed dict 缺 `required` 中的某项 → `missing`
- **unexpected**: parsed dict 含 `properties` 之外的 key（且 schema `additionalProperties=False` 或未指定）
- **type_mismatch**: parsed dict 含的 key 在 properties 中存在，但其值类型与 properties[key].type 不符

类型判定用最朴素的 `isinstance` 映射（string/number/integer/boolean/array/object）。
深层嵌套不做（够用就行；后续 Sprint 引入 jsonschema 时再补）。

### 2.5 ToolRegistry 公共 API（Codex 反馈 #8 修订）

当前 `backend/harness/aether/tools/registry.py:57-93` 的 `ToolRegistry` 只暴露：

```python
class ToolRegistry:
    def register(self, executor) -> None: ...
    def has(self, name: str) -> bool: ...
    def get(self, name: str) -> ToolExecutor: ...     # raises UnknownToolError
    def list_descriptors(self) -> list[ToolDescriptor]: ...
    def dispatch(self, call, context) -> ToolResult: ...
```

PR 4.4 的 `format_unknown_tool_error` 需要 "did you mean" 候选列表，
`format_schema_error` 需要按工具名取 schema。两个使用都不希望走 `get(name)` 抛异常的路径，
所以**新增两个只读的便捷封装**作为公共 API：

```python
def get_descriptor(self, name: str) -> ToolDescriptor:
    """Convenience accessor; returns the descriptor for ``name``.

    Raises ``UnknownToolError`` (same as ``get``).  Wrapping ``get(name).descriptor``
    keeps callers from depending on ``ToolExecutor`` just to read its metadata.
    """
    return self.get(name).descriptor

def list_names(self) -> list[str]:
    """Return all registered tool names in registration order."""
    return list(self._tools.keys())
```

这两个方法是**纯包装**（无新行为），不改既有契约，不影响任何现有调用方。
PR 4.4 同时补两个方法的单测（一个普通 case、一个未注册 case 抛 UnknownToolError）。

## 三、设计

### 3.1 扩展 `runtime/tool_error_format.py`

新增公开函数 `format_schema_error()` 和内部分类逻辑：

```python
"""... existing PR 4.1 docstring ..."""

# ----- New in PR 4.4 -----

@dataclass(frozen=True)
class _SchemaIssue:
    """One issue extracted from a tool descriptor + parsed args comparison."""

    code: str   # "missing" / "unexpected" / "type_mismatch"
    path: str   # e.g. "file_path" or "todos[0].activeForm"
    expected: Optional[str] = None
    received: Optional[str] = None


def format_schema_error(
    tool_name: str,
    parameters_schema: Mapping[str, Any] | None,
    parsed_args: Any,
) -> FormattedToolError:
    """Build a friendly error for a tool call whose ARGS parsed as JSON
    but failed lightweight schema validation.

    ``parameters_schema`` is the tool descriptor's ``parameters`` block
    (JSON Schema-shaped dict).  When ``None`` or unrecognised, returns
    a generic "schema mismatch" message — the model still gets a useful
    signal but without per-field detail.

    The classifier handles:
      - missing ``required`` fields
      - unexpected fields not in ``properties`` (when
        ``additionalProperties=False`` or unset)
      - top-level type mismatches against ``properties[k].type``

    Nested objects/arrays are not deep-validated — caller can use a
    proper jsonschema validator for that and pass the result through
    :func:`format_schema_error_from_issues`.
    """
    issues = _collect_schema_issues(parameters_schema, parsed_args)
    return format_schema_error_from_issues(tool_name, issues)


def format_schema_error_from_issues(
    tool_name: str,
    issues: list[_SchemaIssue],
) -> FormattedToolError:
    """Same output format as ``format_schema_error`` but pre-computed
    issue list. Used when caller has its own schema validator (PR
    Sprint 5+ might bring jsonschema in)."""
    if not issues:
        return FormattedToolError(
            text=f"`{tool_name}` arguments validation failed (no specific issue extracted).",
            category="schema_unknown",
        )

    parts = []
    for issue in issues:
        if issue.code == "missing":
            parts.append(f"The required parameter `{issue.path}` is missing")
        elif issue.code == "unexpected":
            parts.append(f"An unexpected parameter `{issue.path}` was provided")
        elif issue.code == "type_mismatch":
            parts.append(
                f"The parameter `{issue.path}` type is expected as `{issue.expected}` "
                f"but provided as `{issue.received}`"
            )
        else:
            parts.append(f"`{issue.path}`: {issue.code}")

    summary = (
        f"`{tool_name}` failed due to the following "
        f"{'issue' if len(parts) == 1 else f'{len(parts)} issues'}:"
    )
    text = "\n".join([summary, *parts])

    # Pick a single dominant category for routing/observability.
    if any(i.code == "missing" for i in issues):
        category = "schema_missing"
    elif any(i.code == "type_mismatch" for i in issues):
        category = "schema_type_mismatch"
    elif any(i.code == "unexpected" for i in issues):
        category = "schema_unexpected"
    else:
        category = "schema_unknown"
    return FormattedToolError(text=text, category=category)


def format_unknown_tool_error(
    requested_name: str,
    available_names: list[str],
    repair_attempts: list[tuple[str, str]] | None = None,
) -> FormattedToolError:
    """Build a friendly error when the model called a tool that doesn't
    exist (and PR 2.3's repair couldn't fix it).

    ``repair_attempts``: list of (stage, candidate_name) tuples
    capturing what PR 2.3 tried, so the model sees why it failed.
    """
    parts = [f"Tool `{requested_name}` does not exist."]
    if repair_attempts:
        attempts_txt = "; ".join(f"{stage}→`{cand}`" for stage, cand in repair_attempts)
        parts.append(f"Repair attempts: {attempts_txt}.")
    suggestions = _suggest_similar_tools(requested_name, available_names, top_k=3)
    if suggestions:
        parts.append(f"Did you mean: {', '.join('`' + s + '`' for s in suggestions)}?")
    parts.append(
        "Please pick one of the registered tools and retry."
    )
    return FormattedToolError(
        text="\n".join(parts),
        category="unknown_tool",
    )


# ----- internal helpers (continued) -----

_PRIMITIVE_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
}


def _collect_schema_issues(
    schema: Mapping[str, Any] | None,
    parsed_args: Any,
) -> list[_SchemaIssue]:
    """Lightweight schema check.  Returns issue list (possibly empty)."""
    if schema is None or not isinstance(parsed_args, dict):
        return []

    issues: list[_SchemaIssue] = []
    properties = schema.get("properties") or {}
    required = list(schema.get("required") or [])
    additional = schema.get("additionalProperties", None)
    additional_allowed = additional is not False  # default permissive

    # Missing
    for key in required:
        if key not in parsed_args:
            issues.append(_SchemaIssue(code="missing", path=str(key)))

    # Unexpected (only when additional explicitly disallowed)
    if not additional_allowed:
        for key in parsed_args:
            if key not in properties:
                issues.append(_SchemaIssue(code="unexpected", path=str(key)))

    # Type mismatch (top-level only; nested is caller's job)
    for key, prop_schema in properties.items():
        if key not in parsed_args:
            continue
        expected = prop_schema.get("type")
        if not expected:
            continue
        if isinstance(expected, list):
            # Schema like {"type": ["string", "null"]} — allow any in list
            if any(_value_matches_type(parsed_args[key], t) for t in expected):
                continue
            received_t = _python_type_name(parsed_args[key])
            issues.append(
                _SchemaIssue(
                    code="type_mismatch",
                    path=str(key),
                    expected=" or ".join(expected),
                    received=received_t,
                )
            )
        else:
            if _value_matches_type(parsed_args[key], expected):
                continue
            issues.append(
                _SchemaIssue(
                    code="type_mismatch",
                    path=str(key),
                    expected=str(expected),
                    received=_python_type_name(parsed_args[key]),
                )
            )

    return issues


def _value_matches_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    types = _PRIMITIVE_TYPE_MAP.get(expected)
    if types is None:
        return True   # Unknown schema type — be permissive
    if expected == "boolean":
        return isinstance(value, bool)   # don't accept int=True/False here
    if expected in ("number", "integer"):
        return isinstance(value, types) and not isinstance(value, bool)
    return isinstance(value, types)


def _python_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _suggest_similar_tools(
    requested: str,
    available: list[str],
    *,
    top_k: int,
) -> list[str]:
    """Levenshtein-distance suggestion (≤ 2 edits, top_k by smallest distance)."""
    if not available or not requested:
        return []
    scored = []
    for name in available:
        d = _levenshtein(requested.lower(), name.lower())
        if d <= 2:
            scored.append((d, name))
    scored.sort()
    return [n for _, n in scored[:top_k]]


def _levenshtein(a: str, b: str) -> int:
    """Iterative DP Levenshtein.  O(len(a)*len(b)) time, O(min) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]
```

### 3.2 `agent.py::_validate_tool_call_arguments` 接入

修改 inject 路径（PR 1.3 的代码替换）：

```python
# Before (PR 1.3):
invalid_lookup = {name: err for name, err in invalid_json_args}
for c in response.tool_calls:
    if c.name in invalid_lookup:
        err_msg = invalid_lookup[c.name]
        injection.append({
            "role": "tool",
            ...
            "content": (
                f"Error: Invalid JSON arguments. {err_msg}. "
                "For tools with no required parameters, use an empty object: {}. "
                "Please retry with valid JSON."
            ),
            ...
        })

# After (PR 4.4):
from aether.runtime.tool_error_format import format_invalid_tool_args_error

invalid_lookup = {name: err for name, err in invalid_json_args}
# Map tool_name -> the actual exception we caught (we need the
# JSONDecodeError instance, not just its str()).  Track in step 3.
exc_lookup = {name: exc for name, exc in invalid_json_excs}

for c in response.tool_calls:
    if c.name in invalid_lookup:
        if (
            self.config.tool_error_structured_format_enabled
            and c.name in exc_lookup
        ):
            formatted = format_invalid_tool_args_error(
                tool_name=c.name,
                exc=exc_lookup[c.name],
                raw_args=str(getattr(c, "arguments", "") or ""),
            )
            content_text = formatted.text
            error_category = formatted.category
        else:
            err_msg = invalid_lookup[c.name]
            content_text = (
                f"Error: Invalid JSON arguments. {err_msg}. "
                "For tools with no required parameters, use an empty object: {}. "
                "Please retry with valid JSON."
            )
            error_category = "json_syntax_legacy"

        injection.append({
            "role": "tool",
            "name": c.name,
            "tool_call_id": c.id,
            "content": content_text,
            "is_error": True,
            "metadata": {
                "_invalid_json_recovery": True,
                "_tool_error_category": error_category,
            },
        })
        # Observability
        self._record_tool_error(context, c.name, error_category)
    else:
        injection.append({
            "role": "tool",
            ...
            "content": "Skipped: another tool call in this response had invalid JSON.",
            "is_error": True,
            "metadata": {
                "_invalid_json_recovery": True,
                "_tool_error_category": "skipped_due_to_sibling",
            },
        })
        self._record_tool_error(context, c.name, "skipped_due_to_sibling")
```

### 3.3 在解析失败处保留异常实例

`_validate_tool_call_arguments` 当前只记录 `(name, str(exc))`；改为同时记录 `exc`：

```python
# Sprint 4 / PR 4.4: keep the original exception so the structured
# formatter can extract line/column. (PR 1.3 only kept str(exc).)
invalid_json_args: list[tuple[str, str]] = []
invalid_json_excs: list[tuple[str, Exception]] = []   # ← 新增

for call in response.tool_calls:
    ...
    try:
        parsed = json.loads(args_text)
    except json.JSONDecodeError as exc:
        invalid_json_args.append((call.name, str(exc)))
        invalid_json_excs.append((call.name, exc))
        continue
    ...
```

注入路径用 `invalid_json_excs` 找异常对象（key by tool name + index 防重复）。

### 3.4 schema 错误集成（dispatch 时）

PR 1.3 的 invalid-JSON 路径只覆盖**JSON 语法错**。当 JSON 解析成功但参数不符 schema
（例如 read_file 缺 `path`），目前的行为是**让 tool 自己报错**（tool implementation
内部抛 `KeyError` / `ValidationError`，被 ToolError catch）。

PR 4.4 在 dispatch 之前**插入轻量 schema check**，命中时构造与 invalid-JSON 同款的
inject 流（不真正 dispatch tool），让模型早一轮就拿到结构化错误：

```python
# In agent.py, around the tool dispatch loop (currently uses prepare_tool_calls
# from PR 2.3).  Add a schema-check pass on the prepared calls:

def _maybe_inject_schema_errors(
    self,
    *,
    prepared_plan: ToolDispatchPlan,
    response: NormalizedResponse,
    messages: list[dict],
    context: TurnContext,
) -> list[dict] | None:
    """If any prepared call has obviously-wrong args (per descriptor schema),
    inject corrective tool errors and skip dispatch.

    Returns: the injection list if injection happened (caller continues
    the loop); None if no schema errors were detected (caller proceeds
    with dispatch).
    """
    if not self.config.tool_error_structured_format_enabled:
        return None
    if not self.config.tool_schema_precheck_enabled:
        return None

    issues_per_call: list[tuple[ToolCall, FormattedToolError]] = []
    for call in prepared_plan.calls_to_dispatch:
        descriptor = self.tool_registry.get_descriptor(call.name)
        if descriptor is None:
            continue
        schema = descriptor.parameters or {}
        formatted = format_schema_error(call.name, schema, call.arguments)
        if formatted.category == "schema_unknown" and not _has_visible_issue(formatted):
            continue
        issues_per_call.append((call, formatted))

    if not issues_per_call:
        return None

    # Build inject (same shape as PR 1.3 invalid-JSON inject).
    injection: list[dict] = [self._build_assistant_echo(response)]
    err_call_ids = {c.id for c, _ in issues_per_call}
    for call in response.tool_calls:
        if call.id in err_call_ids:
            formatted = next(f for c, f in issues_per_call if c.id == call.id)
            injection.append({
                "role": "tool",
                "name": call.name,
                "tool_call_id": call.id,
                "content": formatted.text,
                "is_error": True,
                "metadata": {
                    "_schema_error_recovery": True,
                    "_tool_error_category": formatted.category,
                },
            })
            self._record_tool_error(context, call.name, formatted.category)
        else:
            injection.append({
                "role": "tool",
                "name": call.name,
                "tool_call_id": call.id,
                "content": "Skipped: another tool call in this response had a schema error.",
                "is_error": True,
                "metadata": {
                    "_schema_error_recovery": True,
                    "_tool_error_category": "skipped_due_to_sibling_schema",
                },
            })
    return injection
```

调用点（在 dispatch 之前）：

```python
prepared_plan = prepare_tool_calls(response.tool_calls, self.tool_registry, ...)
if prepared_plan.exit_reason is not None:
    exit_reason = prepared_plan.exit_reason
    break

# PR 4.4: schema precheck (lightweight; only flags obvious top-level errors)
schema_injection = self._maybe_inject_schema_errors(
    prepared_plan=prepared_plan,
    response=response,
    messages=messages,
    context=context,
)
if schema_injection is not None:
    messages.extend(schema_injection)
    state_machine.transition(LoopState.CHECK_EXIT)
    if budget.exhausted:
        ...
    state_machine.transition(LoopState.PRE_LLM)
    continue

# Continue with normal dispatch.
for prepared in prepared_plan.calls_to_dispatch:
    ...
```

### 3.5 Unknown-tool 错误结构化

PR 2.3 的 `prepare_tool_calls` 在 repair 失败后会构造一个 `synthetic_result`。
本 PR 升级该 synthetic_result 的内容用 `format_unknown_tool_error`：

```python
# In tool_hardening.py / prepare_tool_calls (PR 2.3):
# When repair fails, build the synthetic error using the new formatter.
from aether.runtime.tool_error_format import format_unknown_tool_error

formatted = format_unknown_tool_error(
    requested_name=call.name,
    available_names=list(self.tool_registry.list_names()),
    repair_attempts=repair_log,
)
synthetic_result = {
    "role": "tool",
    "name": call.name,
    "tool_call_id": call.id,
    "content": formatted.text,
    "is_error": True,
    "metadata": {"_unknown_tool_recovery": True, "_tool_error_category": formatted.category},
}
```

### 3.6 observability

`agent.py::_record_tool_error`：

```python
def _record_tool_error(
    self, context: TurnContext, tool_name: str, category: str
) -> None:
    """Per-turn counter of tool errors by category."""
    md = context.metadata.setdefault("tool_errors", {})
    by_category = md.setdefault("by_category", {})
    by_category[category] = int(by_category.get(category, 0)) + 1
    by_tool = md.setdefault("by_tool", {})
    by_tool[tool_name] = int(by_tool.get(tool_name, 0)) + 1
    md["total"] = int(md.get("total", 0)) + 1
```

`_build_result` 输出：

```python
md["tool_errors"] = dict(context.metadata.get("tool_errors") or {})
```

### 3.7 `EngineConfig` 新字段

```python
# Sprint 4 / PR 4.4: schema precheck before dispatch.  When True the
# engine inspects each prepared tool call against its descriptor
# parameters schema and injects a structured error (skipping dispatch)
# if obvious top-level schema issues are found.  Saves one LLM round
# trip per buggy call vs letting the tool implementation fail at runtime.
tool_schema_precheck_enabled: bool = True
```

> `tool_error_structured_format_enabled` 已由 PR 4.1 声明（默认 True）。

## 四、文件改动清单

| 文件 | 改动类型 | 改动详情 | 行数估算 |
|---|---|---|---|
| `backend/harness/aether/runtime/tool_error_format.py` | 修改 | 加 `_SchemaIssue` + `format_schema_error` + `format_schema_error_from_issues` + `format_unknown_tool_error` + `_collect_schema_issues` + `_value_matches_type` + `_python_type_name` + `_suggest_similar_tools` + `_levenshtein` | ~250 净增（基础 PR 4.1 已有 150） |
| `backend/harness/aether/tools/registry.py` | 修改 | **Codex 反馈 #8**：新增 `get_descriptor(name) -> ToolDescriptor`（包装 `get(name).descriptor`，沿用 UnknownToolError）+ `list_names() -> list[str]`（包装 `list(self._tools.keys())`） | ~10 |
| `backend/harness/aether/agents/core/agent.py` | 修改 | `_validate_tool_call_arguments` 注入处接入 formatter；新增 `_maybe_inject_schema_errors`；新增 `_record_tool_error`；`_build_result` 加 `tool_errors` 子字典；保留 invalid_json_excs 列表 | ~120 净增 |
| `backend/harness/aether/agents/core/tool_hardening.py` | 修改 | repair 失败构造 synthetic_result 时改用 `format_unknown_tool_error`（实现里调 `tool_registry.list_names()` 取候选） | ~25 |
| `backend/harness/aether/config/schema.py` | 修改 | 加 `tool_schema_precheck_enabled` | ~10 |
| `backend/harness/aether/tests/tools/test_tool_registry_public_api.py` | **新文件** | 验证新增的 `get_descriptor` / `list_names`（4 case：普通、未注册抛 UnknownToolError、空 registry、注册顺序） | ~80 |
| `backend/harness/aether/tests/runtime/test_tool_error_format_schema.py` | **新文件** | 见 § 五 测试组 A-C（约 22 case） | ~350 |
| `backend/harness/aether/tests/agents/core/test_invalid_json_inject_structured.py` | **新文件** | 测试组 D（既有 PR 1.3 用例 + 升级）| ~200 |
| `backend/harness/aether/tests/agents/core/test_schema_precheck.py` | **新文件** | 测试组 E（约 12 case） | ~250 |
| `backend/harness/aether/tests/agents/core/test_unknown_tool_inject.py` | **新文件** | 测试组 F（约 5 case） | ~120 |

## 五、测试用例（详细）

### 5.1 `test_tool_error_format_schema.py`

**测试组 A：`format_schema_error` 单元**

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | parameters={"required":["path"], "properties":{"path":{"type":"string"}}}, parsed={} | category="schema_missing"；text 含 "The required parameter `path` is missing"（**Codex 反馈 #7**：read_file 真实参数名是 `path`） |
| **T-A2** | parsed={"path":123} | category="schema_type_mismatch"；text 含 "expected as `string` but provided as `integer`" |
| **T-A3** | parsed={"path":"a", "bad_key":"x"}, additionalProperties=False | category="schema_unexpected"；text 含 "An unexpected parameter `bad_key`" |
| **T-A4** | 缺 + 类型错 + 多 同时存在 | text 列出 3 行；category 优先 "schema_missing" |
| **T-A5** | parsed={"path":"a"}, schema 完全 ok | category="schema_unknown"；text 含 "no specific issue extracted" |
| **T-A6** | parameters_schema=None | 走 schema_unknown 分支 |
| **T-A7** | parsed_args 不是 dict（是 list） | 直接返回 schema_unknown（issues=[]） |
| **T-A8** | additionalProperties=True；parsed 有多 key | 不报 unexpected |
| **T-A9** | type=["string","null"]；parsed={"x": None} | match null → 无 issue |
| **T-A10** | type=["string","null"]；parsed={"x": 1} | type_mismatch；expected="string or null" |
| **T-A11** | type="boolean"；parsed={"x": 1} | type_mismatch（不接受 1 当 bool） |
| **T-A12** | type="integer"；parsed={"x": True} | type_mismatch（不接受 True 当 int） |

**测试组 B：`format_unknown_tool_error`**

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | requested="reed_file"；available=["read_file", "write_file"]；suggestion="read_file"（dist=1） | text 含 "Did you mean: `read_file`" |
| **T-B2** | requested="completely_unrelated_xyz"；available=["read_file"] | suggestion 部分省略；text 仍含基础说明 |
| **T-B3** | repair_attempts=[("exact", "x"), ("levenshtein", "read")] | text 含 "Repair attempts: exact→`x`; levenshtein→`read`" |
| **T-B4** | available=[]（注册表空） | text 不含 "Did you mean"；其它正常 |
| **T-B5** | category 总是 "unknown_tool" | 字段验证 |

**测试组 C：辅助函数**

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | `_levenshtein("read_file", "reed_file")` | == 1 |
| **T-C2** | `_levenshtein("", "abc")` | == 3 |
| **T-C3** | `_value_matches_type(True, "boolean")` | True |
| **T-C4** | `_value_matches_type(1, "boolean")` | False |
| **T-C5** | `_value_matches_type(1.0, "integer")` | False |
| **T-C6** | `_python_type_name(None)` | "null" |

### 5.2 `test_invalid_json_inject_structured.py`

**测试组 D：JSON 错误注入（升级 PR 1.3 用例）**

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 模型 emit `read_file({"path":"/x" "bad":1})`（缺 comma）；3 次 silent retry 后注入 | inject 的 tool message content 含 "line 1 column XX"；含 "Hint:"；含 raw snippet；不再含裸 "Expecting ',' delimiter" |
| **T-D2** | metadata["_tool_error_category"]="json_syntax" | 标记正确 |
| **T-D3** | tool_error_structured_format_enabled=False | 退回 PR 1.3 原始 message；category="json_syntax_legacy" |
| **T-D4** | 多个 tool_calls 同时 invalid | 每个都用 formatter；其它 tool 用 "Skipped" |
| **T-D5** | metadata["tool_errors"]["by_category"]["json_syntax"] >= 1 | observability 正确 |
| **T-D6** | metadata["tool_errors"]["total"] 累加跨多次 inject | 正确累加 |
| **T-D7** | 既有 PR 1.3 行为：silent retry 3 次后才 inject | 不变 |
| **T-D8** | 既有 PR 1.3 行为：truncated 路径不变 | 不变 |

### 5.3 `test_schema_precheck.py`

**测试组 E：schema 预校验**

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | tool="read_file" 缺 `path`（schema 要求）| dispatch 跳过；inject schema_missing 错误；continue |
| **T-E2** | tool="read_file" `path` 是 integer | inject schema_type_mismatch |
| **T-E3** | tool 没有 descriptor（mock）| precheck 不报错；正常 dispatch |
| **T-E4** | tool_schema_precheck_enabled=False | 完全跳过 precheck；走原 dispatch |
| **T-E5** | descriptor.parameters=None | precheck 跳过该 tool |
| **T-E6** | 多 tool 全部 schema 正确 | 不 inject；正常 dispatch |
| **T-E7** | 多 tool 1 个 schema 错 | inject 全部（错的用真实错；其它用 sibling skip） |
| **T-E8** | metadata["tool_errors"]["by_category"]["schema_missing"]=1 | 正确 |
| **T-E9** | precheck inject 后 next turn 模型修正 | 正常完成（端到端） |
| **T-E10** | precheck 命中 "additional permissive"（默认）| 不报 unexpected；多余 key 仅 type-mismatch 触发 |
| **T-E11** | precheck 与 invalid-JSON 互斥：args 是 invalid JSON 时 precheck 不跑 | 走 PR 1.3 原路径 |
| **T-E12** | precheck **不**触发 dispatch budget 消耗 | iteration_budget 不被 precheck 计 |

### 5.4 `test_unknown_tool_inject.py`

**测试组 F：unknown tool 错误**

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | 模型 emit "reed_file"；PR 2.3 repair 失败 | synthetic_result.content 用 format_unknown_tool_error；含 "Did you mean: `read_file`" |
| **T-F2** | 模型 emit "totally_made_up_tool" | text 含 "does not exist"；可能无 suggestion |
| **T-F3** | metadata["_tool_error_category"]="unknown_tool" | 标记正确 |
| **T-F4** | 既有 PR 2.3 INVALID_TOOL_REPEATED 路径不变 | 不回归 |
| **T-F5** | tool_error_structured_format_enabled=False | repair 失败仍用 PR 2.3 原始 synthetic message |

## 六、验收门

- [ ] 所有新测试 green（约 39 case）
- [ ] PR 1.3 既有测试不回归（重要：注入路径输出格式变了，可能需要更新固定字符串断言）
- [ ] PR 2.3 既有测试不回归（unknown_tool 路径输出格式变了，同样需要更新）
- [ ] `result.metadata["tool_errors"]` schema 完整（by_category / by_tool / total）
- [ ] 手工 mock：模型 emit read_file 时用 `{"pth":"a"}`（typo）→ 看 inject content 是 "The required parameter `path` is missing" + "An unexpected parameter `pth` was provided"
- [ ] 手工 mock：模型 emit read_file 时用 `{"path":123}` → 看 inject content 是 "type is expected as `string` but provided as `integer`"
- [ ] `ToolRegistry.get_descriptor("read_file").parameters["required"] == ["path"]`（验证新 API + 真实 schema）
- [ ] `"read_file" in ToolRegistry.list_names()`（验证新 API）

## 七、回滚开关

- `tool_error_structured_format_enabled=False`：JSON 错误回到 PR 1.3 文本；schema precheck 也跳过
- `tool_schema_precheck_enabled=False`：仅关闭 schema precheck（JSON 错误仍走结构化）
- 完全 revert PR：删除 `tool_error_format.py` 中 PR 4.4 新增函数；agent.py 的 inject 内容回退到字符串模板

## 八、实施顺序（建议 1 天）

| 步骤 | 时长 | 输出 |
|---|---|---|
| 1. `tool_error_format.py` 加 `format_schema_error` + `_collect_schema_issues` + 测试组 A | 2h | 12 case |
| 2. 加 `format_unknown_tool_error` + `_suggest_similar_tools` + `_levenshtein` + 测试组 B-C | 1.5h | 11 case |
| 3. `agent.py::_validate_tool_call_arguments` 接入 + 保留 exception 实例 | 1h | inject 路径升级 |
| 4. 测试组 D（升级 PR 1.3 用例 + 验证 fallback） | 1.5h | 8 case |
| 5. `agent.py::_maybe_inject_schema_errors` + dispatch 调用点 | 1h | precheck 路径打通 |
| 6. 测试组 E（schema precheck） | 2h | 12 case |
| 7. `tool_hardening.py` 接入 + 测试组 F | 1h | 5 case |
| 8. observability + result.metadata | 30min | _record_tool_error 等 |
| 9. PR 1.3 / PR 2.3 回归（修少量字符串断言） | 1h | full suite green |

## 九、风险与缓解

| 风险 | 概率 | 缓解 |
|---|---|---|
| 既有测试断言精确 string 改后大量需要更新 | 高 | 接受；PR 4.4 commit 同时更新；review 时聚焦"行为变化"而不是"字符串变化" |
| schema precheck 误判（轻量逻辑覆盖不全）| 中 | 仅做 top-level 必填 / 显式禁多 / 顶层类型；嵌套交给 tool 自己；T-E10/E11 覆盖典型边界 |
| `_value_matches_type` 在 `bool` vs `int` 处理上太严格导致已有 tool 失败 | 中 | 测试组 A 严格覆盖；T-A11/A12 验证；如果有 tool 实际依赖 1=True 行为，由该 tool 在 schema 写 `type=["integer","boolean"]` |
| `format_unknown_tool_error` 推荐过多噪音 | 低 | top_k=3 + Levenshtein ≤ 2 限制 |
| precheck 改了 dispatch 顺序导致 PR 2.3 budget / dedup 计数错乱 | 中 | precheck 在 prepare_tool_calls **之后**，dispatch **之前**；T-E12 验证不影响 budget |
| tool_errors metadata 累计跨 turn 泄漏 | 低 | metadata 是 per-turn 的；新 turn 自动 reset；但 cumulative_session 视图需要 Sprint 5 collector 扩展（不在本 PR 范围）|
| Levenshtein 在大 tool registry 上太慢 | 低 | 实测 50 tools × 平均长度 15 ≈ 微秒级；不优化 |

## 十、与后续 PR 的衔接

- **Sprint 5 / P1-7 MessageBuilder**：MessageBuilder 在构建 API messages 时会经过 inject 内容；
  本 PR 输出的 inject 消息已经是结构化文本，对 MessageBuilder 透明。
- **Sprint 6+ 引入正式 jsonschema**：届时 `_collect_schema_issues` 替换为 jsonschema 的 issue 列表
  → 直接走 `format_schema_error_from_issues`，本 PR 接口不变。
- 与 PR 4.1/4.2/4.3 完全独立；可与 PR 4.2 并行实施。
