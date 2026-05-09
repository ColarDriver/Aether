# PR 3.5.2 — `FileEditTool`：局部 search/replace 编辑

> **角色**：补齐 Aether 工具集中**最高优先级**的缺口。
> 没有它，模型只能用 `write_file` 整文件覆盖，
> 上下文消耗、版本控制噪声、误改风险全部放大。

## 一、目标

1. 实现 `FileEditTool`，支持 `old_string` → `new_string` 的局部替换。
2. 默认 **unique-match** 强约束：`old_string` 在文件中必须**恰好出现一次**，
   否则报错让模型重写更具体的 context（claude-code 同款；最关键的安全性约束）。
3. 提供 `replace_all=True` 选项，覆盖批量替换场景（典型例子：变量重命名）。
4. 输出**最小 diff 摘要**给模型（哪行改了、变更前后片段），不打印整文件。

## 二、为什么要做

### 2.1 现状

Aether 现有 `write_file` 只支持整文件覆盖：

```python
# tools/builtins/write_file.py - 现状
def execute(self, call, context):
    path = call.arguments["path"]
    content = call.arguments["content"]
    Path(path).write_text(content)
    return ToolResult(content=f"wrote {len(content)} bytes to {path}")
```

模型想改一个 200 行文件的某个函数时，必须：
1. `read_file` 读出整个 200 行（消耗 ~200 行 prompt token）
2. 在 prompt 里复述整文件 + 改动（消耗 ~400 行 completion token）
3. `write_file` 把整文件写回（消耗 ~200 行 prompt token）

**总共 ~800 行 token**，且每一步都可能引入复制错误（漏空格、改格式、误删某行）。

### 2.2 引入 FileEdit 后

模型只需：
1. `read_file` 看一次（200 行）
2. `file_edit` 调用，参数 `old_string=<10 行待改片段>` + `new_string=<10 行新片段>`
3. 工具校验、应用、返回 diff

**总共 ~230 行 token**，且工具自身校验 unique-match，模型即使想错也改不到不该改的地方。

### 2.3 claude-code 设计参考

`/workspace/open-claude-code/src/tools/FileEditTool/FileEditTool.ts`（625 行）。
核心算法在 `utils.ts`：

* `findActualString(content, search)` — 先精确匹配，失败则尝试**引号归一化**（curly/straight）后匹配
* unique-match 校验 — 如果 `content.split(search).length > 2`（有多于一处匹配）报错
* `applyEditToFile` — 应用替换后写回
* `getPatchForEdit` — 生成 unified diff 给模型

我们 v1 实现**不做引号归一化**（claude-code 那部分 ~150 行，复杂度高且边界 case 多），
只做最核心的精确 unique-match。后续如有真实需求再加。

## 三、设计

### 3.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to the file (absolute or relative to cwd).",
        },
        "old_string": {
            "type": "string",
            "description": (
                "The exact string to replace. Must appear exactly once "
                "in the file unless replace_all=true. Include enough "
                "surrounding context (3-5 lines before and after the "
                "actual change) to make it unique."
            ),
        },
        "new_string": {
            "type": "string",
            "description": "The replacement string.",
        },
        "replace_all": {
            "type": "boolean",
            "default": False,
            "description": (
                "When true, replace ALL occurrences of old_string. "
                "Use only for variable / symbol renames where every "
                "occurrence should change identically."
            ),
        },
    },
    "required": ["path", "old_string", "new_string"],
}
```

### 3.2 算法

```python
def execute(self, call, context):
    path = self._resolve_path(call.arguments["path"])
    old_string = call.arguments["old_string"]
    new_string = call.arguments["new_string"]
    replace_all = bool(call.arguments.get("replace_all", False))

    # 1. 读文件
    if not path.exists():
        return ToolResult(
            content=f"file not found: {path}",
            is_error=True,
        )
    original = path.read_text(encoding="utf-8")

    # 2. 防自环（new_string == old_string 是 no-op）
    if old_string == new_string:
        return ToolResult(
            content="old_string and new_string are identical — no change applied",
            is_error=True,
        )

    # 3. 匹配 + 唯一性校验
    occurrences = original.count(old_string)
    if occurrences == 0:
        return ToolResult(
            content=(
                f"old_string not found in {path}. "
                "Read the file first and ensure old_string is an exact match "
                "(preserve all whitespace and indentation)."
            ),
            is_error=True,
        )
    if not replace_all and occurrences > 1:
        return ToolResult(
            content=(
                f"old_string matches {occurrences} places in {path}. "
                "Either expand old_string with more surrounding context to "
                "make it unique, or set replace_all=true if every match "
                "should change identically."
            ),
            is_error=True,
        )

    # 4. 应用替换
    if replace_all:
        modified = original.replace(old_string, new_string)
        change_count = occurrences
    else:
        modified = original.replace(old_string, new_string, 1)
        change_count = 1

    # 5. 写回
    path.write_text(modified, encoding="utf-8")

    # 6. 生成 diff 摘要（输出对模型可读）
    summary = self._build_diff_summary(
        path=path,
        original=original,
        modified=modified,
        old_string=old_string,
        new_string=new_string,
        change_count=change_count,
    )
    return ToolResult(call_id=call.id, content=summary, is_error=False)
```

### 3.3 diff 摘要格式

为了让模型清晰地"知道改成了什么"，但又不打印整文件，输出固定格式：

```
edited <path> (1 change)

@@ around line 42 @@
- def old_function(x):
-     return x + 1
+ def new_function(x):
+     return x + 2
```

`replace_all` 模式下显示 `(N changes)` 和最多 3 处变更摘要 + `... (N more changes elided)`。

实现使用 Python `difflib.unified_diff`：

```python
def _build_diff_summary(self, *, path, original, modified, old_string, new_string, change_count):
    import difflib
    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=str(path),
        tofile=str(path),
        n=2,  # 2 lines of context
    ))
    # 截断 diff（避免 replace_all 巨型变更刷屏）
    if len(diff_lines) > 80:
        diff_lines = diff_lines[:80] + [f"\n... ({len(diff_lines) - 80} more diff lines elided) ...\n"]
    header = f"edited {path} ({change_count} change{'s' if change_count != 1 else ''})\n"
    return header + "".join(diff_lines)
```

### 3.4 spill 决策

**不 spill**。输出是 diff 摘要，本身就是压缩过的内容。如果 diff > `MAX_RESULT_CHARS`（极端 replace_all），按上面的截断策略走。

### 3.5 安全约束

* **不允许**修改 spilled 文件目录（`~/.aether/tool_results/`） — 防止模型误改 spill 缓存
* **不允许**编辑超过 1 GiB 的文件（claude-code 同款；性能保护）
* `path` 必须存在 — 不存在时建议模型用 `write_file` 创建

```python
spilled_root = (Path.home() / ".aether" / "tool_results").resolve()
try:
    if path.resolve().is_relative_to(spilled_root):
        return ToolResult(
            content=f"refusing to edit cached spill file: {path}",
            is_error=True,
        )
except (ValueError, OSError):
    pass

if path.stat().st_size > 1024 * 1024 * 1024:
    return ToolResult(
        content=f"file too large to edit ({path.stat().st_size} bytes); use shell sed instead",
        is_error=True,
    )
```

## 四、文件改动清单

| 文件 | 类型 | 内容 | 行数 |
|---|---|---|---|
| `backend/harness/aether/tools/builtins/file_edit.py` | **新文件** | `FileEditTool` + 算法 + diff 生成 | ~200 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | 注册 `FileEditTool` 到 `build_default_tool_registry` | ~3 |
| `backend/harness/aether/tests/test_file_edit_tool.py` | **新文件** | 见 § 五 | ~350 |

## 五、测试用例

### 5.1 测试组 A：基础替换

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | 单次出现 + replace_all=False | 替换 1 次；diff 含 `(1 change)` |
| **T-A2** | 单次出现 + replace_all=True | 替换 1 次；行为同 A1 |
| **T-A3** | 0 次出现 | `is_error=True`；提示"not found" |
| **T-A4** | 多次出现 + replace_all=False | `is_error=True`；提示"matches N places" |
| **T-A5** | 多次出现 + replace_all=True | 全部替换；diff 含 `(N changes)` |
| **T-A6** | old_string == new_string | `is_error=True`；提示"identical — no change" |

### 5.2 测试组 B：安全约束

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | 编辑 spilled 文件（path 在 `~/.aether/tool_results/`） | `is_error=True`；提示"refusing to edit cached spill" |
| **T-B2** | 文件不存在 | `is_error=True`；提示"file not found" |
| **T-B3** | 文件 > 1 GiB（mock stat）| `is_error=True`；提示"too large" |

### 5.3 测试组 C：edge cases

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | old_string 跨多行 | 替换正确，diff 显示完整范围 |
| **T-C2** | new_string 包含 unicode（中文/emoji） | 写入用 utf-8，可正确读回 |
| **T-C3** | 文件无尾随 `\n`，编辑后不改变 | 保持原有行尾约定 |
| **T-C4** | Windows 行尾（`\r\n`） | 替换不破坏行尾约定 |

### 5.4 测试组 D：diff 输出

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | 简单替换 | diff 含 `+`/`-` 行；context 2 行 |
| **T-D2** | replace_all=True 替换 100 处 | diff 截断到 80 行 + `... (N more elided) ...` |
| **T-D3** | 替换在文件末尾 | diff 上下文不越界 |

### 5.5 测试组 E：与 PR 3.5.1 spill 集成

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | 编辑结果 diff 输出未 spill（默认应该不 spill） | 无 spill 文件；`tier1_spilled_count` 不变 |

## 六、验收门

* [ ] 30+ 个测试 case 全绿
* [ ] 真实跑：编辑一个 100 行 Python 文件中的某个函数 → diff 正确，文件正确
* [ ] 真实跑：模型用 `read_file` + `file_edit` 完成一次"重命名变量"任务 → replace_all 验证

## 七、回滚开关

工具是新增的，回滚 = 不注册到 `build_default_tool_registry`。
通过 `EngineConfig.use_builtin_tools=False` 也可以全部禁用。

## 八、实施顺序（建议 1.5 天）

| 步骤 | 时长 |
|---|---|
| 1. 新文件 `tools/builtins/file_edit.py` 算法主体 | 2h |
| 2. 注册到 `__init__.py` | 15min |
| 3. `tests/test_file_edit_tool.py` | 4h |
| 4. 真实模型 smoke 验证 | 1h |
| 5. 既有测试回归 | 30min |

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| 模型频繁触发 unique-match 失败 | error 文本明确提示"扩大 old_string 上下文"；不增加 retry 直接让模型自己决策 |
| Windows / Unix 行尾不一致 | T-C4 测试覆盖 |
| 大文件编辑性能差（整文件读写） | 1 GiB 上限 + `read_text` 已经是常用 path；性能不是 v1 关注点 |
| 模型不知道何时用 file_edit vs write_file | 描述里明确："use file_edit for in-place changes; use write_file only for new files or full rewrites" |

## 十、与后续 PR 的接合

* **PR 3.5.4 NotebookEditTool** 是 FileEditTool 的 Jupyter 单元格变体，可以复用本 PR 的 diff 生成
* **Sprint 4 / file history**：未来可加文件编辑历史 tracking（claude-code 有 `fileHistoryTrackEdit`）
