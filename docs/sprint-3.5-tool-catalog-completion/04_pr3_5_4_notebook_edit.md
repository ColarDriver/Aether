# PR 3.5.4 — `NotebookEditTool`：Jupyter 单元格编辑

> **角色**：补齐对 `.ipynb` 文件的结构化编辑能力。
> Notebook 是 JSON-of-cells 结构，用 `FileEdit` 改 raw JSON 极容易破坏文件。
> 需要专门的工具理解 cell 概念。

## 一、目标

1. 实现 `NotebookEditTool`，三种 `edit_mode`：`replace` / `insert` / `delete`。
2. 单元格按 `cell_id` 定位（与 `EditNotebook` 工具的 `cell_idx` 都支持，模型选用方便的）。
3. 输出**变更摘要**给模型，不打印整 notebook（notebook JSON 可能巨大）。
4. 保护 `.ipynb` 文件结构完整性（合法 JSON、必要字段、cell IDs 唯一）。

## 二、为什么要做

### 2.1 `.ipynb` 结构

Jupyter notebook 是这样的 JSON：

```json
{
  "cells": [
    {
      "cell_type": "code",
      "id": "abc123",
      "source": ["import numpy as np\n", "x = np.array([1, 2, 3])"],
      "outputs": [],
      "execution_count": null,
      "metadata": {}
    },
    ...
  ],
  "metadata": {"kernelspec": {...}},
  "nbformat": 4,
  "nbformat_minor": 5
}
```

用 `FileEdit` 改单元格 source 时，模型需要：
1. 知道 source 是 array of strings 而非单字符串
2. 维护原本的 JSON 结构不被破坏
3. 处理 escape（quote、newline、backslash）

**容易出错**。专门的 NotebookEditTool 处理 JSON 解析 / 序列化，模型只关心 cell 内容。

### 2.2 实际工作流场景

数据科学 / ML 用户大量使用 notebook：
* 加新 cell 跑测试代码
* 改某 cell 修 bug 重跑
* 删错误的 cell 清理 history

## 三、设计

### 3.1 输入 schema

```python
{
    "type": "object",
    "properties": {
        "notebook_path": {
            "type": "string",
            "description": "Path to the .ipynb file (absolute or relative to cwd).",
        },
        "edit_mode": {
            "type": "string",
            "enum": ["replace", "insert", "delete"],
            "default": "replace",
            "description": "Operation to perform on the cell.",
        },
        "cell_id": {
            "type": "string",
            "description": (
                "Cell ID for replace/delete; for insert, new cell is "
                "inserted AFTER this cell. Either cell_id OR cell_idx "
                "must be provided."
            ),
        },
        "cell_idx": {
            "type": "integer",
            "description": (
                "Zero-based cell index — alternative to cell_id. "
                "Useful when models can see the index from a previous read."
            ),
        },
        "new_source": {
            "type": "string",
            "description": "New cell content (for replace/insert).",
        },
        "cell_type": {
            "type": "string",
            "enum": ["code", "markdown"],
            "description": (
                "Required for insert; for replace, defaults to current cell type."
            ),
        },
    },
    "required": ["notebook_path", "edit_mode"],
}
```

### 3.2 算法

```python
class NotebookEditTool(ToolExecutor):
    NAME = "notebook_edit"
    MAX_RESULT_CHARS = 60_000

    def execute(self, call, context):
        path = self._resolve_path(call.arguments["notebook_path"])
        mode = call.arguments.get("edit_mode", "replace")
        cell_id = call.arguments.get("cell_id")
        cell_idx = call.arguments.get("cell_idx")

        # 1. 校验扩展名
        if path.suffix != ".ipynb":
            return ToolResult(
                content=f"not a notebook file (expected .ipynb): {path}",
                is_error=True,
            )

        # 2. 读 + 解析
        if not path.exists():
            return ToolResult(content=f"file not found: {path}", is_error=True)
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return ToolResult(
                content=f"invalid JSON in notebook: {exc}",
                is_error=True,
            )
        cells = notebook.get("cells", [])

        # 3. 定位 cell（必须二选一）
        idx = self._locate_cell(cells, cell_id=cell_id, cell_idx=cell_idx, mode=mode)
        if isinstance(idx, ToolResult):  # error
            return idx

        # 4. 应用操作
        if mode == "replace":
            cells[idx]["source"] = self._normalize_source(call.arguments["new_source"])
            new_type = call.arguments.get("cell_type")
            if new_type and new_type != cells[idx].get("cell_type"):
                cells[idx]["cell_type"] = new_type
            summary = f"replaced cell {idx} (id={cells[idx].get('id', '?')})"
        elif mode == "insert":
            new_cell = self._build_new_cell(
                source=call.arguments["new_source"],
                cell_type=call.arguments.get("cell_type") or "code",
            )
            insert_at = idx + 1 if idx >= 0 else 0
            cells.insert(insert_at, new_cell)
            summary = f"inserted new {new_cell['cell_type']} cell at position {insert_at}"
        elif mode == "delete":
            removed = cells.pop(idx)
            summary = f"deleted cell {idx} (id={removed.get('id', '?')})"
        else:
            return ToolResult(content=f"unknown edit_mode: {mode}", is_error=True)

        # 5. 写回（保留原始格式 indent=1 是 Jupyter 默认）
        notebook["cells"] = cells
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")

        return ToolResult(call_id=call.id, content=summary, is_error=False)
```

### 3.3 source 归一化

Jupyter 把 source 存为 array of strings（每行一个字符串，行尾含 `\n` 除最后一行）。
模型传单字符串方便，工具内部归一化：

```python
def _normalize_source(self, text: str) -> list[str]:
    """Convert plain text to Jupyter source array."""
    if not text:
        return []
    lines = text.splitlines(keepends=True)
    return lines
```

读取时反向（处理 `read_file .ipynb` 的输出）— 不在本工具范围。

### 3.4 cell ID 生成

新 cell 的 ID 用 8 字符 hex（与 Jupyter 默认一致）：

```python
import uuid
def _new_cell_id() -> str:
    return uuid.uuid4().hex[:8]
```

### 3.5 spill 决策

按 § 3.1 § 3.2 表中规定 `MAX_RESULT_CHARS = 60_000`。
但实际输出（变更摘要）通常 < 200 字符，spill 几乎不会触发。
保留阈值是为了防御 — 防止某些 edge case（如 v2 加 diff 输出）爆炸。

### 3.6 安全约束

* 拒绝编辑 spilled 文件（同 PR 3.5.2）
* 拒绝非 `.ipynb` 路径
* 注释保护非 cell 字段（metadata / nbformat）— 工具只动 `cells`，其他字段原样保留

## 四、文件改动清单

| 文件 | 类型 | 内容 | 行数 |
|---|---|---|---|
| `backend/harness/aether/tools/builtins/notebook_edit.py` | **新文件** | 工具主体 | ~250 |
| `backend/harness/aether/tools/builtins/__init__.py` | 修改 | 注册 | ~3 |
| `backend/harness/aether/tests/test_notebook_edit_tool.py` | **新文件** | 见 § 五 | ~300 |
| `backend/harness/aether/tests/fixtures/sample.ipynb` | **新文件** | 测试用 notebook | ~50 |

## 五、测试用例

### 5.1 测试组 A：基础三模式

| ID | 场景 | 验证 |
|---|---|---|
| **T-A1** | replace 第 0 个 cell（按 cell_idx） | source 更新，cells 数量不变 |
| **T-A2** | replace 按 cell_id | 找到正确 cell，更新成功 |
| **T-A3** | insert 在第 1 个 cell 后 | 新 cell 出现在 index=2，原 cell 推后 |
| **T-A4** | insert 在头部（idx=-1） | 新 cell 出现在 index=0 |
| **T-A5** | delete 按 cell_idx | cells 数量 -1 |
| **T-A6** | delete 按 cell_id | 同 A5 |

### 5.2 测试组 B：参数校验

| ID | 场景 | 验证 |
|---|---|---|
| **T-B1** | path 不是 .ipynb | `is_error=True` |
| **T-B2** | path 不存在 | `is_error=True` |
| **T-B3** | replace 但 cell_id / cell_idx 都没给 | `is_error=True` |
| **T-B4** | cell_id 不存在 | `is_error=True` |
| **T-B5** | cell_idx 越界 | `is_error=True` |
| **T-B6** | insert 没给 cell_type | 默认 "code" |
| **T-B7** | edit_mode 是无效值 | `is_error=True` |

### 5.3 测试组 C：JSON 完整性

| ID | 场景 | 验证 |
|---|---|---|
| **T-C1** | 写回后用 nbformat 校验 | `nbformat.read()` 不抛错 |
| **T-C2** | metadata 字段保留 | edit 前后 `notebook["metadata"]` 相等 |
| **T-C3** | nbformat / nbformat_minor 保留 | 相等 |
| **T-C4** | 损坏的 notebook（非合法 JSON） | `is_error=True` |

### 5.4 测试组 D：source 归一化

| ID | 场景 | 验证 |
|---|---|---|
| **T-D1** | new_source = "a\nb\nc" | source = ["a\n", "b\n", "c"] |
| **T-D2** | new_source = "" | source = [] |
| **T-D3** | new_source = "single line" | source = ["single line"] |
| **T-D4** | unicode (中文 / emoji) | utf-8 写入正确 |

### 5.5 测试组 E：安全约束

| ID | 场景 | 验证 |
|---|---|---|
| **T-E1** | path 在 `~/.aether/tool_results/` | `is_error=True` |

### 5.6 测试组 F：cell_type 切换

| ID | 场景 | 验证 |
|---|---|---|
| **T-F1** | replace + cell_type=markdown（原是 code） | cell_type 切换为 markdown |
| **T-F2** | insert cell_type=markdown | 新 cell 是 markdown |

## 六、验收门

* [ ] 25+ case 全绿
* [ ] 真实跑：编辑一个有 5 cell 的 .ipynb，replace + insert + delete 各一次
* [ ] Jupyter 打开编辑后的 notebook 能正常显示

## 七、回滚开关

不注册即回滚。

## 八、实施顺序（建议 1 天）

| 步骤 | 时长 |
|---|---|
| 1. `notebook_edit.py` 算法 | 2.5h |
| 2. fixture sample.ipynb | 30min |
| 3. 测试 | 3h |
| 4. nbformat 校验集成 | 1h |
| 5. smoke + 回归 | 1h |

## 九、依赖

* `nbformat` — Jupyter 官方库，**仅测试用**（生产代码用 stdlib `json`）
  * 加到 `pyproject.toml` 的 `dev-dependencies` 或 `test-extras`

## 十、风险与缓解

| 风险 | 缓解 |
|---|---|
| Notebook v3 vs v4 格式差异 | v1 只支持 v4（最常见）；v3 explicit 报错 |
| outputs 字段被无意清除 | replace 模式只改 source，不动 outputs |
| 巨型 notebook（千 cell） | 性能足够（json.loads 单 GB 仍可接受） |
| cell ID 冲突（手工编辑 notebook 有重复 ID） | 重复 ID 时取第一个匹配，warn 但不报错 |
