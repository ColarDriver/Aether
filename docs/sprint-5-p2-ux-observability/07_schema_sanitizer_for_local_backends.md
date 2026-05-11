# PR 5.7 — Schema Sanitizer For Local Backends

## 目标

本地 OpenAI-compatible 后端，尤其 llama.cpp grammar，对 JSON schema 的 `pattern` / `format` 等字段支持不完整。该 PR 在 classifier 识别相关错误后，剥离不兼容字段并 retry 一次。

## 设计

新增 helper：

```python
def strip_pattern_and_format(tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    ...
```

规则：

- 深拷贝 tools，不原地污染 registry 中的原始 tool schema。
- 递归删除所有 dict 中 key 为 `pattern` 或 `format` 的字段。
- 返回 `(new_tools, changed)`。
- 仅在错误被分类为 `llama_cpp_grammar_pattern` 时触发。
- 每次 provider attempt 最多触发一次 sanitizer retry。

ErrorClassifier 扩展：

- 识别 llama.cpp / local backend 返回的 grammar compile error。
- 典型文本包含 `grammar`、`pattern`、`format`、`json schema`、`unsupported` 等组合。
- 分类为 `FailoverReason.llama_cpp_grammar_pattern`，`should_retry=True`，`should_fallback=False`。

## 文件改动

- 新增 `runtime/schema_sanitizer.py` 或 `tools/schema_sanitizer.py`；推荐 runtime 层，因为它服务 provider payload recovery。
- `runtime/error_classifier.py`：恢复并测试 `llama_cpp_grammar_pattern` reason。
- `runtime/recovery.py` 或 `agents/core/agent.py`：接入 sanitizer retry。
- metadata：记录 `schema_sanitizer_applied=True`、`schema_sanitizer_removed_count`。

## 参考实现

Hermes 的 `tools/schema_sanitizer.py` 做得更广，会处理 nullable unions、anyOf/oneOf、bare schema strings 等。Aether 本 PR 只做 roadmap 要求的最小闭环：剥离 `pattern` / `format` 并 retry。更激进的 schema normalization 留给后续 PR。

## 测试

- `tests/runtime/test_schema_sanitizer.py`
- schema 顶层、properties 深层、items 内部的 `pattern` / `format` 都被删除。
- 原始 tools 对象不变。
- 无相关字段时 `changed=False`。
- classifier 对 llama.cpp grammar pattern error 返回指定 reason。
- provider 第一次抛 grammar error，sanitizer retry 后成功，断言只 retry 一次。
- sanitizer 后仍失败，走正常 recovery/fallback 路径。

## 验收门

- 不改变非 llama.cpp grammar 错误路径。
- 不对所有请求预先 sanitizer，避免损失云 provider schema 约束能力。
- retry 上限明确，不能无限剥 schema 重试。
