# PR 5.8 — `image_too_large` Shrink Recovery

## 目标

当 provider 拒绝过大的 base64 图片时，自动缩小消息中的 data URL / base64 image block 并 retry 一次。

## 设计

新增 helper：

```python
def shrink_image_parts_in_messages(
    messages: list[dict[str, Any]],
    *,
    max_base64_bytes: int = 5 * 1024 * 1024,
    target_base64_bytes: int = 4 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], ImageShrinkStats]:
    ...
```

支持节点：

- OpenAI chat style：`{"type": "image_url", "image_url": {"url": "data:image/...;base64,..."}}`。
- OpenAI responses style：`{"type": "input_image", "image_url": "data:image/...;base64,..."}`。
- Anthropic style：`{"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}`。

行为：

- 只处理 base64/data URL，不处理 http/https URL。
- 超过阈值才 shrink。
- 优先保持原格式；失败时可转 JPEG。
- shrink 后若仍不小于原图，视为未 changed。
- classifier 识别 `image_too_large` 后只 retry 一次。

## ErrorClassifier 扩展

识别典型 provider 文本：

- `image_too_large`
- `image exceeds`
- `base64 image`
- `5 MB`
- `image size`

分类为 `FailoverReason.image_too_large`，`should_retry=True`。

## 文件改动

- 新增 `runtime/image_shrink.py`。
- `runtime/error_classifier.py`：恢复 `image_too_large` reason。
- `agents/core/agent.py` 或 recovery strategy：接入 shrink retry。
- 如果项目依赖中没有 Pillow，本 PR 应把 shrink helper 设计为 optional dependency：Pillow 缺失时返回 `changed=False` 并走正常错误路径，不在 import 时崩溃。

## 参考实现

- open-claude-code 把 Anthropic base64 image hard limit 记录为 5MB，并以 raw target 约 3.75MB 做 downsample。
- Hermes 在收到 image-too-large 之后扫描 API messages，把超过 4MB 的 data URL 重编码到 4MB 以下并 retry。

Aether 推荐采用 error-triggered shrink，不在所有图片输入时主动压缩，减少质量损失。

## 测试

- `tests/runtime/test_image_shrink.py`
- 构造 6MB base64 image block，shrink 后小于 target。
- 小于阈值的图片不变。
- http/https URL 不变。
- OpenAI chat、Responses、Anthropic 三种 shape 都能处理。
- Pillow 不可用时 helper 不抛异常。
- provider 第一次抛 `image_too_large`，shrink changed 后 retry 成功。
- shrink 未 changed 时不 retry 或 retry 后按正常错误处理，避免无限循环。

## 验收门

- 只在明确 image-too-large 错误后触发。
- 原始 session history 不应被不必要污染；优先在 API-call copy 上修改。
- metadata 记录 shrink count、original/target size 区间，不记录完整 base64。
