# P3 高级功能（按产品决定）

> 范围：阶段 4（外部 memory provider）、阶段 11.x（账单 / 凭证细分）等
>
> P3 的特点是"实现成本与产品形态强相关"，不一定每个 Aether 部署都需要。
> 本文件只列方向，不展开细节，等具体场景明确后再单独写设计文档。

---

## P3-1 外部 Memory Provider 接入（阶段 4 进阶）

### 场景
用户希望像 Hermes 那样集成 mem0 / Letta / 自研向量记忆服务，让 Agent 在每轮自动注入相关上下文。

### 设计方向

1. **`MemoryProvider` 接口**（新文件 `agents/memory/provider.py`）：
   ```python
   class MemoryProvider(Protocol):
       def on_turn_start(self, turn_count: int, user_message: str) -> None: ...
       def prefetch_all(self, query: str) -> str: ...        # 返回 fenced 上下文块
       def sync_turn(self, user: str, assistant: str) -> None: ...
       def shutdown(self) -> None: ...
   ```
2. 复用 P2-1 的 `pre_llm_call` HookOutcome 协议把 prefetch 结果注入 user message。
3. 在 turn 收尾调 `sync_turn` 写回。
4. `EngineConfig.memory_provider: MemoryProvider | None = None`。

### 与 Hermes 的差异
- Hermes 的 `_memory_manager` 同时管"prefetch + 写回 + 节奏控制"，体量大，不建议照搬。
- Aether 应只定义协议，让产品侧选实现（mem0 / 自研）。

### 验收（待产品形态）
- 注册一个 mock MemoryProvider，断言 prefetch 的内容出现在 LLM 请求里。

---

## P3-2 后台 memory / skill review fork（阶段后续）

### 场景
turn 结束后的"自我提升"流程：开一个后台线程让 LLM 自检本轮交互，自动抽 memory / skill 写到 store。

### 设计方向

1. `EngineConfig.background_review_enabled: bool = False`。
2. `_spawn_background_review(messages_snapshot, review_memory, review_skills)` 在 thread 上跑：
   - 用同一 provider（或专用小模型）发一次 review prompt。
   - 写入 memory_store / skill_store。
3. 注意：这是异步的，不能阻塞 `run_loop` 返回。

### 与生产的取舍
- 后台 review 会消耗额外 token，必须有"按钮"控制。
- 适合 CLI 场景，不适合 API 服务（用户付费却不知道）。

---

## P3-3 Trajectory 压缩（阶段 17.② 进阶）

### 场景
长会话的 trajectory 文件会很大，训练数据采集时希望压缩中段。

### 设计方向
1. 复用 `ContextCompressor`（P1-1）。
2. `_save_trajectory` 之前调 `compressor.compress(messages)` 把中段摘要化。
3. 配置 `EngineConfig.trajectory_compression: bool = False`。

---

## P3-4 多模态 image 通道（阶段 12.2 配套）

### 场景
启用 vision 工具或者直接接收 image 输入。

### 设计方向
1. `MessageBuilder` 识别 `content` 是 list 时按 provider 协议路由：
   - Anthropic：`{"type": "image", "source": {...}}`
   - OpenAI：`{"type": "image_url", "image_url": {...}}`
2. 与 P2-4 image_too_large 缩图协同。
3. 把图像转换工具（base64 ↔ data URL ↔ HTTP URL）放在 `runtime/multimodal.py`。

---

## P3-5 流式工具调用（streaming tool args）

### 场景
模型支持流式输出 tool_call.arguments（OpenAI 已支持），可以在还没全部生成完时就开始预热（例如 IDE 自动补全）。

### 设计方向
1. `ToolDescriptor` 增加 `supports_streaming_args: bool`。
2. provider 流式回调里把 partial arguments 透出给 tool runtime（仅声明支持的工具）。
3. 风险：截断 / 复制粘贴问题加剧。一般不建议在 P3 之前做。

---

## P3-6 Subagent / Delegate 子代理体系（阶段配套）

### 场景
Aether 已经在 `agents/core/agent.py` 里有 `run_subagents` / `_active_children`，但缺：
- 子代理预算限额（防止递归爆炸，已在 P0-6 cap 一级）。
- 子代理交互式追溯（parent 看不到 child 的进度）。
- 子代理跨 provider 路由。

### 设计方向
单独写设计文档，本文件只占位。

---

## P3-7 计费 / 配额 / 用量上报

### 场景
SaaS 化部署，需要实时把 token 用量上报到计费系统。

### 设计方向
1. `EngineHooks.post_api_request` 钩子（已在 P2-1 加）携带 `usage` 字段。
2. 用户在外部实现 hook，把数据推到 Stripe / 自研计费系统。
3. 引擎不内嵌任何计费逻辑（保持纯净）。

---

## P3-8 模型自适应配置（context_length / max_tokens 学习）

### 场景
不同模型 / provider 的真实上下文窗口、max_output_tokens 差异很大，需要程序化探测并缓存。

### 设计方向
1. 错误分类器从 4xx body 解析 `context_length=200000` 等数字。
2. 写到 `~/.aether/model_caps.json`。
3. `OpenAICompatibleModel.__init__` 启动时优先读取该文件初始化 `_max_output_tokens` / `context_length`。

---

## P3 总结

P3 全部不是"必须项"，建议在 Sprint 6 完成之后按以下决策树评估：

```
是否要做长期记忆？
├── 是 → P3-1（MemoryProvider 协议）→ P3-2（后台 review）
└── 否 → 跳过

是否要支持图像 / 视觉工具？
├── 是 → P3-4（多模态）→ P2-4（image shrink）配套
└── 否 → 跳过

是否做 SaaS 计费？
├── 是 → P3-7（配额上报）+ P1-4（token 累计）联动
└── 否 → 跳过

是否要训练数据 / Replay？
├── 是 → P3-3（trajectory 压缩）+ P2-8（trajectory 落盘）
└── 否 → 跳过
```

下一步：阅读 [06_aether_structural_issues.md](./06_aether_structural_issues.md) 看动手前必须先修的"地基"问题。
