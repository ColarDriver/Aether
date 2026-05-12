# Memory 实现对比：Hermes vs open-claude-code

> 调研日期：2026-05-12
>
> 对比对象：
> - Hermes：`/workspace/hermes-agent`
> - open-claude-code：`/workspace/open-claude-code`
> - Aether 现有规划：`docs/run-loop-roadmap/05_p3_advanced_features.md`

---

## 结论摘要

Hermes 和 open-claude-code 的 Memory 不是同一类实现。

- **Hermes 是 provider/backend 型 Memory**：核心是 `MemoryProvider` 抽象和 `MemoryManager` 生命周期。每轮开始前由 provider 检索相关上下文，注入当前 user message；每轮结束后把 user/assistant turn 写回外部 backend。适合接 mem0、Honcho、Hindsight、自研向量服务等。
- **open-claude-code 是文件系统/prompt 型 Memory**：核心是 markdown memory directory、`MEMORY.md` 索引、frontmatter taxonomy、system prompt 指导模型直接维护文件。长期记忆主要是模型通过 Read/Edit/Write 修改本地 markdown 文件，不是 provider 回调。
- **Aether P3-1 应该借鉴 Hermes 的薄协议，而不是照搬 open-claude-code 的 memdir 产品系统**。open-claude-code 值得作为可选产品实现参考，但不应进入 run-loop core。

一句话：

> Aether core 需要的是 Hermes 式 `MemoryProvider` 协议；open-claude-code 式 markdown memory 更适合作为一个可选的 `FileMemoryProvider` 或产品层功能。

---

## Aether roadmap 当前方向

`05_p3_advanced_features.md` 已经把方向定得比较清楚：

- 新增 `agents/memory/provider.py`，定义 `MemoryProvider`。
- turn 开始前 `prefetch_all(query)`，通过 `pre_llm_call`/HookOutcome 注入 user message。
- turn 收尾 `sync_turn(user, assistant)` 写回。
- `EngineConfig.memory_provider: MemoryProvider | None = None`。

同时 roadmap 也明确写了：

- Hermes 的 `_memory_manager` 同时管 prefetch、写回和节奏控制，体量大，不建议照搬。
- Aether 应只定义协议，让产品侧选择 mem0、自研等实现。

这次对比支持这个方向。

---

## Hermes 的 Memory 实现

### 1. 两层 Memory：内置 file-backed store + 外部 provider

Hermes 有两套相关但不同的 Memory：

1. **内置 curated memory**
   - 文件：`tools/memory_tool.py`
   - 存储：`$HERMES_HOME/memories/MEMORY.md` 和 `$HERMES_HOME/memories/USER.md`
   - `MEMORY.md`：agent 的个人 notes，例如环境事实、项目约定、工具 quirks。
   - `USER.md`：用户画像，例如偏好、沟通方式、工作习惯。
   - 通过单个 `memory` tool 修改，支持 `add / replace / remove`。
   - session 开始时加载，注入 system prompt 的是 frozen snapshot。session 中写文件会立刻落盘，但不会改变当前 system prompt，以保持 prompt cache 稳定。

2. **外部 memory provider**
   - 文件：`agent/memory_provider.py`、`agent/memory_manager.py`、`plugins/memory/*`
   - 支持 mem0、Honcho、Hindsight、Supermemory、RetainDB 等 plugin。
   - 由 `memory.provider` 配置选择一个外部 provider。
   - 外部 provider 与内置 `memory` tool 可以并存。

内置 store 偏“小型人工 curated facts”；外部 provider 偏“长期语义记忆 backend”。

### 2. `MemoryProvider` 生命周期

Hermes 的 `MemoryProvider` 是一个明确的 backend 插件接口。核心方法包括：

- `is_available()`：检查配置和依赖是否可用。
- `initialize(session_id, **kwargs)`：初始化 session、平台、用户、profile 等作用域。
- `system_prompt_block()`：返回静态说明，放进 system prompt。
- `prefetch(query, session_id="")`：返回当前 turn 要注入的相关上下文。
- `queue_prefetch(query, session_id="")`：为下一轮预热。
- `sync_turn(user_content, assistant_content, session_id="")`：turn 结束后写回。
- `get_tool_schemas()`：暴露 provider 专属工具。
- `handle_tool_call(tool_name, args)`：路由 provider 工具调用。
- `shutdown()`：flush/close。

还有一些可选 hook：

- `on_turn_start(turn_number, message, **kwargs)`
- `on_session_end(messages)`
- `on_session_switch(new_session_id, ...)`
- `on_pre_compress(messages)`
- `on_memory_write(action, target, content, metadata=None)`
- `on_delegation(task, result, child_session_id=...)`

这些 optional hook 让 provider 能处理节奏、session 轮换、压缩前提取、内置 memory tool 的镜像写入和 subagent 结果观察。

### 3. `MemoryManager` 是集中调度器

`agent/memory_manager.py` 做几件事：

- 注册 providers。
- 限制只有一个 external provider，避免工具 schema 膨胀和 backend 冲突。
- 汇总 `system_prompt_block()`。
- 汇总 `prefetch()`。
- turn 结束统一 `sync_turn()` 和 `queue_prefetch()`。
- 汇总 tool schemas，并把 tool call 路由到正确 provider。
- provider 失败时不阻塞主流程，基本是 best-effort。

它还会把 provider prefetch 的内容包成：

```text
<memory-context>
[System note: ... NOT new user input ...]

...
</memory-context>
```

并有 scrubber 防止这类内部 context 泄漏到 UI 或 assistant 输出中。

### 4. Hermes run loop 的 Memory 时序

Hermes run loop 里的外部 Memory 时序大致是：

1. Agent 初始化：
   - 读取 `memory.provider`。
   - `load_memory_provider(provider_name)`。
   - `provider.is_available()` 后加入 `MemoryManager`。
   - 调用 `initialize_all(session_id, platform, hermes_home, user_id, chat_id, agent_identity, ...)`。
   - 把 provider tool schemas 加入可用工具。

2. 构建 system prompt：
   - 注入内置 `MEMORY.md` / `USER.md` frozen snapshot。
   - 追加 external provider 的 `system_prompt_block()`。

3. 每轮开始：
   - `MemoryManager.on_turn_start(turn_count, original_user_message)`。
   - `MemoryManager.prefetch_all(original_user_message)`。
   - prefetch 只调用一次，并缓存到 `_ext_prefetch_cache`，避免 tool loop 每个 iteration 重复查 memory。

4. API call 前：
   - 把 prefetch 结果 fenced 后追加到当前 user message。
   - 这是 API-call-time only，不修改 `messages` 原始数组，不落 session DB。

5. turn 完成：
   - 如果没有 interrupt，调用 `sync_all(original_user_message, final_response)`。
   - 同时 `queue_prefetch_all(original_user_message)` 预热下一轮。

6. session 结束或切换：
   - `on_session_end(messages)`。
   - `shutdown_all()` 或仅 commit 当前 session memory。

这个流程非常接近 Aether P3-1 的规划。

### 5. provider 示例

#### mem0

`plugins/memory/mem0/__init__.py` 的模式是：

- `system_prompt_block()` 告诉模型 mem0 已启用，并提供 `mem0_search / mem0_conclude / mem0_profile`。
- `queue_prefetch()` 后台线程调用 `client.search(query, filters, top_k=5)`。
- `prefetch()` 读取后台线程结果，返回 `## Mem0 Memory ...`。
- `sync_turn()` 后台线程把 user/assistant messages 交给 `client.add()`，由 mem0 server 侧抽取事实。
- 有 circuit breaker，避免连续失败拖垮主流程。

#### Honcho

`plugins/memory/honcho/__init__.py` 的模式更复杂：

- 支持 `context / tools / hybrid` 三种 recall mode。
- `context`：自动注入 context，不暴露工具。
- `tools`：不自动注入，模型必须调用 `honcho_*` tools。
- `hybrid`：自动注入加工具都可用。
- prefetch 分两层：
  - base context：peer representation/card，按 cadence 缓存刷新。
  - dialectic supplement：调用 Honcho reasoning/chat，按 cadence 和 depth 控制。
- `sync_turn()` 把 user/assistant 消息追加到 Honcho session。
- `on_memory_write()` 可以把内置 `memory` tool 对 `USER.md` 的写入镜像为 Honcho conclusion。

Honcho 是为什么 roadmap 说 “Hermes 的 `_memory_manager` 体量大，不建议照搬” 的典型例子：它把 provider 接入、节奏控制、上下文缓存、工具模式和 session scope 都纳入了实现。

---

## open-claude-code 的 Memory 实现

open-claude-code 的 Memory 不是 provider 接口，而是一组文件系统、prompt、attachment 和后台 fork agent 机制。

### 1. Auto-memory 默认启用

`build-src/src/memdir/paths.ts` 定义 `isAutoMemoryEnabled()`：

- `CLAUDE_CODE_DISABLE_AUTO_MEMORY=true` 时关闭。
- `CLAUDE_CODE_SIMPLE`/`--bare` 时关闭。
- remote 模式没有持久存储时关闭。
- `settings.json` 的 `autoMemoryEnabled` 可覆盖。
- 默认启用。

auto-memory 目录由 `getAutoMemPath()` 计算：

```text
<memoryBase>/projects/<sanitized-git-root>/memory/
```

也支持：

- `CLAUDE_COWORK_MEMORY_PATH_OVERRIDE`
- `autoMemoryDirectory`
- `CLAUDE_CODE_REMOTE_MEMORY_DIR`

### 2. 存储结构：`MEMORY.md` 是索引，topic files 才是实体

`build-src/src/memdir/memdir.ts` 生成 memory prompt。核心规则是：

- memory directory 是一个持久化 markdown 目录。
- 每条 memory 写到自己的文件，例如 `user_role.md`、`feedback_testing.md`。
- 每个 memory 文件使用 frontmatter：
  - `name`
  - `description`
  - `type`
- `MEMORY.md` 只是索引，不是 memory 本体。
- `MEMORY.md` 每行应是短链接，例如 `- [Title](file.md) - one-line hook`。
- `MEMORY.md` 总是加载到上下文，因此要限制长度。

它还定义四类 memory：

- `user`：用户角色、目标、知识背景、偏好。
- `feedback`：用户对工作方式的纠正或确认。
- `project`：项目中不可从代码/git 推导出来的背景、目标、事故、时间点。
- `reference`：外部系统指针，例如 Linear、Grafana、Slack channel。

一个关键差异是：open-claude-code 的 memory prompt 明确要求不要保存可从当前项目状态推导出的信息，例如代码结构、文件路径、架构、git history。这和 Hermes 内置 `MEMORY.md` 会记录环境事实、项目约定的倾向不同。

### 3. Memory prompt 与内容注入是两条链路

open-claude-code 把“如何维护 memory”的说明放进 system prompt：

- `constants/prompts.ts` 里有 `systemPromptSection('memory', () => loadMemoryPrompt())`。
- `loadMemoryPrompt()` 调 `buildMemoryLines('auto memory', autoDir, ...)`。
- 这部分主要是行为规则，不一定包含 `MEMORY.md` 内容。

实际 memory 文件内容通过 `utils/claudemd.ts` 进入 user context：

- `getMemoryFiles()` 读取 managed/user/project/local `CLAUDE.md`。
- 如果 auto-memory 开启，也读取 auto memory 的 `MEMORY.md` entrypoint。
- `getClaudeMds(...)` 把这些文件格式化为 “Contents of path ...”。
- `context.ts` 的 `getUserContext()` 把它作为 `claudeMd` 注入。

也就是说，open-claude-code 把 memory 视作 Claude instructions/context 文件体系的一部分，而不是一个单独 provider。

### 4. 相关 memory 文件的 query-time prefetch

在某个 feature gate 打开时，open-claude-code 不只注入 `MEMORY.md` 索引，还会按当前 query 选择相关 topic files：

- `memdir/findRelevantMemories.ts`
  - 扫描 memory 目录下 markdown 文件的 frontmatter。
  - 用 side query 让模型从 manifest 中选最多 5 个相关文件。
  - 排除已经 surfaced 或模型已经读过的文件。

- `utils/attachments.ts`
  - `startRelevantMemoryPrefetch(messages, toolUseContext)` 每个用户 turn 启动一次。
  - prefetch 不阻塞主 loop。
  - 如果结果在某个 tool iteration 后已经 ready，就把 memory 文件作为 `relevant_memories` attachment 注入。

- `query.ts`
  - turn 开始 `using pendingMemoryPrefetch = startRelevantMemoryPrefetch(...)`。
  - 每轮 tool result 后检查 prefetch 是否 settled。
  - settled 就转成 attachment message，加入 `toolResults`。

attachment 最终会被转换成 system reminder 形式的 meta user message。

这和 Hermes 的差异很大：

- Hermes 的 prefetch 是 provider 返回一段文本，直接 fenced 注入当前 user message。
- open-claude-code 的 relevant memory 是先选择文件，再把文件内容作为 attachment/system reminder 进入上下文。

### 5. 写入方式：模型直接改文件

open-claude-code 没有类似 Hermes 的 `sync_turn(user, assistant)` provider 回调作为主写入路径。

主路径是：

- system prompt 告诉模型如何写 memory 文件。
- 模型使用 Read/Edit/Write 工具直接维护 memory directory。
- `/memory` 命令只是打开 memory 文件让用户编辑。

这意味着 memory 的质量很大程度依赖 prompt 约束、文件工具权限和模型执行纪律。

### 6. 后台 `extractMemories`

`services/extractMemories/extractMemories.ts` 是一个后台补写机制：

- turn 结束后 fork 一个 agent。
- 这个 fork agent 看到父会话上下文。
- 它只允许：
  - Read/Grep/Glob
  - read-only Bash
  - memory directory 内的 Edit/Write
- 它扫描最近若干 messages，更新 auto-memory files。
- 如果主 agent 已经写过 memory 文件，则跳过，避免重复。

但在当前 checkout 中，`stopHooks.ts` 里的调用被 `false &&` 包住，也就是这条路径被 feature gate/dead-code elimination 关闭。实现存在，但不能假设当前构建实际启用。

### 7. `autoDream` 后台整理

`services/autoDream/autoDream.ts` 是另一类后台 memory consolidation：

- 按时间和 session 数量门槛触发。
- fork 一个 agent 扫描 memory dir 和 transcript。
- 合并、修剪、更新 `MEMORY.md` 索引。
- 受 `autoDreamEnabled` 和 feature config 控制。

这更像“周期性整理 memory store”，不是每轮 recall provider。

### 8. `SessionMemory` 不是长期 Memory

open-claude-code 还有 `services/SessionMemory`：

- 维护当前 session 的 `summary.md`。
- 路径：`{projectDir}/{sessionId}/session-memory/summary.md`。
- 达到 token/tool-call 阈值后，fork agent 更新 session notes。
- 用于 compaction/continuity。

这不是跨 session 的用户长期记忆，不能和 auto-memory 或 Hermes external provider 混为一谈。

### 9. 子 Agent 专属 Memory

open-claude-code 还支持 custom agent 的 memory scope：

- `user`：`<memoryBase>/agent-memory/<agentType>/`
- `project`：`<cwd>/.claude/agent-memory/<agentType>/`
- `local`：`<cwd>/.claude/agent-memory-local/<agentType>/`

当 agent definition 声明 `memory` 时：

- 自动给 agent 加 Read/Edit/Write 工具。
- 把 `loadAgentMemoryPrompt(agentType, scope)` 追加到该 agent 的 system prompt。

这是文件系统 memory 的细粒度扩展，不是 provider 抽象。

---

## 直接对比表

| 维度 | Hermes | open-claude-code |
|---|---|---|
| 核心模型 | provider/backend 型 | markdown 文件系统 + prompt 型 |
| 核心抽象 | `MemoryProvider` + `MemoryManager` | `memdir`、`CLAUDE.md` loader、attachments、fork agents |
| 长期存储 | 外部 backend，或内置 `MEMORY.md`/`USER.md` | `~/.claude/projects/<project>/memory/` 下的 markdown topic files |
| 索引 | provider 自己决定，内置 store 是 delimiter entries | `MEMORY.md` 是一行索引，topic files 存正文 |
| 读取/检索 | 每轮 `prefetch(query)` | 加载 `MEMORY.md`，可选 relevance side query 选择 topic files |
| 注入位置 | fenced `<memory-context>` 注入当前 user message | system prompt 说明 + user context + relevant memory attachments |
| 写入 | turn 结束 `sync_turn(user, assistant)`，外部 backend 抽取或保存 | 模型通过 Read/Edit/Write 改 markdown；后台 fork agent 可补写 |
| 工具 | provider 可暴露专属 tools，如 `honcho_search` | 没有 memory tool，主要使用通用文件工具 |
| 节奏控制 | provider/manager 内部控制 cadence、prefetch queue、sync thread | prompt cache、memoized file load、async attachment prefetch、后台 hooks |
| 多 backend | 支持插件，但限制同一时间一个 external provider | 没有 backend 插件概念 |
| 失败处理 | provider best-effort，不阻塞 turn | 文件读写和后台任务 best-effort，prompt 仍可继续 |
| 产品形态 | 适合“接入外部记忆服务” | 适合“本地可编辑、可审计的记忆文件系统” |

---

## 对 Aether 的设计含义

### 1. Aether core 不应该吸收 open-claude-code 的 memdir 全套

open-claude-code 的实现跨越很多系统：

- system prompt sections
- `CLAUDE.md` loader
- file permission carve-out
- attachment injection
- side query relevance selection
- background fork agent
- memory file UI
- compaction/session memory
- agent-specific memory scope

这是一套完整产品，不是一个 run-loop primitive。如果直接搬进 Aether core，会把 Memory 与文件系统、CLI 行为、Claude-specific prompt、任务 UI、feature gate 深度耦合。

### 2. Aether P3-1 应保持 Hermes 式薄接口

建议 Aether core 只提供：

```python
class MemoryProvider(Protocol):
    def on_turn_start(self, turn_count: int, user_message: str) -> None: ...
    def prefetch_all(self, query: str) -> str: ...
    def sync_turn(self, user: str, assistant: str) -> None: ...
    def shutdown(self) -> None: ...
```

并约定：

- `prefetch_all()` 返回已经格式化好的 context block，或由 engine 统一 fence。
- prefetch 结果通过现有 `pre_llm_call` HookOutcome 注入当前 user message。
- 注入是 API-call-time only，不写入 transcript。
- `sync_turn()` 只在完整 turn 成功结束后调用，interrupt/partial response 不写入。
- provider 错误永远不影响主 turn。

### 3. 可选扩展可以分层做

可以按产品需求拆成三个可选层：

#### Level 1：ExternalBackendMemoryProvider

目标：接 mem0、Letta、自研向量库。

实现：

- `initialize()`
- `prefetch_all(query)`
- `sync_turn(user, assistant)`
- `shutdown()`

这是 P3-1 的最小闭环。

#### Level 2：FileMemoryProvider

目标：借鉴 open-claude-code 的 markdown memory。

实现为 provider，而不是写入 engine core：

- 存储目录：`~/.aether/projects/<project>/memory/`
- `MEMORY.md` 作为索引。
- topic files 使用 frontmatter。
- `prefetch_all(query)` 可以先只返回 `MEMORY.md`，后续再做 relevance selection。
- `sync_turn()` 初期可 no-op，依赖显式 memory tool 或后台 review。

这能获得 open-claude-code 的可审计文件优势，同时不污染 run-loop。

#### Level 3：BackgroundReview

目标：turn 后自动抽取 memory/skill。

实现：

- 使用 roadmap P3-2 的 `_spawn_background_review(...)`。
- 由配置开关控制。
- 不阻塞 run_loop。
- 可写入 Level 1 backend 或 Level 2 markdown store。

这个层级类似 open-claude-code 的 `extractMemories`，但应该作为可选后台任务，不进入基础 provider 协议。

### 4. 不建议照搬 Hermes `MemoryManager`

Hermes `MemoryManager` 解决了很多产品问题：

- provider tool schema 路由
- 外部 provider 单例限制
- stream context scrubber
- session switch hook
- on_pre_compress
- on_delegation
- 内置 memory write mirror

这些对 Aether P3-1 来说过重。Aether 可以先做：

- 一个 provider 字段。
- 一个 prefetch 注入点。
- 一个 turn-end sync 点。
- 一个 shutdown 点。

等真实 provider 接入后，再决定是否需要 manager。

---

## 建议的 Aether 验收测试

### P3-1 最小验收

1. 注册 mock provider。
2. 用户输入 `"hello"`。
3. mock `prefetch_all("hello")` 返回 `"known user preference: terse"`。
4. 断言 LLM request 的当前 user message 中包含 memory context。
5. 断言 session transcript 不包含注入的 memory context。
6. assistant final response 后，断言 `sync_turn("hello", assistant_text)` 被调用。
7. interrupt 或异常 turn 下，断言 `sync_turn()` 不被调用。

### 可选 FileMemoryProvider 验收

1. 给定临时 memory dir 和 `MEMORY.md`。
2. `prefetch_all(query)` 返回 fenced index 内容。
3. 超长 `MEMORY.md` 被截断并带提示。
4. frontmatter topic files 可被扫描。
5. relevance selector 关闭时不做 side query。

### 可选 BackgroundReview 验收

1. turn 返回不等待 review 完成。
2. review 使用 messages snapshot，不读可变 messages。
3. review 失败只记录日志，不影响主 turn。
4. 配置关闭时不产生额外 token 消耗。

---

## 源码定位

### Hermes

- `agent/memory_provider.py`
  - `MemoryProvider` 生命周期和 optional hooks。
- `agent/memory_manager.py`
  - provider 注册、prefetch、sync、tool 路由、context fence。
- `run_agent.py`
  - provider 初始化。
  - turn start 的 `on_turn_start`/`prefetch_all`。
  - user message API-only 注入。
  - turn end 的 `sync_all`/`queue_prefetch_all`。
- `tools/memory_tool.py`
  - 内置 `MEMORY.md`/`USER.md` curated store。
- `plugins/memory/mem0/__init__.py`
  - mem0 search/add provider 示例。
- `plugins/memory/honcho/__init__.py`
  - Honcho hybrid/context/tools 模式和 cadence 示例。

### open-claude-code

- `build-src/src/memdir/paths.ts`
  - auto-memory 开关和目录解析。
- `build-src/src/memdir/memdir.ts`
  - memory prompt、`MEMORY.md` 索引规则、topic file 规则。
- `build-src/src/memdir/memoryTypes.ts`
  - `user / feedback / project / reference` taxonomy。
- `build-src/src/utils/claudemd.ts`
  - `CLAUDE.md` 和 AutoMem `MEMORY.md` 的加载、过滤、注入格式。
- `build-src/src/context.ts`
  - user context 注入。
- `build-src/src/utils/attachments.ts`
  - relevant memory prefetch 和 attachment。
- `build-src/src/memdir/findRelevantMemories.ts`
  - side query 选择相关 topic files。
- `build-src/src/services/extractMemories/extractMemories.ts`
  - 后台 fork agent 抽取 memory。
- `build-src/src/services/autoDream/autoDream.ts`
  - 后台 memory consolidation。
- `build-src/src/services/SessionMemory/*`
  - session-local summary memory。
- `build-src/src/tools/AgentTool/agentMemory.ts`
  - custom agent scoped memory。

---

## 最终建议

Aether 短期只实现 roadmap P3-1 的薄 `MemoryProvider`：

- `on_turn_start`
- `prefetch_all`
- `sync_turn`
- `shutdown`

不要在 engine core 内建：

- markdown memory directory
- relevance side query
- background extraction fork
- provider-specific tools
- memory UI
- agent scoped memory

这些都可以作为 provider 或产品层功能逐步加入。

如果未来要同时支持两种产品形态，推荐命名上分清：

- `ExternalMemoryProvider`：mem0/Letta/Honcho/自研服务。
- `FileMemoryProvider`：open-claude-code 风格 markdown memory。
- `SessionSummaryMemory`：会话内 continuity/compaction 用 summary，不属于长期 memory。

