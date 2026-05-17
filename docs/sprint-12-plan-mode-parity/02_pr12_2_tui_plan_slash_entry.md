# PR 12.2 — TUI `/plan` Slash Entry

## 目标 / Goal

在 TS TUI 中补齐用户侧 `/plan` 入口：

- `/plan` 进入 plan mode，或在已处于 plan mode 时展示当前 plan。
- `/plan <description>` 进入 plan mode，并把 description 作为普通用户消息继续提交给 `agent.run`。
- `/plan open` 打开当前 session 的 plan artifact。
- TUI store 显示并同步当前 session mode。
- `/plan` 展示 plan path 和 editor hint，便于用户知道 artifact 的真实位置。

本 PR 依赖 PR 12.1 的 `plan.mode_get`、`plan.mode_set`、`plan.current` RPC。plan artifact 的真实文件内容由 PR 12.4 接入；本 PR 先按 wire contract 编写。

## 当前问题 / Current Problem

当前用户无法从 TUI 主动进入 plan mode。即使模型可以调用 `enter_plan_mode`，用户也没有明确、可发现、可恢复的规划入口。

缺口：

- `tui/src/slash/dispatcher.ts` 没有 `/plan` command。
- slash command 结果类型通常只支持 note / handled，无法表达 "执行 slash command 后继续提交 query"。
- session store 没有 `mode`，banner / activity 区域无法提示当前是 plan mode。
- `/plan open` 没有统一行为：无 plan、editor 不可用、打开失败都需要可读输出。

## 改动 / Changes

### 1. 新增 command 文件

新增：

```text
tui/src/slash/commands/plan.ts
```

并在：

```text
tui/src/slash/dispatcher.ts
```

注册 `planCommand`。

command 名称：

```ts
name: "/plan"
description: "Enable plan mode or view the current session plan"
```

description 与后端 catalog 保持一致，避免 `/help` 和 gateway catalog 出现两套文案。

### 2. 扩展 SlashResult 支持 query continuation

新增一种 slash result，例如：

```ts
type SlashResult =
  | { kind: "handled"; message?: string }
  | { kind: "note"; message: string }
  | { kind: "query"; query: string; message?: string };
```

dispatcher 处理 `/plan <description>` 时返回：

```ts
{ kind: "query", query: description, message: "Enabled plan mode" }
```

app 收到 `kind: "query"` 后，应像普通用户输入一样继续提交一轮 `agent.run`。这样对齐 open-claude-code 的 `shouldQuery: true` 语义：进入 plan mode 只是第一步，真正的规划请求仍作为用户消息进入 transcript。

### 3. `/plan` 行为

#### 当前不是 plan mode

流程：

1. 调用 `plan.mode_set({ session_id, mode: "plan" })`。
2. 更新 TUI session store：`mode = "plan"`。
3. transcript 输出简短 note：`Enabled plan mode`。

如果用户输入的是裸 `/plan`，到此结束。

OCC parity note：`/plan open` 在当前不是 plan mode 时也走这一支，只进入 plan mode，不打开文件；`open` 被当作保留子命令，不作为 query continuation。

#### `/plan <description>`

流程：

1. 先进入 plan mode。
2. 输出 `Enabled plan mode`。
3. 把 `<description>` 作为 query continuation 返回给 app。
4. app 继续发起普通 `agent.run`。

示例：

```text
/plan add auth flow
```

等价于：

1. session mode 切到 `plan`。
2. 用户消息 `add auth flow` 进入下一轮模型调用。

#### 当前已经是 plan mode

裸 `/plan` 不重复 set mode，而是调用 `plan.current`：

- 如果 `has_plan=true` 且有 `plan_content`，渲染当前 plan。
- 如果没有 plan 内容，输出：

```text
Already in plan mode. No plan written yet.
```

如果已经是 plan mode 但输入 `/plan <description>` 且 description 不是单独的 `open`，仍应把 description 继续提交为 query。不要因为已经处于 plan mode 就吞掉用户的新规划请求。

当前 plan 的推荐展示格式：

```text
Current Plan
/home/user/.aether/plans/7f8d4f67.md

# Plan
...

"/plan open" to edit this plan in $EDITOR
```

要求：

- plan path 必须显示。
- plan 内容原样或 markdown-rendered 均可，但不能丢失代码块和列表。
- editor hint 只有在 editor 可用时显示；不可用时不显示 hint 或显示清晰缺省提示。

#### `/plan open`

OCC parity 行为：

1. 当前不是 plan mode：只进入 plan mode，输出 `Enabled plan mode`，不打开文件。
2. 当前已经是 plan mode：调用 `plan.current`。
3. 如果 `has_plan=false`，输出：

```text
No plan written yet.
```

4. 如果 `has_plan=true` 但 `plan_path` 缺失，输出 `Could not open plan: plan path is unavailable.`。
5. 如果有 `plan_path`，调用本地 editor 打开。
6. 如果 editor 环境不可用，输出清晰提示，例如：

```text
Could not open plan: EDITOR is not configured.
```

7. 如果 editor 进程启动失败，显示失败原因，但不要退出 TUI。

命令解析注意：

- `/plan open` 是保留子命令。
- `/plan open auth flow` 应按 description 处理还是报错需要固定。推荐第一版把只有单独 `open` 视为子命令，其余内容作为 description。

如果 Aether 产品选择保留 "agent mode 下 `/plan open` 也可打开旧 artifact" 的能力，必须在测试和 help 中标注为 deliberate divergence，不要把它误写成 OCC parity。

### 4. TUI session store 增加 mode

扩展 session state：

```ts
type SessionMode = "agent" | "plan";

interface SessionState {
  ...
  mode: SessionMode;
}
```

同步来源：

- 初始连接时优先读 `session.current.mode`，如果没有该字段则 fallback 到 `plan.mode_get`。
- `plan.mode_set` 成功后立即更新本地 store。
- `plan.current` 返回 mode 时同步本地 store。
- `exit_plan_mode` approval approve/reject 后由 PR 12.5 更新。
- `/new` / `/resume` 由 PR 12.6 更新。

### 5. Banner / activity 区域显示 plan 标识

在现有状态区域显示一个短标识，例如：

```text
plan
```

要求：

- 不占用大量横向空间。
- agent mode 下不显示或显示默认 mode，按现有 UI 风格决定。
- plan 标识不能和 model / branch / session id 文案重叠。

### 6. Help 和 completer

确保 `/help`、slash completer、command palette 如果存在，都能看到：

```text
/plan  Enable plan mode or view the current session plan
```

前端可以消费 gateway catalog，也可以本地注册；但同一 UI 中不要出现重复项。

## 测试 / Tests

TS 单测建议：

- `plan command enters plan mode`：mock RPC，执行 `/plan`，断言调用 `plan.mode_set(plan)` 并更新 session store。
- `plan command with description returns query result`：执行 `/plan add auth`，断言 result 为 `kind: "query"` 且 `query === "add auth"`。
- `plan command in plan mode renders current plan`：mock `plan.current` 返回 content，断言 transcript note 包含 plan。
- `plan command renders plan path and editor hint`：断言输出包含 `plan_path` 和 `/plan open` hint。
- `plan command in plan mode with no artifact shows empty message`：断言输出 `Already in plan mode. No plan written yet.`。
- `plan open outside plan mode only enables plan mode`：按 OCC parity，断言调用 `plan.mode_set`，不调用 editor。
- `plan open without plan is readable`：mock `has_plan=false`，断言输出 `No plan written yet.`。
- `plan open with plan invokes editor`：mock editor service，断言使用 `plan_path`。
- `plan open editor failure is readable`：editor 抛错时输出清晰错误。
- `help and completer include plan`：`/help` 和 completion source 均包含 `/plan`。

## 验收 / Acceptance

- 输入 `/plan` 后，TUI 显示 `Enabled plan mode`，状态区域出现 `plan`。
- 输入 `/plan add auth flow` 后，TUI 进入 plan mode，并立即向 engine 发起一轮内容为 `add auth flow` 的用户消息。
- 已经处于 plan mode 时输入 `/plan`，能查看最近计划或看到 no-plan 提示。
- `/plan` 展示 plan path 和 `/plan open` hint。
- `/plan open` 覆盖无 plan、有 plan、editor 失败三种情况，TUI 不崩溃。
- 如果按 OCC parity，agent mode 下 `/plan open` 只进入 plan mode；如果 Aether deliberate divergence，测试和文档都要明确。
- `/help` 和 slash completion 能发现 `/plan`。

## 不在本 PR / Deferred

- plan artifact 的实际写入和读取实现：PR 12.4。
- `exit_plan_mode` approval UI 样式和 mode transition：PR 12.5。
- resume / clear lifecycle：PR 12.6。
