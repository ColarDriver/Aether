# PR 6 · TS scaffold + GatewayClient

## 摘要

新建 `tui/` 工程：Ink + React 19 + nanostores + Vitest。实现 `GatewayClient`：负责拉起 `aether-gateway` 子进程、按行解析 JSON、维护 pending request map、缓冲事件、捕获 stderr 与日志。本 PR 之后，可以从 Node 端连通 gateway 并跑通 `gateway.ping` / `commands.catalog` / `agent.run`，但还没有可视 UI（只是 minimal 文本输出）。

## Scope

In scope:

- 新建 `tui/` 顶级目录（位于 repo 根，与 `aether/` 同级）
- `package.json`、`tsconfig.json`、`tsconfig.build.json`、`vitest.config.ts`、`eslint.config.mjs`、`.prettierrc`
- `src/entry.tsx`（最小化：检查 TTY → 起 GatewayClient → 打印 `gateway.ready` 后退出，便于 smoke）
- `src/gatewayClient.ts`：子进程编排
- `src/gatewayTypes.ts`：手写镜像 PR 2~5 的 wire schema
- `src/lib/circularBuffer.ts`：复用 Hermes 设计的环形缓冲
- 单元测试（vitest）：解析、超时、断流、并发 pending、未知 id

Out of scope:

- 任何可视 UI 组件（PR 7）
- 状态管理 stores（PR 7）
- approval / permission overlay（PR 8）

## Contracts

### Directory layout

```
tui/
├── package.json
├── tsconfig.json
├── tsconfig.build.json
├── vitest.config.ts
├── eslint.config.mjs
├── .prettierrc
├── README.md
├── src/
│   ├── entry.tsx                    # process entrypoint
│   ├── gatewayClient.ts             # subprocess + RPC client
│   ├── gatewayTypes.ts              # hand-mirrored wire schema
│   ├── lib/
│   │   └── circularBuffer.ts
│   └── __tests__/
│       ├── gatewayClient.test.ts
│       └── circularBuffer.test.ts
└── scripts/
    └── (placeholders for PR 7+)
```

### `package.json` 关键字段

```json
{
  "name": "tui",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "tsx --watch src/entry.tsx",
    "start": "tsx src/entry.tsx",
    "build": "tsc -p tsconfig.build.json && chmod +x dist/entry.js",
    "type-check": "tsc --noEmit -p tsconfig.json",
    "lint": "eslint src/",
    "fmt": "prettier --write 'src/**/*.{ts,tsx}'",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "ink": "^6.0.0",
    "ink-text-input": "^6.0.0",
    "react": "^19.0.0",
    "nanostores": "^1.0.0",
    "@nanostores/react": "^1.0.0"
  },
  "devDependencies": {
    "tsx": "^4.0.0",
    "typescript": "^5.7.0",
    "@types/node": "^25.0.0",
    "@types/react": "^19.0.0",
    "vitest": "^4.0.0",
    "eslint": "^9.0.0",
    "prettier": "^3.0.0"
  }
}
```

### `GatewayClient` API

```ts
export type StartOptions = {
  pythonPath?: string             // override; else auto-resolve venv → python3
  cwd?: string                    // override; else process.cwd()
  pythonSrcRoot?: string          // sets PYTHONPATH + HERMES-style guard
  env?: NodeJS.ProcessEnv
  startupTimeoutMs?: number       // default 15000
  requestTimeoutMs?: number       // default 120000
}

export class GatewayClient extends EventEmitter {
  start(opts?: StartOptions): Promise<void>            // resolves after gateway.ready
  stop(opts?: { signal?: NodeJS.Signals; graceMs?: number }): Promise<void>
  request<T = unknown>(method: string, params?: object): Promise<T>
  notify(method: string, params?: object): void        // for future use; mostly unused
  respond(id: string, result: unknown): void           // for server-initiated requests
  subscribe(): AsyncIterable<GatewayEvent>             // iterates buffered + future events
  on(event: 'event', cb: (ev: GatewayEvent) => void): this
  on(event: 'log', cb: (line: string) => void): this
  on(event: 'exit', cb: (code: number | null) => void): this
  readonly logs: readonly string[]                     // ring buffer
}
```

### Python 解析器顺序

```ts
const resolvePython = (root: string) => {
  const explicit = process.env.AETHER_PYTHON?.trim() || process.env.PYTHON?.trim()
  if (explicit) return explicit

  const venv = process.env.VIRTUAL_ENV?.trim()
  const candidates = [
    venv && resolve(venv, 'bin/python'),
    venv && resolve(venv, 'Scripts/python.exe'),
    resolve(root, '.venv/bin/python'),
    resolve(root, '.venv/bin/python3'),
    resolve(root, 'venv/bin/python'),
    resolve(root, 'venv/bin/python3')
  ].filter(Boolean)
  for (const c of candidates) if (existsSync(c)) return c
  return process.platform === 'win32' ? 'python' : 'python3'
}
```

### Wire schema mirror（`src/gatewayTypes.ts`）

每个 Python Pydantic model 对应一个 TS interface，命名一致。例：

```ts
export interface TextDelta {
  type: 'text.delta'
  text: string
  sequence: number
  session_id?: string
  run_id?: string
}

export interface ToolCall { ... }
// ... 其它

export type GatewayEvent =
  | { type: 'gateway.ready'; version: string; capabilities: string[] }
  | { type: 'gateway.error'; message: string; where?: string }
  | TextDelta
  | ToolCall
  | ToolResult
  | Reasoning
  | IterationStart
  | IterationEnd
  | LoopStateChanged
  | Status
  | TokenUsage
  | { type: 'gateway.stderr'; line: string }                // synthetic, emitted by client
  | { type: 'gateway.protocol_error'; raw: string; reason: string }   // synthetic

export type ApprovalRequestParams = { ... }
export type PermissionRequestParams = { ... }
```

`type` 字段是 discriminant，TS 端用 switch 收窄。

## 设计要点

**为什么放在 `tui/`（与 `aether/` 同级）而不是 `aether/tui/`。** TS 工程要有独立的 `package.json`、构建产物、依赖树，放在 Python 包内会让 `setuptools` / `uv` 误把 `node_modules` 当成 package data。Hermes 也用 `ui-tui/` 同级。

**为什么用 Ink 6 + React 19。** 这是 Hermes 实际跑过的组合，React 19 的 compiler 让组件性能更好。`tsx` 当 dev runner，`tsc` build。

**为什么没有 React DevTools / hot reload。** Ink 在终端里，DevTools 接不上；hot reload 靠 `tsx --watch` 重启进程。本 PR 不做更复杂的方案。

**`subscribe()` 的事件回放语义。** `start()` 之前发出的事件先进环形 buffer（容量 ~2000）。首次 `subscribe()` 把 buffer 全部 drain 出去，之后切换成实时推送。这是 Hermes 的设计，避免 UI 启动竞态丢事件。

**`stop()` 的 grace。** SIGTERM → 等 `gateway.exit` 事件 / 进程退出，超时（默认 2s）SIGKILL。Hermes 在测试里见过 Python `atexit` 卡几百 ms，2s 留足空间。

**`logs` 的两路。** Python stderr 每行进环形 buffer，同时作为合成事件 `gateway.stderr` emit 给 UI（PR 8 的活动条可显示最近一条）。stdout 的非 JSON 行（理论上不存在）emit 为 `gateway.protocol_error`，方便排错。

**Type 镜像策略。** 一开始手写。新增 wire 字段时人工同步。PR 9 上线前在 Python 侧加 `tests/gateway/test_schema_snapshot.py`，仓库内放一份 `gateway-schema.json`；TS 测试读这份 snapshot 校验关键字段都被 TS interface 覆盖。

**为什么不直接用 `node:net` + tcp。** stdio + json-line 已经能满足 TUI 用法；TCP 增加端口冲突、firewall、隐私 surface。web UI 的 `WebSocketTransport` 在 PR 9 之后单独引入。

**为什么不打包 esbuild。** 当前用 `tsc` + `tsx` 足够，且方便调试。等 PR 9 cutover 之前再决定 bundle 策略；Hermes 用了 `babel` 是因为 `react-compiler`，我们等需要再加。

## Files touched

- new: 整个 `tui/` 目录（按上面 layout）
- modified: 仓库根 `.gitignore`（追加 `tui/node_modules`, `tui/dist`）

## Dependencies

PR 1 ~ 5 已合并：`tui` 启动后会立刻调 `gateway.ping`，期望 `commands.catalog` / `agent.run` 在 PR 7 之前已经可用以便 smoke。

## Acceptance criteria

- `cd tui && npm install && npm run type-check` 通过。
- `npm test` 通过；至少 12 个单测覆盖 GatewayClient（启动、解析、超时、断流、未知 id、并发、stderr 捕获、协议错误、stop 优雅退出、subscribe 回放、environment vars override、Python 解析器优先级）。
- `npm start` 启动后能看到 `gateway.ready` 一行，进程不退出；按 Ctrl+C 后 Python 子进程在 2s 内退出且无 zombie。
- `GatewayClient.request("gateway.ping")` 返回 `{pong: true}`。
- `GatewayClient.request("commands.catalog")` 返回完整 slash 命令清单。
- 模拟 Python 进程在启动后 1s 内退出 → client 抛出 `GatewayCrashedError` 含 stderr ring 内容。
- `GatewayClient.subscribe()` 在 `start()` 后开始监听仍然能拿到启动期 3 条以上的 `status` / `gateway.ready` 事件。

## Manual verification

```bash
cd tui
npm install
AETHER_PYTHON_SRC_ROOT="$PWD/.." npm start

# 单测
npm test

# 类型对齐 smoke：枚举所有 wire 事件 type 字段
node --import tsx -e "
import { GatewayClient } from './src/gatewayClient.js';
const c = new GatewayClient();
await c.start();
console.log('catalog:', await c.request('commands.catalog'));
await c.stop();
"
```
