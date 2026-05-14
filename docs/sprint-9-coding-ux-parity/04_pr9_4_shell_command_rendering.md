# PR 9.4 — Shell Command Rendering Parity

## 目标 / Goal

两件事：

1. **权限弹框** —— 把多命令 shell 拆开渲染，让 `cd /workspace/Aether && cp .env.example .env && uv sync && uv run aether` 显示成 4 行而不是 1 行（截图 `image copy 16.png`）；链式 `&&` / `||` / `;` / `|` 在分支处换行并缩进续行。
2. **聊天结果 footer** —— 在 `ToolCallPanel` 折叠头部下方加一行 `[exit 0 · 12ms]` / `[exit 1 · 39ms · stderr_lines=N]`（截图 `image copy 13.png` 是 Claude Code 的对照）。

依赖：**PR 9.1 必须先合入**（footer 需要 `result.metadata.exit_code` / `duration_ms` / `stderr_lines`）。

## 当前问题 / Current Problem

### 1. 权限弹框 shell 分支只按 `\n` 拆

`tui/src/overlays/PermissionModal.tsx:225-235`：

```typescript
if (preview?.command) {
  return (
    <Box marginTop={1} flexDirection="column">
      {preview.command.split('\n').map((line, idx) => (
        <Text key={idx}>
          <Text bold {...theme.colorProps('accent')}>{idx === 0 ? '$ ' : '  '}</Text>
          <Text {...theme.colorProps('text')}>{line || ' '}</Text>
        </Text>
      ))}
    </Box>
  )
}
```

当模型送出 `cmd1 && cmd2 && cmd3` 这种**单行**字符串时（无换行），整段命令挤成一行。

### 2. 模型有时也会送出无空格的拼接

截图 `image copy 16.png` 显示 `cd /workspace/Aether cp .env.example .env uv sync uv run aether` —— 这是模型把多条命令以**空格**拼接的失误。我们 fix 不了模型，但可以在权限层提示用户这看起来像多个命令（heuristic）。

### 3. ToolCallPanel 没有 shell footer

`tui/src/components/ToolCallPanel.tsx:28-48`：折叠状态下渲染 `▸ shell · $ cmd · ⏱ 234ms`，仅此而已。Claude Code 在第二行追加 `[exit 0 · 12ms · stderr_lines=0]`，我们没有。

### 4. Python `shell.py` 持有 exit/duration 但不出 wire

`aether/tools/builtins/shell.py:232-258` 的 `_format_output` 把 `[exit X · Ums · stderr_lines=Y]` 拼进 result content 字符串。PR 9.1 已经把 metadata 通道打开 —— 这里只需把同样的字段也写进 `ToolResult.metadata`。

## 改动 / Changes

### 1. 新组件：`ShellCommandPreview`

**新文件 `tui/src/components/ShellCommandPreview.tsx`**：

```typescript
import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'

const MAX_LINE_WIDTH = 120
const CHAIN_OPERATORS = ['&&', '||', ';', '|']

export interface ShellCommandPreviewProps {
  command: string
}

interface RenderedLine {
  prefix: string         // '$ ' 或 '  '
  text: string
  suspicious?: boolean   // 看起来像被空格拼接的多命令
}

export function ShellCommandPreview({ command }: ShellCommandPreviewProps): ReactElement {
  const lines = formatShellCommand(command)
  return (
    <Box marginTop={1} flexDirection="column">
      {lines.map((line, idx) => (
        <Text key={idx}>
          <Text bold {...theme.colorProps('accent')}>{line.prefix}</Text>
          <Text {...theme.colorProps('text')}>{line.text || ' '}</Text>
          {line.suspicious ? (
            <Text {...theme.colorProps('warning')}> ⚠</Text>
          ) : null}
        </Text>
      ))}
    </Box>
  )
}

/**
 * 解析逻辑（导出供测试）：
 *
 * 1. 先按真实换行拆。
 * 2. 对每个 segment，扫描 chain operator —— `&&` / `||` / `;` / `|`，并在 operator 后换行。
 *    operator 保留在前一行末尾（`cmd1 &&`），续行用 2 空格缩进。
 * 3. 启发式：若 segment 不含任何 chain operator 但包含 ≥3 个看起来像独立子命令的 token
 *    （首词是已知命令名 + 后续 token 中再次出现已知命令名），mark suspicious=true。
 * 4. 每行最终长度截到 MAX_LINE_WIDTH，超出用 `…` 截断。
 */
export function formatShellCommand(raw: string): RenderedLine[] {
  const out: RenderedLine[] = []
  const hardLines = raw.split('\n')
  for (let hi = 0; hi < hardLines.length; hi++) {
    const segments = splitOnChainOperators(hardLines[hi] ?? '')
    for (let si = 0; si < segments.length; si++) {
      const isFirst = hi === 0 && si === 0
      const segment = segments[si]
      const suspicious = isSuspiciousMultiCommand(segment.text)
      out.push({
        prefix: isFirst ? '$ ' : '  ',
        text: truncate(segment.text + (segment.trailing ?? ''), MAX_LINE_WIDTH),
        suspicious,
      })
    }
  }
  return out
}

interface ShellSegment {
  text: string
  trailing?: string  // ' &&' / ' ;' / 等
}

function splitOnChainOperators(line: string): ShellSegment[] {
  const segments: ShellSegment[] = []
  let cursor = 0
  while (cursor < line.length) {
    let nextIdx = -1
    let nextOp = ''
    for (const op of CHAIN_OPERATORS) {
      const idx = line.indexOf(op, cursor)
      if (idx !== -1 && (nextIdx === -1 || idx < nextIdx)) {
        nextIdx = idx
        nextOp = op
      }
    }
    if (nextIdx === -1) {
      segments.push({ text: line.slice(cursor).trim() })
      break
    }
    segments.push({
      text: line.slice(cursor, nextIdx).trim(),
      trailing: ` ${nextOp}`,
    })
    cursor = nextIdx + nextOp.length
  }
  return segments.filter((s) => s.text.length > 0)
}

const COMMON_CMD_NAMES = new Set([
  'cd','ls','cp','mv','rm','mkdir','touch','cat','echo','grep','find',
  'git','npm','yarn','pnpm','uv','python','python3','node','make','docker',
  'sed','awk','curl','wget','tar','unzip','chmod','chown','sudo',
])

function isSuspiciousMultiCommand(text: string): boolean {
  const tokens = text.split(/\s+/).filter(Boolean)
  if (tokens.length < 4) return false
  let cmdCount = 0
  for (const tok of tokens) {
    if (COMMON_CMD_NAMES.has(tok)) cmdCount++
    if (cmdCount >= 2) return true
  }
  return false
}

function truncate(value: string, max: number): string {
  if (value.length <= max) return value
  return `${value.slice(0, max - 1)}…`
}
```

### 2. 接入 PermissionModal

**`tui/src/overlays/PermissionModal.tsx:225-235`**：

```typescript
import { ShellCommandPreview } from '../components/ShellCommandPreview.js'

...

if (preview?.command && preview.command.trim().length > 0) {
  return <ShellCommandPreview command={preview.command} />
}
```

注意：`preview.command.trim().length > 0` 已在 PR 9.3 里收紧。

### 3. 新组件：`ShellResultFooter`

**新文件 `tui/src/components/ShellResultFooter.tsx`**：

```typescript
import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'
import type { ToolResultMetadata } from '../gatewayTypes.js'

export interface ShellResultFooterProps {
  metadata: ToolResultMetadata
}

export function ShellResultFooter({ metadata }: ShellResultFooterProps): ReactElement | null {
  const exitCode = typeof metadata.exit_code === 'number' ? metadata.exit_code : null
  const durationMs = typeof metadata.duration_ms === 'number' ? metadata.duration_ms : null
  const stderrLines = typeof metadata.stderr_lines === 'number' ? metadata.stderr_lines : null
  const truncated = metadata.truncated === true
  const timedOut = metadata.timed_out === true
  if (exitCode === null && durationMs === null) return null
  const parts: string[] = []
  if (exitCode !== null) parts.push(`exit ${exitCode}`)
  if (durationMs !== null) parts.push(formatDuration(durationMs))
  if (stderrLines !== null && stderrLines > 0) parts.push(`stderr_lines=${stderrLines}`)
  if (truncated) parts.push('truncated')
  if (timedOut) parts.push('timed out')
  const color = exitCode !== null && exitCode !== 0 ? 'error' : 'dim'
  return (
    <Box marginLeft={2}>
      <Text {...theme.colorProps(color)}>[{parts.join(' · ')}]</Text>
    </Box>
  )
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  const seconds = ms / 1000
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`
  const minutes = Math.floor(seconds / 60)
  const remaining = Math.round(seconds % 60)
  return `${minutes}m${remaining.toString().padStart(2, '0')}s`
}
```

### 4. 接入 ToolCallPanel

**`tui/src/components/ToolCallPanel.tsx:28-48`** —— 在折叠 `Box` 内、`InlineResultPreview` 之前插入：

```typescript
import { ShellResultFooter } from './ShellResultFooter.js'
import { categoryFor } from '../lib/toolCategory.js'

...

const isShellCategory = categoryFor(item.toolName) === 'bash'

return (
  <Box flexDirection="column" marginTop={item.iteration > 0 ? 0 : 1}>
    <Box>
      <Text color={headerColor}>▸ {item.toolName}</Text>
      ...
    </Box>
    {isShellCategory && item.result?.metadata ? (
      <ShellResultFooter metadata={item.result.metadata} />
    ) : null}
    {item.result ? <InlineResultPreview text={item.result.text} isError={isError} /> : null}
  </Box>
)
```

展开状态下同样插入 footer —— 用户展开时 `[exit 0 · 12ms]` 依然显示。

### 5. Python：让 shell 工具填 metadata

**`aether/tools/builtins/shell.py`** —— 在 `execute` 构造 `ToolResult` 处（搜索 `return ToolResult(`）扩展 metadata：

```python
return ToolResult(
    tool_call_id=call.id,
    name=call.name,
    content=formatted_output,
    is_error=(exit_code != 0),
    metadata={
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "cwd": str(cwd),
        "command": command,
        "truncated": truncated,
        "timed_out": timed_out,
        "stderr_lines": stderr_line_count,
    },
)
```

具体变量名遵从该文件现有命名（execute 体里这些值大多已经存在；多数情况只需把它们打成 dict 而不是仅拼进 content 字符串）。

`_format_output` 中现有的 `[exit X · Ums]` 文本行可以**保留**（旧客户端的兼容路径），但 TUI 优先用 metadata。

### 6. shell 权限 preview 保持原始换行

**`aether/tools/builtins/shell.py:84-101`** —— `build_permission_preview` 中 `ToolPermissionPreview.command` 字段应该传入 raw `args.command`，不要做任何 normalize（`.replace('\n', ' ')` 之类）。当前实现已经是 raw —— 加一条单测保护。

## 测试 / Tests

### TS

新建 `tui/src/__tests__/shellCommandPreview.test.tsx`：

- `formatShellCommand splits on &&` —— 输入 `'a && b && c'` 输出 3 行 `['$ a &&', '  b &&', '  c']`
- `formatShellCommand splits on |` —— 输入 `'cat x | grep y | wc -l'` 输出 3 行
- `formatShellCommand respects hard newlines first` —— 输入 `'line1\nline2 && line3'` 输出 3 行
- `formatShellCommand flags suspicious space-joined multi-command` —— 输入 `'cd /a cp x y uv sync'` 第一行 `suspicious=true`
- `formatShellCommand does NOT flag legit single command with many args` —— 输入 `'ls -la --color=auto /tmp'` 不 flag
- `formatShellCommand truncates each line at MAX_LINE_WIDTH`
- `renders $ prefix on first line and 2-space indent on continuation`

新建 `tui/src/__tests__/shellResultFooter.test.tsx`：

- `renders [exit 0 · 12ms] when metadata has both`
- `renders error color when exit_code != 0`
- `renders stderr_lines when > 0`
- `renders truncated/timed_out flags`
- `returns null when no exit_code and no duration_ms`

扩展 `tui/src/__tests__/permissionModal.test.tsx`：

- `renders ShellCommandPreview for shell tool with chained command`
- `does not render ShellCommandPreview when command is empty/null`

### Python

扩展 `aether/tests/tools/test_shell_tool.py`（或对应文件）：

- `test_shell_result_includes_exit_metadata` —— 执行 `echo ok`，`result.metadata['exit_code'] == 0`，`duration_ms > 0`
- `test_shell_result_includes_stderr_lines` —— 执行 `python -c "import sys; sys.stderr.write('a\\nb\\n')"`，`stderr_lines == 2`
- `test_shell_preview_keeps_original_newlines` —— 给 args.command 一个含 `\n` 的字符串，build_permission_preview 后 `preview.command` 等值（无修改）

## 验收 / Acceptance

- `uv run pytest aether/tests/tools` 全绿
- `npm test --prefix tui` 全绿
- 手动场景：
  1. 让模型跑 `ls -la && pwd && date` —— 权限弹框 3 行，每行 `$ `/`  ` 前缀，`&&` 留在前一行末尾
  2. 让模型跑一个失败命令（`ls /nonexistent`）—— 批准后 ToolCallPanel 折叠头下方出现红色 `[exit 2 · Nms · stderr_lines=1]`
  3. 让模型跑一个被 PR 9.4 启发式判定为可疑的命令 —— 权限弹框第一行末尾出现 `⚠`
  4. 让模型跑一个会被截断的长输出（spill triggered）—— footer 含 `truncated`
  5. 让模型跑一个超时命令 —— footer 含 `timed out`

## 不在本 PR

- 完整 shell parser（POSIX `sh` 语法、引号、heredoc）—— 启发式足够 Claude Code parity
- stdout/stderr 分块着色（仍按 ToolCallPanel 原样渲染）
- 后台进程跟踪 UI
