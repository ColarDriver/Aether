import { Box, Text, useInput } from 'ink'
import { useMemo, useState, type ReactElement } from 'react'

import { ShellCommandPreview } from '../components/ShellCommandPreview.js'
import { DiffView } from '../lib/diffView.js'
import { reportPermissionPreviewFallback } from '../lib/permissionTelemetry.js'
import type {
  JsonObject,
  PermissionPreview,
  PermissionRequestParams,
  PermissionToolRequest
} from '../gatewayTypes.js'
import {
  commandFromArguments,
  matchesRule,
  pathFromArguments,
  permissionRulesActions,
  permissionRulesState,
  type PermissionRule
} from '../store/permissionRulesStore.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'
import { useStore } from '@nanostores/react'
import { theme } from '../lib/theme.js'
import { formatTodoPreviewLines, todosFromArgs } from '../lib/todos.js'

// Mirrors `ToolPermissionDecisionType` on the wire side. The TUI commits one
// of these four values, the gateway maps them back to the engine via
// `permission.response`.
type Choice = 'allow_once' | 'allow_session' | 'deny' | 'abort'

interface OptionRow {
  decision: Choice
  label: string
}

export function PermissionModal({
  overlay,
  maxRows
}: {
  overlay: OverlayState<PermissionRequestParams>
  maxRows?: number
}): ReactElement {
  const params = overlay.payload
  const request = params.request
  const preview: PermissionPreview | null = request.preview ?? null
  const allowSession = request.allow_session !== false

  const rulesSnapshot = useStore(permissionRulesState)
  const cachedRule = useMemo(
    () => matchesRule(request, rulesSnapshot.rules),
    [request, rulesSnapshot]
  )

  const options = useMemo<OptionRow[]>(() => {
    // Mirrors Python `_permission_options` in `aether/cli/app.py:895-914`:
    // the session-allow label is tool-specific so the user sees what they
    // are actually green-lighting.
    if (!allowSession) {
      return [
        { decision: 'allow_once', label: 'Yes' },
        { decision: 'deny', label: 'No' }
      ]
    }
    let sessionLabel: string
    if (request.tool_name === 'shell') {
      sessionLabel = 'Yes, allow this command prefix in this session'
    } else if (preview?.path) {
      sessionLabel = 'Yes, allow edits in this path during this session'
    } else {
      sessionLabel = 'Yes, allow similar calls during this session'
    }
    return [
      { decision: 'allow_once', label: 'Yes' },
      { decision: 'allow_session', label: sessionLabel },
      { decision: 'deny', label: 'No' }
    ]
  }, [allowSession, preview, request.tool_name])

  const [selectedIndex, setSelectedIndex] = useState(0)

  function commit(decision: Choice): void {
    const payload: JsonObject = { type: decision }
    if (decision === 'allow_session') {
      // The Python TUI sends `{type: "allow_session"}` with no rule and lets
      // the engine derive one via `build_session_rule_for_request`. We mirror
      // that wire shape exactly, and only synthesise the rule locally so the
      // TUI's "matched cached session rule" banner can fire on the next prompt
      // before the next permission.request arrives.
      const cachedRuleForCache = deriveSessionRule(request)
      if (cachedRuleForCache) {
        permissionRulesActions.add(cachedRuleForCache)
      }
    }
    overlayActions.dismiss(overlay.id, 'commit', payload)
  }

  useInput(
    (input, key) => {
      if (key.ctrl && input === 'c') {
        commit('abort')
        return
      }
      if (key.escape) {
        commit('abort')
        return
      }

      if (input >= '1' && input <= '9') {
        const targetIndex = Number.parseInt(input, 10) - 1
        if (targetIndex >= 0 && targetIndex < options.length) {
          const choice = options[targetIndex]
          if (choice) {
            commit(choice.decision)
          }
        }
        return
      }

      if (key.upArrow || key.leftArrow) {
        setSelectedIndex((current) => (current - 1 + options.length) % options.length)
        return
      }
      if (key.downArrow || key.rightArrow) {
        setSelectedIndex((current) => (current + 1) % options.length)
        return
      }

      if (key.return) {
        const choice = options[selectedIndex]
        if (choice) {
          commit(choice.decision)
        }
        return
      }
    },
    { isActive: true }
  )

  const titleText = preview?.title || `Use ${request.tool_name}`
  const subtitleText = preview?.subtitle || preview?.path || null
  const question = permissionQuestion(request)
  const dividerColor = theme.isColorEnabled() ? theme.palette.primaryDim : theme.color('border')
  const dividerProps = dividerColor ? { color: dividerColor } : {}
  const brandTextProps = theme.colorProps('brand')
  const accentTextProps = theme.colorProps('accent')
  const dimTextProps = theme.colorProps('dim')
  const textTextProps = theme.colorProps('text')
  const backgroundColor = theme.color('popup_bg')
  const backgroundProps = backgroundColor ? { backgroundColor } : {}
  const hasLeadIn = Boolean(request.reason || cachedRule)
  const previewRows = maxRows ? Math.max(1, maxRows - 12 - options.length) : undefined

  return (
    <Box flexDirection="column" width="100%">
      <Text {...backgroundProps} {...dividerProps}>{permissionDivider()}</Text>
      <Box flexDirection="column" paddingX={1} paddingBottom={1} {...backgroundProps}>
        {request.reason ? (
          <Box>
            <Text {...accentTextProps}>reason: </Text>
            <Text {...textTextProps}>{request.reason}</Text>
          </Box>
        ) : null}

        {cachedRule ? (
          <Box marginTop={1}>
            <Text {...dimTextProps}>
              (matched cached session rule for {cachedRule.toolName}
              {cachedRule.pathPrefix ? ` · ${cachedRule.pathPrefix}` : ''}
              {cachedRule.commandPrefix ? ` · "${cachedRule.commandPrefix}"` : ''})
            </Text>
          </Box>
        ) : null}

        <Box marginTop={hasLeadIn ? 1 : 0} flexDirection="column">
          <Text bold {...brandTextProps}>{titleText}</Text>
          {subtitleText ? <Text {...dimTextProps}>{subtitleText}</Text> : null}
        </Box>

        <PreviewBody
          preview={preview}
          request={request}
          {...(previewRows !== undefined ? { maxRows: previewRows } : {})}
        />

        <Box marginTop={1}>
          <Text bold {...textTextProps}>{question}</Text>
        </Box>

        <Box marginTop={1} flexDirection="column">
          {options.map((option, idx) => {
            const focused = idx === selectedIndex
            const marker = focused ? '› ' : '  '
            const numberLabel = `${idx + 1}.`
            if (focused) {
              return (
                <Text key={option.decision}>
                  <Text {...accentTextProps}>{marker}</Text>
                  <Text bold {...accentTextProps}>{`${numberLabel} ${option.label}`}</Text>
                </Text>
              )
            }
            return (
              <Text key={option.decision}>
                <Text {...textTextProps}>{marker}</Text>
                <Text {...textTextProps}>{`${numberLabel} ${option.label}`}</Text>
              </Text>
            )
          })}
          <Box marginTop={1}>
            <Text {...dimTextProps}>
              ↑/↓ navigate · Enter pick · 1-{options.length} quick · ESC abort
            </Text>
          </Box>
        </Box>
      </Box>
      <Text {...backgroundProps} {...dividerProps}>{permissionDivider()}</Text>
    </Box>
  )
}

function PreviewBody({
  preview,
  request,
  maxRows
}: {
  preview: PermissionPreview | null
  request: PermissionToolRequest
  maxRows?: number
}): ReactElement | null {
  if (request.tool_name === 'todo_write') {
    const todos = todosFromArgs(request.arguments)
    if (todos.length > 0) {
      const width = Math.max(20, (process.stdout?.columns ?? 100) - 4)
      return (
        <Box marginTop={1} flexDirection="column">
          {formatTodoPreviewLines(todos, {
            ascii: !theme.isUnicodeAllowed(),
            width
          })
            .slice(0, maxRows ?? 8)
            .map((line, idx) => (
              <Text key={idx} {...theme.colorProps('dim')}>{line}</Text>
            ))}
        </Box>
      )
    }
  }
  // Falsy-string-aware guard: a tool may set `diff` / `command` / `body`
  // to "" when the relevant data was structurally empty (e.g., overwriting
  // a file with identical content). Treat whitespace-only values the same
  // as null so we do not render an empty Diff/Shell block AND we do not
  // fall through to the raw-JSON escape hatch when the tool actually had
  // something to say via a sibling field.
  if (preview?.diff && preview.diff.trim().length > 0) {
    const foldThreshold = maxRows ? Math.max(3, maxRows) : undefined
    return (
      <Box marginTop={1} flexDirection="column">
        <DiffView
          diff={preview.diff}
          expanded={false}
          {...(foldThreshold !== undefined ? { foldThreshold } : {})}
        />
      </Box>
    )
  }
  if (preview?.command && preview.command.trim().length > 0) {
    return <ShellCommandPreview command={preview.command} />
  }
  if (preview?.body && preview.body.trim().length > 0) {
    return (
      <Box marginTop={1} flexDirection="column">
        {preview.body.split('\n').slice(0, maxRows ?? 8).map((line, idx) => (
          <Text key={idx} {...theme.colorProps('dim')}>{line || ' '}</Text>
        ))}
      </Box>
    )
  }
  // Fall back: dump arguments preview so the user is never asked to approve a
  // black-box request. Tools should populate at least one of
  // diff/command/body; fire a one-shot telemetry
  // signal so a regression here is observable.
  reportPermissionPreviewFallback({
    toolName: request.tool_name,
    hasPreview: preview !== null,
    previewKeys: preview
      ? (Object.keys(preview) as Array<keyof PermissionPreview>).filter(
          (k) => preview[k] != null && preview[k] !== ''
        )
      : []
  })
  const args = JSON.stringify(request.arguments ?? {}, null, 2)
  return (
    <Box marginTop={1}>
      <Text {...theme.colorProps('dim')}>{truncate(args, 600)}</Text>
    </Box>
  )
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

/**
 * Mirrors Python `_permission_question` in `aether/cli/app.py:1018-1029`.
 * Each tool gets a sentence tailored to what the user is about to authorise
 * — generic "Do you want to allow X" prose only kicks in for tools without
 * a specialised string.
 */
function permissionQuestion(request: PermissionToolRequest): string {
  const preview = request.preview ?? null
  if (request.tool_name === 'file_edit') {
    const target = basename(preview?.path) || 'this file'
    return `Do you want to make this edit to ${target}?`
  }
  if (request.tool_name === 'write_file') {
    const target = basename(preview?.path) || 'this file'
    return `Do you want to write ${target}?`
  }
  if (request.tool_name === 'shell') {
    return 'Do you want to run this command?'
  }
  return `Do you want to allow ${request.tool_name}?`
}

function basename(path: string | null | undefined): string {
  if (!path) {
    return ''
  }
  const lastSlash = path.lastIndexOf('/')
  if (lastSlash === -1) {
    return path
  }
  return path.slice(lastSlash + 1)
}

/**
 * Local mirror of Python `build_session_rule_for_request` in
 * `aether/runtime/tools/tool_permissions.py:216`. Used only to populate the
 * client-side rule cache so a subsequent same-tool prompt shows the "matched
 * cached session rule" banner before the engine's own rule kicks in.
 */
function deriveSessionRule(request: PermissionToolRequest): PermissionRule | null {
  if (request.tool_name === 'shell') {
    const command =
      request.preview?.command || commandFromArguments(request)
    const prefix = shellCommandPrefix(command)
    if (!prefix) {
      return { toolName: request.tool_name }
    }
    return { toolName: request.tool_name, commandPrefix: prefix }
  }
  const path =
    (request.preview && request.preview.path) || pathFromArguments(request)
  if (!path) {
    return { toolName: request.tool_name }
  }
  return { toolName: request.tool_name, pathPrefix: path }
}

/**
 * Mirrors Python `_shell_command_prefix` — slice off chains/pipes, keep the
 * leading word (or first two if a flag is present). Returns null for empty
 * input so the caller can fall back to a tool-name-only rule.
 */
function shellCommandPrefix(command: string | null | undefined): string | null {
  if (!command) {
    return null
  }
  const trimmed = command.trim()
  if (!trimmed) {
    return null
  }
  let head = trimmed
  for (const sep of ['&&', '||', ';', '|']) {
    const idx = head.indexOf(sep)
    if (idx !== -1) {
      head = head.slice(0, idx).trim()
    }
  }
  const parts = head.split(/\s+/).filter(Boolean)
  if (parts.length === 0) {
    return null
  }
  if (parts.length === 1) {
    return parts[0] ?? null
  }
  return `${parts[0]} ${parts[1]}`
}

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value
  }
  return `${value.slice(0, max - 3)}...`
}

function permissionDivider(): string {
  const cols = process.stdout?.columns
  // Keep one spare column so terminals do not auto-wrap a full-width rule
  // onto the next row, which visually creates a fake blank gap below it.
  const width = Number.isFinite(cols) && cols ? Math.max(20, cols - 3) : 79
  return '─'.repeat(width)
}
