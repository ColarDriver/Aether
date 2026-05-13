import { Box, Text, useInput } from 'ink'
import TextInput from 'ink-text-input'
import { useEffect, useMemo, useState, type ReactElement } from 'react'

import { DiffView } from '../lib/diffView.js'
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

type Choice = 'allow_once' | 'allow_session' | 'deny'

type ModalState = 'choose' | 'session-prefix'

const RISK_COLOR: Record<string, string> = {
  high: 'red',
  medium: 'yellow',
  low: 'green'
}

export function PermissionModal({
  overlay
}: {
  overlay: OverlayState<PermissionRequestParams>
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
  const initialChoice: Choice = cachedRule ? 'allow_once' : 'allow_once'

  const [choice, setChoice] = useState<Choice>(initialChoice)
  const [stage, setStage] = useState<ModalState>('choose')
  const [diffExpanded, setDiffExpanded] = useState(false)
  const [prefixDraft, setPrefixDraft] = useState<string>(() =>
    suggestPrefix(request)
  )

  function commit(decision: Choice, rule?: PermissionRule): void {
    const payload: JsonObject = { type: decision }
    if (decision === 'allow_session' && rule) {
      payload.rule = serialiseRule(rule)
      permissionRulesActions.add(rule)
    }
    overlayActions.dismiss(overlay.id, 'commit', payload)
  }

  function startSessionPrefix(): void {
    if (!allowSession) {
      // Server says session-allow is not offered for this request.
      commit('allow_once')
      return
    }
    setStage('session-prefix')
  }

  function confirmSessionPrefix(): void {
    const trimmed = prefixDraft.trim()
    const rule: PermissionRule = { toolName: request.tool_name }
    if (trimmed) {
      if (preview?.command || (request.arguments && 'command' in request.arguments)) {
        rule.commandPrefix = trimmed
      } else {
        rule.pathPrefix = trimmed
      }
    }
    commit('allow_session', rule)
  }

  useInput(
    (input, key) => {
      if (key.escape) {
        if (stage === 'session-prefix') {
          setStage('choose')
          return
        }
        commit('deny')
        return
      }

      if (stage === 'session-prefix') {
        return
      }

      if (input === 'y' || input === 'Y') {
        commit('allow_once')
        return
      }
      if (input === 'n' || input === 'N') {
        commit('deny')
        return
      }
      if ((input === 's' || input === 'S') && allowSession) {
        startSessionPrefix()
        return
      }
      if (input === 'e' || input === 'E') {
        setDiffExpanded((value) => !value)
        return
      }
      if (key.leftArrow) {
        setChoice((current) => prevChoice(current, allowSession))
        return
      }
      if (key.rightArrow) {
        setChoice((current) => nextChoice(current, allowSession))
        return
      }
      if (key.return) {
        if (choice === 'allow_session') {
          startSessionPrefix()
        } else {
          commit(choice)
        }
        return
      }
    },
    { isActive: true }
  )

  const riskColor = RISK_COLOR[request.risk?.toLowerCase() ?? ''] ?? 'magenta'
  const titleText = preview?.title || `${request.tool_name} request`

  return (
    <Box flexDirection="column" borderStyle="round" borderColor={riskColor} paddingX={1}>
      <Box>
        <Text bold color={riskColor}>
          Permission · {request.tool_name}
        </Text>
        <Text dimColor> · risk </Text>
        <Text color={riskColor}>{request.risk || 'unknown'}</Text>
        <Text dimColor> · {overlay.id}</Text>
      </Box>

      {request.reason ? (
        <Box marginTop={1}>
          <Text color="cyan">reason: </Text>
          <Text>{request.reason}</Text>
        </Box>
      ) : null}

      {cachedRule ? (
        <Box marginTop={1}>
          <Text color="green" dimColor>
            (matched cached session rule for {cachedRule.toolName}
            {cachedRule.pathPrefix ? ` · ${cachedRule.pathPrefix}` : ''}
            {cachedRule.commandPrefix ? ` · "${cachedRule.commandPrefix}"` : ''})
          </Text>
        </Box>
      ) : null}

      <Box marginTop={1} flexDirection="column">
        <Text bold>{titleText}</Text>
        {preview?.subtitle ? <Text dimColor>{preview.subtitle}</Text> : null}
      </Box>

      <PreviewBody preview={preview} request={request} expanded={diffExpanded} />

      {stage === 'choose' ? (
        <ChoiceFooter
          choice={choice}
          allowSession={allowSession}
          hasDiff={Boolean(preview?.diff)}
        />
      ) : (
        <SessionPrefixEditor
          request={request}
          draft={prefixDraft}
          onChange={setPrefixDraft}
          onConfirm={confirmSessionPrefix}
          onCancel={() => setStage('choose')}
        />
      )}
    </Box>
  )
}

function PreviewBody({
  preview,
  request,
  expanded
}: {
  preview: PermissionPreview | null
  request: PermissionToolRequest
  expanded: boolean
}): ReactElement | null {
  if (preview?.diff) {
    return (
      <Box marginTop={1} flexDirection="column">
        <DiffView diff={preview.diff} expanded={expanded} />
      </Box>
    )
  }
  if (preview?.command) {
    return (
      <Box
        marginTop={1}
        borderStyle="single"
        borderColor="gray"
        paddingX={1}
      >
        <Text>$ {preview.command}</Text>
      </Box>
    )
  }
  if (preview?.body) {
    return (
      <Box marginTop={1} flexDirection="column">
        {preview.body.split('\n').map((line, idx) => (
          <Text key={idx}>{line || ' '}</Text>
        ))}
      </Box>
    )
  }
  // Fall back: dump arguments preview so the user is never asked to approve a
  // black-box request.
  const args = JSON.stringify(request.arguments ?? {}, null, 2)
  return (
    <Box marginTop={1}>
      <Text dimColor>{truncate(args, 600)}</Text>
    </Box>
  )
}

function ChoiceFooter({
  choice,
  allowSession,
  hasDiff
}: {
  choice: Choice
  allowSession: boolean
  hasDiff: boolean
}): ReactElement {
  const formatLabel = (value: Choice, label: string, hint: string): ReactElement => {
    const focused = value === choice
    const focusProps = focused ? { color: 'cyan' } : {}
    return (
      <Text key={value}>
        {focused ? <Text color="cyan">› </Text> : <Text>  </Text>}
        <Text {...focusProps}>[{hint}]</Text>{' '}
        <Text {...focusProps}>{label}</Text>
      </Text>
    )
  }
  return (
    <Box marginTop={1} flexDirection="column">
      {formatLabel('allow_once', 'allow once', 'Y')}
      {allowSession ? formatLabel('allow_session', 'allow for session…', 'S') : null}
      {formatLabel('deny', 'deny', 'N')}
      <Box marginTop={1}>
        <Text dimColor>
          ←/→ navigate · Enter pick · ESC deny{hasDiff ? ' · E to expand diff' : ''}
        </Text>
      </Box>
    </Box>
  )
}

function SessionPrefixEditor({
  request,
  draft,
  onChange,
  onConfirm,
  onCancel
}: {
  request: PermissionToolRequest
  draft: string
  onChange: (value: string) => void
  onConfirm: () => void
  onCancel: () => void
}): ReactElement {
  const isCommand = Boolean(commandFromArguments(request))
  // Wire ESC back into the prefix editor explicitly — the parent `useInput`
  // already steals ESC when stage === 'session-prefix' to flip back to
  // 'choose', so we only need to expose the Enter / submit path here.
  useEffect(() => {
    return () => {
      // no-op cleanup just to keep eslint react-hooks happy if we later add
      // event subscriptions.
    }
  }, [])
  void onCancel
  return (
    <Box marginTop={1} flexDirection="column">
      <Text bold color="cyan">
        Allow for session matching:
      </Text>
      <Text>
        <Text dimColor>tool: </Text>
        <Text>{request.tool_name}</Text>
      </Text>
      <Box>
        <Text dimColor>{isCommand ? 'command prefix: ' : 'path prefix: '}</Text>
        <TextInput value={draft} onChange={onChange} onSubmit={onConfirm} />
      </Box>
      <Box marginTop={1}>
        <Text dimColor>Enter confirm · ESC back · empty allows ALL {request.tool_name}</Text>
      </Box>
    </Box>
  )
}

// ──────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────

function suggestPrefix(request: PermissionToolRequest): string {
  const command = commandFromArguments(request)
  if (command) {
    // Suggest the leading word — e.g. "ls -la /tmp" → "ls"
    const head = command.split(/\s+/, 1)[0]
    return head ?? ''
  }
  const path = pathFromArguments(request)
  if (!path) {
    return ''
  }
  // Suggest the parent directory; users almost always want directory-scoped
  // rules, not the exact file.
  const lastSlash = path.lastIndexOf('/')
  if (lastSlash === -1) {
    return ''
  }
  return path.slice(0, lastSlash + 1)
}

function nextChoice(current: Choice, allowSession: boolean): Choice {
  const order: Choice[] = allowSession
    ? ['allow_once', 'allow_session', 'deny']
    : ['allow_once', 'deny']
  const index = order.indexOf(current)
  return order[(index + 1) % order.length] ?? order[0] ?? 'deny'
}

function prevChoice(current: Choice, allowSession: boolean): Choice {
  const order: Choice[] = allowSession
    ? ['allow_once', 'allow_session', 'deny']
    : ['allow_once', 'deny']
  const index = order.indexOf(current)
  const next = (index - 1 + order.length) % order.length
  return order[next] ?? order[0] ?? 'deny'
}

function serialiseRule(rule: PermissionRule): JsonObject {
  const payload: JsonObject = {
    tool_name: rule.toolName,
    behavior: 'allow',
    scope: 'session'
  }
  if (rule.pathPrefix) {
    payload.path_prefix = rule.pathPrefix
  }
  if (rule.commandPrefix) {
    payload.command_prefix = rule.commandPrefix
  }
  if (rule.reason) {
    payload.reason = rule.reason
  }
  return payload
}

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value
  }
  return `${value.slice(0, max - 3)}...`
}
