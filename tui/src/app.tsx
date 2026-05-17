import { homedir } from 'node:os'
import { basename, dirname, resolve } from 'node:path'
import { readdir, readFile } from 'node:fs/promises'

import { Box, Text, useApp, useStdout } from 'ink'
import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import { ActivityBar } from './components/ActivityBar.js'
import { Banner } from './components/Banner.js'
import { ChatTranscript } from './components/ChatTranscript.js'
import { Composer } from './components/Composer.js'
import { ReasoningLine } from './components/ReasoningLine.js'
import type { GatewayClient } from './gatewayClient.js'
import type { JsonObject, SessionInfo } from './gatewayTypes.js'
import { useGatewayEvents } from './hooks/useGatewayEvents.js'
import { useReverseRpc } from './hooks/useReverseRpc.js'
import { augmentSystemPrompt } from './lib/environment.js'
import { stripToolBlocks } from './lib/phantomTool.js'
import { envBaseUrl, resolveDiscoveredBaseUrl } from './lib/sessionBaseUrl.js'
import { OverlayFrame } from './overlays/OverlayFrame.js'
import { activityActions } from './store/activityStore.js'
import { chatActions, chatItems } from './store/chatStore.js'
import { composerActions } from './store/composerStore.js'
import {
  OVERLAY_PRIORITY,
  overlayActions,
  overlayStack,
  snapshotHasKind
} from './store/overlayStore.js'
import { sessionActions, sessionState } from './store/sessionStore.js'
import {
  dispatchSlash,
  isSlashCommandLine,
  type CommandsCatalogResult,
  type PrefsGetResult,
  type ProvidersModelsResult,
  type SessionCreateResult,
  type SessionCurrentResult,
  type SessionListResult,
  type SessionResumeResult,
  type SessionUpdateResult,
  type SlashCtx,
  type SlashResult,
  type ToolsListResult
} from './slash/dispatcher.js'

export function App({
  client,
  repoRoot,
  workspaceCwd = repoRoot
}: {
  client: GatewayClient
  repoRoot: string
  workspaceCwd?: string
}) {
  const app = useApp()
  const { stdout } = useStdout()
  const session = useStore(sessionState)
  const transcript = useStore(chatItems)
  const overlays = useStore(overlayStack)
  const [bootError, setBootError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [terminalWidth, setTerminalWidth] = useState<number>(() =>
    readTerminalWidth(stdout as NodeJS.WriteStream | undefined)
  )
  const initialSessionId = useRef(process.env.AETHER_SESSION_ID?.trim() || null)
  const hasVisibleStreamingAssistant = transcript.some(
    (item) => item.kind === 'assistant' && item.streaming && stripToolBlocks(item.text).trim().length > 0
  )
  // Mirrors Python `_has_permission_prompt` gating in `aether/cli/app.py`:
  // while a permission overlay is up the modal is the only interactable
  // surface — input box, activity bar, and reasoning excerpt are hidden so
  // the user can't accidentally split focus away from the prompt.
  const hasPermissionPrompt = snapshotHasKind(overlays, 'permission')
  const showBottomActivity =
    running && !hasVisibleStreamingAssistant && !hasPermissionPrompt

  useGatewayEvents(client)
  useReverseRpc(client)

  useEffect(() => {
    const stream = stdout as NodeJS.WriteStream | undefined
    if (!stream) {
      return
    }
    const syncWidth = () => {
      setTerminalWidth(readTerminalWidth(stream))
    }
    syncWidth()
    stream.on('resize', syncWidth)
    return () => {
      stream.off('resize', syncWidth)
    }
  }, [stdout])

  useEffect(() => {
    return () => {
      // Drain any pending overlays on unmount so dismiss callbacks (which write
      // deny responses) always run. Without this a process-wide exit could
      // leave the gateway holding pending Futures.
      overlayActions.dismissAll('cancel')
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    async function boot() {
      try {
        sessionActions.setStatus('starting')
        activityActions.setStatus('starting')
        await client.start({
          cwd: workspaceCwd,
          pythonSrcRoot: process.env.AETHER_PYTHON_SRC_ROOT ?? repoRoot
        })
        const catalog = await client.request<CommandsCatalogResult>('commands.catalog')
        if (cancelled) {
          return
        }
        sessionActions.setCatalog(catalog.commands)

        const current = await client.request<SessionCurrentResult>('session.current')
        if (cancelled) {
          return
        }
        if (current.info) {
          const info = await normaliseSessionInfo(current.info)
          if (cancelled) {
            return
          }
          sessionActions.setSession(info)
        } else if (hasAutoResumeFlag()) {
          await maybeAutoResume(client, normaliseSessionInfo)
        } else {
          await createSession()
        }
        const bannerData = await loadBannerData(client, repoRoot, workspaceCwd)
        if (cancelled) {
          return
        }
        sessionActions.setBannerData(bannerData)
        sessionActions.setStatus('idle')
        // The activity bar reads from its own store — keep it in lockstep so
        // it does not get stuck on the initial 'starting' value after boot
        // completes (until then the spinner reads as "the gateway hasn't
        // come up yet"). Subsequent transitions go through the gateway's
        // own status events as a turn progresses.
        activityActions.setStatus('idle')
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error)
        setBootError(message)
        chatActions.pushNote(message, 'error')
        sessionActions.setStatus('idle')
        activityActions.setStatus('error', message)
      }
    }
    void boot()
    return () => {
      cancelled = true
    }
  }, [client, repoRoot, workspaceCwd])

  // Wire the persistent on-disk REPL history once on mount. Tests can opt
  // out by attaching their own shim before mounting.
  useEffect(() => {
    composerActions.attachHistoryFile()
  }, [])

  // Honour `--verbose` / `AETHER_VERBOSE=1` so a user who asked for noisy
  // diagnostics from the launcher gets them from the first turn instead of
  // having to retype `/verbose`. Idempotent — only flips the flag if it's
  // not already on.
  useEffect(() => {
    if (process.env.AETHER_VERBOSE === '1') {
      const enabled = chatActions.toggleVerbose()
      if (!enabled) {
        chatActions.toggleVerbose()
      }
    }
  }, [])

  async function discoverProviderModels(
    provider: string,
    baseUrl: string | null
  ): Promise<ProvidersModelsResult | null> {
    const params: Record<string, unknown> = { provider }
    if (baseUrl) {
      params.base_url = baseUrl
    }
    return client
      .request<ProvidersModelsResult>('providers.models', params)
      .catch(() => null)
  }

  async function normaliseSessionInfo(info: SessionInfo): Promise<SessionInfo> {
    if (info.provider !== 'openai') {
      return info
    }
    const baseUrl = info.base_url ?? envBaseUrl(info.provider)
    const discovery = await discoverProviderModels(info.provider, baseUrl)
    const nextBaseUrl = resolveDiscoveredBaseUrl(baseUrl, discovery?.discovery)
    if (!nextBaseUrl || nextBaseUrl === (info.base_url ?? null)) {
      return info
    }
    try {
      const result = await client.request<SessionUpdateResult>('session.update', {
        session_id: info.session_id,
        base_url: nextBaseUrl
      })
      return result.info
    } catch {
      return info
    }
  }

  async function createSession(input: {
    provider?: string
    model?: string
    system?: string | null
    sessionId?: string | null
    baseUrl?: string | null
  } = {}): Promise<SessionInfo> {
    const current = sessionState.get()
    const provider = input.provider || current.provider || process.env.AETHER_PROVIDER || 'openai'
    const rawBaseUrl = input.baseUrl ?? current.baseUrl ?? envBaseUrl(provider)
    const discovery = await discoverProviderModels(provider, rawBaseUrl)
    const baseUrl =
      provider === 'openai'
        ? resolveDiscoveredBaseUrl(rawBaseUrl, discovery?.discovery)
        : rawBaseUrl
    const model =
      input.model ||
      current.model ||
      (await resolveInitialModel(client, provider, discovery, baseUrl))
    const params: Record<string, unknown> = { provider, model }
    const requestedSessionId = input.sessionId ?? consumeInitialSessionId()
    if (requestedSessionId) {
      params.session_id = requestedSessionId
    }
    if (provider === 'openai' && baseUrl) {
      params.base_url = baseUrl
    }
    const system = input.system ?? current.systemOverride
    params.system = augmentSystemPrompt(system, workspaceCwd)
    const result = await client.request<SessionCreateResult>('session.create', params)
    sessionActions.setSession(result.info)
    return result.info
  }

  function consumeInitialSessionId(): string | null {
    const sessionId = initialSessionId.current
    initialSessionId.current = null
    return sessionId
  }

  async function handleSubmit(text: string): Promise<void> {
    if (isSlashCommandLine(text)) {
      const ctx: SlashCtx = {
        client,
        catalog: sessionState.get().catalog,
        getSession: () => sessionState.get(),
        createSession,
        setSystemOverride: sessionActions.setSystemOverride,
        toggleVerbose: chatActions.toggleVerbose
      }
      const result = await dispatchSlash(text, ctx)
      await applySlashResult(result)
      return
    }

    await submitUserMessage(text)
  }

  async function submitUserMessage(text: string): Promise<void> {
    if (running) {
      chatActions.pushNote('A turn is already running. Use Ctrl+C or /interrupt.', 'warn')
      return
    }

    setRunning(true)
    chatActions.addUserMessage(text)
    sessionActions.setStatus('thinking')
    activityActions.beginTurn()
    try {
      const info = sessionState.get().sessionId ? null : await createSession()
      const sessionId = sessionState.get().sessionId || info?.session_id
      if (!sessionId) {
        throw new Error('No active session_id')
      }
      const current = sessionState.get()
      const params: Record<string, unknown> = {
        session_id: sessionId,
        user_message: text
      }
      if (current.systemOverride) {
        params.system_override = current.systemOverride
      }
      if (current.maxIterations !== null) {
        params.max_iterations = current.maxIterations
      }
      if (current.temperature !== null) {
        params.temperature = current.temperature
      }
      if (current.maxTokens !== null) {
        params.max_tokens = current.maxTokens
      }
      if (current.disableBuiltinTools) {
        params.disable_builtin_tools = true
      }
      // agent.run blocks until the engine finishes the turn (which can take
      // minutes for tool-heavy runs and is gated by reverse-RPC prompts the
      // user themselves owns). Disabling the client-side timer mirrors the
      // Python CLI, where `EngineRequest` is awaited synchronously with no
      // wall clock. Cancellation still goes through `agent.cancel`.
      await client.request('agent.run', params, { timeoutMs: null })
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      chatActions.pushNote(message, 'error')
      sessionActions.setStatus('idle')
    } finally {
      setRunning(false)
      // Drain any messages the user queued while the turn ran. We submit them
      // sequentially so the next-turn pipeline matches the same path as a
      // fresh user submission (slash detection, system override, etc.).
      void drainQueued()
    }
  }

  async function drainQueued(): Promise<void> {
    const next = composerActions.shiftQueued()
    if (!next) {
      return
    }
    await handleSubmit(next)
  }

  async function applySlashResult(result: SlashResult): Promise<void> {
    switch (result.kind) {
      case 'note':
        chatActions.pushNote(result.text, result.level)
        break
      case 'replace-history':
        chatActions.replaceTranscript(result.messages)
        if (result.info) {
          const info = await normaliseSessionInfo(result.info)
          sessionActions.setSession(info)
          chatActions.pushNote(`resumed session ${info.session_id.slice(0, 8)}`, 'success')
        }
        break
      case 'session':
        sessionActions.setSession(await normaliseSessionInfo(result.info))
        if (result.note) {
          chatActions.pushNote(result.note, 'success')
        }
        break
      case 'clear':
        chatActions.reset()
        break
      case 'refresh':
        process.stdout.write('\u001b[2J\u001b[H')
        break
      case 'toggle-verbose':
        chatActions.pushNote(`verbose ${result.enabled ? 'on' : 'off'}`, 'success')
        break
      case 'query':
        if (result.note) {
          chatActions.pushNote(result.note, result.level ?? 'info')
        }
        await submitUserMessage(result.query)
        break
      case 'exit':
        await client.stop()
        app.exit()
        break
      case 'noop':
        break
    }
  }

  async function handleCancel(): Promise<void> {
    const current = sessionState.get()
    if (running && current.sessionId) {
      // Mirror Python `app.py:_handle_esc` — flip the visual-pending latch
      // BEFORE issuing the cancel RPC so the activity-bar spinner and any
      // streaming-state UI disappears instantly. The gateway round-trip
      // takes time; without this latch the user sees the bar still
      // "thinking" for a beat after they pressed ESC.
      activityActions.markInterruptPending()
      composerActions.clearQueued()
      await client.request('agent.cancel', { session_id: current.sessionId }).catch(() => undefined)
      chatActions.pushNote('interrupt', 'warn')
      return
    }
    await client.stop()
    app.exit()
  }

  return (
    <Box flexDirection="column" paddingX={1} width={terminalWidth}>
      <Banner />
      {bootError ? <Text color="red">{bootError}</Text> : null}
      <ChatTranscript />
      <OverlayFrame />
      {showBottomActivity ? (
        <Box flexDirection="column" marginTop={1} width="100%">
          <ReasoningLine />
          <ActivityBar />
        </Box>
      ) : null}
      {hasPermissionPrompt ? null : (
        <Box marginTop={showBottomActivity ? 1 : 0} width="100%">
          <Composer
            disabled={session.status === 'starting'}
            busy={running}
            onSubmit={(text) => void handleSubmit(text)}
            onCancel={() => void handleCancel()}
          />
        </Box>
      )}
    </Box>
  )
}

async function resolveInitialModel(
  client: GatewayClient,
  provider: string,
  discoveryResult?: ProvidersModelsResult | null,
  baseUrl?: string | null
): Promise<string> {
  const explicit = process.env.AETHER_MODEL
  if (explicit) {
    return explicit
  }

  const remembered = await client
    .request<PrefsGetResult<string>>('prefs.get', {
      key: `last_model_by_provider.${provider}`
    })
    .then((result) => result.value)
    .catch(() => null)
  if (remembered) {
    return remembered
  }

  const models =
    discoveryResult?.models ??
    (await client
      .request<ProvidersModelsResult>('providers.models', {
        provider,
        ...(baseUrl ? { base_url: baseUrl } : {})
      })
      .then((result) => result.models)
      .catch(() => []))
  const first = models[0]?.id
  if (first) {
    return first
  }

  return fallbackModel(provider)
}

function fallbackModel(provider: string): string {
  if (provider === 'claude') {
    return 'claude-sonnet-4-6'
  }
  if (provider === 'codex') {
    return 'gpt-5.4'
  }
  return 'gpt-4o'
}

function hasAutoResumeFlag(): boolean {
  return Boolean(process.env.AETHER_RESUME?.trim())
}

/**
 * Honour `AETHER_RESUME` env vars set by the launcher so
 * users can land directly in a resumed session: if the value is a session id
 * we restore that session; if the flag is set with no/empty value we open the
 * SessionPicker overlay.
 */
async function maybeAutoResume(
  client: GatewayClient,
  normaliseSessionInfo: (info: SessionInfo) => Promise<SessionInfo>
): Promise<void> {
  const flag = process.env.AETHER_RESUME
  if (!flag) {
    return
  }
  const trimmed = flag.trim()
  if (trimmed && trimmed !== '1') {
    try {
      const result = await client.request<SessionResumeResult>('session.resume', {
        session_id: trimmed
      })
      const info = await normaliseSessionInfo(result.info)
      sessionActions.setSession(info)
      chatActions.replaceTranscript(result.messages)
      chatActions.pushNote(`resumed session ${info.session_id.slice(0, 8)}`, 'success')
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      chatActions.pushNote(`auto-resume failed: ${message}`, 'error')
    }
    return
  }
  // Empty / `1` value → open the picker so the user can choose.
  try {
    const sessions = await client.request<SessionListResult>('session.list', { limit: 20 })
    overlayActions.push({
      kind: 'session-picker',
      id: `picker_boot_${Date.now()}`,
      payload: {
        sessions: sessions.sessions,
        async resolveResume(sessionId: string) {
          const result = await client.request<SessionResumeResult>('session.resume', {
            session_id: sessionId
          })
          return {
            info: await normaliseSessionInfo(result.info),
            messages: result.messages
          }
        },
        onResume(info: SessionInfo, messages: import('./gatewayTypes.js').TranscriptMessage[]) {
          sessionActions.setSession(info)
          chatActions.replaceTranscript(messages)
          chatActions.pushNote(
            `resumed session ${info.session_id.slice(0, 8)}`,
            'success'
          )
        }
      },
      createdAt: Date.now(),
      priority: OVERLAY_PRIORITY['session-picker'],
      onDismiss: () => undefined
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    chatActions.pushNote(`auto-resume picker failed: ${message}`, 'error')
  }
}

export function noteTextFromUsage(usage: JsonObject | undefined): string {
  if (!usage) {
    return ''
  }
  return JSON.stringify(usage)
}

function readTerminalWidth(stream: NodeJS.WriteStream | undefined): number {
  const columns = stream?.columns
  return typeof columns === 'number' && columns > 0 ? columns : 80
}

async function loadBannerData(
  client: GatewayClient,
  repoRoot: string,
  workspaceCwd: string
): Promise<{
  recentSessions: SessionInfo[]
  bannerTools: string[]
  bannerToolCount: number
  bannerSkills: string[]
  bannerSkillCount: number
}> {
  const [sessionsResult, toolsResult, skills] = await Promise.all([
    client
      .request<SessionListResult>('session.list', { limit: 4 })
      .catch(() => ({ sessions: [] })),
    client
      .request<ToolsListResult>('tools.list')
      .catch(() => ({ tools: [] })),
    discoverSkillNames(repoRoot, workspaceCwd).catch(() => [])
  ])

  const recentSessions = sessionsResult.sessions ?? []
  const toolNames = (toolsResult.tools ?? []).map((tool) => shortToolName(tool.name))

  return {
    recentSessions,
    bannerTools: toolNames.slice(0, 4),
    bannerToolCount: toolNames.length,
    bannerSkills: skills.slice(0, 4),
    bannerSkillCount: skills.length
  }
}

async function discoverSkillNames(
  repoRoot: string,
  workspaceCwd: string
): Promise<string[]> {
  const roots = Array.from(
    new Set([
      resolve(workspaceCwd, 'skills'),
      resolve(repoRoot, 'skills'),
      resolve(homedir(), '.aether', 'skills')
    ])
  )
  const names = new Set<string>()
  for (const root of roots) {
    await walkSkillRoot(root, 0, names)
  }
  return Array.from(names).sort((left, right) => left.localeCompare(right))
}

async function walkSkillRoot(
  root: string,
  depth: number,
  names: Set<string>
): Promise<void> {
  if (depth > 4) {
    return
  }
  let entries
  try {
    entries = await readdir(root, { withFileTypes: true })
  } catch {
    return
  }
  for (const entry of entries) {
    const fullPath = resolve(root, entry.name)
    if (entry.isFile() && entry.name === 'SKILL.md') {
      names.add(await readSkillName(fullPath))
      continue
    }
    if (entry.isDirectory()) {
      await walkSkillRoot(fullPath, depth + 1, names)
    }
  }
}

async function readSkillName(path: string): Promise<string> {
  try {
    const text = await readFile(path, 'utf8')
    const match = text.match(/^name:\s*(.+)\s*$/m)
    if (match?.[1]) {
      return match[1].trim().replace(/^['"]|['"]$/g, '')
    }
  } catch {
    // ignore and fall back to directory name
  }
  return basename(dirname(path))
}

function shortToolName(name: string): string {
  let normalized = (name ?? '').trim()
  if (!normalized) {
    return ''
  }
  if (normalized.includes('__')) {
    const parts = normalized.split('__')
    normalized = parts[parts.length - 1] ?? normalized
  }
  if (normalized.includes('.')) {
    const parts = normalized.split('.')
    normalized = parts[parts.length - 1] ?? normalized
  }
  if (normalized.includes(':')) {
    const parts = normalized.split(':')
    normalized = parts[parts.length - 1] ?? normalized
  }
  return normalized
}
