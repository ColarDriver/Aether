import { Box, Text, useApp } from 'ink'
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
import { OverlayFrame } from './overlays/OverlayFrame.js'
import { activityActions } from './store/activityStore.js'
import { chatActions } from './store/chatStore.js'
import { composerActions } from './store/composerStore.js'
import { OVERLAY_PRIORITY, overlayActions } from './store/overlayStore.js'
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
  type SlashCtx,
  type SlashResult
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
  const session = useStore(sessionState)
  const [bootError, setBootError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const initialSessionId = useRef(process.env.AETHER_SESSION_ID?.trim() || null)

  useGatewayEvents(client)
  useReverseRpc(client)

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
          sessionActions.setSession(current.info)
        } else if (hasAutoResumeFlag()) {
          await maybeAutoResume(client)
        } else {
          const info = await createSession()
          chatActions.pushNote(`started session ${info.session_id.slice(0, 8)}`, 'info')
        }
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

  async function createSession(input: {
    provider?: string
    model?: string
    system?: string | null
    sessionId?: string | null
    baseUrl?: string | null
  } = {}): Promise<SessionInfo> {
    const current = sessionState.get()
    const provider = input.provider || current.provider || process.env.AETHER_PROVIDER || 'openai'
    const model = input.model || current.model || (await resolveInitialModel(client, provider))
    const params: Record<string, unknown> = { provider, model }
    const requestedSessionId = input.sessionId ?? consumeInitialSessionId()
    if (requestedSessionId) {
      params.session_id = requestedSessionId
    }
    // Base URL precedence mirrors Python `aether/cli/main.py`: explicit
    // launcher flag (AETHER_BASE_URL) > legacy provider env > nothing.
    const baseUrl =
      input.baseUrl ||
      current.baseUrl ||
      process.env.OPENAI_BASE_URL ||
      process.env.ANTHROPIC_BASE_URL
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
      await client.request('agent.run', params)
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
          sessionActions.setSession(result.info)
          chatActions.pushNote(`resumed session ${result.info.session_id.slice(0, 8)}`, 'info')
        }
        break
      case 'session':
        sessionActions.setSession(result.info)
        if (result.note) {
          chatActions.pushNote(result.note, 'info')
        }
        break
      case 'clear':
        chatActions.reset()
        break
      case 'refresh':
        process.stdout.write('\u001b[2J\u001b[H')
        break
      case 'toggle-verbose':
        chatActions.pushNote(`verbose ${result.enabled ? 'on' : 'off'}`, 'info')
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
      await client.request('agent.cancel', { session_id: current.sessionId }).catch(() => undefined)
      chatActions.pushNote('interrupt requested', 'warn')
      return
    }
    await client.stop()
    app.exit()
  }

  return (
    <Box flexDirection="column" paddingX={1}>
      <Banner />
      {bootError ? <Text color="red">{bootError}</Text> : null}
      <ChatTranscript />
      <OverlayFrame />
      <ReasoningLine />
      <ActivityBar />
      <Composer
        disabled={session.status === 'starting'}
        busy={running}
        onSubmit={(text) => void handleSubmit(text)}
        onCancel={() => void handleCancel()}
      />
    </Box>
  )
}

async function resolveInitialModel(client: GatewayClient, provider: string): Promise<string> {
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

  const models = await client
    .request<ProvidersModelsResult>('providers.models', { provider })
    .then((result) => result.models)
    .catch(() => [])
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
 * Honour `AETHER_RESUME` env vars set by the launcher (PR 9 territory) so
 * users can land directly in a resumed session: if the value is a session id
 * we restore that session; if the flag is set with no/empty value we open the
 * SessionPicker overlay.
 */
async function maybeAutoResume(client: GatewayClient): Promise<void> {
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
      sessionActions.setSession(result.info)
      chatActions.replaceTranscript(result.messages)
      chatActions.pushNote(`resumed session ${result.info.session_id.slice(0, 8)}`, 'info')
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
          return { info: result.info, messages: result.messages }
        },
        onResume(info: SessionInfo, messages: import('./gatewayTypes.js').TranscriptMessage[]) {
          sessionActions.setSession(info)
          chatActions.replaceTranscript(messages)
          chatActions.pushNote(
            `resumed session ${info.session_id.slice(0, 8)}`,
            'info'
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
