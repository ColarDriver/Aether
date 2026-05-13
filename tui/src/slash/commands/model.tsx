import type { ModelInfo, SessionInfo } from '../../gatewayTypes.js'
import { OVERLAY_PRIORITY, overlayActions } from '../../store/overlayStore.js'
import { chatActions } from '../../store/chatStore.js'
import { sessionActions } from '../../store/sessionStore.js'
import { Box, Text } from 'ink'
import type { ReactElement } from 'react'
import type { PickerItem, PickerPayload } from '../../overlays/Picker.js'
import type {
  ModelDiscovery,
  ProvidersModelsResult,
  SessionUpdateResult,
  SlashCommand
} from '../dispatcher.js'

export const modelCommand: SlashCommand = {
  name: '/model',
  category: 'remote',
  async execute(args, ctx) {
    const state = ctx.getSession()
    const provider = state.provider || process.env.AETHER_PROVIDER || 'openai'
    const target = args[0]?.trim()

    const result = await ctx.client.request<ProvidersModelsResult>('providers.models', {
      provider
    })
    const models = result.models
    const discovery = result.discovery

    if (!target) {
      // No arg → open an interactive picker (mirrors Python `_cmd_model` no-arg
      // path which jumps into the alternate-buffer picker).
      if (models.length === 0) {
        return {
          kind: 'note',
          level: 'warn',
          text: `provider ${provider} returned no models`
        }
      }
      // When live discovery failed, surface the reason as a note above the
      // picker so the user can debug their endpoint without digging through
      // gateway logs. The picker still opens so they can pick a fallback.
      if (discovery && discovery.kind === 'static') {
        chatActions.pushNote(formatDiscoveryFallbackNote(discovery), 'warn')
      }
      // When discovery succeeded but the gateway noticed the base_url is
      // missing /v1 (or whatever prefix the chat endpoint needs), warn
      // BEFORE the user sends a turn that will 404 in 30 seconds.
      if (discovery && discovery.kind === 'live' && discovery.warning) {
        chatActions.pushNote(`⚠ ${discovery.warning}`, 'warn')
      }
      const items: PickerItem<ModelInfo>[] = models.map((model) => ({
        id: model.id,
        value: model,
        searchKey: `${model.id} ${model.display_name ?? ''}`
      }))
      const overlayId = `picker_model_${Date.now()}`
      const payload: PickerPayload<ModelInfo> = {
        title: formatPickerTitle(provider, models.length, discovery),
        items,
        ...(state.model ? { currentId: state.model } : {}),
        pendingVerb: 'switching',
        renderRow(item) {
          return renderModelRow(item.value)
        },
        async onSelect(item) {
          await applyModel(ctx, provider, item.id, discovery)
        }
      }
      overlayActions.push({
        kind: 'picker',
        id: overlayId,
        payload,
        createdAt: Date.now(),
        priority: OVERLAY_PRIORITY.picker,
        onDismiss: () => undefined
      })
      return { kind: 'noop' }
    }

    const known = models.some((model) => model.id === target)
    if (!known && provider !== 'openai') {
      return {
        kind: 'note',
        level: 'warn',
        text: `model not in ${provider} catalog: ${target}`
      }
    }

    const info = await switchModel(ctx, provider, target, discovery)
    const suffix = known
      ? ''
      : ' (not in static catalog; accepted for OpenAI-compatible provider)'
    return {
      kind: 'session',
      info,
      note: `model set to ${target}; session ${info.session_id.slice(0, 8)}${suffix}`
    }
  }
}

async function applyModel(
  ctx: Parameters<SlashCommand['execute']>[1],
  provider: string,
  modelId: string,
  discovery: ModelDiscovery | undefined
): Promise<void> {
  const info = await switchModel(ctx, provider, modelId, discovery)
  sessionActions.setSession(info)
  chatActions.pushNote(
    `model set to ${modelId}; session ${info.session_id.slice(0, 8)}`,
    'info'
  )
}

async function switchModel(
  ctx: Parameters<SlashCommand['execute']>[1],
  provider: string,
  modelId: string,
  discovery: ModelDiscovery | undefined
): Promise<SessionInfo> {
  await ctx.client.request('prefs.set', {
    key: `last_model_by_provider.${provider}`,
    value: modelId
  })
  const baseUrl = discovery?.suggested_base_url || discovery?.base_url || null
  const state = ctx.getSession()
  if (!state.sessionId) {
    return ctx.createSession({ provider, model: modelId, baseUrl })
  }
  const params: Record<string, unknown> = {
    session_id: state.sessionId,
    provider,
    model: modelId
  }
  if (baseUrl) {
    params.base_url = baseUrl
  }
  const result = await ctx.client.request<SessionUpdateResult>('session.update', params)
  return result.info
}

function renderModelRow(model: ModelInfo): ReactElement {
  // Match the Python picker layout: just the model id, no display-name
  // column. Most live providers (Kimi, DeepSeek, vLLM, etc.) only have an
  // id anyway, and the cluttered two-column look from the previous
  // revision did not match the screenshot the user sent.
  return (
    <Box>
      <Text>{model.id}</Text>
    </Box>
  )
}

function formatPickerTitle(
  provider: string,
  count: number,
  discovery: ModelDiscovery | undefined
): string {
  if (!discovery) {
    return `Select model · ${provider} · ${count} available`
  }
  if (discovery.kind === 'live') {
    const baseUrl = discovery.base_url ? ` · ${discovery.base_url}` : ''
    return `Select model · ${provider}${baseUrl} · ${count} live`
  }
  // Static fallback — make it visible in the title so the user does not
  // think the 4 gpt-* entries are what their endpoint actually offers.
  return `Select model · ${provider} · ${count} (static fallback)`
}

function formatDiscoveryFallbackNote(discovery: ModelDiscovery): string {
  const parts: string[] = ['live model discovery skipped']
  if (discovery.reason) {
    parts.push(discovery.reason.replace(/_/g, ' '))
  }
  if (discovery.base_url) {
    parts.push(`base_url=${discovery.base_url}`)
  } else {
    parts.push('no base_url configured')
  }
  if (discovery.error) {
    parts.push(discovery.error)
  }
  if (discovery.body_preview) {
    parts.push(`body=${discovery.body_preview}`)
  }
  return parts.join(' · ')
}
