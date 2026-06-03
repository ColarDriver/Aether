import type { RunStatusSnapshot, TokenUsage } from '../../chat-rendering'
import { spinnerVerbForSeed, tokenUsageSummary } from '../../chat-rendering'
import { AetherAvatar } from './blocks/AetherAvatar'

type Props = {
  activeRunId: string | null
  // True once the inline name-row status has scrolled above the viewport.
  visible: boolean
  status?: RunStatusSnapshot
  tokens?: TokenUsage
  model?: string | null
}

// The overflow state: once the inline status scrolls out of view during a long
// streaming reply, this full-width bar pins to the top of the conversation. It
// speaks the same language as the inline row (static Aether mark with a breathing
// edge + shimmer verb), and adds run meta (tokens, model) since there's room.
export function RunStatusHeader({ activeRunId, visible, status, tokens, model }: Props) {
  const show = Boolean(activeRunId) && visible
  const state = status?.state || 'thinking'
  const verb = spinnerVerbForSeed(activeRunId ?? status?.runId)
  const tokenSummary = tokenUsageSummary(tokens ?? status?.tokens)
  // ↑ while waiting on the model (prompt still flowing up), ↓ once output streams back.
  const tokenArrow = state === 'requesting' ? '↑' : '↓'
  const hasMeta = Boolean(tokenSummary) || Boolean(model)
  return (
    <div
      className={'run-status-header' + (show ? ' run-status-header-visible' : '')}
      role="status"
      aria-hidden={!show}
      aria-label="Current activity"
    >
      <AetherAvatar active className="aether-avatar-sm" />
      <span className="run-status-name">Aether</span>
      <span className="run-status-sep" aria-hidden="true">·</span>
      <strong className="aether-shimmer-text run-status-verb">{verb}…</strong>
      {hasMeta ? (
        <span className="run-status-meta">
          {tokenSummary ? (
            <span className="run-status-tokens">
              <span className="run-status-arrow">{tokenArrow}</span> {tokenSummary}
            </span>
          ) : null}
          {tokenSummary && model ? <span className="run-status-pipe" aria-hidden="true" /> : null}
          {model ? <span className="run-status-model">{model}</span> : null}
        </span>
      ) : null}
      <span className="run-status-scan" aria-hidden="true" />
    </div>
  )
}
