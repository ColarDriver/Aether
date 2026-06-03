import type { StreamingStatusBlock as StreamingStatus } from '../../../chat-rendering'
import { spinnerVerbForSeed } from '../../../chat-rendering'
import { AetherAvatar } from './AetherAvatar'

type Props = {
  block: StreamingStatus
}

// Inline activity, living in the assistant turn's name row: a static Aether mark
// whose edge breathes while the run is live, then "Aether · <verb>…" with the
// shimmer riding the whimsical verb. The explicit mode word (Requesting/…) is
// intentionally dropped — the verb plus the breathing mark carry the signal.
export function StreamingStatusBlock({ block }: Props) {
  const verb = spinnerVerbForSeed(block.runId ?? block.id)
  return (
    <div className="chat-block chat-status-inline" role="status" aria-label="Aether is working">
      <AetherAvatar active />
      <span className="chat-status-name">Aether</span>
      <span className="chat-status-sep" aria-hidden="true">·</span>
      <strong className="aether-shimmer-text chat-status-verb">{verb}…</strong>
    </div>
  )
}
