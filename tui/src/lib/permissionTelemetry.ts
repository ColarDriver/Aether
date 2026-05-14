/**
 * Lightweight, fire-and-forget telemetry for permission-modal rendering
 * fallbacks. The JSON-args dump in `PermissionModal.PreviewBody` is a
 * worst-case escape hatch — tools should always send `diff` / `command` /
 * `body`, so a fallback hit should be rare enough to
 * notice.
 *
 * We log to stderr (rather than open a new gateway RPC) so the signal is
 * captured by any wrapping log collector without needing a wire-level
 * change. Failures here never propagate — telemetry must not break the
 * permission flow.
 */

export interface PermissionPreviewFallbackEvent {
  toolName: string
  hasPreview: boolean
  previewKeys: string[]
}

export function reportPermissionPreviewFallback(
  event: PermissionPreviewFallbackEvent
): void {
  try {
    const line = JSON.stringify({
      event: 'permission_preview_fallback',
      ...event,
      ts: Date.now()
    })
    // stderr is fine here — gateway/log collectors pick it up, but it
    // does not interleave with Ink's stdout rendering.
    process.stderr.write(`${line}\n`)
  } catch {
    // Swallowed on purpose: telemetry is best-effort.
  }
}
