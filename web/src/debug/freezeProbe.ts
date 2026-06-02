// Temporary diagnostic for the mid-stream freeze. Two signals, printed to the
// browser console, that together tell us whether a stall is a blocked main
// thread (render) or a stalled stream (backend/transport):
//
//   [freeze] main thread blocked for <N>ms   -> render froze the tab (Bug A)
//   [frame] <type> seq=<n> +<gap>ms          -> per inbound frame; if these
//                                                 stop while the UI looks stuck,
//                                                 the backend stopped sending (Bug B)
//
// Remove this module (and its two call sites) once the freeze is diagnosed.

let lastFrameAt = 0
let frameCount = 0

export function logInboundFrame(type: string, sequence: unknown): void {
  const now = performance.now()
  const gap = lastFrameAt === 0 ? 0 : Math.round(now - lastFrameAt)
  lastFrameAt = now
  frameCount += 1
  const seq = typeof sequence === 'number' ? sequence : '-'
  // Content deltas are the high-frequency ones; keep the line terse.
  console.log(`[frame] ${type} seq=${seq} +${gap}ms #${frameCount}`)
}

export function startMainThreadStallProbe(): void {
  const TICK_MS = 1000
  const STALL_THRESHOLD_MS = 1500
  let expected = performance.now() + TICK_MS
  const tick = () => {
    const now = performance.now()
    const drift = Math.round(now - expected)
    if (drift > STALL_THRESHOLD_MS - TICK_MS) {
      console.warn(`[freeze] main thread blocked for ~${drift + TICK_MS}ms`)
    }
    expected = now + TICK_MS
    setTimeout(tick, TICK_MS)
  }
  setTimeout(tick, TICK_MS)
}
