type Listener = () => void

const HEARTBEAT_MS = 50

const listeners = new Set<Listener>()
let timer: NodeJS.Timeout | null = null
let startedAt = 0

export function nowMs(): number {
  return Date.now()
}

export function subscribe(listener: Listener): () => void {
  listeners.add(listener)
  if (timer === null) {
    startedAt = nowMs()
    timer = setInterval(() => {
      for (const current of Array.from(listeners)) {
        current()
      }
    }, HEARTBEAT_MS)
  }

  let unsubscribed = false
  return () => {
    if (unsubscribed) {
      return
    }
    unsubscribed = true
    listeners.delete(listener)
    if (listeners.size === 0 && timer !== null) {
      clearInterval(timer)
      timer = null
    }
  }
}

export function clockStartedAt(): number {
  return startedAt
}

export function _clockStateForTests(): { listeners: number; running: boolean } {
  return {
    listeners: listeners.size,
    running: timer !== null
  }
}

export function _resetClockForTests(): void {
  if (timer !== null) {
    clearInterval(timer)
    timer = null
  }
  listeners.clear()
  startedAt = 0
}
