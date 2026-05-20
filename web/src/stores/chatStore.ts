import { create } from 'zustand'
import { runSocket } from '../api/runSocket'
import type { RunSocketFrame } from '../api/types'

type ChatState = {
  connected: boolean
  frames: RunSocketFrame[]
  activeRunId: string | null
  connect: () => void
  startRun: (sessionId: string, message: string) => string
  cancelRun: (sessionId: string) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  connected: false,
  frames: [],
  activeRunId: null,
  connect: () => {
    runSocket.connect()
    runSocket.onFrame((frame) => {
      set((state) => ({
        frames: [...state.frames, frame],
        connected: frame.type === 'ready' ? true : state.connected,
        activeRunId:
          frame.type === 'run.accepted' && typeof frame.payload?.run_id === 'string'
            ? frame.payload.run_id
            : state.activeRunId,
      }))
    })
  },
  startRun: (sessionId, message) => {
    get().connect()
    const runId = runSocket.startRun(sessionId, message)
    set({ activeRunId: runId })
    return runId
  },
  cancelRun: (sessionId) => {
    runSocket.cancelRun(sessionId, get().activeRunId ?? undefined)
  },
}))
