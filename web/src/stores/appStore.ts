import { create } from 'zustand'
import { api } from '../api/client'
import type { ConsoleView, HealthStatus, StatusResponse } from '../api/types'

type AppState = {
  status: StatusResponse | null
  health: HealthStatus | null
  activeView: ConsoleView
  isLoading: boolean
  error: string | null
  setActiveView: (view: ConsoleView) => void
  bootstrap: () => Promise<void>
}

export const useAppStore = create<AppState>((set) => ({
  status: null,
  health: null,
  activeView: 'chat',
  isLoading: false,
  error: null,
  setActiveView: (activeView) => set({ activeView }),
  bootstrap: async () => {
    set({ isLoading: true, error: null })
    try {
      const [status, health] = await Promise.all([api.status(), api.health()])
      set({ status, health, isLoading: false })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), isLoading: false })
    }
  },
}))
