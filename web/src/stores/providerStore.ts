import { create } from 'zustand'
import { api } from '../api/client'
import type { ProviderRuntimeStatus, ProviderSummary } from '../api/types'

type ProviderState = {
  providers: ProviderSummary[]
  current: ProviderRuntimeStatus | null
  isLoading: boolean
  error: string | null
  loadProviders: () => Promise<void>
}

export const useProviderStore = create<ProviderState>((set) => ({
  providers: [],
  current: null,
  isLoading: false,
  error: null,
  loadProviders: async () => {
    set({ isLoading: true, error: null })
    try {
      const [{ providers }, current] = await Promise.all([api.providers(), api.currentProvider()])
      set({ providers, current, isLoading: false })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), isLoading: false })
    }
  },
}))
