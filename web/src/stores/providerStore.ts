import { create } from 'zustand'
import { api } from '../api/client'
import type { ModelSummary, ProviderRuntimeStatus, ProviderSummary } from '../api/types'

type ProviderState = {
  providers: ProviderSummary[]
  current: ProviderRuntimeStatus | null
  modelsByProvider: Record<string, ModelSummary[]>
  isLoading: boolean
  error: string | null
  loadProviders: () => Promise<void>
  loadModels: (provider: string) => Promise<void>
  selectModel: (provider: string, model: string) => Promise<void>
}

export const useProviderStore = create<ProviderState>((set, get) => ({
  providers: [],
  current: null,
  modelsByProvider: {},
  isLoading: false,
  error: null,
  loadProviders: async () => {
    set({ isLoading: true, error: null })
    try {
      const [{ providers }, current] = await Promise.all([api.providers(), api.currentProvider()])
      set({ providers, current, isLoading: false })
      if (current.provider_name) void get().loadModels(current.provider_name)
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), isLoading: false })
    }
  },
  loadModels: async (provider) => {
    if (get().modelsByProvider[provider]) return
    const result = await api.providerModels(provider)
    set((state) => ({
      modelsByProvider: { ...state.modelsByProvider, [provider]: result.models },
    }))
  },
  selectModel: async (provider, model) => {
    const result = await api.selectModel({ provider, model, persist_last_model: true })
    set((state) => ({
      current: {
        family: result.family,
        provider_name: result.provider,
        model: result.model,
        base_url: result.base_url,
        api_key_env_names: result.missing_credentials,
        model_env_names: [],
        base_url_env_names: [],
        source: 'web',
        credential: result.credential ?? null,
      },
      providers: state.providers,
    }))
  },
}))
