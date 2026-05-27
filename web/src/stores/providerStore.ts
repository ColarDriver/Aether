import { create } from 'zustand'
import { api } from '../api/client'
import type { ModelSummary, ProviderModelList, ProviderPreflightStatus, ProviderRuntimeStatus, ProviderSelectionResult, ProviderSummary } from '../api/types'

type ProviderState = {
  providers: ProviderSummary[]
  current: ProviderRuntimeStatus | null
  preflight: ProviderPreflightStatus | null
  modelsByProvider: Record<string, ModelSummary[]>
  discoveryByProvider: Record<string, ProviderModelList['discovery']>
  isLoading: boolean
  error: string | null
  loadProviders: () => Promise<void>
  loadModels: (provider: string, options?: { force?: boolean }) => Promise<void>
  loadPreflight: (params?: { provider?: string | null; model?: string | null; baseUrl?: string | null }) => Promise<ProviderPreflightStatus>
  selectModel: (provider: string, model: string) => Promise<ProviderSelectionResult>
}

export const useProviderStore = create<ProviderState>((set, get) => ({
  providers: [],
  current: null,
  preflight: null,
  modelsByProvider: {},
  discoveryByProvider: {},
  isLoading: false,
  error: null,
  loadProviders: async () => {
    set({ isLoading: true, error: null })
    try {
      const [{ providers }, current] = await Promise.all([api.providers(), api.currentProvider()])
      set({ providers, current, isLoading: false })
      if (current.provider_name) void get().loadModels(current.provider_name)
      void get().loadPreflight({
        provider: current.provider_name,
        model: current.model,
        baseUrl: current.base_url ?? null,
      }).catch(() => undefined)
    } catch (error) {
      set({ error: error instanceof Error ? error.message : String(error), isLoading: false })
    }
  },
  loadModels: async (provider, options = {}) => {
    if (!options.force && get().modelsByProvider[provider]) return
    const result = await api.providerModels(provider)
    set((state) => ({
      modelsByProvider: { ...state.modelsByProvider, [provider]: result.models },
      discoveryByProvider: { ...state.discoveryByProvider, [provider]: result.discovery },
    }))
  },
  loadPreflight: async (params = {}) => {
    const result = await api.providerPreflight(params)
    set({ preflight: result })
    return result
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
      preflight: null,
      providers: state.providers,
    }))
    return result
  },
}))
