export interface DiscoveryBaseUrlLike {
  base_url?: string | null
  suggested_base_url?: string | null
}

function clean(value: string | null | undefined): string | null {
  if (!value) {
    return null
  }
  const trimmed = value.trim()
  return trimmed ? trimmed : null
}

export function envBaseUrl(provider: string | null | undefined = null): string | null {
  const explicit = clean(process.env.AETHER_BASE_URL)
  if (explicit) {
    return explicit
  }
  if (provider === 'claude') {
    return clean(process.env.ANTHROPIC_BASE_URL)
  }
  if (provider === 'openai') {
    return clean(process.env.OPENAI_BASE_URL)
  }
  return clean(process.env.OPENAI_BASE_URL) ?? clean(process.env.ANTHROPIC_BASE_URL)
}

export function resolveDiscoveredBaseUrl(
  currentBaseUrl: string | null | undefined,
  discovery?: DiscoveryBaseUrlLike | null
): string | null {
  return (
    clean(discovery?.suggested_base_url) ??
    clean(discovery?.base_url) ??
    clean(currentBaseUrl)
  )
}
