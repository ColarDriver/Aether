import type { TokenUsage } from './blocks'

export function tokenUsageFromRecord(payload: Record<string, unknown>): TokenUsage {
  const usage = recordOrNull(payload.usage) ?? payload
  const promptDetails = recordOrNull(usage.prompt_tokens_details) ?? recordOrNull(usage.promptTokensDetails)
  const completionDetails = recordOrNull(usage.completion_tokens_details) ?? recordOrNull(usage.completionTokensDetails)

  const inputTokens = numberField(usage, 'input_tokens', 'inputTokens', 'prompt_tokens', 'promptTokens')
  const outputTokens = numberField(usage, 'output_tokens', 'outputTokens', 'completion_tokens', 'completionTokens')
  const cacheReadTokens = numberField(usage, 'cache_read_tokens', 'cacheReadTokens', 'cached_tokens', 'cachedTokens')
    ?? numberField(promptDetails, 'cached_tokens', 'cachedTokens')
  const cacheWriteTokens = numberField(usage, 'cache_write_tokens', 'cacheWriteTokens')
  const reasoningTokens = numberField(usage, 'reasoning_tokens', 'reasoningTokens')
    ?? numberField(completionDetails, 'reasoning_tokens', 'reasoningTokens')
  const explicitTotal = numberField(usage, 'total_tokens', 'totalTokens', 'total')

  const partial: TokenUsage = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    cache_read_tokens: cacheReadTokens,
    cache_write_tokens: cacheWriteTokens,
    reasoning_tokens: reasoningTokens,
  }
  const totalTokens = explicitTotal ?? derivedTokenTotal(partial)
  return {
    ...partial,
    total_tokens: totalTokens,
  }
}

export function tokenUsageTotal(tokens?: TokenUsage | null): number {
  if (!tokens) return 0
  return tokens.total_tokens ?? derivedTokenTotal(tokens) ?? 0
}

export function tokenUsageBreakdown(tokens?: TokenUsage | null): string[] {
  if (!tokens) return []
  const cacheTokens = Math.max(0, tokens.cache_read_tokens ?? 0) + Math.max(0, tokens.cache_write_tokens ?? 0)
  return [
    positivePart(tokens.input_tokens, 'in'),
    positivePart(tokens.output_tokens, 'out'),
    positivePart(cacheTokens, 'cache'),
    positivePart(tokens.reasoning_tokens, 'reasoning'),
  ].filter((item): item is string => Boolean(item))
}

export function tokenUsageSummary(tokens?: TokenUsage | null): string | null {
  const total = tokenUsageTotal(tokens)
  if (total <= 0) return null
  const breakdown = tokenUsageBreakdown(tokens)
  return formatCompactTokens(total) + ' tokens' + (breakdown.length > 0 ? ' (' + breakdown.join(' / ') + ')' : '')
}

export function formatCompactTokens(value: number): string {
  if (value < 1000) return String(value)
  if (value < 1_000_000) return (value / 1000).toFixed(1).replace(/.0$/, '') + 'k'
  return (value / 1_000_000).toFixed(1).replace(/.0$/, '') + 'M'
}

function derivedTokenTotal(tokens: TokenUsage): number | undefined {
  const values = [
    tokens.input_tokens,
    tokens.output_tokens,
    tokens.cache_read_tokens,
    tokens.cache_write_tokens,
    tokens.reasoning_tokens,
  ].filter((value): value is number => typeof value === 'number' && Number.isFinite(value) && value > 0)
  if (values.length === 0) return undefined
  return values.reduce((sum, value) => sum + value, 0)
}

function positivePart(value: number | undefined, label: string): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) return null
  return formatCompactTokens(value) + ' ' + label
}

function numberField(record: Record<string, unknown> | null, ...keys: string[]): number | undefined {
  if (!record) return undefined
  for (const key of keys) {
    const value = record[key]
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
    }
  }
  return undefined
}

function recordOrNull(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
