/**
 * Port of `aether/cli/theme.py` — palette + glyph table + colour toggle.
 *
 * Components should call `theme.color('user' | 'assistant' | …)` and
 * `theme.icon('assistant' | …)` instead of hard-coding hex values; that way
 * a future contributor can recolour the whole TUI from one place and the
 * `AETHER_ASCII=1` / `NO_COLOR` toggles remain authoritative.
 */

const PALETTE = {
  primary: '#7C5CFF',
  primaryDim: '#5B3FE0',
  accent: '#22D3EE',
  accentDim: '#0E7490',
  text: '#E5E7EB',
  dim: '#64748B',
  border: '#334155',
  panel: '#0F172A',
  // `popupBg` matches the `bg:#1E293B` background Python's slash completion
  // menu paints (`completion-menu.*` styles in `aether/cli/app.py`). Kept
  // separate from `panel` so the popup retains its visual identity even if
  // the broader panel colour shifts.
  popupBg: '#1E293B',
  success: '#22C55E',
  warning: '#F59E0B',
  error: '#EF4444',
  info: '#38BDF8',
  toolAccent: '#F97316',
  toolDim: '#9A3412',
  toolShimmer: '#FED7AA'
} as const

type SemanticName =
  | 'brand'
  | 'accent'
  | 'dim'
  | 'text'
  | 'border'
  | 'popup_bg'
  | 'user'
  | 'assistant'
  | 'system'
  | 'tool'
  | 'tool_args'
  | 'tool_result'
  | 'tool_error'
  | 'success'
  | 'warning'
  | 'error'
  | 'info'
  | 'status'

const SEMANTIC_COLORS: Record<SemanticName, string> = {
  brand: PALETTE.primary,
  accent: PALETTE.accent,
  dim: PALETTE.dim,
  text: PALETTE.text,
  border: PALETTE.border,
  popup_bg: PALETTE.popupBg,
  user: PALETTE.primary,
  assistant: PALETTE.primary,
  system: PALETTE.dim,
  tool: PALETTE.toolAccent,
  tool_args: PALETTE.dim,
  tool_result: PALETTE.text,
  tool_error: PALETTE.error,
  success: PALETTE.success,
  warning: PALETTE.warning,
  error: PALETTE.error,
  info: PALETTE.info,
  status: PALETTE.primary
}

const ICONS_UNICODE: Record<string, string> = {
  logo: '✦',
  user: '›',
  assistant: '●',
  tool: '⚙',
  tool_done: '✓',
  tool_error: '✗',
  info: 'ℹ',
  warn: '⚠',
  error: '✗',
  success: '✓',
  thinking: '✷',
  arrow: '→',
  bullet: '•',
  dot: '·',
  session: '◈',
  model: '◇',
  provider: '◆',
  iter: '↻',
  interrupt: '⏹',
  spark: '⚡'
}

const ICONS_ASCII: Record<string, string> = {
  logo: '*',
  user: '>',
  assistant: '*',
  tool: 'T',
  tool_done: 'v',
  tool_error: 'x',
  info: 'i',
  warn: '!',
  error: 'x',
  success: 'v',
  thinking: '*',
  arrow: '->',
  bullet: '-',
  dot: '.',
  session: '#',
  model: 'M',
  provider: 'P',
  iter: '@',
  interrupt: '[]',
  spark: '!'
}

function isUnicodeAllowed(): boolean {
  if (process.env.AETHER_ASCII === '1') {
    return false
  }
  const encoding = (process.env.LC_ALL || process.env.LC_CTYPE || process.env.LANG || '')
    .toLowerCase()
  if (!encoding) {
    // node defaults to UTF-8 on most modern systems; assume yes unless an
    // operator explicitly opts out via AETHER_ASCII.
    return true
  }
  return encoding.includes('utf')
}

function isColorEnabled(): boolean {
  if (process.env.NO_COLOR !== undefined) {
    return false
  }
  if (process.env.TERM === 'dumb') {
    return false
  }
  if (process.stdout && process.stdout.isTTY === false) {
    return false
  }
  return true
}

export const theme = {
  palette: PALETTE,
  /**
   * Return the hex colour for a semantic role, or `undefined` when colour
   * is disabled. The returned value is always safe to splat into Ink's
   * `<Text color={…}>` prop with `exactOptionalPropertyTypes: true` so
   * call sites do not need to special-case the `undefined` branch.
   */
  color(name: SemanticName): string | undefined {
    if (!isColorEnabled()) {
      return undefined
    }
    return SEMANTIC_COLORS[name]
  },
  /** Helper that returns `{ color }` props or `{}` when colour is disabled. */
  colorProps(name: SemanticName): { color?: string } {
    const value = theme.color(name)
    return value === undefined ? {} : { color: value }
  },
  icon(name: string): string {
    const table = isUnicodeAllowed() ? ICONS_UNICODE : ICONS_ASCII
    return table[name] ?? ''
  },
  isUnicodeAllowed,
  isColorEnabled
}
