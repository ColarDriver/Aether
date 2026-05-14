const TURN_FOOTER_RE = /^(?:[✓⏹✗vx]|\[\])\s+(?:done|cancelled|failed)\b/

export function isTurnFooterText(text: string): boolean {
  return TURN_FOOTER_RE.test(text.trim())
}
