const ESC = String.fromCharCode(27)
const SGR_MOUSE_RE = new RegExp(`(?:${ESC})?\\[<(\\d+);\\d+;\\d+[mM]`, 'g')

export function stripMouseTrackingSequences(input: string): string {
  if (!input) {
    return input
  }
  return input.replace(SGR_MOUSE_RE, '')
}

export function isOnlyMouseTrackingInput(input: string): boolean {
  return input.length > 0 && stripMouseTrackingSequences(input).length === 0
}

export function mouseButtonCodes(input: string): number[] {
  const buttons: number[] = []
  for (const match of input.matchAll(SGR_MOUSE_RE)) {
    const button = Number.parseInt(match[1] ?? '', 10)
    if (Number.isFinite(button)) {
      buttons.push(button)
    }
  }
  return buttons
}
