export type InlineSegment =
  | { kind: 'text'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'code'; text: string }

export type MarkdownBlock =
  | { kind: 'paragraph'; segments: InlineSegment[] }
  | { kind: 'code'; text: string }

export function parseMarkdownLite(input: string): MarkdownBlock[] {
  const blocks: MarkdownBlock[] = []
  const lines = input.split('\n')
  let codeBuffer: string[] | null = null
  let paragraph: string[] = []

  const flushParagraph = () => {
    if (paragraph.length === 0) {
      return
    }
    blocks.push({ kind: 'paragraph', segments: parseInline(paragraph.join('\n')) })
    paragraph = []
  }

  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (codeBuffer) {
        blocks.push({ kind: 'code', text: codeBuffer.join('\n') })
        codeBuffer = null
      } else {
        flushParagraph()
        codeBuffer = []
      }
      continue
    }
    if (codeBuffer) {
      codeBuffer.push(line)
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      continue
    }
    paragraph.push(line)
  }

  if (codeBuffer) {
    blocks.push({ kind: 'code', text: codeBuffer.join('\n') })
  }
  flushParagraph()
  return blocks
}

export function parseInline(input: string): InlineSegment[] {
  const segments: InlineSegment[] = []
  let cursor = 0

  while (cursor < input.length) {
    const boldStart = input.indexOf('**', cursor)
    const codeStart = input.indexOf('`', cursor)
    const candidates = [boldStart, codeStart].filter((value) => value >= 0)
    const nextStart = candidates.length ? Math.min(...candidates) : -1

    if (nextStart === -1) {
      pushText(segments, input.slice(cursor))
      break
    }
    if (nextStart > cursor) {
      pushText(segments, input.slice(cursor, nextStart))
    }

    if (nextStart === boldStart) {
      const end = input.indexOf('**', nextStart + 2)
      if (end === -1) {
        pushText(segments, input.slice(nextStart))
        break
      }
      segments.push({ kind: 'bold', text: input.slice(nextStart + 2, end) })
      cursor = end + 2
    } else {
      const end = input.indexOf('`', nextStart + 1)
      if (end === -1) {
        pushText(segments, input.slice(nextStart))
        break
      }
      segments.push({ kind: 'code', text: input.slice(nextStart + 1, end) })
      cursor = end + 1
    }
  }

  return segments
}

function pushText(segments: InlineSegment[], text: string): void {
  if (!text) {
    return
  }
  const last = segments.at(-1)
  if (last?.kind === 'text') {
    last.text += text
    return
  }
  segments.push({ kind: 'text', text })
}
