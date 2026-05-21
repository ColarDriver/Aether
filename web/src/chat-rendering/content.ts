import type { AskUserQuestion, AskUserQuestionOption, DiffContent } from './blocks'

export function stringFromUnknown(value: unknown): string {
  if (typeof value === 'string') return value
  if (value == null) return ''
  if (value instanceof Error) return value.message
  if (typeof value === 'number' || typeof value === 'boolean' || typeof value === 'bigint') {
    return String(value)
  }
  if (Array.isArray(value)) {
    return value.map((item) => stringFromUnknown(item)).filter(Boolean).join('\n')
  }
  if (typeof value === 'object') {
    const text = textFieldFromRecord(value)
    if (text) return text
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }
  return String(value)
}

export function recordFromUnknown(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

export function jsonPreview(value: unknown, options: { maxChars?: number } = {}): string {
  const maxChars = options.maxChars ?? 2000
  const text = typeof value === 'string' ? value : stringFromUnknown(value)
  if (text.length <= maxChars) return text
  return text.slice(0, Math.max(0, maxChars - 1)) + '…'
}

export function firstNonEmptyLine(text: string): string {
  return text.split(/\r?\n/).map((line) => line.trim()).find(Boolean) ?? ''
}

export function extractDiffFromMetadata(metadata: Record<string, unknown>): DiffContent | null {
  const diff = stringOrNull(metadata.diff)
  const oldText = firstString(metadata.oldText, metadata.old_text, metadata.old_string)
  const newText = firstString(metadata.newText, metadata.new_text, metadata.new_string)
  const path = firstString(metadata.path, metadata.file_path, metadata.filePath)
  const language = firstString(metadata.language, metadata.lang)

  if (!diff && oldText == null && newText == null) return null

  return {
    path,
    diff,
    oldText,
    newText,
    language,
  }
}

export function parseAskUserQuestions(value: unknown): AskUserQuestion[] {
  const input = recordFromUnknown(value)
  const rawQuestions = Array.isArray(input.questions) ? input.questions : null

  if (rawQuestions) {
    return rawQuestions.map(questionFromUnknown).filter((question): question is AskUserQuestion => Boolean(question))
  }

  const single = questionFromUnknown(input)
  return single ? [single] : []
}

export function answersFromMetadata(metadata: Record<string, unknown>): Record<string, string> {
  const normalized: Record<string, string> = {}
  const answers = recordFromUnknown(metadata.answers)
  for (const [key, value] of Object.entries(answers)) {
    const text = stringFromUnknown(value).trim()
    if (text) normalized[key] = text
  }

  if (Array.isArray(metadata.answer_pairs)) {
    for (const pair of metadata.answer_pairs) {
      const record = recordFromUnknown(pair)
      const label = stringOrNull(record.label)
      const value = stringOrNull(record.value)
      if (label && value != null) normalized[label] = value
    }
  }

  return normalized
}

function questionFromUnknown(value: unknown): AskUserQuestion | null {
  const input = recordFromUnknown(value)
  const question = stringOrNull(input.question) ?? stringOrNull(input.prompt)
  if (!question) return null
  const id = stringOrUndefined(input.id)
  const header = stringOrUndefined(input.header)
  const options = parseOptions(input.options)
  const multiSelect = Boolean(input.multiSelect ?? input.multi_select ?? input.allow_multiple)
  const freeText = Boolean(input.freeText ?? input.free_text)
  return {
    ...(id ? { id } : {}),
    question,
    ...(header ? { header } : {}),
    ...(options.length > 0 ? { options } : {}),
    ...(multiSelect ? { multiSelect } : {}),
    ...(freeText ? { freeText } : {}),
  }
}

function parseOptions(value: unknown): AskUserQuestionOption[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((option) => {
    if (typeof option === 'string' && option.trim()) return [{ label: option.trim() }]
    const record = recordFromUnknown(option)
    const label = stringOrNull(record.label)
    if (!label) return []
    const id = stringOrUndefined(record.id)
    const description = stringOrUndefined(record.description)
    return [{
      ...(id ? { id } : {}),
      label,
      ...(description ? { description } : {}),
    }]
  })
}

function textFieldFromRecord(value: object): string {
  const record = value as Record<string, unknown>
  return firstString(record.text, record.content, record.message) ?? ''
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    const text = stringOrNull(value)
    if (text != null) return text
  }
  return null
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function stringOrUndefined(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}
