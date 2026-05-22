import type { AskUserQuestion, AskUserQuestionOption, ChatAttachment, DiagnosticEntry, DiagnosticFileGroup, DiffContent, TaskNotificationBlock } from './blocks'

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

export function attachmentsFromUnknown(value: unknown): ChatAttachment[] {
  if (!Array.isArray(value)) return []
  return value.map(attachmentFromUnknown).filter((attachment): attachment is ChatAttachment => Boolean(attachment))
}

export function attachmentsFromMetadata(metadata: Record<string, unknown>): ChatAttachment[] {
  for (const key of ['attachments', 'displayAttachments', 'display_attachments']) {
    const attachments = attachmentsFromUnknown(metadata[key])
    if (attachments.length > 0) return attachments
  }
  return []
}


export type DiagnosticsBlockContent = {
  content: string
  files: DiagnosticFileGroup[]
}

export function parseDiagnosticsBlock(text: string): DiagnosticsBlockContent | null {
  const xml = diagnosticsXml(text)
  if (!xml) return null
  const body = xml.replace(/^<diagnostics>\s*/i, '').replace(/\s*<\/diagnostics>$/i, '')
  const files: DiagnosticFileGroup[] = []
  let current: DiagnosticFileGroup | null = null

  for (const rawLine of body.split('\n')) {
    const line = rawLine.trimEnd()
    const heading = line.match(/^##\s+(.+)$/)
    if (heading) {
      current = { path: decodeXmlText(heading[1]!.trim()), diagnostics: [] }
      files.push(current)
      continue
    }
    if (!current) continue
    const diagnostic = parseDiagnosticLine(line)
    if (diagnostic) current.diagnostics.push(diagnostic)
  }

  const nonEmptyFiles = files.filter((file) => file.diagnostics.length > 0)
  if (nonEmptyFiles.length === 0) return null
  return { content: xml, files: nonEmptyFiles }
}

export function isDiagnosticsText(text: string): boolean {
  const trimmed = text.trim()
  const xml = diagnosticsXml(trimmed)
  return Boolean(xml && xml === trimmed)
}

export type TaskNotificationContent = Pick<
  TaskNotificationBlock,
  'taskId' | 'subagentType' | 'status' | 'durationSeconds' | 'summary' | 'error' | 'outputFile'
>

export function parseTaskNotification(text: string): TaskNotificationContent | null {
  const xml = taskNotificationXml(text)
  if (!xml) return null
  const taskId = readXmlTag(xml, 'task_id') ?? readXmlTag(xml, 'task-id') ?? readXmlTag(xml, 'tool-use-id')
  const status = readXmlTag(xml, 'status')
  if (!taskId || !status) return null
  return {
    taskId,
    status,
    subagentType: readXmlTag(xml, 'subagent_type') ?? readXmlTag(xml, 'subagent-type') ?? null,
    durationSeconds: numberOrNull(readXmlTag(xml, 'duration_seconds') ?? readXmlTag(xml, 'duration-seconds')),
    summary: readXmlTag(xml, 'summary') ?? null,
    error: readXmlTag(xml, 'error') ?? null,
    outputFile: readXmlTag(xml, 'output_file') ?? readXmlTag(xml, 'output-file') ?? null,
  }
}

export function isTaskNotificationText(text: string): boolean {
  const trimmed = text.trim()
  const xml = taskNotificationXml(trimmed)
  return Boolean(xml && xml === trimmed)
}

function attachmentFromUnknown(value: unknown): ChatAttachment | null {
  const input = recordFromUnknown(value)
  const rawType = stringOrNull(input.type) ?? stringOrNull(input.kind)
  const type = rawType === 'image' || rawType === 'text' || rawType === 'file'
    ? rawType
    : firstString(input.data, input.url, input.previewUrl)
      ? 'image'
      : 'file'
  const name = firstString(input.name, input.filename)
  const path = firstString(input.path, input.file_path, input.filePath)
  const url = firstString(input.url, input.previewUrl, input.preview_url)
  const mimeType = firstString(input.mimeType, input.mime_type, input.media_type)
  const data = stringOrUndefined(input.data)
  if (!rawType && !name && !path && !url && !mimeType && !data) return null
  const lineStart = numberOrUndefined(input.lineStart ?? input.line_start)
  const lineEnd = numberOrUndefined(input.lineEnd ?? input.line_end)
  const note = stringOrUndefined(input.note)
  const quote = stringOrUndefined(input.quote)
  return {
    type,
    ...(name ? { name } : path ? { name: path.split('/').filter(Boolean).pop() ?? path } : {}),
    ...(path ? { path } : {}),
    ...(url ? { url } : {}),
    ...(mimeType ? { mimeType } : {}),
    ...(data ? { data } : {}),
    ...(Boolean(input.isDirectory ?? input.is_directory) ? { isDirectory: true } : {}),
    ...(lineStart != null ? { lineStart } : {}),
    ...(lineEnd != null ? { lineEnd } : {}),
    ...(note ? { note } : {}),
    ...(quote ? { quote } : {}),
  }
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

function numberOrUndefined(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function numberOrNull(value: unknown): number | null {
  return numberOrUndefined(value) ?? null
}


function parseDiagnosticLine(line: string): DiagnosticEntry | null {
  const match = line.match(/^\s*(ERROR|WARNING|INFO|HINT)\s+(\d+):(\d+)\s+([^:]+):\s*(.*)$/i)
  if (!match) return null
  const sourceAndCode = decodeXmlText(match[4]!.trim())
  const sourceMatch = sourceAndCode.match(/^(.*?)\s+\[([^\]]+)\]$/)
  return {
    severity: match[1]!.toLowerCase(),
    line: Number(match[2]),
    column: Number(match[3]),
    source: (sourceMatch?.[1] ?? sourceAndCode).trim(),
    ...(sourceMatch?.[2] ? { code: sourceMatch[2] } : {}),
    message: decodeXmlText(match[5]!.trim()),
  }
}

function diagnosticsXml(text: string): string | null {
  const trimmed = text.trim()
  const full = trimmed.match(/^<diagnostics>\s*[\s\S]*<\/diagnostics>$/i)
  if (full) return trimmed
  return trimmed.match(/<diagnostics>\s*[\s\S]*?<\/diagnostics>/i)?.[0] ?? null
}

function taskNotificationXml(text: string): string | null {
  const trimmed = text.trim()
  const full = trimmed.match(/^<task-notification>\s*[\s\S]*<\/task-notification>$/i)
  if (full) return trimmed
  return trimmed.match(/<task-notification>\s*[\s\S]*?<\/task-notification>/i)?.[0] ?? null
}

function readXmlTag(xml: string, tag: string): string | undefined {
  const match = xml.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, 'i'))
  return match?.[1] ? decodeXmlText(match[1].trim()) : undefined
}

function decodeXmlText(text: string): string {
  return text
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&amp;/g, '&')
}
