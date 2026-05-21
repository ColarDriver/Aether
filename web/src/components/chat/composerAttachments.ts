import type { ChatAttachment } from '../../chat-rendering'

const INLINE_TEXT_LIMIT_BYTES = 128 * 1024

export async function attachmentsFromFiles(files: Iterable<File>): Promise<ChatAttachment[]> {
  const pending = Array.from(files).map(fileToAttachment)
  return (await Promise.all(pending)).filter((attachment): attachment is ChatAttachment => Boolean(attachment))
}

export function filesFromDataTransfer(dataTransfer: DataTransfer | null): File[] {
  if (!dataTransfer) return []
  if (dataTransfer.files.length > 0) return Array.from(dataTransfer.files)
  return Array.from(dataTransfer.items)
    .filter((item) => item.kind === 'file')
    .map((item) => item.getAsFile())
    .filter((file): file is File => Boolean(file))
}

async function fileToAttachment(file: File): Promise<ChatAttachment> {
  const mimeType = file.type || undefined
  if (mimeType?.startsWith('image/')) {
    return {
      type: 'image',
      name: file.name || 'image',
      ...(mimeType ? { mimeType } : {}),
      data: await readFile(file, 'data-url'),
      note: formatBytes(file.size),
    }
  }

  if (isTextFile(file) && file.size <= INLINE_TEXT_LIMIT_BYTES) {
    return {
      type: 'text',
      name: file.name || 'text',
      ...(mimeType ? { mimeType } : {}),
      data: await readFile(file, 'text'),
      note: formatBytes(file.size),
    }
  }

  return {
    type: 'file',
    name: file.name || 'file',
    ...(mimeType ? { mimeType } : {}),
    note: formatBytes(file.size),
  }
}

function readFile(file: File, mode: 'data-url' | 'text'): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file.'))
    if (mode === 'data-url') {
      reader.readAsDataURL(file)
    } else {
      reader.readAsText(file)
    }
  })
}

function isTextFile(file: File): boolean {
  if (file.type.startsWith('text/')) return true
  return /\.(md|markdown|txt|json|jsonl|yaml|yml|toml|csv|ts|tsx|js|jsx|py|rs|go|java|c|cc|cpp|h|hpp|css|html|xml|sh|bash|zsh)$/i.test(file.name)
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B'
  if (bytes < 1024) return bytes + ' B'
  const kib = bytes / 1024
  if (kib < 1024) return kib.toFixed(kib >= 10 ? 0 : 1) + ' KB'
  const mib = kib / 1024
  return mib.toFixed(mib >= 10 ? 0 : 1) + ' MB'
}
