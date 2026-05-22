import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { CodeBlock } from './CodeBlock'
import { TerminalChrome } from './TerminalChrome'
import { canPreviewToolResult, ToolResultPreview } from './ToolResultPreview'

type Props = {
  block: ToolResult
  command?: string | null
  toolArguments?: Record<string, unknown>
}

export function ToolResultBlock({ block, command, toolArguments = {} }: Props) {
  if (isTerminalResult(block)) {
    return (
      <TerminalChrome
        command={command || metadataString(block.metadata, 'command') || metadataString(block.metadata, 'cmd')}
        output={block.content}
        isError={block.isError}
        exitCode={metadataNumber(block.metadata, 'exit_code') ?? metadataNumber(block.metadata, 'exitCode')}
        durationMs={metadataNumber(block.metadata, 'duration_ms') ?? metadataNumber(block.metadata, 'durationMs')}
      />
    )
  }

  if (canPreviewToolResult(block)) return <ToolResultPreview block={block} toolArguments={toolArguments} />

  return (
    <div className={'tool-result-block' + (block.isError ? ' tool-result-block-error' : '')}>
      <div className="tool-result-header">{block.isError ? 'Error output' : 'Tool output'}</div>
      <CodeBlock code={block.content} language={languageFromMetadata(block.metadata)} wrap />
    </div>
  )
}

function languageFromMetadata(metadata: Record<string, unknown>): string {
  return typeof metadata.language === 'string'
    ? metadata.language
    : typeof metadata.lang === 'string'
      ? metadata.lang
      : 'text'
}

function isTerminalResult(block: ToolResult): boolean {
  const toolName = (block.toolName || '').toLowerCase()
  if (['bash', 'shell', 'exec', 'exec_command', 'terminal', 'run_shell'].includes(toolName)) return true
  const kind = metadataString(block.metadata, 'kind') || metadataString(block.metadata, 'view')
  if (kind && ['terminal', 'shell', 'command'].includes(kind.toLowerCase())) return true
  const language = languageFromMetadata(block.metadata).toLowerCase()
  return ['bash', 'sh', 'shell', 'zsh'].includes(language)
}

function metadataString(metadata: Record<string, unknown>, key: string): string | null {
  return typeof metadata[key] === 'string' ? metadata[key] : null
}

function metadataNumber(metadata: Record<string, unknown>, key: string): number | null {
  return typeof metadata[key] === 'number' && Number.isFinite(metadata[key]) ? metadata[key] : null
}
