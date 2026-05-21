import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { CodeBlock } from './CodeBlock'

type Props = {
  block: ToolResult
}

export function ToolResultBlock({ block }: Props) {
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
