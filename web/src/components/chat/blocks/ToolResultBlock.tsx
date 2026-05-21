import type { ToolResultBlock as ToolResult } from '../../../chat-rendering'

type Props = {
  block: ToolResult
}

export function ToolResultBlock({ block }: Props) {
  return (
    <div className={'tool-result-block' + (block.isError ? ' tool-result-block-error' : '')}>
      <div className="tool-result-header">{block.isError ? 'Error output' : 'Tool output'}</div>
      <pre>{block.content}</pre>
    </div>
  )
}
