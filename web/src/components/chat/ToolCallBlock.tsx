import type { ToolBlock } from '../../stores/chatStore'
import { DiffViewer } from './DiffViewer'

type Props = {
  tool: ToolBlock
}

export function ToolCallBlock({ tool }: Props) {
  const diff = typeof tool.metadata?.diff === 'string' ? tool.metadata.diff : null
  return (
    <div className={'tool-block' + (tool.isError ? ' tool-block-error' : '')}>
      <div className="tool-block-header">
        <strong>{tool.toolName}</strong>
        <span>{tool.status}</span>
      </div>
      <pre>{JSON.stringify(tool.arguments, null, 2)}</pre>
      {diff ? <DiffViewer diff={diff} /> : null}
      {tool.content ? <pre className="tool-result">{tool.content}</pre> : null}
    </div>
  )
}
