import type { ToolBlock } from '../../stores/chatStore'

type Props = {
  tool: ToolBlock
}

export function ToolCallBlock({ tool }: Props) {
  return (
    <div className={`tool-block${tool.isError ? ' tool-block-error' : ''}`}>
      <div className="tool-block-header">
        <strong>{tool.toolName}</strong>
        <span>{tool.status}</span>
      </div>
      <pre>{JSON.stringify(tool.arguments, null, 2)}</pre>
      {tool.content ? <pre className="tool-result">{tool.content}</pre> : null}
    </div>
  )
}
