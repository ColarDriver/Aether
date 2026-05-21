import type { DiffBlock, ToolCallBlock as ToolCall, ToolResultBlock as ToolResult } from '../../../chat-rendering'
import { ToolCallBlock } from './ToolCallBlock'

type Props = {
  toolCalls: ToolCall[]
  results: Map<string, ToolResult>
  diffs: Map<string, DiffBlock[]>
}

export function ToolCallGroup({ toolCalls, results, diffs }: Props) {
  return (
    <div className="tool-call-group">
      {toolCalls.map((toolCall) => (
        <ToolCallBlock
          block={toolCall}
          key={toolCall.id}
          result={results.get(toolCall.toolCallId)}
          diffs={diffs.get(toolCall.toolCallId) ?? []}
        />
      ))}
    </div>
  )
}
