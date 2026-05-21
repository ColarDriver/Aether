import type { ChatBlock, DiffBlock, ToolCallBlock, ToolResultBlock } from './blocks'

export type ToolGroupItem = {
  kind: 'tool_group'
  id: string
  toolCalls: ToolCallBlock[]
}

export type ChatRenderItem =
  | { kind: 'block'; block: ChatBlock }
  | ToolGroupItem

export type ChatRenderModel = {
  items: ChatRenderItem[]
  toolResultsByCallId: Map<string, ToolResultBlock>
  diffsByToolCallId: Map<string, DiffBlock[]>
  childToolCallsByParent: Map<string, ToolCallBlock[]>
}

export function buildChatRenderModel(blocks: ChatBlock[]): ChatRenderModel {
  const items: ChatRenderItem[] = []
  const toolResultsByCallId = new Map<string, ToolResultBlock>()
  const diffsByToolCallId = new Map<string, DiffBlock[]>()
  const childToolCallsByParent = new Map<string, ToolCallBlock[]>()
  const toolCallIds = new Set<string>()
  const associatedResultIds = new Set<string>()
  let pendingToolCalls: ToolCallBlock[] = []

  for (const block of blocks) {
    if (block.kind === 'tool_call') {
      toolCallIds.add(block.toolCallId)
      associatedResultIds.add(block.toolCallId)
    }
    if (block.kind === 'ask_user_question' && block.toolCallId) associatedResultIds.add(block.toolCallId)
    if (block.kind === 'tool_result') toolResultsByCallId.set(block.toolCallId, block)
    if (block.kind === 'diff') {
      const toolCallId = toolCallIdFromDiff(block)
      if (toolCallId) {
        const diffs = diffsByToolCallId.get(toolCallId) ?? []
        diffs.push(block)
        diffsByToolCallId.set(toolCallId, diffs)
      }
    }
  }

  const flushTools = () => {
    if (pendingToolCalls.length === 0) return
    items.push({
      kind: 'tool_group',
      id: 'tool-group-' + pendingToolCalls[0]!.id,
      toolCalls: pendingToolCalls,
    })
    pendingToolCalls = []
  }

  for (const block of blocks) {
    if (block.kind === 'assistant_message' && !block.content.trim()) continue

    if (block.kind === 'tool_result' && associatedResultIds.has(block.toolCallId)) {
      continue
    }

    if (block.kind === 'diff') {
      const toolCallId = toolCallIdFromDiff(block)
      if (toolCallId && associatedResultIds.has(toolCallId)) continue
    }

    if (block.kind === 'tool_call') {
      if (block.parentToolCallId && toolCallIds.has(block.parentToolCallId)) {
        flushTools()
        const siblings = childToolCallsByParent.get(block.parentToolCallId) ?? []
        siblings.push(block)
        childToolCallsByParent.set(block.parentToolCallId, siblings)
      } else {
        pendingToolCalls.push(block)
      }
      continue
    }

    flushTools()
    items.push({ kind: 'block', block })
  }

  flushTools()
  return { items, toolResultsByCallId, diffsByToolCallId, childToolCallsByParent }
}

function toolCallIdFromDiff(block: DiffBlock): string | null {
  const metadataToolCallId = block.path?.startsWith('tool:')
    ? block.path.slice('tool:'.length)
    : null
  return metadataToolCallId || block.id.match(/(?:diff-)(.+)$/)?.[1] || null
}
