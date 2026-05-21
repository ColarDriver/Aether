import type { TranscriptMessage } from '../api/types'
import type { ChatBlock, ToolStatus } from './blocks'
import { extractDiffFromMetadata, parseAskUserQuestions, recordFromUnknown } from './content'

export function normalizeTranscript(sessionId: string, transcript: TranscriptMessage[]): ChatBlock[] {
  const blocks: ChatBlock[] = []

  transcript.forEach((message, index) => {
    const timestamp = index
    const runId = 'persisted-' + index

    if (message.role === 'user') {
      blocks.push({
        id: 'persisted-' + index + '-user',
        sessionId,
        runId,
        timestamp,
        source: 'transcript',
        kind: 'user_message',
        content: message.text ?? '',
      })
      return
    }

    if (message.role === 'system') {
      blocks.push({
        id: 'persisted-' + index + '-system',
        sessionId,
        runId,
        timestamp,
        source: 'transcript',
        kind: 'system_notice',
        content: message.text ?? '',
      })
      return
    }

    if (message.role === 'assistant') {
      const text = message.text ?? ''
      if (text.trim() || !message.tool_calls?.length) {
        blocks.push({
          id: 'persisted-' + index + '-assistant',
          sessionId,
          runId,
          timestamp,
          source: 'transcript',
          kind: 'assistant_message',
          content: text,
          isError: Boolean(message.is_error),
        })
      }

      for (const toolCall of message.tool_calls ?? []) {
        blocks.push(toolCallBlock({
          id: 'persisted-' + index + '-tool-' + toolCall.id,
          sessionId,
          runId,
          timestamp,
          toolCallId: toolCall.id,
          toolName: toolCall.name || 'tool',
          args: recordFromUnknown(toolCall.arguments),
          status: 'running',
        }))
      }
      return
    }

    if (message.role === 'tool') {
      const metadata = recordFromUnknown(message.metadata)
      const toolCallId = message.tool_call_id || 'tool-' + index
      blocks.push({
        id: 'persisted-' + index + '-result-' + toolCallId,
        sessionId,
        runId,
        timestamp,
        source: 'transcript',
        kind: 'tool_result',
        toolCallId,
        toolName: message.name || 'tool',
        content: message.text ?? '',
        isError: Boolean(message.is_error),
        metadata,
      })

      const diff = extractDiffFromMetadata(metadata)
      if (diff) {
        blocks.push({
          id: 'persisted-' + index + '-diff-' + toolCallId,
          sessionId,
          runId,
          timestamp,
          source: 'transcript',
          kind: 'diff',
          origin: 'tool_result',
          ...diff,
        })
      }
    }
  })

  return blocks
}

function toolCallBlock(input: {
  id: string
  sessionId: string
  runId: string
  timestamp: number
  toolCallId: string
  toolName: string
  args: Record<string, unknown>
  status: ToolStatus
}): ChatBlock {
  if (isAskUserQuestionTool(input.toolName)) {
    return {
      id: input.id,
      sessionId: input.sessionId,
      runId: input.runId,
      timestamp: input.timestamp,
      source: 'transcript',
      kind: 'ask_user_question',
      toolCallId: input.toolCallId,
      questions: parseAskUserQuestions(input.args),
      state: 'pending',
    }
  }

  return {
    id: input.id,
    sessionId: input.sessionId,
    runId: input.runId,
    timestamp: input.timestamp,
    source: 'transcript',
    kind: 'tool_call',
    toolCallId: input.toolCallId,
    toolName: input.toolName,
    arguments: input.args,
    status: input.status,
  }
}

function isAskUserQuestionTool(toolName: string): boolean {
  return toolName.toLowerCase() === 'ask_user_question' || toolName === 'AskUserQuestion'
}
