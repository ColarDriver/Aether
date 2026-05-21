import type { DiffBlock as DiffChatBlock } from '../../../chat-rendering'
import { DiffViewer } from '../DiffViewer'

type Props = {
  block: DiffChatBlock
}

export function DiffBlock({ block }: Props) {
  const diff = block.diff ?? diffFromOldNew(block.oldText ?? '', block.newText ?? '')
  if (!diff.trim()) return null
  return (
    <div className="chat-block chat-block-diff">
      {block.path ? <div className="diff-path">{block.path}</div> : null}
      <DiffViewer diff={diff} />
    </div>
  )
}

function diffFromOldNew(oldText: string, newText: string): string {
  const oldLines = oldText ? oldText.split('\n').map((line) => '-' + line) : []
  const newLines = newText ? newText.split('\n').map((line) => '+' + line) : []
  return [...oldLines, ...newLines].join('\n')
}
