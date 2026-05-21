import type { SystemNoticeBlock as SystemNotice } from '../../../chat-rendering'
import { MarkdownRenderer } from '../MarkdownRenderer'

type Props = {
  block: SystemNotice
}

export function SystemNoticeBlock({ block }: Props) {
  if (block.content.includes('\n')) {
    return (
      <article className="chat-block chat-block-system chat-block-system-rich">
        <MarkdownRenderer text={block.content} />
      </article>
    )
  }
  return <div className="chat-block chat-block-system">{block.content}</div>
}
