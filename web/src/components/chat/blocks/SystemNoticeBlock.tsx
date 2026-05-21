import { Info } from 'lucide-react'
import type { SystemNoticeBlock as SystemNotice } from '../../../chat-rendering'
import { MarkdownRenderer } from '../MarkdownRenderer'

type Props = {
  block: SystemNotice
}

export function SystemNoticeBlock({ block }: Props) {
  if (block.content.includes('\n')) {
    return (
      <article className="chat-block chat-block-system chat-block-system-rich">
        <header>
          <span className="system-notice-icon" aria-hidden="true"><Info size={14} /></span>
          <strong>System notice</strong>
        </header>
        <div className="system-notice-body">
          <MarkdownRenderer text={block.content} />
        </div>
      </article>
    )
  }
  return <div className="chat-block chat-block-system">{block.content}</div>
}
