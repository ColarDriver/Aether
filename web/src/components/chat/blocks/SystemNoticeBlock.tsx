import type { SystemNoticeBlock as SystemNotice } from '../../../chat-rendering'

type Props = {
  block: SystemNotice
}

export function SystemNoticeBlock({ block }: Props) {
  return <div className="chat-block chat-block-system">{block.content}</div>
}
