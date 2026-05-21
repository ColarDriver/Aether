import { CircleAlert } from 'lucide-react'
import type { ErrorBlock as ErrorChatBlock } from '../../../chat-rendering'

type Props = {
  block: ErrorChatBlock
}

export function ErrorBlock({ block }: Props) {
  return (
    <article className="chat-block chat-block-error" role="alert">
      <header>
        <span className="error-icon" aria-hidden="true"><CircleAlert size={15} /></span>
        <div>
          <strong>Error</strong>
          {block.code ? <small>{block.code}</small> : null}
        </div>
      </header>
      <p>{block.message}</p>
    </article>
  )
}
