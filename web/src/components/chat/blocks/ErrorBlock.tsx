import type { ErrorBlock as ErrorChatBlock } from '../../../chat-rendering'

type Props = {
  block: ErrorChatBlock
}

export function ErrorBlock({ block }: Props) {
  return (
    <article className="chat-block chat-block-error" role="alert">
      <strong>Error</strong>
      <p>{block.message}</p>
    </article>
  )
}
