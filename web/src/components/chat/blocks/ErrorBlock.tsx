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
      {block.details && block.details.length > 0 ? (
        <dl className="error-detail-list" aria-label="Error diagnostics">
          {block.details.map((detail) => (
            <div key={detail.label + detail.value}>
              <dt>{detail.label}</dt>
              <dd>{detail.value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {block.suggestions && block.suggestions.length > 0 ? (
        <div className="error-suggestions" aria-label="Suggested fixes">
          <strong>Suggested fixes</strong>
          <ul>
            {block.suggestions.map((suggestion) => <li key={suggestion}>{suggestion}</li>)}
          </ul>
        </div>
      ) : null}
    </article>
  )
}
