import type { ApprovalRequestBlock as ApprovalRequest } from '../../../chat-rendering'
import { MarkdownRenderer } from '../MarkdownRenderer'

type Props = {
  block: ApprovalRequest
  onRespond?: (result: Record<string, unknown>) => void
}

export function ApprovalRequestBlock({ block, onRespond }: Props) {
  return (
    <article className="chat-block prompt-inline-block">
      <header>
        <strong>{block.approvalKind === 'plan' ? 'Plan approval' : 'Approval request'}</strong>
        <span>{block.state}</span>
      </header>
      {block.planPath ? <div className="muted">{block.planPath}</div> : null}
      {block.planText ? <MarkdownRenderer text={block.planText} /> : null}
      {block.questions.length > 0 ? <pre>{JSON.stringify(block.questions, null, 2)}</pre> : null}
      {block.state === 'pending' && onRespond ? (
        <footer>
          <button type="button" onClick={() => onRespond({ confirmed: false })}>Reject</button>
          <button type="button" onClick={() => onRespond({ confirmed: true })}>Approve</button>
        </footer>
      ) : null}
    </article>
  )
}
