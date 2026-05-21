import type { ApprovalPrompt } from '../../stores/chatStore'
import { MarkdownRenderer } from './MarkdownRenderer'

type Props = {
  prompt: ApprovalPrompt
  onApprove: () => void
  onReject: () => void
}

export function ApprovalDialog({ prompt, onApprove, onReject }: Props) {
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="prompt-modal" role="dialog" aria-modal="true" aria-label="Approval request">
        <header>
          <strong>{prompt.kind === 'plan' ? 'Approve plan' : 'Answer questions'}</strong>
          {prompt.planPath ? <span>{prompt.planPath}</span> : null}
        </header>
        <div className="prompt-body">
          {prompt.planText ? <MarkdownRenderer text={prompt.planText} /> : null}
          {prompt.questions.length > 0 ? (
            <pre>{JSON.stringify(prompt.questions, null, 2)}</pre>
          ) : null}
        </div>
        <footer>
          <button type="button" onClick={onReject}>Reject</button>
          <button type="button" className="primary-action" onClick={onApprove}>Approve</button>
        </footer>
      </section>
    </div>
  )
}
