import { CheckCircle2, ClipboardCheck, HelpCircle, XCircle } from 'lucide-react'
import type { ApprovalRequestBlock as ApprovalRequest } from '../../../chat-rendering'
import { ApprovalContent } from '../PromptContent'

type Props = {
  block: ApprovalRequest
  onRespond?: (result: Record<string, unknown>) => void
}

export function ApprovalRequestBlock({ block, onRespond }: Props) {
  const isQuestionApproval = block.approvalKind === 'questions' || block.questions.length > 0
  const Icon = approvalIcon(block.state, isQuestionApproval)
  return (
    <article className={'chat-block prompt-inline-block prompt-inline-' + block.state}>
      <header>
        <span className="prompt-inline-icon"><Icon size={16} /></span>
        <div>
          <strong>{block.approvalKind === 'plan' ? 'Plan approval' : isQuestionApproval ? 'Answer questions' : 'Approval request'}</strong>
          {block.planPath ? <small>{block.planPath}</small> : null}
        </div>
        <span>{block.state}</span>
      </header>
      <ApprovalContent
        approvalKind={block.approvalKind}
        disabled={block.state !== 'pending'}
        planPath={block.planPath}
        planText={block.planText}
        questions={block.questions}
        onSubmitAnswers={(answers) => onRespond?.({ confirmed: true, answers })}
      />
      {block.statusMessage ? <p className="prompt-inline-status">{block.statusMessage}</p> : null}
      {block.state === 'pending' && onRespond && !isQuestionApproval ? (
        <footer>
          <button type="button" onClick={() => onRespond({ confirmed: false })}>Reject</button>
          <button type="button" onClick={() => onRespond({ confirmed: true })}>Approve</button>
        </footer>
      ) : null}
    </article>
  )
}

function approvalIcon(state: string, isQuestionApproval: boolean) {
  if (state === 'approved' || state === 'answered') return CheckCircle2
  if (state === 'rejected') return XCircle
  if (state === 'expired' || state === 'stale' || state === 'missing' || state === 'disconnected') return XCircle
  if (isQuestionApproval) return HelpCircle
  return ClipboardCheck
}
