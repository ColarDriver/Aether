import type { ApprovalRequestBlock as ApprovalRequest } from '../../../chat-rendering'
import { ApprovalContent } from '../PromptContent'

type Props = {
  block: ApprovalRequest
  onRespond?: (result: Record<string, unknown>) => void
}

export function ApprovalRequestBlock({ block, onRespond }: Props) {
  const isQuestionApproval = block.approvalKind === 'questions' || block.questions.length > 0
  return (
    <article className="chat-block prompt-inline-block">
      <header>
        <strong>{block.approvalKind === 'plan' ? 'Plan approval' : isQuestionApproval ? 'Answer questions' : 'Approval request'}</strong>
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
      {block.state === 'pending' && onRespond && !isQuestionApproval ? (
        <footer>
          <button type="button" onClick={() => onRespond({ confirmed: false })}>Reject</button>
          <button type="button" onClick={() => onRespond({ confirmed: true })}>Approve</button>
        </footer>
      ) : null}
    </article>
  )
}
