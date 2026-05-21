import type { ApprovalPrompt } from '../../stores/chatStore'
import { parseAskUserQuestions } from '../../chat-rendering'
import { ApprovalContent } from './PromptContent'

type Props = {
  prompt: ApprovalPrompt
  onApprove: (answers?: Record<string, string>) => void
  onReject: () => void
}

export function ApprovalDialog({ prompt, onApprove, onReject }: Props) {
  const questions = parseAskUserQuestions({ questions: prompt.questions })
  const isQuestionApproval = prompt.kind === 'questions' || questions.length > 0
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="prompt-modal" role="dialog" aria-modal="true" aria-label="Approval request">
        <header>
          <strong>{prompt.kind === 'plan' ? 'Approve plan' : isQuestionApproval ? 'Answer questions' : 'Approval request'}</strong>
          {prompt.planPath ? <span>{prompt.planPath}</span> : null}
        </header>
        <div className="prompt-body">
          <ApprovalContent
            approvalKind={prompt.kind}
            planPath={prompt.planPath}
            planText={prompt.planText}
            questions={questions}
            showPlanPath={false}
            onSubmitAnswers={(answers) => onApprove(answers)}
          />
        </div>
        <footer>
          <button type="button" onClick={onReject}>Reject</button>
          {!isQuestionApproval ? (
            <button type="button" className="primary-action" onClick={() => onApprove()}>Approve</button>
          ) : null}
        </footer>
      </section>
    </div>
  )
}
