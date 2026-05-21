import { ClipboardCheck, HelpCircle } from 'lucide-react'
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
  const Icon = isQuestionApproval ? HelpCircle : ClipboardCheck
  return (
    <div className="modal-backdrop" role="presentation">
      <section className="prompt-modal" role="dialog" aria-modal="true" aria-label="Approval request">
        <header>
          <span className="prompt-modal-icon" aria-hidden="true"><Icon size={17} /></span>
          <div className="prompt-modal-title">
            <strong>{prompt.kind === 'plan' ? 'Approve plan' : isQuestionApproval ? 'Answer questions' : 'Approval request'}</strong>
            <span>{prompt.planPath || (isQuestionApproval ? 'model is waiting for input' : 'approval required')}</span>
          </div>
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
          <button type="button" className="prompt-action prompt-action-danger" onClick={onReject}>Reject</button>
          {!isQuestionApproval ? (
            <button type="button" className="prompt-action prompt-action-primary" onClick={() => onApprove()}>Approve</button>
          ) : null}
        </footer>
      </section>
    </div>
  )
}
