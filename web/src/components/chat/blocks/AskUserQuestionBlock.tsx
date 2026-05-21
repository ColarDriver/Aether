import { HelpCircle, MessageCircleQuestion } from 'lucide-react'
import type { AskUserQuestionBlock as AskUserQuestion } from '../../../chat-rendering'

type Props = {
  block: AskUserQuestion
}

export function AskUserQuestionBlock({ block }: Props) {
  return (
    <article className={'chat-block prompt-inline-block prompt-inline-' + block.state}>
      <header>
        <span className="prompt-inline-icon"><HelpCircle size={16} /></span>
        <div>
          <strong>Input requested</strong>
          <small>{block.questions.length} question{block.questions.length === 1 ? '' : 's'}</small>
        </div>
        <span>{block.state}</span>
      </header>
      {block.questions.map((question, index) => (
        <div className="question-preview" key={index}>
          {question.header ? (
            <div className="question-preview-header">
              <MessageCircleQuestion size={14} />
              <span>{question.header}</span>
            </div>
          ) : null}
          <p>{question.question}</p>
          {question.id ? <code>{question.id}</code> : null}
          {question.options?.length ? (
            <div className="question-options">
              {question.options.map((option) => (
                <span key={option.label}>
                  <strong>{option.label}</strong>
                  {option.description ? <small>{option.description}</small> : null}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ))}
      {block.answers && Object.keys(block.answers).length > 0 ? (
        <div className="question-answers">
          <strong>User answered</strong>
          {Object.entries(block.answers).map(([label, value]) => (
            <div key={label}>
              <code>{label}</code>
              <p>{value}</p>
            </div>
          ))}
        </div>
      ) : null}
    </article>
  )
}
