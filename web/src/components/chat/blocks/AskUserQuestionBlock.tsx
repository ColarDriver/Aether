import type { AskUserQuestionBlock as AskUserQuestion } from '../../../chat-rendering'

type Props = {
  block: AskUserQuestion
}

export function AskUserQuestionBlock({ block }: Props) {
  return (
    <article className="chat-block prompt-inline-block">
      <header>
        <strong>Input requested</strong>
        <span>{block.state}</span>
      </header>
      {block.questions.map((question, index) => (
        <div className="question-preview" key={index}>
          {question.header ? <div className="muted">{question.header}</div> : null}
          <p>{question.question}</p>
          {question.id ? <code>{question.id}</code> : null}
          {question.options?.length ? (
            <div className="question-options">
              {question.options.map((option) => <span key={option.label}>{option.label}</span>)}
            </div>
          ) : null}
        </div>
      ))}
    </article>
  )
}
