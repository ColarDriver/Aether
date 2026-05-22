import { CheckCircle2, HelpCircle, MessageCircleQuestion } from 'lucide-react'
import type { AskUserQuestion as Question, AskUserQuestionBlock as AskUserQuestion } from '../../../chat-rendering'

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
        <QuestionPreview question={question} index={index} answers={block.answers} state={block.state} key={questionKey(question) + '-' + index} />
      ))}
      {block.answers && unmatchedAnswers(block.questions, block.answers).length > 0 ? (
        <div className="question-answers question-answers-unmatched">
          <strong>Additional answers</strong>
          {unmatchedAnswers(block.questions, block.answers).map(([label, value]) => (
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

function QuestionPreview({ question, index, answers, state }: { question: Question; index: number; answers?: Record<string, string>; state: AskUserQuestion['state'] }) {
  const answer = answerForQuestion(question, answers)
  const selectedValues = selectedAnswerValues(answer)
  return (
    <div className="question-preview">
      {question.header ? (
        <div className="question-preview-header">
          <MessageCircleQuestion size={14} />
          <span>{question.header}</span>
        </div>
      ) : null}
      <p>{question.question}</p>
      <div className="question-preview-meta">
        <code>{question.id || 'question ' + (index + 1)}</code>
        {question.multiSelect ? <span>multi-select</span> : null}
        {question.freeText || !question.options?.length ? <span>free text</span> : null}
      </div>
      {question.options?.length ? (
        <div className="question-options question-options-preview">
          {question.options.map((option) => {
            const selected = optionSelected(option, selectedValues)
            return (
              <span className={selected ? 'question-option-selected' : undefined} key={option.id ?? option.label}>
                <strong>{option.label}</strong>
                {option.description ? <small>{option.description}</small> : null}
                {selected ? <em><CheckCircle2 size={12} aria-hidden="true" />selected</em> : null}
              </span>
            )
          })}
        </div>
      ) : null}
      {answer ? (
        <div className="question-answer-summary">
          <strong>{question.options?.length ? 'Selected answer' : 'Response'}</strong>
          <p>{answer}</p>
        </div>
      ) : state === 'pending' ? (
        <div className="question-answer-summary question-answer-pending">
          <strong>Awaiting answer</strong>
        </div>
      ) : null}
    </div>
  )
}

function answerForQuestion(question: Question, answers?: Record<string, string>): string | null {
  if (!answers) return null
  const keys = [question.id, question.header, question.question].filter((value): value is string => Boolean(value))
  for (const key of keys) {
    const direct = answers[key]
    if (direct?.trim()) return direct
  }
  const normalizedKeys = keys.map(normalizeAnswerKey)
  for (const [key, value] of Object.entries(answers)) {
    if (!value?.trim()) continue
    if (normalizedKeys.includes(normalizeAnswerKey(key))) return value
  }
  return null
}

function unmatchedAnswers(questions: Question[], answers?: Record<string, string>): Array<[string, string]> {
  if (!answers) return []
  return Object.entries(answers).filter(([key, value]) => {
    if (!value?.trim()) return false
    return !questions.some((question) => answerKeyMatchesQuestion(key, question))
  })
}

function answerKeyMatchesQuestion(key: string, question: Question): boolean {
  const normalized = normalizeAnswerKey(key)
  return [question.id, question.header, question.question]
    .filter((value): value is string => Boolean(value))
    .some((candidate) => normalizeAnswerKey(candidate) === normalized)
}

function optionSelected(option: { id?: string; label: string }, selectedValues: string[]): boolean {
  const candidates = [option.label, option.id].filter((value): value is string => Boolean(value)).map(normalizeAnswerKey)
  return selectedValues.some((value) => candidates.includes(normalizeAnswerKey(value)))
}

function selectedAnswerValues(answer: string | null): string[] {
  if (!answer) return []
  return answer
    .split(/[,;\n]/)
    .map((value) => value.trim())
    .filter(Boolean)
}

function normalizeAnswerKey(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9一-鿿]+/g, '')
}

function questionKey(question: Question): string {
  return question.id || question.header || question.question
}
