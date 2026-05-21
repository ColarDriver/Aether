import { useMemo, useRef, useState } from 'react'
import type { AskUserQuestion, PermissionPreview } from '../../chat-rendering'
import { MarkdownRenderer } from './MarkdownRenderer'
import { DiffViewer } from './DiffViewer'

type PermissionPreviewContentProps = {
  preview?: PermissionPreview | null
  reason?: string | null
  args?: Record<string, unknown>
  showArguments?: boolean
}

export function PermissionPreviewContent({
  preview,
  reason,
  args = {},
  showArguments = true,
}: PermissionPreviewContentProps) {
  return (
    <div className="prompt-content">
      {preview?.subtitle ? <p className="muted">{preview.subtitle}</p> : null}
      {reason ? <p>{reason}</p> : null}
      {preview?.command ? <pre className="command-preview">{preview.command}</pre> : null}
      {preview?.body ? <p>{preview.body}</p> : null}
      {preview?.diff ? <DiffViewer diff={preview.diff} /> : null}
      {showArguments && Object.keys(args).length > 0 ? (
        <pre className="prompt-json-preview">{JSON.stringify(args, null, 2)}</pre>
      ) : null}
    </div>
  )
}

type ApprovalContentProps = {
  approvalKind: string
  planPath?: string | null
  planText?: string | null
  questions?: AskUserQuestion[]
  disabled?: boolean
  showPlanPath?: boolean
  onSubmitAnswers?: (answers: Record<string, string>) => void
}

export function ApprovalContent({
  planPath,
  planText,
  questions = [],
  disabled,
  showPlanPath = true,
  onSubmitAnswers,
}: ApprovalContentProps) {
  return (
    <div className="prompt-content">
      {showPlanPath && planPath ? <div className="muted">{planPath}</div> : null}
      {planText ? <MarkdownRenderer text={planText} /> : null}
      {questions.length > 0 ? (
        <QuestionAnswerForm
          disabled={disabled}
          questions={questions}
          onSubmit={(answers) => onSubmitAnswers?.(answers)}
        />
      ) : null}
    </div>
  )
}

export function QuestionAnswerForm({
  questions,
  disabled,
  onSubmit,
}: {
  questions: AskUserQuestion[]
  disabled?: boolean
  onSubmit: (answers: Record<string, string>) => void
}) {
  const [selected, setSelected] = useState<Record<string, string[]>>({})
  const [freeText, setFreeText] = useState<Record<string, string>>({})
  const composingRef = useRef(false)
  const answers = useMemo(() => buildQuestionAnswers(questions, selected, freeText), [freeText, questions, selected])
  const canSubmit = questions.length > 0 && questions.every((question) => Boolean(answers[questionKey(question)]?.trim()))

  const toggle = (question: AskUserQuestion, label: string) => {
    const key = questionKey(question)
    setSelected((current) => {
      const values = current[key] ?? []
      if (question.multiSelect) {
        const nextValues = values.includes(label)
          ? values.filter((value) => value !== label)
          : [...values, label]
        return { ...current, [key]: nextValues }
      }
      return { ...current, [key]: values[0] === label ? [] : [label] }
    })
    setFreeText((current) => ({ ...current, [key]: '' }))
  }

  const submit = () => {
    if (disabled || !canSubmit) return
    onSubmit(answers)
  }

  return (
    <div className="question-form">
      {questions.map((question, index) => {
        const key = questionKey(question)
        const values = selected[key] ?? []
        return (
          <section className="question-form-item" key={key}>
            {question.header ? <div className="muted">{question.header}</div> : null}
            <p>{question.question}</p>
            {question.options?.length ? (
              <div className="question-options">
                {question.options.map((option) => {
                  const active = values.includes(option.label)
                  return (
                    <button
                      aria-pressed={active}
                      disabled={disabled}
                      key={option.id ?? option.label}
                      type="button"
                      className={active ? 'question-option question-option-active' : 'question-option'}
                      onClick={() => toggle(question, option.label)}
                    >
                      <span>{option.label}</span>
                      {option.description ? <small>{option.description}</small> : null}
                    </button>
                  )
                })}
              </div>
            ) : null}
            {question.freeText || !question.options?.length ? (
              <input
                aria-label={'Answer ' + (index + 1)}
                disabled={disabled}
                value={freeText[key] ?? ''}
                onChange={(event) => {
                  setFreeText((current) => ({ ...current, [key]: event.target.value }))
                  setSelected((current) => ({ ...current, [key]: [] }))
                }}
                onCompositionStart={() => {
                  composingRef.current = true
                }}
                onCompositionEnd={() => {
                  composingRef.current = false
                }}
                onKeyDown={(event) => {
                  if (composingRef.current || event.nativeEvent.isComposing || event.keyCode === 229) return
                  if (event.key === 'Enter') submit()
                }}
                placeholder="Type an answer"
                type="text"
              />
            ) : null}
          </section>
        )
      })}
      {!disabled ? (
        <footer>
          <button type="button" disabled={!canSubmit} onClick={submit}>Submit answers</button>
        </footer>
      ) : null}
    </div>
  )
}

export function buildQuestionAnswers(
  questions: AskUserQuestion[],
  selected: Record<string, string[]>,
  freeText: Record<string, string>,
): Record<string, string> {
  const answers: Record<string, string> = {}
  for (const question of questions) {
    const key = questionKey(question)
    const typed = freeText[key]?.trim()
    if (typed) {
      answers[key] = typed
      continue
    }
    const values = selected[key] ?? []
    if (values.length > 0) answers[key] = values.join(', ')
  }
  return answers
}

export function questionKey(question: AskUserQuestion): string {
  return question.id || question.question
}
