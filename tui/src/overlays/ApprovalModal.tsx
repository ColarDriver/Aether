import { spawn } from 'node:child_process'
import { readFile } from 'node:fs/promises'

import { Box, Text, useInput } from 'ink'
import TextInput from 'ink-text-input'
import { useEffect, useMemo, useState, type ReactElement } from 'react'

import type {
  ApprovalQuestion,
  ApprovalRequestParams,
  JsonObject
} from '../gatewayTypes.js'
import { Markdown } from '../lib/markdown.js'
import { theme } from '../lib/theme.js'
import { overlayActions, type OverlayState } from '../store/overlayStore.js'

export interface ApprovalAnswerMap {
  [questionId: string]: string
}

export function ApprovalModal({
  overlay,
  maxRows
}: {
  overlay: OverlayState<ApprovalRequestParams>
  maxRows?: number
}): ReactElement {
  const params = overlay.payload
  if (params.kind === 'plan') {
    return (
      <PlanApproval
        overlay={overlay}
        {...(maxRows !== undefined ? { maxRows } : {})}
      />
    )
  }
  return <QuestionApproval overlay={overlay} />
}

// ──────────────────────────────────────────────────────────────────────────
// PLAN
// ──────────────────────────────────────────────────────────────────────────

function PlanApproval({
  overlay,
  maxRows
}: {
  overlay: OverlayState<ApprovalRequestParams>
  maxRows?: number
}): ReactElement {
  const params = overlay.payload
  const remaining = useCountdown(params.deadline_ms ?? 0)
  const border = theme.color('border')
  const title = theme.colorProps('info')
  const originalPlan = params.plan_text ?? '(no plan text provided)'
  const [currentPlan, setCurrentPlan] = useState(originalPlan)
  const [editMessage, setEditMessage] = useState<string | null>(null)
  const planRows = maxRows ? Math.max(1, maxRows - 6) : undefined

  function approvalPayload(confirmed: boolean): JsonObject {
    const payload: JsonObject = { confirmed, answers: {} }
    if (confirmed && currentPlan !== originalPlan) {
      payload.updated_input = { plan: currentPlan }
    }
    return payload
  }

  useInput((input, key) => {
    if (key.ctrl && input === 'c') {
      overlayActions.dismiss(overlay.id, 'cancel')
      return
    }
    if (key.ctrl && input === 'g') {
      if (!params.plan_path) {
        setEditMessage('Plan path is unavailable')
        return
      }
      void openPlanInEditor(params.plan_path).then(
        (content) => {
          setCurrentPlan(content)
          setEditMessage('Plan updated from editor')
        },
        (error: unknown) => {
          const message = error instanceof Error ? error.message : String(error)
          setEditMessage(`Could not edit plan: ${message}`)
        }
      )
      return
    }
    if (key.escape) {
      overlayActions.dismiss(overlay.id, 'commit', approvalPayload(false))
      return
    }
    if (input === 'y' || input === 'Y' || key.return) {
      overlayActions.dismiss(overlay.id, 'commit', approvalPayload(true))
      return
    }
    if (input === 'n' || input === 'N') {
      overlayActions.dismiss(overlay.id, 'commit', approvalPayload(false))
      return
    }
  })

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      {...(border ? { borderColor: border } : {})}
      paddingX={1}
    >
      <Box>
        <Text bold {...title}>
          Plan approval
        </Text>
      </Box>
      {params.plan_path ? (
        <Box>
          <Text dimColor>{params.plan_path}</Text>
        </Box>
      ) : null}
      <Box marginTop={1} flexDirection="column">
        <PlanText
          text={currentPlan}
          {...(planRows !== undefined ? { maxRows: planRows } : {})}
        />
      </Box>
      <Box marginTop={1}>
        <Text>
          [<Text color="green">Y</Text>] approve · [<Text color="red">N</Text>] reject ·
          ESC reject
        </Text>
        {params.plan_path ? <Text dimColor> · Ctrl+G edit</Text> : null}
        {remaining !== null ? (
          <Text dimColor> · {formatRemaining(remaining)}</Text>
        ) : null}
      </Box>
      {editMessage ? (
        <Box>
          <Text dimColor>{editMessage}</Text>
        </Box>
      ) : null}
    </Box>
  )
}

function PlanText({ text, maxRows }: { text: string; maxRows?: number }): ReactElement {
  return <Markdown text={limitMarkdownRows(text || '(no plan text provided)', maxRows)} />
}

function limitMarkdownRows(text: string, maxRows: number | undefined): string {
  if (!maxRows || maxRows < 1) {
    return text
  }
  const lines = text.split('\n')
  if (lines.length <= maxRows) {
    return text
  }
  return [...lines.slice(0, Math.max(1, maxRows - 1)), '... plan truncated in terminal view ...'].join('\n')
}

async function openPlanInEditor(path: string): Promise<string> {
  const editor = process.env.AETHER_EDITOR || process.env.VISUAL || process.env.EDITOR
  if (!editor?.trim()) {
    throw new Error('EDITOR is not configured')
  }
  await new Promise<void>((resolve, reject) => {
    const child = spawn(editor, [path], {
      shell: true,
      stdio: 'inherit'
    })
    child.once('error', reject)
    child.once('exit', (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`editor exited with code ${code ?? 'unknown'}`))
      }
    })
  })
  return readFile(path, 'utf8')
}

// ──────────────────────────────────────────────────────────────────────────
// QUESTIONS
// ──────────────────────────────────────────────────────────────────────────

type RowKind = 'option' | 'free' | 'chat' | 'skip'

interface QuestionRow {
  kind: RowKind
  number: number
  label: string
  description?: string
  optionLabel?: string
}

function buildRows(question: ApprovalQuestion | undefined, planMode: boolean): QuestionRow[] {
  const rows: QuestionRow[] = []
  if (!question) {
    return rows
  }
  const options = question.options ?? []
  options.forEach((opt, idx) => {
    rows.push({
      kind: 'option',
      number: idx + 1,
      label: opt.label,
      ...(opt.description !== undefined ? { description: opt.description } : {}),
      optionLabel: opt.label
    })
  })
  rows.push({ kind: 'free', number: rows.length + 1, label: 'Type something.' })
  rows.push({ kind: 'chat', number: rows.length + 1, label: 'Chat about this' })
  if (planMode) {
    rows.push({
      kind: 'skip',
      number: rows.length + 1,
      label: 'Skip interview and plan immediately'
    })
  }
  return rows
}

function QuestionApproval({
  overlay
}: {
  overlay: OverlayState<ApprovalRequestParams>
}): ReactElement {
  const params = overlay.payload
  const questions = useMemo(() => params.questions ?? [], [params])
  const planMode = Boolean(params.plan_path)
  const [cursor, setCursor] = useState(0) // 0..questions.length-1 is a question, == length is the Submit chip
  const [answers, setAnswers] = useState<ApprovalAnswerMap>({})
  const [selectIndex, setSelectIndex] = useState<number>(0)
  const [mode, setMode] = useState<'list' | 'freeText'>('list')
  const [freeTextValue, setFreeTextValue] = useState<string>('')
  const remaining = useCountdown(params.deadline_ms ?? 0)

  const current = cursor < questions.length ? questions[cursor] : undefined
  const rows = useMemo(() => buildRows(current, planMode), [current, planMode])

  // Defensive: a questions request with no questions is malformed; commit
  // empty so the engine does not hang waiting for a response.
  useEffect(() => {
    if (questions.length === 0) {
      overlayActions.dismiss(overlay.id, 'commit', { confirmed: true, answers: {} })
    }
  }, [overlay.id, questions])

  function commitAnswers(
    map: ApprovalAnswerMap,
    extra: { action?: 'chat' | 'skip'; confirmed?: boolean } = {}
  ): void {
    const payload: JsonObject = {
      confirmed: extra.confirmed ?? true,
      answers: map
    }
    if (extra.action) {
      payload.action = extra.action
    }
    overlayActions.dismiss(overlay.id, 'commit', payload)
  }

  function advanceCursor(nextAnswers: ApprovalAnswerMap): void {
    // Skip questions that already have answers; land on the next pending
    // question, or the Submit chip when all are answered.
    let next = cursor + 1
    while (next < questions.length) {
      const q = questions[next]
      if (q && nextAnswers[q.id] === undefined) {
        break
      }
      next += 1
    }
    setCursor(Math.min(next, questions.length))
    setSelectIndex(0)
    setMode('list')
    setFreeTextValue('')
  }

  function recordAnswer(value: string): void {
    if (!current) {
      commitAnswers(answers)
      return
    }
    const next: ApprovalAnswerMap = { ...answers, [current.id]: value }
    setAnswers(next)
    if (questions.every((q) => next[q.id] !== undefined)) {
      commitAnswers(next)
      return
    }
    advanceCursor(next)
  }

  function activateRow(row: QuestionRow): void {
    if (row.kind === 'option') {
      recordAnswer(row.optionLabel ?? row.label)
      return
    }
    if (row.kind === 'free') {
      setMode('freeText')
      setFreeTextValue('')
      return
    }
    if (row.kind === 'chat') {
      commitAnswers(answers, { confirmed: false, action: 'chat' })
      return
    }
    if (row.kind === 'skip') {
      commitAnswers({}, { confirmed: true, action: 'skip' })
    }
  }

  useInput(
    (input, key) => {
      if (key.ctrl && input === 'c') {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (key.escape) {
        if (mode === 'freeText') {
          setMode('list')
          setFreeTextValue('')
          return
        }
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (mode === 'freeText') {
        // TextInput owns Enter / character keys via onChange/onSubmit below.
        return
      }
      if (key.tab) {
        // Single-question prompts auto-commit when the user picks an option,
        // so the chip bar (and Tab navigation) is hidden — swallow Tab to
        // avoid trapping the user on an invisible Submit chip.
        if (questions.length <= 1) {
          return
        }
        const total = questions.length + 1 // +1 for the Submit chip
        const direction = key.shift ? -1 : 1
        const nextCursor = (cursor + direction + total) % total
        setCursor(nextCursor)
        setSelectIndex(0)
        return
      }
      if (cursor === questions.length) {
        // Submit chip is focused — Enter commits.
        if (key.return) {
          commitAnswers(answers)
        }
        return
      }
      if (rows.length === 0) {
        return
      }
      if (key.upArrow) {
        setSelectIndex((idx) => (idx <= 0 ? rows.length - 1 : idx - 1))
        return
      }
      if (key.downArrow) {
        setSelectIndex((idx) => (idx + 1) % rows.length)
        return
      }
      if (key.return) {
        const row = rows[selectIndex]
        if (row) {
          activateRow(row)
        }
        return
      }
      if (input >= '1' && input <= '9') {
        const num = Number.parseInt(input, 10)
        const row = rows.find((r) => r.number === num)
        if (row) {
          setSelectIndex(rows.indexOf(row))
          activateRow(row)
        }
      }
    },
    { isActive: true }
  )

  if (questions.length === 0) {
    return (
      <Box borderStyle="round" borderColor="yellow" paddingX={1}>
        <Text>Empty questionnaire — auto-confirming.</Text>
      </Box>
    )
  }

  const onSubmitChip = cursor === questions.length

  return (
    <Box flexDirection="column">
      {params.plan_path ? (
        <Box>
          <Text dimColor>Planning: {params.plan_path}</Text>
        </Box>
      ) : null}
      {questions.length > 1 ? (
        <ChipBar
          questions={questions}
          answers={answers}
          cursor={cursor}
        />
      ) : null}
      <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
        {onSubmitChip ? (
          <Box flexDirection="column">
            <Text bold color="yellow">
              Submit
            </Text>
            <Box marginTop={1}>
              <Text>Press Enter to send answers to the agent.</Text>
            </Box>
          </Box>
        ) : (
          <>
            <Box>
              <Text bold color="yellow">
                {current?.text || `Question ${cursor + 1}/${questions.length}`}
              </Text>
            </Box>
            <Box marginTop={1} flexDirection="column">
              {rows.map((row, idx) => {
                const focused = idx === selectIndex
                if (row.kind === 'free' && mode === 'freeText' && focused) {
                  return (
                    <Box key={`${row.kind}-${row.number}`}>
                      <Text color="cyan">
                        › {row.number}.{' '}
                      </Text>
                      <TextInput
                        value={freeTextValue}
                        onChange={setFreeTextValue}
                        onSubmit={(value) => {
                          if (value.trim().length === 0) {
                            return
                          }
                          recordAnswer(value)
                        }}
                      />
                    </Box>
                  )
                }
                if (row.kind === 'chat' && idx === firstFooterIndex(rows)) {
                  return (
                    <Box key={`${row.kind}-${row.number}`} flexDirection="column">
                      <Text dimColor>{'─'.repeat(40)}</Text>
                      <QuestionRowView row={row} focused={focused} />
                    </Box>
                  )
                }
                return (
                  <QuestionRowView
                    key={`${row.kind}-${row.number}`}
                    row={row}
                    focused={focused}
                  />
                )
              })}
            </Box>
          </>
        )}
        <Box marginTop={1}>
          <Text dimColor>
            Enter to select · Tab/Arrow keys to navigate · Esc to cancel
          </Text>
          {remaining !== null ? (
            <Text dimColor> · {formatRemaining(remaining)}</Text>
          ) : null}
        </Box>
      </Box>
    </Box>
  )
}

function firstFooterIndex(rows: QuestionRow[]): number {
  return rows.findIndex((r) => r.kind === 'chat')
}

function QuestionRowView({ row, focused }: { row: QuestionRow; focused: boolean }): ReactElement {
  const pointerProps = focused ? { color: 'cyan' as const } : {}
  return (
    <Box flexDirection="column">
      <Box>
        <Text {...pointerProps}>{focused ? '›' : ' '} </Text>
        <Text {...pointerProps} bold={focused}>
          {row.number}. {row.label}
        </Text>
      </Box>
      {row.description ? (
        <Box>
          <Text>{'     '}</Text>
          <Text dimColor>{row.description}</Text>
        </Box>
      ) : null}
    </Box>
  )
}

function ChipBar({
  questions,
  answers,
  cursor
}: {
  questions: ApprovalQuestion[]
  answers: ApprovalAnswerMap
  cursor: number
}): ReactElement {
  return (
    <Box>
      <Text dimColor>← </Text>
      {questions.map((q, idx) => {
        const answered = answers[q.id] !== undefined
        const focused = idx === cursor
        // ASCII glyphs avoid CJK width-overlap that `☐` / `✓` exhibit in
        // monospace fonts when followed by Chinese / Japanese / Korean text.
        const glyph = answered ? '[x]' : '[ ]'
        const label = chipLabel(q, idx)
        const textProps = focused
          ? { color: 'cyan' as const, bold: true }
          : answered
            ? { color: 'green' as const }
            : { dimColor: true }
        return (
          <Text key={q.id} {...textProps}>
            {glyph} {label}
            {idx < questions.length - 1 ? '  ' : ' '}
          </Text>
        )
      })}
      <Text
        {...(cursor === questions.length
          ? { color: 'cyan' as const, bold: true }
          : { color: 'green' as const })}
      >
        [Submit]{' '}
      </Text>
      <Text dimColor>→</Text>
    </Box>
  )
}

function chipLabel(question: ApprovalQuestion, index: number): string {
  const header = (question.header ?? '').trim()
  if (header) {
    return header
  }
  const text = (question.text ?? '').trim()
  if (text) {
    return text.length > 12 ? `${text.slice(0, 11)}…` : text
  }
  return `Q${index + 1}`
}

// ──────────────────────────────────────────────────────────────────────────
// Shared question utilities
// ──────────────────────────────────────────────────────────────────────────

export function questionDefaultValue(question: ApprovalQuestion): string {
  if ((question.kind ?? 'open') === 'select') {
    return question.options?.[0]?.label ?? ''
  }
  return ''
}

// ──────────────────────────────────────────────────────────────────────────
// Countdown — small reusable hook
// ──────────────────────────────────────────────────────────────────────────

function useCountdown(deadlineMs: number): number | null {
  return deadlineMs > 0 ? deadlineMs : null
}

function formatRemaining(ms: number): string {
  if (ms <= 0) {
    return '0s left'
  }
  if (ms < 60_000) {
    return `${Math.ceil(ms / 1000)}s left`
  }
  const minutes = Math.floor(ms / 60_000)
  const seconds = Math.ceil((ms % 60_000) / 1000)
  return `${minutes}m${seconds.toString().padStart(2, '0')}s left`
}
