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
  overlay
}: {
  overlay: OverlayState<ApprovalRequestParams>
}): ReactElement {
  const params = overlay.payload
  if (params.kind === 'plan') {
    return <PlanApproval overlay={overlay} />
  }
  return <QuestionApproval overlay={overlay} />
}

// ──────────────────────────────────────────────────────────────────────────
// PLAN
// ──────────────────────────────────────────────────────────────────────────

function PlanApproval({
  overlay
}: {
  overlay: OverlayState<ApprovalRequestParams>
}): ReactElement {
  const params = overlay.payload
  const remaining = useCountdown(params.deadline_ms ?? 0)
  const border = theme.color('border')
  const title = theme.colorProps('info')
  const originalPlan = params.plan_text ?? '(no plan text provided)'
  const [currentPlan, setCurrentPlan] = useState(originalPlan)
  const [editMessage, setEditMessage] = useState<string | null>(null)

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
        <Text dimColor> · {overlay.id}</Text>
      </Box>
      {params.plan_path ? (
        <Box>
          <Text dimColor>{params.plan_path}</Text>
        </Box>
      ) : null}
      <Box marginTop={1} flexDirection="column">
        <PlanText text={currentPlan} />
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

function PlanText({ text }: { text: string }): ReactElement {
  return <Markdown text={text || '(no plan text provided)'} />
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

function QuestionApproval({
  overlay
}: {
  overlay: OverlayState<ApprovalRequestParams>
}): ReactElement {
  const params = overlay.payload
  const questions = useMemo(() => params.questions ?? [], [params])
  const [cursor, setCursor] = useState(0)
  const [answers, setAnswers] = useState<ApprovalAnswerMap>({})
  const [openValue, setOpenValue] = useState<string>('')
  const [selectIndex, setSelectIndex] = useState<number>(0)
  const remaining = useCountdown(params.deadline_ms ?? 0)

  const current = questions[cursor]
  const isOpen = !current || (current.kind ?? 'open') === 'open'

  // Defensive: a questions request with no questions is malformed; commit
  // empty so the engine does not hang waiting for a response.
  useEffect(() => {
    if (questions.length === 0) {
      overlayActions.dismiss(overlay.id, 'commit', { confirmed: true, answers: {} })
    }
  }, [overlay.id, questions])

  function commitAll(map: ApprovalAnswerMap): void {
    const payload: JsonObject = { confirmed: true, answers: map }
    overlayActions.dismiss(overlay.id, 'commit', payload)
  }

  function recordAndAdvance(value: string): void {
    if (!current) {
      commitAll(answers)
      return
    }
    const next: ApprovalAnswerMap = { ...answers, [current.id]: value }
    if (cursor + 1 >= questions.length) {
      commitAll(next)
      return
    }
    setAnswers(next)
    setCursor(cursor + 1)
    setOpenValue('')
    setSelectIndex(0)
  }

  useInput(
    (input, key) => {
      if (key.ctrl && input === 'c') {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (key.escape) {
        overlayActions.dismiss(overlay.id, 'cancel')
        return
      }
      if (!current) {
        return
      }
      if (isOpen) {
        // Open input — TextInput owns most keys; we still let Enter commit
        // through onSubmit.
        return
      }
      const options = current.options ?? []
      if (key.upArrow) {
        setSelectIndex((idx) => (idx <= 0 ? Math.max(options.length - 1, 0) : idx - 1))
        return
      }
      if (key.downArrow) {
        setSelectIndex((idx) => (options.length === 0 ? 0 : (idx + 1) % options.length))
        return
      }
      if (key.return) {
        const choice = options[selectIndex] ?? ''
        recordAndAdvance(choice)
        return
      }
      if (input >= '1' && input <= '9') {
        const idx = Number.parseInt(input, 10) - 1
        if (idx < options.length) {
          recordAndAdvance(options[idx] ?? '')
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

  return (
    <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1}>
      <Box>
        <Text bold color="yellow">
          Questions ({cursor + 1}/{questions.length})
        </Text>
        <Text dimColor> · {overlay.id}</Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        <Text>{current?.text}</Text>
      </Box>
      <Box marginTop={1} flexDirection="column">
        {isOpen ? (
          <Box>
            <Text color="cyan">› </Text>
            <TextInput
              value={openValue}
              onChange={setOpenValue}
              onSubmit={(value) => recordAndAdvance(value)}
            />
          </Box>
        ) : (
          <SelectList
            options={current?.options ?? []}
            selectedIndex={selectIndex}
            onSelect={(value) => recordAndAdvance(value)}
          />
        )}
      </Box>
      <Box marginTop={1}>
        <Text dimColor>
          {isOpen ? 'Enter to submit · ESC reject' : '↑/↓ navigate · Enter pick · 1-9 quick · ESC reject'}
        </Text>
        {remaining !== null ? (
          <Text dimColor> · {formatRemaining(remaining)}</Text>
        ) : null}
      </Box>
    </Box>
  )
}

function SelectList({
  options,
  selectedIndex,
  onSelect
}: {
  options: string[]
  selectedIndex: number
  onSelect: (value: string) => void
}): ReactElement {
  // Render-only — actual key handling lives in QuestionApproval so the parent
  // owns the answers map and progression.
  void onSelect
  return (
    <Box flexDirection="column">
      {options.map((option, idx) => {
        const isFocused = idx === selectedIndex
        const props = isFocused ? { color: 'cyan' } : {}
        return (
          <Text key={idx} {...props}>
            {isFocused ? '› ' : '  '}
            {option}
          </Text>
        )
      })}
    </Box>
  )
}

// ──────────────────────────────────────────────────────────────────────────
// Shared question utilities
// ──────────────────────────────────────────────────────────────────────────

export function questionDefaultValue(question: ApprovalQuestion): string {
  if ((question.kind ?? 'open') === 'select') {
    return question.options?.[0] ?? ''
  }
  return ''
}

// ──────────────────────────────────────────────────────────────────────────
// Countdown — small reusable hook
// ──────────────────────────────────────────────────────────────────────────

function useCountdown(deadlineMs: number): number | null {
  const [remaining, setRemaining] = useState<number | null>(() =>
    deadlineMs > 0 ? deadlineMs : null
  )

  useEffect(() => {
    if (deadlineMs <= 0) {
      setRemaining(null)
      return
    }
    const start = Date.now()
    setRemaining(deadlineMs)
    const handle = setInterval(() => {
      const elapsed = Date.now() - start
      const next = Math.max(0, deadlineMs - elapsed)
      setRemaining(next)
      if (next <= 0) {
        clearInterval(handle)
      }
    }, 500)
    return () => {
      clearInterval(handle)
    }
  }, [deadlineMs])

  return remaining
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
