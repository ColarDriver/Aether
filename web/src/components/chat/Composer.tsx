import { Send, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { SlashCommandInfo } from '../../api/types'
import { Button } from '../shared/Button'
import { SlashPopover, type SlashPopoverHandle } from './SlashPopover'

type Props = {
  disabled: boolean
  running: boolean
  onSend: (message: string) => void
  onCancel: () => void
  slashCommands?: SlashCommandInfo[]
}

export function Composer({ disabled, running, onSend, onCancel, slashCommands }: Props) {
  const [value, setValue] = useState('')
  const [cursorPosition, setCursorPosition] = useState(0)
  const [loadedCommands, setLoadedCommands] = useState<SlashCommandInfo[]>([])
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const slashPopoverRef = useRef<SlashPopoverHandle>(null)
  const commands = slashCommands ?? loadedCommands

  useEffect(() => {
    if (slashCommands || disabled) return
    let cancelled = false
    api.commands()
      .then((result) => {
        if (!cancelled) setLoadedCommands(result.commands)
      })
      .catch(() => {
        if (!cancelled) setLoadedCommands([])
      })
    return () => {
      cancelled = true
    }
  }, [disabled, slashCommands])

  const submit = () => {
    const text = value.trim()
    if (!text || disabled || running) return
    setValue('')
    setCursorPosition(0)
    onSend(text)
  }

  const applySlashCommand = (nextValue: string, nextCursorPosition: number) => {
    setValue(nextValue)
    setCursorPosition(nextCursorPosition)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
    })
  }

  const updateCursorPosition = () => {
    setCursorPosition(textareaRef.current?.selectionStart ?? value.length)
  }

  return (
    <div className="composer">
      <SlashPopover
        commands={commands}
        cursorPosition={cursorPosition}
        disabled={disabled || running}
        onApply={applySlashCommand}
        ref={slashPopoverRef}
        value={value}
      />
      <textarea
        ref={textareaRef}
        value={value}
        disabled={disabled}
        placeholder={disabled ? 'Select a session' : 'Ask Aether'}
        onChange={(event) => {
          setValue(event.target.value)
          setCursorPosition(event.target.selectionStart)
        }}
        onClick={updateCursorPosition}
        onKeyDown={(event) => {
          if (slashPopoverRef.current?.handleKey(event)) return
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
        onKeyUp={updateCursorPosition}
      />
      {running ? (
        <Button aria-label="Stop run" onClick={onCancel}>
          <Square size={16} />
        </Button>
      ) : (
        <Button aria-label="Send message" onClick={submit} disabled={disabled || !value.trim()}>
          <Send size={16} />
        </Button>
      )}
    </div>
  )
}
