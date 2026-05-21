import { Boxes, CircleGauge, Folder, Paperclip, Route, Send, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { SlashCommandInfo } from '../../api/types'
import type { WorkspaceEntry } from '../../api/types'
import type { ChatAttachment } from '../../chat-rendering'
import { Button } from '../shared/Button'
import { AttachmentGallery } from './AttachmentGallery'
import { SlashPopover, type SlashPopoverHandle } from './SlashPopover'
import { WorkspaceReferencePopover, type WorkspaceReferencePopoverHandle } from './WorkspaceReferencePopover'
import { attachmentsFromFiles, filesFromDataTransfer } from './composerAttachments'
import { isSlashCommandInput } from './slashExecute'
import { mergeWorkspaceAttachment } from './workspaceReferences'

type Props = {
  disabled: boolean
  running: boolean
  onSend: (message: string, attachments?: ChatAttachment[]) => void
  onCancel: () => void
  onSlashCommand?: (command: string) => void
  slashCommands?: SlashCommandInfo[]
  provider?: string | null
  model?: string | null
  mode?: string | null
  inputTokens?: number | null
  outputTokens?: number | null
}

export function Composer({
  disabled,
  running,
  onSend,
  onCancel,
  onSlashCommand,
  slashCommands,
  provider,
  model,
  mode,
  inputTokens,
  outputTokens,
}: Props) {
  const [value, setValue] = useState('')
  const [attachments, setAttachments] = useState<ChatAttachment[]>([])
  const [cursorPosition, setCursorPosition] = useState(0)
  const [loadedCommands, setLoadedCommands] = useState<SlashCommandInfo[]>([])
  const [dragActive, setDragActive] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const slashPopoverRef = useRef<SlashPopoverHandle>(null)
  const workspaceReferencePopoverRef = useRef<WorkspaceReferencePopoverHandle>(null)
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
    const hasAttachments = attachments.length > 0
    if ((!text && !hasAttachments) || disabled || running) return
    const sentAttachments = attachments
    setValue('')
    setAttachments([])
    setCursorPosition(0)
    if (onSlashCommand && !hasAttachments && isSlashCommandInput(text)) {
      onSlashCommand(text)
      return
    }
    if (sentAttachments.length > 0) {
      onSend(text, sentAttachments)
    } else {
      onSend(text)
    }
  }

  const applySlashCommand = (nextValue: string, nextCursorPosition: number) => {
    setValue(nextValue)
    setCursorPosition(nextCursorPosition)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
    })
  }

  const applyWorkspaceReference = (nextValue: string, nextCursorPosition: number, entry: WorkspaceEntry) => {
    setValue(nextValue)
    setCursorPosition(nextCursorPosition)
    setAttachments((current) => mergeWorkspaceAttachment(current, entry))
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
    })
  }

  const updateCursorPosition = () => {
    setCursorPosition(textareaRef.current?.selectionStart ?? value.length)
  }

  const addFiles = async (files: Iterable<File>) => {
    if (disabled || running) return
    const nextAttachments = await attachmentsFromFiles(files)
    if (nextAttachments.length > 0) {
      setAttachments((current) => [...current, ...nextAttachments])
    }
  }

  const handleDataTransfer = (dataTransfer: DataTransfer | null) => {
    const files = filesFromDataTransfer(dataTransfer)
    if (files.length === 0) return false
    void addFiles(files)
    return true
  }

  return (
    <div
      className={'composer' + (dragActive ? ' composer-drag-active' : '')}
      onDragEnter={(event) => {
        if (disabled || running) return
        if (filesFromDataTransfer(event.dataTransfer).length > 0) setDragActive(true)
      }}
      onDragLeave={(event) => {
        if (event.currentTarget.contains(event.relatedTarget as Node | null)) return
        setDragActive(false)
      }}
      onDragOver={(event) => {
        if (disabled || running) return
        if (filesFromDataTransfer(event.dataTransfer).length === 0) return
        event.preventDefault()
        setDragActive(true)
      }}
      onDrop={(event) => {
        setDragActive(false)
        if (handleDataTransfer(event.dataTransfer)) event.preventDefault()
      }}
    >
      <SlashPopover
        commands={commands}
        cursorPosition={cursorPosition}
        disabled={disabled || running || attachments.length > 0}
        onApply={applySlashCommand}
        ref={slashPopoverRef}
        value={value}
      />
      <WorkspaceReferencePopover
        cursorPosition={cursorPosition}
        disabled={disabled || running}
        onApply={applyWorkspaceReference}
        ref={workspaceReferencePopoverRef}
        value={value}
      />
      {attachments.length > 0 ? (
        <div className="composer-attachments">
          <AttachmentGallery
            attachments={attachments}
            onRemove={(index) => setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index))}
          />
        </div>
      ) : null}
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
        onPaste={(event) => {
          if (handleDataTransfer(event.clipboardData)) event.preventDefault()
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey && isExactSlashCommand(value, commands)) {
            event.preventDefault()
            submit()
            return
          }
          if (workspaceReferencePopoverRef.current?.handleKey(event)) return
          if (slashPopoverRef.current?.handleKey(event)) return
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            submit()
          }
        }}
        onKeyUp={updateCursorPosition}
      />
      <div className="composer-footer">
        <div className="composer-context">
          <span className="composer-chip composer-chip-model" title={provider && model ? provider + ' / ' + model : 'Provider not loaded'}>
            <Boxes size={14} />
            <span>{provider && model ? model : 'Model'}</span>
          </span>
          <span className="composer-chip" title="Workspace references use @path search">
            <Folder size={14} />
            <span>Workspace</span>
          </span>
          {mode ? (
            <span className={mode === 'plan' ? 'composer-chip composer-chip-plan' : 'composer-chip'} title="Current session mode">
              <Route size={14} />
              <span>{mode}</span>
            </span>
          ) : null}
          <ContextRing inputTokens={inputTokens} outputTokens={outputTokens} />
        </div>
        <div className="composer-actions">
          <input
            ref={fileInputRef}
            className="composer-file-input"
            type="file"
            multiple
            onChange={(event) => {
              const files = event.target.files ? Array.from(event.target.files) : []
              event.target.value = ''
              void addFiles(files)
            }}
          />
          <Button
            aria-label="Attach files"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || running}
          >
            <Paperclip size={16} />
          </Button>
          {running ? (
            <Button aria-label="Stop run" onClick={onCancel}>
              <Square size={16} />
            </Button>
          ) : (
            <Button aria-label="Send message" onClick={submit} disabled={disabled || (!value.trim() && attachments.length === 0)}>
              <Send size={16} />
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}

function isExactSlashCommand(value: string, commands: SlashCommandInfo[]): boolean {
  const trimmed = value.trim()
  if (!isSlashCommandInput(trimmed)) return false
  const head = trimmed.split(/\s+/, 1)[0]
  if (head !== trimmed) return true
  return commands.some((command) => command.name === head)
}

function ContextRing({ inputTokens, outputTokens }: { inputTokens?: number | null; outputTokens?: number | null }) {
  const total = Math.max(0, (inputTokens ?? 0) + (outputTokens ?? 0))
  const percent = total > 0 ? Math.min(99, Math.max(1, Math.round(total / 2000))) : 0
  const circumference = 61.261
  const offset = circumference - (circumference * percent) / 100
  const title = total > 0
    ? `${total.toLocaleString()} active-run tokens (${inputTokens ?? 0} in / ${outputTokens ?? 0} out)`
    : 'No active-run token usage yet'

  return (
    <span className="composer-context-ring" title={title} aria-label={title}>
      <CircleGauge size={14} className="composer-context-ring-icon" />
      <span className="ctx-ring">
        <svg className="ctx-ring-svg" viewBox="0 0 24 24" aria-hidden="true">
          <circle className="ctx-ring-track" cx="12" cy="12" r="9.75" />
          <circle
            className="ctx-ring-value"
            cx="12"
            cy="12"
            r="9.75"
            style={{ strokeDashoffset: offset }}
          />
        </svg>
        <span className="ctx-ring-center">{percent > 0 ? percent : '-'}</span>
      </span>
    </span>
  )
}
