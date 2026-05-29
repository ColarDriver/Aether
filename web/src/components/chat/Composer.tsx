import { Activity, AtSign, BarChart3, Boxes, Brain, ChevronDown, ChevronLeft, ChevronRight, CircleGauge, Command, Eye, FileText, Folder, Paperclip, Plus, Route, Send, Server, Sparkles, Square, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { KeyboardEvent, RefObject } from 'react'
import { api } from '../../api/client'
import type { ContextStatus, PermissionMode, SlashCommandInfo, WorkspaceEntry, WorkspaceFile } from '../../api/types'
import type { ChatAttachment, TokenUsage } from '../../chat-rendering'
import { tokenUsageBreakdown, tokenUsageTotal } from '../../chat-rendering'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'
import { Button } from '../shared/Button'
import { CopyButton } from '../shared/CopyButton'
import { AttachmentGallery } from './AttachmentGallery'
import { ComposerInspectorPanel, type ComposerInspectorKind } from './ComposerInspectorPanel'
import { MarkdownRenderer } from './MarkdownRenderer'
import { PermissionModeSelector } from './PermissionModeSelector'
import { SlashPopover, type SlashPopoverHandle } from './SlashPopover'
import { WorkspaceReferencePopover, type WorkspaceReferencePopoverHandle } from './WorkspaceReferencePopover'
import { attachmentsFromFiles, filesFromDataTransfer } from './composerAttachments'
import { isSlashCommandInput, WEB_LOCAL_INSPECTOR_COMMANDS } from './slashExecute'
import { mergeWorkspaceAttachment, syncWorkspaceReferenceAttachmentsForValue, workspaceReferenceTokenExists } from './workspaceReferences'

type ComposerDraft = {
  value: string
  attachments: ChatAttachment[]
  cursorPosition: number
}

export type ComposerDraftPatch = {
  id: number
  mode: 'replace' | 'append'
  text: string
  attachments?: ChatAttachment[]
}

const NEW_SESSION_DRAFT_KEY = '__aether_new_session__'

type WorkspacePreviewState =
  | { status: 'idle' }
  | { status: 'loading'; path: string }
  | { status: 'ready'; path: string; file: WorkspaceFile }
  | { status: 'error'; path: string; message: string }

type Props = {
  disabled: boolean
  running: boolean
  sessionId?: string | null
  onSend: (message: string, attachments?: ChatAttachment[]) => void
  onCancel: () => void
  onSlashCommand?: (command: string) => void
  slashCommands?: SlashCommandInfo[]
  provider?: string | null
  model?: string | null
  mode?: string | null
  permissionMode?: PermissionMode | string | null
  onPermissionModeChange?: (mode: PermissionMode) => Promise<void> | void
  inputTokens?: number | null
  outputTokens?: number | null
  tokens?: TokenUsage | null
  runMetadata?: Record<string, unknown> | null
  sessionSummary?: string | null
  messageCount?: number | null
  draftPatch?: ComposerDraftPatch | null
  workspaceRootVersion?: number
}

export function Composer({
  disabled,
  running,
  sessionId,
  onSend,
  onCancel,
  onSlashCommand,
  slashCommands,
  provider,
  model,
  mode,
  permissionMode,
  onPermissionModeChange,
  inputTokens,
  outputTokens,
  tokens,
  runMetadata,
  sessionSummary,
  messageCount,
  draftPatch,
  workspaceRootVersion = 0,
}: Props) {
  const [value, setValue] = useState('')
  const [attachments, setAttachments] = useState<ChatAttachment[]>([])
  const [cursorPosition, setCursorPosition] = useState(0)
  const [loadedCommands, setLoadedCommands] = useState<SlashCommandInfo[]>([])
  const [dragActive, setDragActive] = useState(false)
  const [inspectorKind, setInspectorKind] = useState<ComposerInspectorKind | null>(null)
  const [controlMenuOpen, setControlMenuOpen] = useState(false)
  const [workspaceRoot, setWorkspaceRoot] = useState<string | null>(null)
  const [workspaceRootError, setWorkspaceRootError] = useState<string | null>(null)
  const [workspacePreview, setWorkspacePreview] = useState<WorkspacePreviewState>({ status: 'idle' })
  const [contextEstimate, setContextEstimate] = useState<ContextStatus | null>(null)
  const [contextEstimateError, setContextEstimateError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const controlMenuRef = useRef<HTMLDivElement>(null)
  const slashPopoverRef = useRef<SlashPopoverHandle>(null)
  const workspaceReferencePopoverRef = useRef<WorkspaceReferencePopoverHandle>(null)
  const appliedDraftPatchIdRef = useRef<number | null>(null)
  const workspaceRootVersionRef = useRef(workspaceRootVersion)
  const draftKey = sessionId ?? NEW_SESSION_DRAFT_KEY
  const draftMapRef = useRef(new Map<string, ComposerDraft>())
  const draftKeyRef = useRef(draftKey)
  const valueRef = useRef(value)
  const attachmentsRef = useRef(attachments)
  const cursorPositionRef = useRef(cursorPosition)
  const commands = mergeLocalInspectorCommands(slashCommands ?? loadedCommands)

  useEffect(() => {
    valueRef.current = value
    attachmentsRef.current = attachments
    cursorPositionRef.current = cursorPosition
    draftMapRef.current.set(draftKeyRef.current, { value, attachments, cursorPosition })
  }, [attachments, cursorPosition, value])

  useEffect(() => {
    if (draftKeyRef.current === draftKey) return
    draftMapRef.current.set(draftKeyRef.current, {
      value: valueRef.current,
      attachments: attachmentsRef.current,
      cursorPosition: cursorPositionRef.current,
    })
    draftKeyRef.current = draftKey
    const draft = draftMapRef.current.get(draftKey)
    setValue(draft?.value ?? '')
    setAttachments(draft?.attachments ?? [])
    setCursorPosition(draft?.cursorPosition ?? 0)
    setInspectorKind(null)
    setControlMenuOpen(false)
  }, [draftKey])

  useEffect(() => {
    if (!draftPatch || appliedDraftPatchIdRef.current === draftPatch.id || disabled) return
    appliedDraftPatchIdRef.current = draftPatch.id
    const currentValue = valueRef.current
    const nextValue = draftPatch.mode === 'append'
      ? appendDraftText(currentValue, draftPatch.text)
      : draftPatch.text
    const nextAttachments = draftPatch.mode === 'append'
      ? mergeDraftAttachments(attachmentsRef.current, draftPatch.attachments ?? [])
      : draftPatch.attachments ?? []
    const nextCursorPosition = nextValue.length
    setValue(nextValue)
    setAttachments(nextAttachments)
    setCursorPosition(nextCursorPosition)
    setInspectorKind(null)
    setControlMenuOpen(false)
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
    })
  }, [disabled, draftPatch])

  useEffect(() => {
    if (!controlMenuOpen) return
    const handler = (event: MouseEvent) => {
      if (controlMenuRef.current && !controlMenuRef.current.contains(event.target as Node)) setControlMenuOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [controlMenuOpen])

  useEffect(() => {
    if (slashCommands || disabled) return
    let cancelled = false
    api.commands()
      .then((result) => {
        if (!cancelled) setLoadedCommands(Array.isArray(result.commands) ? result.commands : [])
      })
      .catch(() => {
        if (!cancelled) setLoadedCommands([])
      })
    return () => {
      cancelled = true
    }
  }, [disabled, slashCommands])

  useEffect(() => {
    const hasDraftContext = value.trim().length > 0 || attachments.length > 0
    if (!sessionId || disabled || !hasDraftContext) {
      setContextEstimate(null)
      setContextEstimateError(null)
      return
    }
    let cancelled = false
    const handle = window.setTimeout(() => {
      api.estimateContext(sessionId, {
        draft: value,
        attachments: attachments.map(contextEstimateAttachment),
      })
        .then((estimate) => {
          if (cancelled) return
          setContextEstimate(estimate)
          setContextEstimateError(null)
        })
        .catch((error: unknown) => {
          if (cancelled) return
          setContextEstimate(null)
          setContextEstimateError(error instanceof Error ? error.message : String(error))
        })
    }, 350)
    return () => {
      cancelled = true
      window.clearTimeout(handle)
    }
  }, [attachments, disabled, model, provider, sessionId, value])

  useEffect(() => {
    if (disabled) return
    const rootChanged = workspaceRootVersionRef.current !== workspaceRootVersion
    workspaceRootVersionRef.current = workspaceRootVersion
    if (rootChanged) {
      const paths = attachmentsRef.current
        .filter((attachment) => attachment.note === 'workspace reference' && attachment.path)
        .map((attachment) => attachment.path || '')
      if (paths.length > 0) {
        setAttachments((current) => current.filter((attachment) => attachment.note !== 'workspace reference'))
        setValue((current) => removeWorkspaceReferenceTokens(current, paths))
      }
      setWorkspacePreview({ status: 'idle' })
    }
    let cancelled = false
    api.workspaceRoot()
      .then((result) => {
        if (cancelled) return
        setWorkspaceRoot(result.root)
        setWorkspaceRootError(null)
      })
      .catch((error) => {
        if (cancelled) return
        setWorkspaceRoot(null)
        setWorkspaceRootError(error instanceof Error ? error.message : String(error))
      })
    return () => {
      cancelled = true
    }
  }, [disabled, workspaceRootVersion])

  const submit = () => {
    const text = value.trim()
    const hasAttachments = attachments.length > 0
    if ((!text && !hasAttachments) || disabled || running) return
    const sentAttachments = attachments
    setValue('')
    setAttachments([])
    setCursorPosition(0)
    const inspectorCommand = localInspectorKindForCommand(text)
    if (!hasAttachments && inspectorCommand) {
      setInspectorKind(inspectorCommand)
      return
    }
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

  const browseWorkspaceReference = (nextValue: string, nextCursorPosition: number) => {
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

  const focusTextareaAt = (nextCursorPosition: number) => {
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
    })
  }

  const syncWorkspaceContextForValue = (nextValue: string) => {
    setAttachments((current) => syncWorkspaceReferenceAttachmentsForValue(current, nextValue))
    setWorkspacePreview((current) => {
      if (current.status === 'idle') return current
      return workspaceReferenceTokenExists(nextValue, current.path) ? current : { status: 'idle' }
    })
  }

  const insertAtCursor = (token: string) => {
    if (disabled || running) return
    const start = textareaRef.current?.selectionStart ?? cursorPosition
    const end = textareaRef.current?.selectionEnd ?? cursorPosition
    const prefix = value.slice(0, start)
    const suffix = value.slice(end)
    const insert = prefix.length > 0 && !/\s$/.test(prefix) ? ' ' + token : token
    const nextValue = prefix + insert + suffix
    const nextCursorPosition = prefix.length + insert.length
    setValue(nextValue)
    setCursorPosition(nextCursorPosition)
    setControlMenuOpen(false)
    focusTextareaAt(nextCursorPosition)
  }

  const removeAttachmentAt = (index: number) => {
    setAttachments((current) => current.filter((_, itemIndex) => itemIndex !== index))
  }

  const removeWorkspaceReferenceAt = (index: number) => {
    const attachment = attachments[index]
    removeAttachmentAt(index)
    if (attachment?.path) {
      setValue((current) => removeWorkspaceReferenceTokens(current, [attachment.path || '']))
      setWorkspacePreview((current) => current.status !== 'idle' && current.path === attachment.path ? { status: 'idle' } : current)
    }
  }

  const clearWorkspaceReferences = () => {
    const paths = attachments
      .filter((attachment) => attachment.note === 'workspace reference' && attachment.path)
      .map((attachment) => attachment.path || '')
    setAttachments((current) => current.filter((attachment) => attachment.note !== 'workspace reference'))
    if (paths.length > 0) {
      setValue((current) => removeWorkspaceReferenceTokens(current, paths))
    }
    setWorkspacePreview({ status: 'idle' })
  }

  const moveWorkspaceReferenceAt = (index: number, direction: -1 | 1) => {
    setAttachments((current) => {
      const workspaceIndices = current
        .map((attachment, itemIndex) => ({ attachment, itemIndex }))
        .filter((item) => item.attachment.note === 'workspace reference')
        .map((item) => item.itemIndex)
      const position = workspaceIndices.indexOf(index)
      const swapIndex = workspaceIndices[position + direction]
      if (position < 0 || swapIndex == null) return current
      const next = [...current]
      const currentItem = next[index]
      next[index] = next[swapIndex]!
      next[swapIndex] = currentItem!
      return next
    })
  }

  const previewWorkspaceReference = (attachment: ChatAttachment) => {
    const path = attachment.path
    if (!path || attachment.isDirectory) return
    if (workspacePreview.status === 'ready' && workspacePreview.path === path) {
      setWorkspacePreview({ status: 'idle' })
      return
    }
    setWorkspacePreview({ status: 'loading', path })
    api.workspaceFile(path)
      .then((file) => setWorkspacePreview({ status: 'ready', path, file }))
      .catch((error: unknown) => {
        setWorkspacePreview({ status: 'error', path, message: error instanceof Error ? error.message : String(error) })
      })
  }

  const openInspector = (kind: ComposerInspectorKind) => {
    setInspectorKind(kind)
    setControlMenuOpen(false)
    focusTextareaAt(cursorPosition)
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

  const workspaceReferenceItems = attachments
    .map((attachment, index) => ({ attachment, index }))
    .filter((item) => item.attachment.note === 'workspace reference')
  const manualAttachmentItems = attachments
    .map((attachment, index) => ({ attachment, index }))
    .filter((item) => item.attachment.note !== 'workspace reference')

  return (
    <div
      className={'composer' + (dragActive ? ' composer-drag-active' : '') + (running ? ' composer-running' : '')}
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
        onBrowse={browseWorkspaceReference}
        ref={workspaceReferencePopoverRef}
        value={value}
      />
      {inspectorKind ? (
        <ComposerInspectorPanel
          kind={inspectorKind}
          sessionId={sessionId}
          sessionSummary={sessionSummary}
          messageCount={messageCount}
          provider={provider}
          model={model}
          mode={mode}
          inputTokens={inputTokens}
          outputTokens={outputTokens}
          tokens={tokens}
          runMetadata={runMetadata}
          contextEstimate={contextEstimate}
          contextEstimateError={contextEstimateError}
          onClose={() => setInspectorKind(null)}
        />
      ) : null}
      <div className="composer-drop-hint" aria-hidden={!dragActive}>
        <Paperclip size={18} />
        <span>Drop files to attach</span>
      </div>
      <WorkspaceContextStrip
        disabled={disabled || running}
        items={workspaceReferenceItems}
        onClear={clearWorkspaceReferences}
        onMove={moveWorkspaceReferenceAt}
        onPreview={previewWorkspaceReference}
        onRemove={removeWorkspaceReferenceAt}
        preview={workspacePreview}
      />
      {manualAttachmentItems.length > 0 ? (
        <div className="composer-attachments">
          <AttachmentGallery
            attachments={manualAttachmentItems.map((item) => item.attachment)}
            onRemove={(index) => {
              const item = manualAttachmentItems[index]
              if (item) removeAttachmentAt(item.index)
            }}
          />
        </div>
      ) : null}
      <textarea
        ref={textareaRef}
        value={value}
        disabled={disabled}
        placeholder={disabled ? 'Select a session' : 'Ask Aether'}
        onChange={(event) => {
          const nextValue = event.target.value
          setValue(nextValue)
          setCursorPosition(event.target.selectionStart)
          syncWorkspaceContextForValue(nextValue)
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
        <div className="composer-tools">
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
          <ComposerControlMenu
            disabled={disabled || running}
            open={controlMenuOpen}
            onToggle={() => setControlMenuOpen((current) => !current)}
            onAttach={() => {
              setControlMenuOpen(false)
              fileInputRef.current?.click()
            }}
            onSlash={() => insertAtCursor('/')}
            onWorkspace={() => insertAtCursor('@')}
            onInspector={openInspector}
            refEl={controlMenuRef}
          />
          <Button
            aria-label="Attach files"
            title="Attach files"
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled || running}
          >
            <Paperclip size={16} />
          </Button>
          <ProjectContextChip
            root={workspaceRoot}
            error={workspaceRootError}
            referenceCount={attachments.filter((attachment) => attachment.note === 'workspace reference').length}
            disabled={disabled || running}
            onInsertReference={() => insertAtCursor('@')}
            onClearReferences={clearWorkspaceReferences}
          />
          {mode ? (
            <span className={mode === 'plan' ? 'composer-chip composer-chip-plan' : 'composer-chip'} title="Current session mode">
              <Route size={14} />
              <span>
                <strong>{mode}</strong>
                <small>mode</small>
              </span>
            </span>
          ) : null}
          {onPermissionModeChange ? (
            <PermissionModeSelector
              value={permissionMode ?? (mode === 'plan' ? 'plan' : 'default')}
              disabled={disabled || running}
              onChange={onPermissionModeChange}
            />
          ) : null}
        </div>
        <div className="composer-runbar">
          <ModelChip provider={provider} model={model} disabled={disabled || running} sessionId={sessionId} />
          <ContextRing inputTokens={inputTokens} outputTokens={outputTokens} tokens={tokens} contextStatus={contextEstimate} contextError={contextEstimateError} />
          <div className="composer-actions">
            {running ? (
              <Button className="composer-stop-button" aria-label="Stop run" title="Stop run" onClick={onCancel}>
                <Square size={16} />
              </Button>
            ) : (
              <Button
                className="composer-send-button"
                aria-label="Send message"
                title="Send message"
                onClick={submit}
                disabled={disabled || (!value.trim() && attachments.length === 0)}
              >
                <Send size={16} />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


function contextEstimateAttachment(attachment: ChatAttachment): Record<string, unknown> {
  const payload: Record<string, unknown> = { type: attachment.type }
  for (const key of ['name', 'path', 'url', 'mimeType', 'data', 'note', 'quote'] as const) {
    const value = attachment[key]
    if (typeof value === 'string' && value.length > 0) payload[key] = value
  }
  if (typeof attachment.isDirectory === 'boolean') payload.isDirectory = attachment.isDirectory
  if (typeof attachment.lineStart === 'number') payload.lineStart = attachment.lineStart
  if (typeof attachment.lineEnd === 'number') payload.lineEnd = attachment.lineEnd
  return payload
}

function mergeLocalInspectorCommands(commands: SlashCommandInfo[] | undefined): SlashCommandInfo[] {
  const remoteCommands = Array.isArray(commands) ? commands : []
  const seen = new Set<string>()
  const merged: SlashCommandInfo[] = []
  for (const command of [...WEB_LOCAL_INSPECTOR_COMMANDS, ...remoteCommands]) {
    if (seen.has(command.name)) continue
    seen.add(command.name)
    merged.push(command)
  }
  return merged
}

function localInspectorKindForCommand(value: string): ComposerInspectorKind | null {
  const head = value.trim().split(/\s+/, 1)[0]
  if (head === '/status') return 'status'
  if (head === '/context') return 'context'
  if (head === '/cost') return 'cost'
  if (head === '/skills') return 'skills'
  if (head === '/mcp') return 'mcp'
  return null
}

function isExactSlashCommand(value: string, commands: SlashCommandInfo[]): boolean {
  const trimmed = value.trim()
  if (!isSlashCommandInput(trimmed)) return false
  const head = trimmed.split(/\s+/, 1)[0]
  if (head !== trimmed) return true
  return commands.some((command) => command.name === head)
}

function ComposerControlMenu({
  disabled,
  open,
  onToggle,
  onAttach,
  onSlash,
  onWorkspace,
  onInspector,
  refEl,
}: {
  disabled: boolean
  open: boolean
  onToggle: () => void
  onAttach: () => void
  onSlash: () => void
  onWorkspace: () => void
  onInspector: (kind: ComposerInspectorKind) => void
  refEl: RefObject<HTMLDivElement | null>
}) {
  return (
    <div className="composer-control-menu" ref={refEl}>
      <Button aria-label="Open composer menu" title="Open composer menu" onClick={onToggle} disabled={disabled}>
        <Plus size={16} />
      </Button>
      {open ? (
        <div className="composer-control-popover" role="menu" aria-label="Composer menu">
          <div className="composer-control-section">
            <button type="button" role="menuitem" onClick={onAttach}>
              <Paperclip size={14} />
              <span>Attach files</span>
            </button>
            <button type="button" role="menuitem" onClick={onSlash}>
              <Command size={14} />
              <span>Slash command</span>
            </button>
            <button type="button" role="menuitem" onClick={onWorkspace}>
              <AtSign size={14} />
              <span>Workspace reference</span>
            </button>
          </div>
          <div className="composer-control-section">
            <button type="button" role="menuitem" onClick={() => onInspector('status')}>
              <Activity size={14} />
              <span>Status</span>
            </button>
            <button type="button" role="menuitem" onClick={() => onInspector('context')}>
              <Brain size={14} />
              <span>Context</span>
            </button>
            <button type="button" role="menuitem" onClick={() => onInspector('cost')}>
              <BarChart3 size={14} />
              <span>Cost</span>
            </button>
            <button type="button" role="menuitem" onClick={() => onInspector('skills')}>
              <Sparkles size={14} />
              <span>Skills</span>
            </button>
            <button type="button" role="menuitem" onClick={() => onInspector('mcp')}>
              <Server size={14} />
              <span>MCP</span>
            </button>
          </div>
        </div>
      ) : null}
    </div>
  )
}

function ProjectContextChip({
  root,
  error,
  referenceCount,
  disabled,
  onInsertReference,
  onClearReferences,
}: {
  root?: string | null
  error?: string | null
  referenceCount: number
  disabled: boolean
  onInsertReference: () => void
  onClearReferences: () => void
}) {
  const label = root ? workspaceRootName(root) : 'Workspace'
  const detail = referenceCount > 0
    ? referenceCount.toLocaleString() + ' ref' + (referenceCount === 1 ? '' : 's')
    : error
      ? 'unavailable'
      : root || '@path context'
  return (
    <span className={'composer-project-context' + (referenceCount > 0 ? ' composer-project-context-active' : '')}>
      <button
        type="button"
        className="composer-chip composer-chip-workspace"
        title={root || error || 'Workspace references use @path search'}
        disabled={disabled}
        aria-label="Add workspace reference"
        onClick={onInsertReference}
      >
        <Folder size={14} />
        <span>
          <strong>{label}</strong>
          <small>{detail}</small>
        </span>
      </button>
      {referenceCount > 0 ? (
        <button
          type="button"
          className="composer-project-clear"
          aria-label="Clear workspace references"
          disabled={disabled}
          onClick={onClearReferences}
        >
          <X size={12} />
        </button>
      ) : null}
    </span>
  )
}

function WorkspaceContextStrip({
  items,
  disabled,
  onRemove,
  onClear,
  onMove,
  onPreview,
  preview,
}: {
  items: Array<{ attachment: ChatAttachment; index: number }>
  disabled: boolean
  onRemove: (index: number) => void
  onClear: () => void
  onMove: (index: number, direction: -1 | 1) => void
  onPreview: (attachment: ChatAttachment) => void
  preview: WorkspacePreviewState
}) {
  if (items.length === 0) return null
  return (
    <section className="composer-workspace-context" aria-label="Workspace context">
      <header>
        <span>
          <AtSign size={14} aria-hidden="true" />
          <strong>Workspace context</strong>
          <small>{items.length.toLocaleString()} ref{items.length === 1 ? '' : 's'}</small>
        </span>
        <button type="button" disabled={disabled} onClick={onClear}>Clear</button>
      </header>
      <div className="composer-workspace-context-list">
        {items.map(({ attachment, index }, itemPosition) => {
          const Icon = attachment.isDirectory ? Folder : FileText
          const label = attachment.name || attachment.path || 'workspace reference'
          const canPreview = !attachment.isDirectory && Boolean(attachment.path)
          const handleChipKeyDown = (event: KeyboardEvent<HTMLSpanElement>) => {
            if (event.currentTarget !== event.target || disabled) return
            if (event.key === 'Delete' || event.key === 'Backspace') {
              event.preventDefault()
              onRemove(index)
            }
            if ((event.key === 'Enter' || event.key === ' ') && canPreview) {
              event.preventDefault()
              onPreview(attachment)
            }
          }
          return (
            <span
              aria-label={'Workspace reference ' + label}
              className="composer-workspace-context-chip"
              key={(attachment.path || label) + '-' + index}
              onKeyDown={handleChipKeyDown}
              role="group"
              tabIndex={disabled ? -1 : 0}
              title="Enter previews, Delete removes"
            >
              <Icon size={14} aria-hidden="true" />
              <span>
                <strong>{label}</strong>
                {attachment.path ? <small>{attachment.path}</small> : null}
              </span>
              <span className="composer-workspace-context-actions">
                <button
                  type="button"
                  aria-label={'Move workspace reference ' + label + ' earlier'}
                  disabled={disabled || itemPosition === 0}
                  onClick={() => onMove(index, -1)}
                >
                  <ChevronLeft size={12} aria-hidden="true" />
                </button>
                <button
                  type="button"
                  aria-label={'Move workspace reference ' + label + ' later'}
                  disabled={disabled || itemPosition === items.length - 1}
                  onClick={() => onMove(index, 1)}
                >
                  <ChevronRight size={12} aria-hidden="true" />
                </button>
                {attachment.path ? (
                  <CopyButton
                    text={attachment.path}
                    label={'Copy workspace reference ' + label}
                    displayLabel="Copy"
                    className="composer-workspace-context-copy"
                    disabled={disabled}
                  />
                ) : null}
                {canPreview ? (
                  <button type="button" aria-label={'Preview workspace reference ' + label} disabled={disabled} onClick={() => onPreview(attachment)}>
                    <Eye size={12} aria-hidden="true" />
                  </button>
                ) : null}
                <button type="button" aria-label={'Remove workspace reference ' + label} disabled={disabled} onClick={() => onRemove(index)}>
                  <X size={12} aria-hidden="true" />
                </button>
              </span>
            </span>
          )
        })}
      </div>
      <WorkspaceContextPreview preview={preview} />
    </section>
  )
}

function WorkspaceContextPreview({ preview }: { preview: WorkspacePreviewState }) {
  if (preview.status === 'idle') return null
  if (preview.status === 'loading') {
    return (
      <div className="composer-workspace-preview" aria-label="Workspace reference preview">
        <header>
          <FileText size={14} aria-hidden="true" />
          <strong>{preview.path}</strong>
          <em>Loading</em>
        </header>
      </div>
    )
  }
  if (preview.status === 'error') {
    return (
      <div className="composer-workspace-preview composer-workspace-preview-error" aria-label="Workspace reference preview">
        <header>
          <FileText size={14} aria-hidden="true" />
          <strong>{preview.path}</strong>
        </header>
        <p>{preview.message}</p>
      </div>
    )
  }

  const file = preview.file
  return (
    <div className="composer-workspace-preview" aria-label="Workspace reference preview">
      <header>
        <FileText size={14} aria-hidden="true" />
        <strong>{file.path}</strong>
        <span>
          <em>{file.language}</em>
          <em>{formatBytes(file.size_bytes)}</em>
          {file.truncated ? <em>truncated</em> : null}
        </span>
      </header>
      {file.binary ? (
        <p>Binary file preview is disabled.</p>
      ) : file.language === 'markdown' ? (
        <div className="composer-workspace-preview-markdown">
          <MarkdownRenderer text={file.content} />
        </div>
      ) : (
        <pre>{file.content}</pre>
      )}
    </div>
  )
}

function appendDraftText(currentValue: string, text: string): string {
  const current = currentValue.trimEnd()
  const insert = text.trimEnd()
  if (!current) return insert
  if (!insert) return current
  return current + '\n\n' + insert
}

function mergeDraftAttachments(current: ChatAttachment[], incoming: ChatAttachment[]): ChatAttachment[] {
  if (incoming.length === 0) return current
  const seen = new Set(current.map((attachment) => attachmentKey(attachment)))
  const next = [...current]
  for (const attachment of incoming) {
    const key = attachmentKey(attachment)
    if (seen.has(key)) continue
    seen.add(key)
    next.push(attachment)
  }
  return next
}

function attachmentKey(attachment: ChatAttachment): string {
  return [attachment.type, attachment.path || '', attachment.url || '', attachment.name || ''].join(':')
}

function removeWorkspaceReferenceTokens(value: string, paths: string[]): string {
  let next = value
  for (const path of paths) {
    if (!path) continue
    const escaped = escapeRegExp(path)
    next = next.replace(new RegExp('(^|\\s)@' + escaped + '(?=\\s|$)\\s?', 'g'), (_match, prefix: string) => prefix || '')
  }
  return next.replace(/[ \t]{2,}/g, ' ')
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
function workspaceRootName(root: string): string {
  const clean = root.replace(/\\+$/g, '').replace(/\/+$/g, '')
  return clean.split(/[\\/]/).filter(Boolean).pop() || root || 'Workspace'
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value)) return '0 B'
  if (value >= 1_000_000) return (value / 1_000_000).toFixed(1) + ' MB'
  if (value >= 1_000) return (value / 1_000).toFixed(1) + ' KB'
  return Math.max(0, value) + ' B'
}

function ContextRing({ inputTokens, outputTokens, tokens, contextStatus, contextError }: { inputTokens?: number | null; outputTokens?: number | null; tokens?: TokenUsage | null; contextStatus?: ContextStatus | null; contextError?: string | null }) {
  const fallbackTotal = Math.max(0, (inputTokens ?? 0) + (outputTokens ?? 0))
  const usageTotal = tokenUsageTotal(tokens) || fallbackTotal
  const estimateTotal = contextStatus?.prompt_tokens ?? contextStatus?.token_estimate ?? 0
  const total = estimateTotal || usageTotal
  const contextWindow = contextStatus?.context_window ?? null
  const breakdown = tokenUsageBreakdown(tokens)
  const usageDetail = breakdown.length > 0
    ? breakdown.join(' / ')
    : (inputTokens ?? 0).toLocaleString() + ' in / ' + (outputTokens ?? 0).toLocaleString() + ' out'
  const percent = contextWindow && contextWindow > 0
    ? Math.min(100, Math.max(1, Math.round((total / contextWindow) * 100)))
    : total > 0
      ? Math.min(99, Math.max(1, Math.round(total / 2000)))
      : 0
  const pressure = contextStatus?.pressure_level || pressureForContext(total, contextWindow)
  const circumference = 61.261
  const offset = circumference - (circumference * percent) / 100
  const title = contextError
    ? 'Context estimate unavailable: ' + contextError
    : estimateTotal > 0
      ? 'Next run estimate: ' + estimateTotal.toLocaleString() + ' tokens' + (contextWindow ? ' / ' + contextWindow.toLocaleString() + ' window' : '') + ' (' + pressure + ' pressure)'
      : usageTotal > 0
        ? usageTotal.toLocaleString() + ' tokens (' + usageDetail + ')'
        : 'No token usage yet'

  return (
    <span className={'composer-context-ring composer-context-ring-pressure-' + pressure} title={title} aria-label={title}>
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
      <span className="composer-context-ring-copy">
        <strong>{total > 0 ? total.toLocaleString() : '-'}</strong>
        <small>{estimateTotal > 0 ? 'context' : 'tokens'}</small>
      </span>
    </span>
  )
}

function pressureForContext(tokens: number, contextWindow: number | null): string {
  if (!contextWindow || contextWindow <= 0 || tokens <= 0) return 'unknown'
  const ratio = tokens / contextWindow
  if (ratio >= 0.98) return 'critical'
  if (ratio >= 0.85) return 'high'
  if (ratio >= 0.65) return 'medium'
  return 'low'
}

function ModelChip({ provider, model, disabled, sessionId }: { provider?: string | null; model?: string | null; disabled: boolean; sessionId?: string | null }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  const popoverRef = useRef<HTMLDivElement>(null)
  const providers = useProviderStore((state) => state.providers)
  const modelsByProvider = useProviderStore((state) => state.modelsByProvider)
  const loadProviders = useProviderStore((state) => state.loadProviders)
  const loadModels = useProviderStore((state) => state.loadModels)
  const selectModel = useProviderStore((state) => state.selectModel)
  const updateSession = useSessionStore((state) => state.updateSession)

  useEffect(() => {
    if (!open) return
    if (!providers.length) void loadProviders()
    const handler = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open, loadProviders, providers.length])

  useEffect(() => {
    if (open) {
      for (const p of providers) void loadModels(p.name)
    }
  }, [open, providers, loadModels])

  useEffect(() => {
    if (!open || !ref.current || !popoverRef.current) return
    const rect = ref.current.getBoundingClientRect()
    const pop = popoverRef.current
    pop.style.right = (window.innerWidth - rect.right) + 'px'
    pop.style.bottom = (window.innerHeight - rect.top + 6) + 'px'
  }, [open])

  const handleSelect = (providerName: string, modelId: string) => {
    void selectModel(providerName, modelId)
    if (sessionId) {
      void updateSession(sessionId, { provider: providerName, model: modelId })
    }
    setOpen(false)
  }

  const title = provider && model ? provider + ' / ' + model : 'Provider not loaded'
  const label = provider && model ? model : 'Model'

  return (
    <div className="composer-model-picker" ref={ref}>
      <button
        type="button"
        className="composer-chip composer-chip-model"
        title={title}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        <Boxes size={14} />
        <span>
          <strong>{label}</strong>
          {provider ? <small>{provider}</small> : null}
        </span>
        <ChevronDown size={12} />
      </button>
      {open ? (
        <div className="composer-model-popover" ref={popoverRef}>
          {providers.map((p) => {
            const models = modelsByProvider[p.name] ?? []
            return (
              <div key={p.name} className="composer-model-group">
                <div className="composer-model-group-header">{p.display_name}</div>
                <div className="composer-model-list">
                  {models.map((m) => (
                    <button
                      key={m.id}
                      type="button"
                      className={m.id === model && p.name === provider ? 'composer-model-option active-model' : 'composer-model-option'}
                      onClick={() => handleSelect(p.name, m.id)}
                    >
                      {m.display_name || m.id}
                    </button>
                  ))}
                  {models.length === 0 ? <span className="muted">Loading...</span> : null}
                </div>
              </div>
            )
          })}
          {providers.length === 0 ? <span className="muted">Loading providers...</span> : null}
        </div>
      ) : null}
    </div>
  )
}
