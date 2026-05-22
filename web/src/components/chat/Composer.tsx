import { Activity, AtSign, BarChart3, Boxes, Brain, ChevronDown, CircleGauge, Command, Folder, Paperclip, Plus, Route, Send, Server, Sparkles, Square, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { RefObject } from 'react'
import { api } from '../../api/client'
import type { SlashCommandInfo } from '../../api/types'
import type { WorkspaceEntry } from '../../api/types'
import type { ChatAttachment } from '../../chat-rendering'
import { useProviderStore } from '../../stores/providerStore'
import { useSessionStore } from '../../stores/sessionStore'
import { Button } from '../shared/Button'
import { AttachmentGallery } from './AttachmentGallery'
import { ComposerInspectorPanel, type ComposerInspectorKind } from './ComposerInspectorPanel'
import { SlashPopover, type SlashPopoverHandle } from './SlashPopover'
import { WorkspaceReferencePopover, type WorkspaceReferencePopoverHandle } from './WorkspaceReferencePopover'
import { attachmentsFromFiles, filesFromDataTransfer } from './composerAttachments'
import { isSlashCommandInput } from './slashExecute'
import { mergeWorkspaceAttachment } from './workspaceReferences'

type ComposerDraft = {
  value: string
  attachments: ChatAttachment[]
  cursorPosition: number
}

const NEW_SESSION_DRAFT_KEY = '__aether_new_session__'

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
  inputTokens?: number | null
  outputTokens?: number | null
  sessionSummary?: string | null
  messageCount?: number | null
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
  inputTokens,
  outputTokens,
  sessionSummary,
  messageCount,
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
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const controlMenuRef = useRef<HTMLDivElement>(null)
  const slashPopoverRef = useRef<SlashPopoverHandle>(null)
  const workspaceReferencePopoverRef = useRef<WorkspaceReferencePopoverHandle>(null)
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
        if (!cancelled) setLoadedCommands(result.commands)
      })
      .catch(() => {
        if (!cancelled) setLoadedCommands([])
      })
    return () => {
      cancelled = true
    }
  }, [disabled, slashCommands])

  useEffect(() => {
    if (disabled) return
    let cancelled = false
    api.workspaceTree('')
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
  }, [disabled])

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

  const updateCursorPosition = () => {
    setCursorPosition(textareaRef.current?.selectionStart ?? value.length)
  }

  const focusTextareaAt = (nextCursorPosition: number) => {
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      textareaRef.current?.setSelectionRange(nextCursorPosition, nextCursorPosition)
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

  const clearWorkspaceReferences = () => {
    setAttachments((current) => current.filter((attachment) => attachment.note !== 'workspace reference'))
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
          onClose={() => setInspectorKind(null)}
        />
      ) : null}
      <div className="composer-drop-hint" aria-hidden={!dragActive}>
        <Paperclip size={18} />
        <span>Drop files to attach</span>
      </div>
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
        </div>
        <div className="composer-runbar">
          <ModelChip provider={provider} model={model} disabled={disabled || running} sessionId={sessionId} />
          <ContextRing inputTokens={inputTokens} outputTokens={outputTokens} />
          <div className="composer-actions">
            {running ? (
              <Button aria-label="Stop run" title="Stop run" onClick={onCancel}>
                <Square size={16} />
              </Button>
            ) : (
              <Button
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

const LOCAL_INSPECTOR_COMMANDS: SlashCommandInfo[] = [
  { name: '/status', description: 'Show runtime and session status', category: 'local' },
  { name: '/context', description: 'Show active context usage', category: 'local' },
  { name: '/cost', description: 'Show local usage analytics', category: 'local' },
  { name: '/skills', description: 'Show available skills', category: 'local' },
  { name: '/mcp', description: 'Show MCP integration status', category: 'local' },
]

function mergeLocalInspectorCommands(commands: SlashCommandInfo[]): SlashCommandInfo[] {
  const seen = new Set<string>()
  const merged: SlashCommandInfo[] = []
  for (const command of [...LOCAL_INSPECTOR_COMMANDS, ...commands]) {
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

function workspaceRootName(root: string): string {
  const clean = root.replace(/\\+$/g, '').replace(/\/+$/g, '')
  return clean.split(/[\\/]/).filter(Boolean).pop() || root || 'Workspace'
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
      <span className="composer-context-ring-copy">
        <strong>{total > 0 ? total.toLocaleString() : '-'}</strong>
        <small>tokens</small>
      </span>
    </span>
  )
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
