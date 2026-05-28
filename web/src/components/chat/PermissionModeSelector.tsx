import { Ban, Bolt, ChevronDown, ClipboardCheck, ShieldQuestion, Unlock } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import type { PermissionMode } from '../../api/types'
import { ConfirmDialog } from '../shared/ConfirmDialog'

type Props = {
  value?: PermissionMode | string | null
  disabled?: boolean
  onChange: (mode: PermissionMode) => Promise<void> | void
}

const permissionModes: Array<{
  value: PermissionMode
  label: string
  shortLabel: string
  description: string
  tone?: 'danger' | 'plan'
  icon: typeof ShieldQuestion
}> = [
  {
    value: 'default',
    label: 'Ask before tools',
    shortLabel: 'Ask',
    description: 'Prompt before write, shell, and delegation tools.',
    icon: ShieldQuestion,
  },
  {
    value: 'acceptEdits',
    label: 'Auto-accept edits',
    shortLabel: 'Edits',
    description: 'Allow file writes and edits while still asking for shell and tasks.',
    icon: Bolt,
  },
  {
    value: 'plan',
    label: 'Plan mode',
    shortLabel: 'Plan',
    description: 'Enter plan mode and block mutating tools until the plan is approved.',
    tone: 'plan',
    icon: ClipboardCheck,
  },
  {
    value: 'bypassPermissions',
    label: 'Bypass permissions',
    shortLabel: 'Bypass',
    description: 'Allow all tool categories without prompting for this session.',
    tone: 'danger',
    icon: Unlock,
  },
  {
    value: 'dontAsk',
    label: 'Auto-deny tools',
    shortLabel: 'Deny',
    description: 'Reject mutating tools instead of prompting.',
    icon: Ban,
  },
]

export function PermissionModeSelector({ value, disabled = false, onChange }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [confirmBypass, setConfirmBypass] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const current = normalizedPermissionMode(value)
  const currentItem = permissionModes.find((item) => item.value === current) ?? permissionModes[0]
  const CurrentIcon = currentItem.icon

  useEffect(() => {
    if (!open) return
    const onPointerDown = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const commit = (mode: PermissionMode) => {
    setSaving(true)
    setError(null)
    void Promise.resolve()
      .then(() => onChange(mode))
      .then(() => {
        setOpen(false)
        setConfirmBypass(false)
      })
      .catch((changeError) => {
        setError(changeError instanceof Error ? changeError.message : String(changeError))
      })
      .finally(() => setSaving(false))
  }

  return (
    <div className="permission-mode-selector" ref={ref}>
      <button
        type="button"
        className={'permission-mode-trigger permission-mode-trigger-' + current}
        disabled={disabled || saving}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={'Permission mode: ' + currentItem.label}
        title={currentItem.label}
        onClick={() => setOpen((state) => !state)}
      >
        <CurrentIcon size={14} />
        <span>{currentItem.shortLabel}</span>
        <ChevronDown size={13} />
      </button>
      {open ? (
        <div className="permission-mode-menu" role="menu" aria-label="Permission mode">
          <header>Permission mode</header>
          {permissionModes.map((item) => {
            const Icon = item.icon
            const selected = item.value === current
            return (
              <button
                key={item.value}
                type="button"
                role="menuitemradio"
                aria-checked={selected}
                className={'permission-mode-option' + (selected ? ' permission-mode-option-selected' : '') + (item.tone ? ' permission-mode-option-' + item.tone : '')}
                onClick={() => {
                  if (item.value === 'bypassPermissions') {
                    setOpen(false)
                    setConfirmBypass(true)
                    return
                  }
                  commit(item.value)
                }}
              >
                <Icon size={15} />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.description}</small>
                </span>
              </button>
            )
          })}
          {error ? <p className="permission-mode-error">{error}</p> : null}
        </div>
      ) : null}
      {confirmBypass ? (
        <ConfirmDialog
          title="Bypass permissions?"
          description="Aether will run write, shell, and delegation tools without prompting in this session."
          confirmLabel={saving ? 'Saving' : 'Bypass'}
          cancelLabel="Cancel"
          onConfirm={() => commit('bypassPermissions')}
          onCancel={() => {
            if (!saving) setConfirmBypass(false)
          }}
        />
      ) : null}
    </div>
  )
}

function normalizedPermissionMode(value: PermissionMode | string | null | undefined): PermissionMode {
  return permissionModes.some((item) => item.value === value) ? value as PermissionMode : 'default'
}
