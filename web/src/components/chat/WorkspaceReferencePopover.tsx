import { FileText, Folder } from 'lucide-react'
import { forwardRef, useEffect, useImperativeHandle, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { api } from '../../api/client'
import type { WorkspaceEntry } from '../../api/types'
import { findWorkspaceReferenceTrigger, replaceWorkspaceReferenceToken } from './workspaceReferences'

export type WorkspaceReferencePopoverHandle = {
  handleKey: (event: KeyboardEvent<HTMLTextAreaElement>) => boolean
}

type Props = {
  value: string
  cursorPosition: number
  disabled?: boolean
  onApply: (value: string, cursorPosition: number, entry: WorkspaceEntry) => void
}

export const WorkspaceReferencePopover = forwardRef<WorkspaceReferencePopoverHandle, Props>(function WorkspaceReferencePopover(
  { value, cursorPosition, disabled = false, onApply },
  ref,
) {
  const trigger = useMemo(() => findWorkspaceReferenceTrigger(value, cursorPosition), [cursorPosition, value])
  const token = trigger ? value.slice(trigger.atPos, cursorPosition) : ''
  const query = trigger?.filter ?? ''
  const [dismissedToken, setDismissedToken] = useState<string | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [entries, setEntries] = useState<WorkspaceEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const visible = !disabled && trigger !== null && dismissedToken !== token

  useEffect(() => {
    setSelectedIndex(0)
    if (dismissedToken && dismissedToken !== token) setDismissedToken(null)
  }, [dismissedToken, token])

  useEffect(() => {
    if (!visible) {
      setEntries([])
      setError(null)
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)
    const request = query
      ? api.workspaceSearch(query, 30).then((result) => result.entries)
      : api.workspaceTree('').then((result) => result.entries.slice(0, 30))

    request
      .then((nextEntries) => {
        if (cancelled) return
        setEntries(nextEntries)
      })
      .catch((nextError) => {
        if (cancelled) return
        setEntries([])
        setError(nextError instanceof Error ? nextError.message : String(nextError))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [query, visible])

  const apply = (entry: WorkspaceEntry | undefined) => {
    if (!entry) return
    const next = replaceWorkspaceReferenceToken(value, cursorPosition, entry.path)
    onApply(next.value, next.cursorPosition, entry)
    setDismissedToken(null)
  }

  useImperativeHandle(ref, () => ({
    handleKey: (event) => {
      if (!visible) return false
      if (event.key === 'Escape') {
        event.preventDefault()
        setDismissedToken(token)
        return true
      }
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        if (entries.length === 0) return true
        setSelectedIndex((current) => (current + 1) % entries.length)
        return true
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        if (entries.length === 0) return true
        setSelectedIndex((current) => (current - 1 + entries.length) % entries.length)
        return true
      }
      if (event.key === 'Tab' || event.key === 'Enter') {
        event.preventDefault()
        if (entries.length === 0) return true
        apply(entries[selectedIndex])
        return true
      }
      return false
    },
  }), [entries, onApply, selectedIndex, token, value, cursorPosition, visible])

  if (!visible) return null

  return (
    <div className="workspace-reference-popover" role="listbox" aria-label="Workspace references">
      <div className="workspace-reference-header">
        <Folder aria-hidden="true" size={13} />
        <span>{query ? '@' + query : 'Workspace'}</span>
        {loading ? <em>Loading</em> : null}
      </div>
      {error ? <div className="workspace-reference-empty">{error}</div> : null}
      {!error && !loading && entries.length === 0 ? (
        <div className="workspace-reference-empty">No matching files</div>
      ) : null}
      {entries.map((entry, index) => {
        const active = index === selectedIndex
        const Icon = entry.kind === 'directory' ? Folder : FileText
        return (
          <button
            type="button"
            role="option"
            aria-selected={active}
            className={'workspace-reference-option' + (active ? ' workspace-reference-option-active' : '')}
            key={entry.path}
            onClick={() => apply(entry)}
            onMouseEnter={() => setSelectedIndex(index)}
          >
            <Icon aria-hidden="true" size={14} />
            <span>{entry.kind === 'directory' ? entry.name + '/' : entry.name}</span>
            <small>{entry.path}</small>
          </button>
        )
      })}
    </div>
  )
})
