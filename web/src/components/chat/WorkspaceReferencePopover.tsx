import { ChevronRight, CornerUpLeft, FileText, Folder } from 'lucide-react'
import { forwardRef, useEffect, useImperativeHandle, useMemo, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { api } from '../../api/client'
import type { WorkspaceEntry } from '../../api/types'
import { findWorkspaceReferenceTrigger, replaceWorkspaceReferenceBrowseToken, replaceWorkspaceReferenceToken } from './workspaceReferences'

export type WorkspaceReferencePopoverHandle = {
  handleKey: (event: KeyboardEvent<HTMLTextAreaElement>) => boolean
}

type Props = {
  value: string
  cursorPosition: number
  disabled?: boolean
  onApply: (value: string, cursorPosition: number, entry: WorkspaceEntry) => void
  onBrowse: (value: string, cursorPosition: number) => void
}

type WorkspaceReferenceOption =
  | { kind: 'parent'; path: string }
  | { kind: 'entry'; entry: WorkspaceEntry }

export const WorkspaceReferencePopover = forwardRef<WorkspaceReferencePopoverHandle, Props>(function WorkspaceReferencePopover(
  { value, cursorPosition, disabled = false, onApply, onBrowse },
  ref,
) {
  const trigger = useMemo(() => findWorkspaceReferenceTrigger(value, cursorPosition), [cursorPosition, value])
  const token = trigger ? value.slice(trigger.atPos, cursorPosition) : ''
  const query = trigger?.filter ?? ''
  const treePath = useMemo(() => treePathFromQuery(query), [query])
  const browseMode = treePath !== null
  const [dismissedToken, setDismissedToken] = useState<string | null>(null)
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [entries, setEntries] = useState<WorkspaceEntry[]>([])
  const [treeCurrentPath, setTreeCurrentPath] = useState<string | null>(null)
  const [treeParentPath, setTreeParentPath] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const visible = !disabled && trigger !== null && dismissedToken !== token
  const options = useMemo<WorkspaceReferenceOption[]>(() => {
    const next: WorkspaceReferenceOption[] = []
    if (browseMode && treeParentPath !== null) next.push({ kind: 'parent', path: treeParentPath })
    for (const entry of entries) next.push({ kind: 'entry', entry })
    return next
  }, [browseMode, entries, treeParentPath])
  const breadcrumbs = useMemo(() => buildBreadcrumbs(treeCurrentPath ?? treePath ?? ''), [treeCurrentPath, treePath])
  const directoryCount = entries.filter((entry) => entry.kind === 'directory').length

  useEffect(() => {
    setSelectedIndex(0)
    if (dismissedToken && dismissedToken !== token) setDismissedToken(null)
  }, [dismissedToken, token])

  useEffect(() => {
    if (selectedIndex < options.length || selectedIndex === 0) return
    setSelectedIndex(Math.max(0, options.length - 1))
  }, [options.length, selectedIndex])

  useEffect(() => {
    if (!visible) {
      setEntries([])
      setTreeCurrentPath(null)
      setTreeParentPath(null)
      setError(null)
      setLoading(false)
      return
    }

    let cancelled = false
    setLoading(true)
    setError(null)

    const request = browseMode
      ? api.workspaceTree(treePath ?? '').then((result) => ({
        entries: result.entries.slice(0, 30),
        treeCurrentPath: result.path,
        treeParentPath: result.parent_path ?? null,
      }))
      : api.workspaceSearch(query, 30).then((result) => ({
        entries: result.entries,
        treeCurrentPath: null,
        treeParentPath: null,
      }))

    request
      .then((result) => {
        if (cancelled) return
        setEntries(result.entries)
        setTreeCurrentPath(result.treeCurrentPath)
        setTreeParentPath(result.treeParentPath)
        setSelectedIndex(result.treeParentPath !== null && result.entries.length > 0 ? 1 : 0)
      })
      .catch((nextError) => {
        if (cancelled) return
        setEntries([])
        setTreeCurrentPath(null)
        setTreeParentPath(null)
        setError(nextError instanceof Error ? nextError.message : String(nextError))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [browseMode, query, treePath, visible])

  const browseTo = (path: string) => {
    const next = replaceWorkspaceReferenceBrowseToken(value, cursorPosition, path)
    onBrowse(next.value, next.cursorPosition)
    setSelectedIndex(0)
    setDismissedToken(null)
  }

  const apply = (entry: WorkspaceEntry) => {
    const next = replaceWorkspaceReferenceToken(value, cursorPosition, entry.path)
    onApply(next.value, next.cursorPosition, entry)
    setDismissedToken(null)
  }

  const choose = (option: WorkspaceReferenceOption | undefined) => {
    if (!option) return
    if (option.kind === 'parent') {
      browseTo(option.path)
      return
    }
    if (option.entry.kind === 'directory') {
      browseTo(option.entry.path)
      return
    }
    apply(option.entry)
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
        if (options.length === 0) return true
        setSelectedIndex((current) => (current + 1) % options.length)
        return true
      }
      if (event.key === 'ArrowUp') {
        event.preventDefault()
        if (options.length === 0) return true
        setSelectedIndex((current) => (current - 1 + options.length) % options.length)
        return true
      }
      if (event.key === 'Home') {
        event.preventDefault()
        setSelectedIndex(0)
        return true
      }
      if (event.key === 'End') {
        event.preventDefault()
        setSelectedIndex(Math.max(0, options.length - 1))
        return true
      }
      if (event.key === 'ArrowRight') {
        const option = options[selectedIndex]
        if (option?.kind === 'entry' && option.entry.kind === 'directory') {
          event.preventDefault()
          browseTo(option.entry.path)
          return true
        }
      }
      if (event.key === 'ArrowLeft' && browseMode && treeParentPath !== null) {
        event.preventDefault()
        browseTo(treeParentPath)
        return true
      }
      if (event.key === 'Tab' || event.key === 'Enter') {
        event.preventDefault()
        if (options.length === 0) return true
        choose(options[selectedIndex])
        return true
      }
      return false
    },
  }), [options, selectedIndex, token, visible, value, cursorPosition, onApply, onBrowse])

  if (!visible) return null

  const headerLabel = browseMode ? 'Browse workspace' : '@' + query
  const emptyLabel = browseMode ? 'No files in this folder' : 'No matching files'
  const countLabel = loading ? null : entries.length.toLocaleString() + ' item' + (entries.length === 1 ? '' : 's') + ' / ' + directoryCount.toLocaleString() + ' dir' + (directoryCount === 1 ? '' : 's')

  return (
    <div className="workspace-reference-popover" role="listbox" aria-label="Workspace references">
      <div className="workspace-reference-header">
        <Folder aria-hidden="true" size={13} />
        <span>{headerLabel}</span>
        {countLabel ? <small>{countLabel}</small> : null}
        {loading ? <em>Loading</em> : null}
      </div>
      {browseMode ? (
        <nav className="workspace-reference-breadcrumb" aria-label="Workspace reference path">
          <button type="button" onClick={() => browseTo('')} title="Workspace root">root</button>
          {breadcrumbs.map((crumb) => (
            <span className="workspace-reference-crumb-segment" key={crumb.path}>
              <ChevronRight size={12} aria-hidden="true" />
              <button type="button" onClick={() => browseTo(crumb.path)} title={crumb.path}>
                {crumb.name}
              </button>
            </span>
          ))}
        </nav>
      ) : null}
      {error ? <div className="workspace-reference-empty">{error}</div> : null}
      {!error && !loading && options.length === 0 ? (
        <div className="workspace-reference-empty">{emptyLabel}</div>
      ) : null}
      {options.map((option, index) => renderOption(option, index, selectedIndex, choose, setSelectedIndex))}
    </div>
  )
})

function renderOption(
  option: WorkspaceReferenceOption,
  index: number,
  selectedIndex: number,
  choose: (option: WorkspaceReferenceOption) => void,
  setSelectedIndex: (index: number) => void,
) {
  const active = index === selectedIndex
  if (option.kind === 'parent') {
    const detail = option.path ? option.path : 'workspace root'
    return (
      <button
        type="button"
        role="option"
        aria-label={'Up to ' + detail}
        aria-selected={active}
        className={'workspace-reference-option workspace-reference-option-parent' + (active ? ' workspace-reference-option-active' : '')}
        key={'__parent__' + option.path}
        onClick={() => choose(option)}
        onMouseEnter={() => setSelectedIndex(index)}
      >
        <CornerUpLeft aria-hidden="true" size={14} />
        <span>..</span>
        <small>{detail}</small>
      </button>
    )
  }

  const entry = option.entry
  const Icon = entry.kind === 'directory' ? Folder : FileText
  return (
    <button
      type="button"
      role="option"
      aria-selected={active}
      className={'workspace-reference-option' + (active ? ' workspace-reference-option-active' : '')}
      key={entry.path}
      onClick={() => choose(option)}
      onMouseEnter={() => setSelectedIndex(index)}
    >
      <Icon aria-hidden="true" size={14} />
      <span>{entry.kind === 'directory' ? entry.name + '/' : entry.name}</span>
      <small>{entry.path}</small>
    </button>
  )
}

type WorkspaceBreadcrumb = {
  name: string
  path: string
}

function buildBreadcrumbs(path: string): WorkspaceBreadcrumb[] {
  const parts = path.split('/').filter(Boolean)
  const crumbs: WorkspaceBreadcrumb[] = []
  for (let index = 0; index < parts.length; index += 1) {
    crumbs.push({ name: parts[index] ?? '', path: parts.slice(0, index + 1).join('/') })
  }
  return crumbs
}

function treePathFromQuery(query: string): string | null {
  if (!query) return ''
  if (!query.endsWith('/')) return null
  return query.replace(/\/+$/g, '')
}
