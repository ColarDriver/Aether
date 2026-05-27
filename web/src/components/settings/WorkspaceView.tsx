import { useCallback, useRef, useState } from 'react'
import { api } from '../../api/client'
import type { WorkspaceEntry, WorkspaceFile, WorkspaceTree } from '../../api/types'
import { WorkspaceFilePanel } from '../chat/WorkspaceFilePanel'
import { WorkspaceRail } from '../chat/WorkspaceRail'

export function WorkspaceView() {
  const autoSelectedInitialFile = useRef(false)
  const [previewPath, setPreviewPath] = useState<string | null>(null)
  const [previewFile, setPreviewFile] = useState<WorkspaceFile | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [previewError, setPreviewError] = useState<string | null>(null)

  const openFile = useCallback((path: string) => {
    setPreviewPath(path)
    setPreviewFile(null)
    setPreviewError(null)
    setPreviewLoading(true)
    api.workspaceFile(path)
      .then((file) => {
        setPreviewFile(file)
        setPreviewError(null)
      })
      .catch((err: unknown) => {
        setPreviewFile(null)
        setPreviewError(err instanceof Error ? err.message : String(err))
      })
      .finally(() => setPreviewLoading(false))
  }, [])

  const closePreview = useCallback(() => {
    setPreviewPath(null)
    setPreviewFile(null)
    setPreviewError(null)
    setPreviewLoading(false)
  }, [])

  const saveFile = useCallback(async (path: string, content: string) => {
    const saved = await api.workspaceSaveFile(path, content)
    const nextFile = { ...saved, path: saved.path || path }
    setPreviewFile(nextFile)
    setPreviewPath(nextFile.path)
    setPreviewError(null)
    return nextFile
  }, [])

  const handleDeletedPath = useCallback((path: string) => {
    const activePath = normalizeWorkspacePath(previewPath ?? '')
    const deletedPath = normalizeWorkspacePath(path)
    if (!activePath || !deletedPath) return
    if (activePath === deletedPath || activePath.startsWith(deletedPath + '/')) {
      closePreview()
    }
  }, [closePreview, previewPath])

  const handleRenamedPath = useCallback((path: string, newPath: string, kind: WorkspaceEntry['kind']) => {
    const activePath = normalizeWorkspacePath(previewPath ?? '')
    const oldPath = normalizeWorkspacePath(path)
    const renamedPath = normalizeWorkspacePath(newPath)
    if (!activePath || !oldPath || !renamedPath) return
    if (activePath === oldPath && kind === 'file') {
      openFile(renamedPath)
      return
    }
    if (kind === 'directory' && activePath.startsWith(oldPath + '/')) {
      openFile(renamedPath + activePath.slice(oldPath.length))
    }
  }, [openFile, previewPath])

  const handleTreeLoaded = useCallback((tree: WorkspaceTree) => {
    if (autoSelectedInitialFile.current || previewPath || tree.path) return
    const firstFile = tree.entries.find((entry) => entry.kind === 'file' && entry.name.toLowerCase() === 'readme.md')
      ?? tree.entries.find((entry) => entry.kind === 'file')
    if (!firstFile) return
    autoSelectedInitialFile.current = true
    openFile(firstFile.path)
  }, [openFile, previewPath])

  const handleWorkspaceRootChanged = useCallback(() => {
    autoSelectedInitialFile.current = false
    closePreview()
  }, [closePreview])

  return (
    <div className="settings-panel workspace-panel workspace-panel-full">
      <header className="panel-header">
        <div>
          <h2>Workspace</h2>
          <p>Browse, edit, create, rename, and delete project files within the configured workspace root.</p>
        </div>
      </header>

      <div className="workspace-layout workspace-full-layout">
        <WorkspaceRail
          selectedFilePath={previewPath}
          onSelectFile={openFile}
          onDeletedPath={handleDeletedPath}
          onRenamedPath={handleRenamedPath}
          onTreeLoaded={handleTreeLoaded}
          onWorkspaceRootChanged={handleWorkspaceRootChanged}
        />
        <WorkspaceFilePanel
          preview={{
            path: previewPath,
            file: previewFile,
            loading: previewLoading,
            error: previewError,
          }}
          onClose={closePreview}
          onSave={saveFile}
        />
      </div>
    </div>
  )
}

function normalizeWorkspacePath(path: string): string {
  return path.trim().replace(/\\/g, '/').replace(/^\/+/, '').replace(/\/+$/, '')
}
