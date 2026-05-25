import { describe, expect, it } from 'vitest'
import {
  findWorkspaceReferenceTrigger,
  mergeWorkspaceAttachment,
  replaceWorkspaceReferenceBrowseToken,
  replaceWorkspaceReferenceToken,
  syncWorkspaceReferenceAttachmentsForValue,
  workspaceEntryToAttachment,
  workspaceReferenceTokenExists,
} from './workspaceReferences'

describe('workspaceReferences', () => {
  it('finds @ triggers only at token boundaries', () => {
    expect(findWorkspaceReferenceTrigger('@src/app.ts', 11)).toEqual({ atPos: 0, filter: 'src/app.ts' })
    expect(findWorkspaceReferenceTrigger('read @src', 9)).toEqual({ atPos: 5, filter: 'src' })
    expect(findWorkspaceReferenceTrigger('email@example.com', 17)).toBeNull()
    expect(findWorkspaceReferenceTrigger('read @src/app.ts now', 20)).toBeNull()
  })

  it('replaces the active @ token with a workspace path token', () => {
    expect(replaceWorkspaceReferenceToken('read @app now', 9, 'src/app.ts')).toEqual({
      value: 'read @src/app.ts now',
      cursorPosition: 17,
    })
  })

  it('converts workspace entries into display attachments', () => {
    expect(workspaceEntryToAttachment({ kind: 'directory', name: 'src', path: 'src' })).toMatchObject({
      type: 'file',
      name: 'src/',
      path: 'src',
      isDirectory: true,
    })
    expect(workspaceEntryToAttachment({ kind: 'file', name: 'app.ts', path: 'src/app.ts' })).toMatchObject({
      type: 'text',
      name: 'app.ts',
      path: 'src/app.ts',
    })
  })

  it('does not duplicate attachments for the same workspace path', () => {
    const entry = { kind: 'file' as const, name: 'app.ts', path: 'src/app.ts' }
    const once = mergeWorkspaceAttachment([], entry)
    const twice = mergeWorkspaceAttachment(once, entry)

    expect(twice).toHaveLength(1)
  })

  it("replaces the active @ token with a browsed folder token", () => {
    expect(replaceWorkspaceReferenceBrowseToken("read @src now", 9, "src/components")).toEqual({
      value: "read @src/components/ now",
      cursorPosition: 21,
    })
    expect(replaceWorkspaceReferenceBrowseToken("read @src now", 9, "")).toEqual({
      value: "read @ now",
      cursorPosition: 6,
    })
  })

  it("detects full workspace reference tokens for selected paths", () => {
    expect(workspaceReferenceTokenExists("read @src/app.ts now", "src/app.ts")).toBe(true)
    expect(workspaceReferenceTokenExists("read @src/app.ts/ now", "src/app.ts")).toBe(true)
    expect(workspaceReferenceTokenExists("read @src/app.tsx now", "src/app.ts")).toBe(false)
    expect(workspaceReferenceTokenExists("read src/app.ts now", "src/app.ts")).toBe(false)
  })

  it("keeps only workspace attachments that still have visible @path tokens", () => {
    const manual = { type: "text" as const, name: "manual.txt", path: "manual.txt" }
    const app = { type: "text" as const, name: "app.ts", path: "src/app.ts", note: "workspace reference" }
    const readme = { type: "text" as const, name: "README.md", path: "README.md", note: "workspace reference" }
    const synced = syncWorkspaceReferenceAttachmentsForValue([manual, app, readme], "inspect @src/app.ts")

    expect(synced).toEqual([manual, app])
  })
})
