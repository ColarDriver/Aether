import { describe, expect, it } from 'vitest'
import {
  findWorkspaceReferenceTrigger,
  mergeWorkspaceAttachment,
  replaceWorkspaceReferenceToken,
  workspaceEntryToAttachment,
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
})
