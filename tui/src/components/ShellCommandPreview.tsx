import { Box, Text } from 'ink'
import type { ReactElement } from 'react'

import { theme } from '../lib/theme.js'

const MAX_LINE_WIDTH = 120
const CHAIN_OPERATORS = ['&&', '||', ';', '|'] as const

export interface ShellCommandPreviewProps {
  command: string
}

export interface RenderedShellLine {
  prefix: '$ ' | '  '
  text: string
  suspicious?: boolean
}

/**
 * Multi-line shell command preview for the permission overlay.
 *
 * Splits the model's `command` first on real newlines, then on chain
 * operators (`&&` / `||` / `;` / `|`) so a one-line `cmd1 && cmd2 && cmd3`
 * still reads as three lines. The leading line gets `$ `, continuation
 * lines get a 2-space indent — matches Claude Code's rendering.
 *
 * If a single segment looks like several space-joined commands jammed
 * together (e.g. the model emitted `cd /a cp x y uv sync` without
 * operators), we flag the row with a small `⚠` so the user is alerted
 * before approving. We can't fix the model's emission, but we can hint
 * that what they're about to run is multi-command.
 */
export function ShellCommandPreview({ command }: ShellCommandPreviewProps): ReactElement {
  const lines = formatShellCommand(command)
  return (
    <Box marginTop={1} flexDirection="column">
      {lines.map((line, idx) => (
        <Text key={idx}>
          <Text bold {...theme.colorProps('accent')}>
            {line.prefix}
          </Text>
          <Text {...theme.colorProps('text')}>{line.text || ' '}</Text>
          {line.suspicious ? (
            <Text {...theme.colorProps('warning')}> ⚠</Text>
          ) : null}
        </Text>
      ))}
    </Box>
  )
}

export function formatShellCommand(raw: string): RenderedShellLine[] {
  const out: RenderedShellLine[] = []
  const hardLines = raw.split('\n')
  let isFirstOverall = true
  for (const hardLine of hardLines) {
    const segments = splitOnChainOperators(hardLine)
    if (segments.length === 0) {
      out.push({
        prefix: isFirstOverall ? '$ ' : '  ',
        text: ''
      })
      isFirstOverall = false
      continue
    }
    for (const segment of segments) {
      const text = segment.text + (segment.trailing ?? '')
      out.push({
        prefix: isFirstOverall ? '$ ' : '  ',
        text: truncate(text, MAX_LINE_WIDTH),
        ...(isSuspiciousMultiCommand(segment.text) ? { suspicious: true } : {})
      })
      isFirstOverall = false
    }
  }
  return out
}

interface ShellSegment {
  text: string
  trailing?: string
}

function splitOnChainOperators(line: string): ShellSegment[] {
  const segments: ShellSegment[] = []
  let cursor = 0
  while (cursor < line.length) {
    let nextIdx = -1
    let nextOp = ''
    for (const op of CHAIN_OPERATORS) {
      const idx = line.indexOf(op, cursor)
      if (idx !== -1 && (nextIdx === -1 || idx < nextIdx)) {
        nextIdx = idx
        nextOp = op
      }
    }
    if (nextIdx === -1) {
      const tail = line.slice(cursor).trim()
      if (tail.length > 0) {
        segments.push({ text: tail })
      }
      break
    }
    const head = line.slice(cursor, nextIdx).trim()
    if (head.length > 0) {
      segments.push({ text: head, trailing: ` ${nextOp}` })
    }
    cursor = nextIdx + nextOp.length
  }
  return segments
}

const COMMON_CMD_NAMES = new Set([
  'cd',
  'ls',
  'cp',
  'mv',
  'rm',
  'mkdir',
  'touch',
  'cat',
  'echo',
  'grep',
  'find',
  'git',
  'npm',
  'yarn',
  'pnpm',
  'uv',
  'python',
  'python3',
  'node',
  'make',
  'docker',
  'sed',
  'awk',
  'curl',
  'wget',
  'tar',
  'unzip',
  'chmod',
  'chown',
  'sudo'
])

function isSuspiciousMultiCommand(text: string): boolean {
  const tokens = text.split(/\s+/).filter(Boolean)
  if (tokens.length < 4) return false
  let cmdCount = 0
  for (const tok of tokens) {
    if (COMMON_CMD_NAMES.has(tok)) {
      cmdCount += 1
      if (cmdCount >= 2) return true
    }
  }
  return false
}

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value
  }
  return `${value.slice(0, max - 1)}…`
}
