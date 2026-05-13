import { Box, Text } from 'ink'
import { useStore } from '@nanostores/react'
import { useMemo, type ReactElement } from 'react'

import { probeEnvironment } from '../lib/environment.js'
import { theme } from '../lib/theme.js'
import { sessionState } from '../store/sessionStore.js'

const TUI_VERSION = '0.1.0'
const BANNER_MAX_WIDTH = 72
const BANNER_MIN_WIDTH = 48

function terminalWidth(): number {
  const cols = process.stdout?.columns
  if (!Number.isFinite(cols) || !cols) {
    return 100
  }
  return cols
}

function bannerWidth(): number {
  const term = terminalWidth()
  return Math.min(BANNER_MAX_WIDTH, Math.max(BANNER_MIN_WIDTH, term - 4))
}

function truncate(value: string, max: number): string {
  if (value.length <= max) {
    return value
  }
  return value.slice(0, Math.max(0, max - 1)) + '…'
}

export function Banner(): ReactElement | null {
  const session = useStore(sessionState)
  const env = useMemo(() => probeEnvironment(), [])

  if (process.env.AETHER_NO_BANNER === '1') {
    return <BootLine />
  }

  const sessionShort = session.sessionId
    ? session.sessionId.length > 9
      ? `${session.sessionId.slice(0, 8)}…`
      : session.sessionId
    : 'starting'
  const methods = session.gatewayReady?.methods?.length ?? 0
  const gatewayVersion = session.gatewayReady?.version ?? null

  const primaryProps = theme.colorProps('brand')
  const primaryDim = theme.palette.primaryDim
  const accent = theme.colorProps('accent')
  const dim = theme.colorProps('dim')
  const success = theme.colorProps('success')
  const warning = theme.colorProps('warning')

  const width = bannerWidth()
  const border = theme.isColorEnabled() ? primaryDim : undefined

  // Mirrors Python `aether/cli/banner.py` _info_table: right-aligned dim
  // icon+label, then value. We keep the label column visually consistent by
  // padding to 11 chars so the value column lines up across rows.
  const labelCell = (icon: string, label: string) => (
    <Box width={11} flexShrink={0}>
      <Text {...dim}>
        <Text {...primaryProps}>{icon}</Text> {label}
      </Text>
    </Box>
  )

  return (
    <Box flexDirection="column">
      <Box
        borderStyle="round"
        {...(border ? { borderColor: border } : {})}
        paddingX={2}
        paddingY={0}
        flexDirection="column"
        width={width}
      >
        <Box>
          <Text bold {...primaryProps}>
            {theme.icon('logo')} Aether
          </Text>
          <Text {...dim} italic>
            {'  '}industrial agent harness v{TUI_VERSION}
          </Text>
        </Box>
        <Box marginTop={1} flexDirection="column">
          <Box>
            {labelCell(theme.icon('provider'), 'provider')}
            <Text bold {...accent}>
              {session.provider}
            </Text>
          </Box>
          <Box>
            {labelCell(theme.icon('model'), 'model')}
            <Text bold>{session.model || 'resolving…'}</Text>
          </Box>
          <Box>
            {labelCell(theme.icon('session'), 'session')}
            <Text>{sessionShort}</Text>
          </Box>
          <Box>
            {labelCell(theme.icon('dot'), 'cwd')}
            <Text {...dim}>{truncate(env.cwd, Math.max(20, width - 18))}</Text>
            {env.branch ? (
              <>
                <Text {...dim}> · </Text>
                <Text {...accent}>{env.branch}</Text>
              </>
            ) : null}
          </Box>
          <Box>
            {labelCell(theme.icon('tool'), 'gateway')}
            <Text {...(gatewayVersion ? success : warning)}>
              {gatewayVersion ?? 'booting'}
            </Text>
            <Text {...dim}> · methods {methods}</Text>
          </Box>
          {session.systemOverride ? (
            <Box>
              {labelCell(theme.icon('info'), 'system')}
              <Text>custom system prompt loaded</Text>
            </Box>
          ) : null}
        </Box>
      </Box>
      <HintLine width={width} />
    </Box>
  )
}

function HintLine({ width }: { width: number }): ReactElement {
  const dim = theme.colorProps('dim')
  const accent = theme.colorProps('accent')
  // Mirror Python `_hint_line`: centered, bullet separators between key
  // commands. Approximate the visible glyph count to centre the hint inside
  // the same column band the panel occupies.
  const segments: { hint: string; key: string }[] = [
    { hint: 'Type your message', key: '' },
    { hint: 'for commands', key: '/help' },
    { hint: 'newline', key: 'Shift+Enter' },
    { hint: 'interrupt', key: 'ESC' },
    { hint: 'exit', key: '/exit' }
  ]
  const visibleLength = segments.reduce((sum, s, i) => {
    const sep = i === 0 ? 0 : 3 // '  ·  '
    return sum + sep + s.key.length + (s.key && s.hint ? 1 : 0) + s.hint.length
  }, 0)
  const pad = Math.max(0, Math.floor((width - visibleLength) / 2))

  return (
    <Box>
      <Text>
        {' '.repeat(pad)}
        {segments.map((seg, i) => (
          <Text key={i}>
            {i > 0 ? <Text {...dim}>{'  ·  '}</Text> : null}
            {seg.key ? (
              <>
                <Text {...accent}>{seg.key}</Text>
                {seg.hint ? <Text {...dim}> {seg.hint}</Text> : null}
              </>
            ) : (
              <Text {...dim}>{seg.hint}</Text>
            )}
          </Text>
        ))}
      </Text>
    </Box>
  )
}

function BootLine(): ReactElement {
  const session = useStore(sessionState)
  const sessionId = session.sessionId ? session.sessionId.slice(0, 8) : '…'
  const dim = theme.colorProps('dim')
  const brand = theme.colorProps('brand')
  return (
    <Box>
      <Text bold {...brand}>
        {theme.icon('logo')} aether{' '}
      </Text>
      <Text {...dim}>
        {session.provider}/{session.model || 'resolving'} · {sessionId}
      </Text>
    </Box>
  )
}
