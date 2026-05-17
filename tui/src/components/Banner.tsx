import { createRequire } from 'node:module'
import { Box, Text } from 'ink'
import stringWidth from 'string-width'
import { useStore } from '@nanostores/react'
import { useMemo, type ReactElement, type ReactNode } from 'react'

import { probeEnvironment } from '../lib/environment.js'
import { theme } from '../lib/theme.js'
import { sessionState } from '../store/sessionStore.js'

const require = createRequire(import.meta.url)
const { version: PACKAGE_VERSION = '1.0.0' } = require('../../package.json') as {
  version?: string
}
const APP_VERSION = process.env.AETHER_VERSION?.trim() || PACKAGE_VERSION
const BANNER_MAX_WIDTH = 72
const BANNER_MIN_WIDTH = 48
const LABEL_WIDTH = 11
const ICON_WIDTH = 2

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

  const toolCount = session.bannerToolCount
  const skillCount = session.bannerSkillCount

  const primaryDim = theme.palette.primaryDim
  const accent = theme.colorProps('accent')
  const dim = theme.colorProps('dim')

  const width = bannerWidth()
  const border = theme.isColorEnabled() ? primaryDim : undefined
  const sessionDetails = buildSessionDetails(session).slice(0, 4)
  const toolDetails = session.bannerTools.filter(Boolean).slice(0, 4)
  const skillDetails = session.bannerSkills.filter(Boolean).slice(0, 4)

  return (
    <Box flexDirection="column">
      <TopBorder width={width} />
      <FrameLine width={width}>
        <Text> </Text>
      </FrameLine>
      <InfoRow
        width={width}
        level={0}
        icon={theme.icon('provider')}
        label="provider"
        value={
          <Text bold {...accent}>
            {session.provider}
          </Text>
        }
      />
      <InfoRow
        width={width}
        level={1}
        icon={theme.icon('model')}
        label="model"
        value={
          <Text>
            <Text bold>{session.model || 'resolving…'}</Text>
          </Text>
        }
      />
      <InfoRow
        width={width}
        level={0}
        icon={theme.icon('session')}
        label="session"
        value={<Text>{sessionDetails.length}</Text>}
      />
      {sessionDetails.map((item) => (
        <InfoDetailRow
          key={item}
          width={width}
          level={1}
          text={item}
          tone="text"
        />
      ))}
      {session.mode === 'plan' ? (
        <InfoRow
          width={width}
          level={1}
          icon={theme.icon('info')}
          label="mode"
          value={<Text {...theme.colorProps('info')}>plan</Text>}
        />
      ) : null}
      <InfoRow
        width={width}
        level={1}
        icon={theme.icon('dot')}
        label="cwd"
        value={<Text {...dim}>{truncate(env.cwd, Math.max(20, width - 22))}</Text>}
      />
      <InfoRow
        width={width}
        level={0}
        icon={theme.icon('tool')}
        label="tools"
        value={<Text>{toolCount}</Text>}
      />
      {toolDetails.length > 0 ? (
        <>
          {toolDetails.map((item) => (
            <InfoDetailRow
              key={item}
              width={width}
              level={1}
              text={item}
            />
          ))}
          {session.bannerToolCount > toolDetails.length ? (
            <InfoDetailRow width={width} level={1} text="…" />
          ) : null}
        </>
      ) : null}
      <InfoRow
        width={width}
        level={0}
        icon={theme.icon('spark')}
        label="skills"
        value={<Text>{skillCount}</Text>}
      />
      {skillDetails.length > 0 ? (
        <>
          {skillDetails.map((item) => (
            <InfoDetailRow
              key={item}
              width={width}
              level={1}
              text={item}
            />
          ))}
          {session.bannerSkillCount > skillDetails.length ? (
            <InfoDetailRow width={width} level={1} text="…" />
          ) : null}
        </>
      ) : null}
      {session.systemOverride ? (
        <InfoRow
          width={width}
          level={0}
          icon={theme.icon('info')}
          label="system"
          value={<Text>custom system prompt loaded</Text>}
        />
      ) : null}
      <BottomBorder width={width} border={border} />
    </Box>
  )
}

function TopBorder({ width }: { width: number }): ReactElement {
  const border = theme.isColorEnabled() ? theme.palette.primaryDim : undefined
  const title = `${theme.icon('logo')} Aether v${APP_VERSION}`
  const filler = '─'.repeat(Math.max(0, width - stringWidth(title) - 5))

  return (
    <Box>
      <Text {...(border ? { color: border } : {})}>╭─ </Text>
      <Text bold {...theme.colorProps('brand')}>
        {theme.icon('logo')} Aether
      </Text>
      <Text {...theme.colorProps('dim')}>{` v${APP_VERSION}`}</Text>
      <Text {...(border ? { color: border } : {})}>{' ' + filler + '╮'}</Text>
    </Box>
  )
}

function BottomBorder({
  width,
  border,
}: {
  width: number
  border: string | undefined
}): ReactElement {
  return (
    <Box>
      <Text {...(border ? { color: border } : {})}>
        {'╰' + '─'.repeat(Math.max(0, width - 2)) + '╯'}
      </Text>
    </Box>
  )
}

function FrameLine({
  width,
  paddingLeft = 2,
  children,
}: {
  width: number
  paddingLeft?: number
  children: ReactNode
}): ReactElement {
  const border = theme.isColorEnabled() ? theme.palette.primaryDim : undefined
  return (
    <Box width={width}>
      <Text {...(border ? { color: border } : {})}>│</Text>
      <Box width={width - 2} paddingLeft={paddingLeft} paddingRight={2}>
        {children}
      </Box>
      <Text {...(border ? { color: border } : {})}>│</Text>
    </Box>
  )
}

function InfoRow({
  width,
  level,
  icon,
  label,
  value,
}: {
  width: number
  level: 0 | 1 | 2
  icon: string
  label: string
  value: ReactNode
}): ReactElement {
  const dim = theme.colorProps('dim')
  const brand = theme.colorProps('brand')
  return (
    <FrameLine width={width} paddingLeft={2 + level * 2}>
      <Box>
        <Box width={LABEL_WIDTH} flexShrink={0}>
          <Box flexDirection="row">
            <Box width={ICON_WIDTH} flexShrink={0}>
              <Text {...brand}>{icon}</Text>
            </Box>
            <Text {...dim}>{label}</Text>
          </Box>
        </Box>
        <Box flexGrow={1}>{value}</Box>
      </Box>
    </FrameLine>
  )
}

function InfoDetailRow({
  width,
  level,
  text,
  tone = 'dim',
}: {
  width: number
  level: 1 | 2
  text: string
  tone?: 'dim' | 'text'
}): ReactElement {
  const props = tone === 'text' ? theme.colorProps('text') : theme.colorProps('dim')
  return (
    <FrameLine width={width} paddingLeft={2 + level * 2}>
      <Text {...props}>{truncate(text, Math.max(16, width - (10 + level * 2)))}</Text>
    </FrameLine>
  )
}

function buildSessionDetails(session: ReturnType<typeof sessionState.get>): string[] {
  const details: string[] = []
  const current = session.sessionId ? session.sessionId.slice(0, 8) : 'starting'
  details.push(current)

  for (const item of session.recentSessions) {
    if (item.session_id === session.sessionId) {
      continue
    }
    const id = item.session_id.slice(0, 8)
    const summary = (item.summary ?? '').replace(/\s+/g, ' ').trim()
    if (summary) {
      details.push(`${id}  ${summary}`)
      continue
    }
    const model = [item.provider, item.model].filter(Boolean).join('/')
    details.push(model ? `${id}  ${model}` : id)
  }
  return details
}

function BootLine(): ReactElement {
  const session = useStore(sessionState)
  const sessionId = session.sessionId ? session.sessionId.slice(0, 8) : '…'
  const mode = session.mode === 'plan' ? ' · plan' : ''
  const dim = theme.colorProps('dim')
  const brand = theme.colorProps('brand')
  return (
    <Box>
      <Text bold {...brand}>
        {theme.icon('logo')} aether{' '}
      </Text>
      <Text {...dim}>
        {session.provider}/{session.model || 'resolving'} · {sessionId}
        {mode}
      </Text>
    </Box>
  )
}
