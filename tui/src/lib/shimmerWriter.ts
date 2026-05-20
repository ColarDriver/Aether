import type { DOMElement } from 'ink'

import { BSU, ESU, SYNC_SUPPORTED } from './atomicSyncStdout.js'
import { nowMs, subscribe } from './animationClock.js'
import { shimmer } from './shimmer.js'

const SAVE_CURSOR = '\x1b7'
const RESTORE_CURSOR = '\x1b8'
const RESET = '\x1b[0m'

export interface ShimmerCell {
  row: number
  col: number
}

export interface ShimmerWriterOptions extends ShimmerCell {
  stdout: NodeJS.WriteStream
  label: string
  baseColor: string
  highlightColor: string
  intervalMs: number
}

export interface RunningShimmerWriter {
  stop: () => void
}

export function isDirectWriteShimmerEnabled(): boolean {
  return process.env.AETHER_SHIMMER_DIRECT_WRITE === '1'
}

export function startShimmerWriter(
  options: ShimmerWriterOptions
): RunningShimmerWriter | null {
  if (!isDirectWriteShimmerEnabled() || !options.stdout.isTTY || !options.label) {
    return null
  }

  let lastTick = -1
  const writeCurrentFrame = () => {
    const tick = Math.floor(nowMs() / options.intervalMs)
    if (tick === lastTick) {
      return
    }
    lastTick = tick
    options.stdout.write(buildShimmerFrame(options, tick))
  }

  writeCurrentFrame()
  const unsubscribe = subscribe(writeCurrentFrame)
  return {
    stop: unsubscribe
  }
}

export function buildShimmerFrame(
  options: Omit<ShimmerWriterOptions, 'stdout' | 'intervalMs'>,
  tick: number,
  syncSupported = SYNC_SUPPORTED
): string {
  const slices = shimmer(options.label, tick)
  const base = ansiColor(options.baseColor)
  const highlight = ansiColor(options.highlightColor)
  const cursorTo = `\x1b[${Math.max(1, Math.floor(options.row))};${Math.max(
    1,
    Math.floor(options.col)
  )}H`
  const payload =
    cursorTo +
    base +
    slices.before +
    highlight +
    slices.highlight +
    base +
    slices.after +
    RESET

  return `${syncSupported ? BSU : ''}${SAVE_CURSOR}${payload}${RESTORE_CURSOR}${
    syncSupported ? ESU : ''
  }`
}

export function measureShimmerCell(
  element: DOMElement | null,
  stdout: NodeJS.WriteStream | undefined,
  colOffset: number
): ShimmerCell | null {
  if (!element?.yogaNode || !stdout?.isTTY) {
    return null
  }

  const rows = typeof stdout.rows === 'number' && stdout.rows > 0 ? stdout.rows : 24
  let absoluteTop = element.yogaNode.getComputedTop()
  let absoluteLeft = element.yogaNode.getComputedLeft()
  let root = element.yogaNode
  let parent = element.parentNode

  while (parent) {
    if (parent.yogaNode) {
      absoluteTop += parent.yogaNode.getComputedTop()
      absoluteLeft += parent.yogaNode.getComputedLeft()
      root = parent.yogaNode
    }
    const scrollTop = (parent as { scrollTop?: number }).scrollTop
    if (scrollTop) {
      absoluteTop -= scrollTop
    }
    parent = parent.parentNode
  }

  const screenHeight = Math.max(1, root.getComputedHeight())
  return {
    row: clamp(rows - screenHeight + absoluteTop + 1, 1, rows),
    col: Math.max(1, Math.floor(absoluteLeft + colOffset + 1))
  }
}

function ansiColor(color: string): string {
  const hex = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color)
  if (hex) {
    const red = Number.parseInt(hex[1] ?? '0', 16)
    const green = Number.parseInt(hex[2] ?? '0', 16)
    const blue = Number.parseInt(hex[3] ?? '0', 16)
    return `\x1b[38;2;${red};${green};${blue}m`
  }
  if (color === 'white') {
    return '\x1b[37m'
  }
  if (color === 'gray' || color === 'grey') {
    return '\x1b[90m'
  }
  return ''
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, Math.floor(value)))
}
