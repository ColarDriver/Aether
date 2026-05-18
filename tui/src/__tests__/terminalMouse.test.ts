import { describe, expect, it } from 'vitest'

import {
  isOnlyMouseTrackingInput,
  mouseButtonCodes,
  stripMouseTrackingSequences
} from '../lib/terminalMouse.js'

describe('terminal mouse tracking helpers', () => {
  it('strips SGR mouse reports before they reach text input', () => {
    expect(stripMouseTrackingSequences('\x1b[<64;85;16M')).toBe('')
    expect(stripMouseTrackingSequences('[<65;85;16M')).toBe('')
    expect(stripMouseTrackingSequences('a\x1b[<64;85;16Mb')).toBe('ab')
  })

  it('detects mouse-only input chunks', () => {
    expect(isOnlyMouseTrackingInput('\x1b[<64;85;16M[<64;85;16M')).toBe(true)
    expect(isOnlyMouseTrackingInput('hello')).toBe(false)
  })

  it('extracts wheel button codes from raw terminal data', () => {
    expect(mouseButtonCodes('\x1b[<64;85;16M[<65;85;16M')).toEqual([64, 65])
  })
})
