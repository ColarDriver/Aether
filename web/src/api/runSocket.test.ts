// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest'
import { setBaseUrl, setSessionToken } from './client'
import { buildRunSocketUrl } from './runSocket'

describe('run socket client', () => {
  beforeEach(() => {
    setBaseUrl('http://127.0.0.1:9120')
    setSessionToken(null)
  })

  it('builds the run websocket URL from the REST base URL', () => {
    expect(buildRunSocketUrl()).toBe('ws://127.0.0.1:9120/api/runs/ws')
  })

  it('includes the local web token when configured', () => {
    setSessionToken('abc')
    expect(buildRunSocketUrl()).toBe('ws://127.0.0.1:9120/api/runs/ws?token=abc')
  })
})
