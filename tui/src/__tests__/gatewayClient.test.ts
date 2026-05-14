import { EventEmitter } from 'node:events'
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join, resolve } from 'node:path'
import { PassThrough } from 'node:stream'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  GatewayClient,
  GatewayCrashedError,
  GatewayRpcError,
  GatewayTimeoutError,
  resolvePython
} from '../gatewayClient.js'
import type { GatewayProcess, GatewaySpawn, StartOptions } from '../gatewayClient.js'
import type { GatewayEvent } from '../gatewayTypes.js'

class FakeGatewayProcess extends EventEmitter implements GatewayProcess {
  readonly stdin = new PassThrough()
  readonly stdout = new PassThrough()
  readonly stderr = new PassThrough()
  readonly signals: Array<NodeJS.Signals | number | undefined> = []
  killed = false
  exitCode: number | null = null

  kill(signal?: NodeJS.Signals | number): boolean {
    this.killed = true
    this.signals.push(signal)
    return true
  }

  writeStdout(frame: unknown): void {
    this.stdout.write(`${JSON.stringify(frame)}\n`)
  }

  writeRawStdout(line: string): void {
    this.stdout.write(`${line}\n`)
  }

  writeStderr(line: string): void {
    this.stderr.write(`${line}\n`)
  }

  exit(code: number | null = 0, signal: NodeJS.Signals | null = null): void {
    this.exitCode = code
    this.emit('exit', code, signal)
    this.stdout.end()
    this.stderr.end()
    this.stdin.end()
  }
}

const spawned: FakeGatewayProcess[] = []
let envSnapshot: NodeJS.ProcessEnv

afterEach(() => {
  for (const proc of spawned) {
    if (proc.exitCode === null) {
      proc.exit(0)
    }
  }
  spawned.length = 0
  vi.restoreAllMocks()
  process.env = envSnapshot
})

describe('GatewayClient', () => {
  envSnapshot = { ...process.env }

  it('starts after gateway.ready and records the event', async () => {
    const { client, proc } = makeClient()
    const readyEvents: GatewayEvent[] = []
    client.on('event', (event) => readyEvents.push(event))

    const start = client.start({ startupTimeoutMs: 50 })
    proc.writeStdout(readyFrame())

    await start

    expect(readyEvents[0]).toMatchObject({
      type: 'gateway.ready',
      version: '0.5.0',
      capabilities: ['ping']
    })
  })

  it('spawns the gateway with resolved python path and PYTHONPATH', async () => {
    const proc = new FakeGatewayProcess()
    const spawn = vi.fn<GatewaySpawn>(() => proc)
    const client = new GatewayClient({ spawn })

    const start = client.start({
      pythonPath: '/custom/python',
      cwd: '/repo',
      pythonSrcRoot: '/repo',
      env: { PYTHONPATH: '/existing' },
      startupTimeoutMs: 50
    })
    proc.writeStdout(readyFrame())
    await start

    expect(spawn).toHaveBeenCalledOnce()
    const spawnCall = spawn.mock.calls[0]
    if (!spawnCall) {
      throw new Error('expected gateway spawn to be called')
    }
    const [command, args, options] = spawnCall
    expect(command).toBe('/custom/python')
    expect(args).toEqual([
      '-u',
      '-c',
      'from aether.gateway.entry import main; raise SystemExit(main())'
    ])
    expect(options.cwd).toBe('/repo')
    expect(options.env?.AETHER_PYTHON_SRC_ROOT).toBe('/repo')
    expect(options.env?.PYTHONPATH).toContain('/repo')
    expect(options.env?.PYTHONPATH).toContain('/existing')
  })

  it('rejects startup when the process exits before ready and includes stderr', async () => {
    const { client, proc } = makeClient()
    const start = client.start({ startupTimeoutMs: 100 })

    proc.writeStderr('boot failed')
    await tick()
    proc.exit(1)

    await expect(start).rejects.toMatchObject({
      name: 'GatewayCrashedError',
      stderr: ['boot failed']
    })
  })

  it('rejects startup on ready timeout and kills the process', async () => {
    const { client, proc } = makeClient()
    const start = client.start({ startupTimeoutMs: 5 })

    await expect(start).rejects.toBeInstanceOf(GatewayTimeoutError)

    expect(proc.signals).toContain('SIGKILL')
  })

  it('resolves a request response by id', async () => {
    const { client, proc } = await makeStartedClient()

    const response = client.request<{ pong: true }>('gateway.ping')
    const requestFrame = await readClientFrame(proc)
    proc.writeStdout({ jsonrpc: '2.0', id: requestFrame.id, result: { pong: true } })

    await expect(response).resolves.toEqual({ pong: true })
  })

  it('resolves concurrent pending requests independently', async () => {
    const { client, proc } = await makeStartedClient()

    const first = client.request('one')
    const firstFrame = await readClientFrame(proc)
    const second = client.request('two')
    const secondFrame = await readClientFrame(proc)

    proc.writeStdout({ jsonrpc: '2.0', id: secondFrame.id, result: { order: 2 } })
    proc.writeStdout({ jsonrpc: '2.0', id: firstFrame.id, result: { order: 1 } })

    await expect(first).resolves.toEqual({ order: 1 })
    await expect(second).resolves.toEqual({ order: 2 })
  })

  it('rejects RPC errors as GatewayRpcError', async () => {
    const { client, proc } = await makeStartedClient()

    const request = client.request('missing.method')
    const frame = await readClientFrame(proc)
    proc.writeStdout({
      jsonrpc: '2.0',
      id: frame.id,
      error: { code: -32601, message: 'unknown method' }
    })

    await expect(request).rejects.toBeInstanceOf(GatewayRpcError)
  })

  it('times out pending requests and removes them', async () => {
    const { client, proc } = await makeStartedClient({ requestTimeoutMs: 5 })

    const request = client.request('slow')
    const frame = await readClientFrame(proc)

    await expect(request).rejects.toBeInstanceOf(GatewayTimeoutError)

    const protocolError = waitForEvent(client, 'gateway.protocol_error')
    proc.writeStdout({ jsonrpc: '2.0', id: frame.id, result: { late: true } })
    await expect(protocolError).resolves.toMatchObject({
      type: 'gateway.protocol_error',
      reason: expect.stringContaining('not pending')
    })
  })

  it('silently drops gateway acks for reverse-RPC responses we sent', async () => {
    // After the TUI writes a response to a server-initiated request
    // (`srv_app_*` / `srv_perm_*`), the gateway dispatcher synthesises an
    // ack response back. Those acks must not show up as protocol errors in
    // the transcript, since `srv_*` ids are not in our `#pending` map.
    const { client, proc } = await makeStartedClient()
    const protocolErrors: GatewayEvent[] = []
    client.on('event', (event) => {
      if (event.type === 'gateway.protocol_error') {
        protocolErrors.push(event)
      }
    })

    proc.writeStdout({ jsonrpc: '2.0', id: 'srv_perm_1', result: { ok: true } })
    proc.writeStdout({ jsonrpc: '2.0', id: 'srv_app_2', result: { ok: true } })
    await tick()

    expect(protocolErrors).toEqual([])
  })

  it('honours timeoutMs:null by disabling the client-side wall clock', async () => {
    // agent.run can legitimately take many minutes; the Python CLI calls the
    // engine synchronously with no client-side timer at all. We mirror that
    // by letting callers opt out per-request.
    const { client, proc } = await makeStartedClient({ requestTimeoutMs: 5 })
    const request = client.request('agent.run', undefined, { timeoutMs: null })
    const frame = await readClientFrame(proc)
    await sleep(20)
    proc.writeStdout({ jsonrpc: '2.0', id: frame.id, result: { exit_reason: 'done' } })
    await expect(request).resolves.toMatchObject({ exit_reason: 'done' })
  })

  it('emits protocol errors for malformed stdout and unknown response ids', async () => {
    const { client, proc } = await makeStartedClient()
    const firstError = waitForEvent(client, 'gateway.protocol_error')

    proc.writeRawStdout('{not-json')

    await expect(firstError).resolves.toMatchObject({
      type: 'gateway.protocol_error',
      raw: '{not-json'
    })

    const secondError = waitForEvent(client, 'gateway.protocol_error')
    proc.writeStdout({ jsonrpc: '2.0', id: 'missing', result: {} })

    await expect(secondError).resolves.toMatchObject({
      type: 'gateway.protocol_error',
      reason: expect.stringContaining('not pending')
    })
  })

  it('captures stderr as logs and synthetic events', async () => {
    const { client, proc } = await makeStartedClient()
    const logs: string[] = []
    client.on('log', (line) => logs.push(line))
    const stderrEvent = waitForEvent(client, 'gateway.stderr')

    proc.writeStderr('warning line')

    await expect(stderrEvent).resolves.toEqual({
      type: 'gateway.stderr',
      line: 'warning line'
    })
    expect(logs).toEqual(['warning line'])
    expect(client.logs).toEqual(['warning line'])
  })

  it('replays buffered events to new subscribers', async () => {
    const { client, proc } = await makeStartedClient()
    proc.writeStdout({
      jsonrpc: '2.0',
      method: 'event',
      params: {
        type: 'status',
        session_id: 's1',
        run_id: 'r1',
        kind: 'thinking'
      }
    })
    await tick()

    const iterator = client.subscribe()[Symbol.asyncIterator]()

    await expect(iterator.next()).resolves.toMatchObject({
      value: { type: 'gateway.ready' },
      done: false
    })
    await expect(iterator.next()).resolves.toMatchObject({
      value: { type: 'status', kind: 'thinking' },
      done: false
    })
    await iterator.return?.()
  })

  it('surfaces server-initiated srv_* requests as events', async () => {
    const { client, proc } = await makeStartedClient()
    const serverRequest = waitForEvent(client, 'gateway.server_request')

    proc.writeStdout({
      jsonrpc: '2.0',
      id: 'srv_app_1',
      method: 'approval.request',
      params: { kind: 'plan', deadline_ms: 1000 }
    })

    await expect(serverRequest).resolves.toMatchObject({
      type: 'gateway.server_request',
      id: 'srv_app_1',
      method: 'approval.request'
    })
  })

  it('writes notifications and reverse-RPC responses', async () => {
    const { client, proc } = await makeStartedClient()

    client.notify('client.ready', { ok: true })
    const notifyFrame = await readClientFrame(proc)
    client.respond('srv_app_1', { approved: true })
    const responseFrame = await readClientFrame(proc)

    expect(notifyFrame).toEqual({
      jsonrpc: '2.0',
      method: 'client.ready',
      params: { ok: true }
    })
    expect(responseFrame).toEqual({
      jsonrpc: '2.0',
      id: 'srv_app_1',
      result: { approved: true }
    })
  })

  it('sends SIGTERM then SIGKILL when stop grace expires', async () => {
    const { client, proc } = await makeStartedClient()
    const stopped = client.stop({ graceMs: 5 })

    expect(proc.signals[0]).toBe('SIGTERM')
    await sleep(15)
    expect(proc.signals[1]).toBe('SIGKILL')

    proc.exit(0)
    await stopped
  })

  it('rejects pending requests when the gateway exits', async () => {
    const { client, proc } = await makeStartedClient()

    const request = client.request('long')
    await readClientFrame(proc)
    proc.exit(1)

    await expect(request).rejects.toBeInstanceOf(GatewayCrashedError)
  })

  it('can start again after a clean process exit', async () => {
    const firstProc = new FakeGatewayProcess()
    const secondProc = new FakeGatewayProcess()
    spawned.push(firstProc, secondProc)
    const spawn = vi.fn<GatewaySpawn>()
    spawn.mockReturnValueOnce(firstProc)
    spawn.mockReturnValueOnce(secondProc)
    const client = new GatewayClient({ spawn })

    const firstStart = client.start({ startupTimeoutMs: 50 })
    firstProc.writeStdout(readyFrame())
    await firstStart
    firstProc.exit(0)
    await tick()

    const secondStart = client.start({ startupTimeoutMs: 50 })
    secondProc.writeStdout(readyFrame())
    await secondStart

    expect(spawn).toHaveBeenCalledTimes(2)
  })
})

describe('resolvePython', () => {
  it('prefers explicit env overrides', () => {
    expect(resolvePython('/repo', { AETHER_PYTHON: '/aether/python' })).toBe('/aether/python')
    expect(resolvePython('/repo', { PYTHON: '/usr/bin/python-custom' })).toBe(
      '/usr/bin/python-custom'
    )
  })

  it('prefers VIRTUAL_ENV before repo-local venvs', () => {
    const root = mkdtempSync(join(tmpdir(), 'aether-root-'))
    const venv = mkdtempSync(join(tmpdir(), 'aether-venv-'))
    const venvBin = join(venv, 'bin')
    const rootBin = join(root, '.venv', 'bin')
    mkdirSync(venvBin, { recursive: true })
    mkdirSync(rootBin, { recursive: true })
    writeFileSync(join(venvBin, 'python'), '')
    writeFileSync(join(rootBin, 'python'), '')

    expect(resolvePython(root, { VIRTUAL_ENV: venv })).toBe(resolve(venvBin, 'python'))
  })

  it('falls back to repo-local .venv before system python', () => {
    const root = mkdtempSync(join(tmpdir(), 'aether-root-'))
    const rootBin = join(root, '.venv', 'bin')
    mkdirSync(rootBin, { recursive: true })
    writeFileSync(join(rootBin, 'python3'), '')

    expect(resolvePython(root, {})).toBe(resolve(rootBin, 'python3'))
  })
})

function makeClient() {
  const proc = new FakeGatewayProcess()
  spawned.push(proc)
  const client = new GatewayClient({ spawn: () => proc })
  return { client, proc }
}

async function makeStartedClient(opts: { requestTimeoutMs?: number } = {}) {
  const { client, proc } = makeClient()
  const startOpts: StartOptions = { startupTimeoutMs: 50 }
  if (opts.requestTimeoutMs !== undefined) {
    startOpts.requestTimeoutMs = opts.requestTimeoutMs
  }
  const start = client.start(startOpts)
  proc.writeStdout(readyFrame())
  await start
  return { client, proc }
}

function readyFrame() {
  return {
    jsonrpc: '2.0',
    method: 'gateway.ready',
    params: {
      version: '0.5.0',
      capabilities: ['ping'],
      methods: ['gateway.ping']
    }
  }
}

function readClientFrame(proc: FakeGatewayProcess): Promise<Record<string, unknown>> {
  return new Promise((resolveFrame) => {
    proc.stdin.once('data', (chunk) => {
      resolveFrame(JSON.parse(String(chunk).trim()))
    })
  })
}

function waitForEvent(client: GatewayClient, type: GatewayEvent['type']): Promise<GatewayEvent> {
  return new Promise((resolveEvent) => {
    client.on('event', (event) => {
      if (event.type === type) {
        resolveEvent(event)
      }
    })
  })
}

function tick(): Promise<void> {
  return Promise.resolve()
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolveSleep) => {
    setTimeout(resolveSleep, ms)
  })
}
