import { spawn as nodeSpawn } from 'node:child_process'
import type { SpawnOptionsWithoutStdio } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { existsSync } from 'node:fs'
import { delimiter, resolve } from 'node:path'
import { createInterface } from 'node:readline'
import type { Interface as ReadlineInterface } from 'node:readline'
import type { Readable, Writable } from 'node:stream'

import { CircularBuffer } from './lib/circularBuffer.js'
import type {
  GatewayEvent,
  GatewayProtocolError as GatewayProtocolErrorEvent,
  GatewayReady,
  GatewayServerRequest,
  JsonObject,
  RpcError,
  RpcId
} from './gatewayTypes.js'

const GATEWAY_MAIN_SNIPPET =
  'from aether.gateway.entry import main; raise SystemExit(main())'

const DEFAULT_STARTUP_TIMEOUT_MS = 15_000
const DEFAULT_REQUEST_TIMEOUT_MS = 120_000
const DEFAULT_STOP_GRACE_MS = 2_000
const DEFAULT_EVENT_BUFFER_SIZE = 2_000
const DEFAULT_LOG_BUFFER_SIZE = 500

export type StartOptions = {
  pythonPath?: string
  cwd?: string
  pythonSrcRoot?: string
  env?: NodeJS.ProcessEnv
  startupTimeoutMs?: number
  requestTimeoutMs?: number
}

export type StopOptions = {
  signal?: NodeJS.Signals
  graceMs?: number
}

export interface GatewayProcess extends EventEmitter {
  stdin: Writable
  stdout: Readable
  stderr: Readable
  killed?: boolean
  exitCode?: number | null
  kill(signal?: NodeJS.Signals | number): boolean
}

export type GatewaySpawn = (
  command: string,
  args: string[],
  options: SpawnOptionsWithoutStdio
) => GatewayProcess

export type GatewayClientOptions = {
  spawn?: GatewaySpawn
  eventBufferSize?: number
  logBufferSize?: number
}

type PendingRequest = {
  method: string
  resolve: (value: unknown) => void
  reject: (reason: unknown) => void
  timer: NodeJS.Timeout
}

class AsyncEventQueue {
  #queue: GatewayEvent[]
  #closed = false
  #waiting: ((result: IteratorResult<GatewayEvent>) => void) | null = null

  constructor(replay: GatewayEvent[]) {
    this.#queue = [...replay]
  }

  push(event: GatewayEvent): void {
    if (this.#closed) {
      return
    }
    const waiting = this.#waiting
    if (waiting) {
      this.#waiting = null
      waiting({ done: false, value: event })
      return
    }
    this.#queue.push(event)
  }

  next(): Promise<IteratorResult<GatewayEvent>> {
    const event = this.#queue.shift()
    if (event) {
      return Promise.resolve({ done: false, value: event })
    }
    if (this.#closed) {
      return Promise.resolve({ done: true, value: undefined })
    }
    return new Promise((resolveNext) => {
      this.#waiting = resolveNext
    })
  }

  close(): void {
    if (this.#closed) {
      return
    }
    this.#closed = true
    const waiting = this.#waiting
    if (waiting) {
      this.#waiting = null
      waiting({ done: true, value: undefined })
    }
  }
}

export class GatewayTimeoutError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GatewayTimeoutError'
  }
}

export class GatewayCrashedError extends Error {
  readonly code: number | null
  readonly signal: NodeJS.Signals | null
  readonly stderr: readonly string[]

  constructor(
    message: string,
    options: {
      code?: number | null
      signal?: NodeJS.Signals | null
      stderr?: readonly string[]
      cause?: unknown
    } = {}
  ) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause })
    this.name = 'GatewayCrashedError'
    this.code = options.code ?? null
    this.signal = options.signal ?? null
    this.stderr = options.stderr ?? []
  }
}

export class GatewayRpcError extends Error {
  readonly code: number
  readonly data: JsonObject | null | undefined
  readonly method: string

  constructor(method: string, error: RpcError) {
    super(error.message)
    this.name = 'GatewayRpcError'
    this.method = method
    this.code = error.code
    this.data = error.data
  }
}

export class GatewayProtocolError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'GatewayProtocolError'
  }
}

export function resolvePython(root: string, env: NodeJS.ProcessEnv = process.env): string {
  const explicit = env.AETHER_PYTHON?.trim() || env.PYTHON?.trim()
  if (explicit) {
    return explicit
  }

  const venv = env.VIRTUAL_ENV?.trim()
  const candidates = [
    venv && resolve(venv, 'bin/python'),
    venv && resolve(venv, 'Scripts/python.exe'),
    resolve(root, '.venv/bin/python'),
    resolve(root, '.venv/bin/python3'),
    resolve(root, 'venv/bin/python'),
    resolve(root, 'venv/bin/python3')
  ].filter((candidate): candidate is string => Boolean(candidate))

  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate
    }
  }

  return process.platform === 'win32' ? 'python' : 'python3'
}

export class GatewayClient extends EventEmitter {
  readonly #spawn: GatewaySpawn
  readonly #events: CircularBuffer<GatewayEvent>
  readonly #logs: CircularBuffer<string>
  readonly #subscribers = new Set<AsyncEventQueue>()
  readonly #pending = new Map<RpcId, PendingRequest>()
  #proc: GatewayProcess | null = null
  #stdoutReader: ReadlineInterface | null = null
  #stderrReader: ReadlineInterface | null = null
  #startPromise: Promise<void> | null = null
  #startResolve: (() => void) | null = null
  #startReject: ((reason: unknown) => void) | null = null
  #startupTimer: NodeJS.Timeout | null = null
  #exitPromise: Promise<void> | null = null
  #exitResolve: (() => void) | null = null
  #ready = false
  #requestTimeoutMs = DEFAULT_REQUEST_TIMEOUT_MS
  #nextRequestId = 1

  constructor(options: GatewayClientOptions = {}) {
    super()
    this.#spawn = options.spawn ?? ((command, args, spawnOptions) => nodeSpawn(command, args, spawnOptions))
    this.#events = new CircularBuffer<GatewayEvent>(
      options.eventBufferSize ?? DEFAULT_EVENT_BUFFER_SIZE
    )
    this.#logs = new CircularBuffer<string>(options.logBufferSize ?? DEFAULT_LOG_BUFFER_SIZE)
  }

  get logs(): readonly string[] {
    return this.#logs.toArray()
  }

  override on(event: 'event', cb: (ev: GatewayEvent) => void): this
  override on(event: 'log', cb: (line: string) => void): this
  override on(event: 'exit', cb: (code: number | null) => void): this
  // EventEmitter's implementation signature is intentionally any[].
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  override on(event: string | symbol, cb: (...args: any[]) => void): this {
    return super.on(event, cb)
  }

  async start(opts: StartOptions = {}): Promise<void> {
    if (this.#ready) {
      return
    }
    if (this.#startPromise) {
      return this.#startPromise
    }
    if (this.#proc) {
      throw new Error('Gateway process has already been started')
    }

    const cwd = opts.cwd ?? process.cwd()
    const env = this.#buildEnv(cwd, opts)
    const pythonSrcRoot = env.AETHER_PYTHON_SRC_ROOT ?? cwd
    const pythonPath = opts.pythonPath ?? resolvePython(pythonSrcRoot, env)
    const startupTimeoutMs = opts.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS
    this.#requestTimeoutMs = opts.requestTimeoutMs ?? DEFAULT_REQUEST_TIMEOUT_MS
    this.#ready = false

    const startPromise = new Promise<void>((resolveStart, rejectStart) => {
      this.#startResolve = resolveStart
      this.#startReject = rejectStart
    })
    this.#startPromise = startPromise

    try {
      const proc = this.#spawn(pythonPath, ['-u', '-c', GATEWAY_MAIN_SNIPPET], {
        cwd,
        env,
        stdio: ['pipe', 'pipe', 'pipe']
      })
      this.#bindProcess(proc)
    } catch (error) {
      const crashed = new GatewayCrashedError('Failed to spawn gateway process', {
        stderr: this.logs,
        cause: error
      })
      this.#failStart(crashed)
      return startPromise
    }

    this.#startupTimer = setTimeout(() => {
      const timeout = new GatewayTimeoutError(
        `Gateway did not emit gateway.ready within ${startupTimeoutMs}ms`
      )
      this.#failStart(timeout)
      this.#proc?.kill('SIGKILL')
    }, startupTimeoutMs)

    return startPromise
  }

  async stop(opts: StopOptions = {}): Promise<void> {
    const proc = this.#proc
    if (!proc) {
      return
    }

    const signal = opts.signal ?? 'SIGTERM'
    const graceMs = opts.graceMs ?? DEFAULT_STOP_GRACE_MS
    const exitPromise = this.#exitPromise ?? Promise.resolve()
    proc.kill(signal)

    const exitedGracefully = await this.#waitForExit(graceMs)
    if (!exitedGracefully) {
      proc.kill('SIGKILL')
    }

    await Promise.race([exitPromise, delay(Math.min(graceMs, 500))])
  }

  request<T = unknown>(method: string, params?: object): Promise<T> {
    const id = `cli_${this.#nextRequestId++}`
    const frame = this.#withOptionalParams({ jsonrpc: '2.0' as const, id, method }, params)

    return new Promise<T>((resolveRequest, rejectRequest) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id)
        rejectRequest(
          new GatewayTimeoutError(
            `Gateway request timed out after ${this.#requestTimeoutMs}ms: ${method}`
          )
        )
      }, this.#requestTimeoutMs)

      this.#pending.set(id, {
        method,
        resolve: (value) => resolveRequest(value as T),
        reject: rejectRequest,
        timer
      })

      try {
        this.#writeFrame(frame)
      } catch (error) {
        clearTimeout(timer)
        this.#pending.delete(id)
        rejectRequest(error)
      }
    })
  }

  notify(method: string, params?: object): void {
    this.#writeFrame(this.#withOptionalParams({ jsonrpc: '2.0' as const, method }, params))
  }

  respond(id: string, result: unknown): void {
    this.#writeFrame({ jsonrpc: '2.0', id, result })
  }

  subscribe(): AsyncIterable<GatewayEvent> {
    const queue = new AsyncEventQueue(this.#events.toArray())
    this.#subscribers.add(queue)
    const subscribers = this.#subscribers

    return {
      async *[Symbol.asyncIterator]() {
        try {
          while (true) {
            const next = await queue.next()
            if (next.done) {
              return
            }
            yield next.value
          }
        } finally {
          queue.close()
          subscribers.delete(queue)
        }
      }
    }
  }

  #bindProcess(proc: GatewayProcess): void {
    this.#proc = proc
    this.#exitPromise = new Promise((resolveExit) => {
      this.#exitResolve = resolveExit
    })

    this.#stdoutReader = createInterface({ input: proc.stdout, crlfDelay: Infinity })
    this.#stderrReader = createInterface({ input: proc.stderr, crlfDelay: Infinity })

    this.#stdoutReader.on('line', (line) => {
      this.#handleStdoutLine(line)
    })
    this.#stderrReader.on('line', (line) => {
      this.#handleStderrLine(line)
    })

    proc.once('error', (error: Error) => {
      const crashed = new GatewayCrashedError(`Gateway process error: ${error.message}`, {
        stderr: this.logs,
        cause: error
      })
      if (!this.#ready) {
        this.#failStart(crashed)
      }
      this.#rejectAllPending(crashed)
    })

    proc.once('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      this.#handleExit(code, signal)
    })
  }

  #handleStdoutLine(raw: string): void {
    const line = raw.trim()
    if (!line) {
      return
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(line)
    } catch (error) {
      const reason = error instanceof Error ? error.message : 'invalid JSON'
      this.#emitProtocolError(raw, reason)
      return
    }

    if (!isRecord(parsed)) {
      this.#emitProtocolError(raw, 'frame must be a JSON object')
      return
    }

    this.#handleFrame(parsed, raw)
  }

  #handleFrame(frame: Record<string, unknown>, raw: string): void {
    if (isResponseFrame(frame)) {
      this.#handleResponseFrame(frame, raw)
      return
    }

    if (isNotificationFrame(frame)) {
      this.#handleNotificationFrame(frame, raw)
      return
    }

    if (isRequestFrame(frame)) {
      this.#handleServerRequestFrame(frame, raw)
      return
    }

    this.#emitProtocolError(raw, 'unrecognised JSON-RPC frame shape')
  }

  #handleResponseFrame(frame: Record<string, unknown>, raw: string): void {
    const id = frame.id
    if (id === null || (typeof id !== 'string' && typeof id !== 'number')) {
      this.#emitProtocolError(raw, 'response id must be a string, number, or null')
      return
    }

    const pending = this.#pending.get(id)
    if (!pending) {
      this.#emitProtocolError(raw, `response id is not pending: ${String(id)}`)
      return
    }

    this.#pending.delete(id)
    clearTimeout(pending.timer)

    if ('error' in frame) {
      if (!isRpcError(frame.error)) {
        pending.reject(new GatewayProtocolError('response error is not a valid RpcError'))
        return
      }
      pending.reject(new GatewayRpcError(pending.method, frame.error))
      return
    }

    pending.resolve(frame.result)
  }

  #handleNotificationFrame(frame: Record<string, unknown>, raw: string): void {
    const method = frame.method
    const params = frame.params

    if (method === 'gateway.ready') {
      const event = normalizeGatewayReady(params)
      this.#emitEvent(event)
      this.#resolveStart()
      return
    }

    if (method === 'gateway.error') {
      if (!isRecord(params)) {
        this.#emitProtocolError(raw, 'gateway.error params must be an object')
        return
      }
      this.#emitEvent({ type: 'gateway.error', ...params } as GatewayEvent)
      return
    }

    if (method === 'event') {
      if (!isRecord(params) || typeof params.type !== 'string') {
        this.#emitProtocolError(raw, 'event params must be an object with a type')
        return
      }
      this.#emitEvent(params as unknown as GatewayEvent)
      return
    }

    this.#emitProtocolError(raw, `unknown notification method: ${method}`)
  }

  #handleServerRequestFrame(frame: Record<string, unknown>, raw: string): void {
    const id = frame.id
    const method = frame.method
    if (typeof method !== 'string') {
      this.#emitProtocolError(raw, 'server request method must be a string')
      return
    }
    if (typeof id !== 'string' || !id.startsWith('srv_')) {
      this.#emitProtocolError(raw, 'server request id must start with srv_')
      return
    }

    this.#emitEvent({
      type: 'gateway.server_request',
      id,
      method,
      params: frame.params ?? {}
    } satisfies GatewayServerRequest)
  }

  #handleStderrLine(line: string): void {
    this.#logs.push(line)
    this.emit('log', line)
    this.#emitEvent({ type: 'gateway.stderr', line })
  }

  #handleExit(code: number | null, signal: NodeJS.Signals | null): void {
    this.#clearStartupTimer()
    this.#stdoutReader?.close()
    this.#stderrReader?.close()
    this.#stdoutReader = null
    this.#stderrReader = null

    const crashed = new GatewayCrashedError(this.#exitMessage(code, signal), {
      code,
      signal,
      stderr: this.logs
    })
    if (!this.#ready) {
      this.#failStart(crashed)
    }
    this.#rejectAllPending(crashed)
    this.#closeSubscribers()

    this.#ready = false
    this.#startPromise = null
    this.#proc = null
    this.emit('exit', code)
    this.#exitResolve?.()
    this.#exitResolve = null
    this.#exitPromise = null
  }

  #resolveStart(): void {
    this.#ready = true
    this.#clearStartupTimer()
    const resolveStart = this.#startResolve
    this.#startResolve = null
    this.#startReject = null
    resolveStart?.()
  }

  #failStart(reason: unknown): void {
    this.#clearStartupTimer()
    const rejectStart = this.#startReject
    this.#startResolve = null
    this.#startReject = null
    this.#startPromise = null
    rejectStart?.(reason)
  }

  #clearStartupTimer(): void {
    if (this.#startupTimer) {
      clearTimeout(this.#startupTimer)
      this.#startupTimer = null
    }
  }

  #rejectAllPending(reason: unknown): void {
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(reason)
    }
    this.#pending.clear()
  }

  #closeSubscribers(): void {
    for (const subscriber of this.#subscribers) {
      subscriber.close()
    }
    this.#subscribers.clear()
  }

  #emitEvent(event: GatewayEvent): void {
    this.#events.push(event)
    for (const subscriber of this.#subscribers) {
      subscriber.push(event)
    }
    this.emit('event', event)
  }

  #emitProtocolError(raw: string, reason: string): void {
    const event: GatewayProtocolErrorEvent = {
      type: 'gateway.protocol_error',
      raw,
      reason
    }
    this.#emitEvent(event)
  }

  #writeFrame(frame: Record<string, unknown>): void {
    const proc = this.#proc
    if (!proc) {
      throw new Error('Gateway process is not running')
    }
    proc.stdin.write(`${JSON.stringify(frame)}\n`)
  }

  #withOptionalParams<T extends Record<string, unknown>>(frame: T, params?: object): T {
    if (params === undefined) {
      return frame
    }
    return { ...frame, params } as T
  }

  #buildEnv(cwd: string, opts: StartOptions): NodeJS.ProcessEnv {
    const env: NodeJS.ProcessEnv = { ...process.env, ...opts.env }
    const pythonSrcRoot = opts.pythonSrcRoot ?? env.AETHER_PYTHON_SRC_ROOT ?? cwd
    env.AETHER_PYTHON_SRC_ROOT = pythonSrcRoot
    env.PYTHONPATH = env.PYTHONPATH
      ? `${pythonSrcRoot}${delimiter}${env.PYTHONPATH}`
      : pythonSrcRoot
    return env
  }

  async #waitForExit(timeoutMs: number): Promise<boolean> {
    if (!this.#exitPromise) {
      return true
    }
    const marker = Symbol('timeout')
    const result = await Promise.race([this.#exitPromise, delay(timeoutMs).then(() => marker)])
    return result !== marker
  }

  #exitMessage(code: number | null, signal: NodeJS.Signals | null): string {
    const reason = signal ? `signal=${signal}` : `code=${String(code)}`
    const stderr = this.logs.length > 0 ? `\nRecent stderr:\n${this.logs.join('\n')}` : ''
    return `Gateway process exited (${reason})${stderr}`
  }
}

function normalizeGatewayReady(params: unknown): GatewayReady {
  const payload = isRecord(params) ? params : {}
  return {
    type: 'gateway.ready',
    version: typeof payload.version === 'string' ? payload.version : 'unknown',
    capabilities: arrayOfStrings(payload.capabilities),
    methods: arrayOfStrings(payload.methods)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isResponseFrame(frame: Record<string, unknown>): boolean {
  return 'id' in frame && !('method' in frame) && ('result' in frame || 'error' in frame)
}

function isNotificationFrame(frame: Record<string, unknown>): frame is Record<string, unknown> & {
  method: string
} {
  return typeof frame.method === 'string' && !('id' in frame)
}

function isRequestFrame(frame: Record<string, unknown>): frame is Record<string, unknown> & {
  id: RpcId
  method: string
} {
  return (
    typeof frame.method === 'string' &&
    'id' in frame &&
    !('result' in frame) &&
    !('error' in frame)
  )
}

function isRpcError(value: unknown): value is RpcError {
  return (
    isRecord(value) &&
    typeof value.code === 'number' &&
    typeof value.message === 'string' &&
    (!('data' in value) || value.data === null || isRecord(value.data))
  )
}

function arrayOfStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string')
}

function delay(ms: number): Promise<void> {
  return new Promise((resolveDelay) => {
    setTimeout(resolveDelay, ms)
  })
}
