export const BSU = '\x1b[?2026h'
export const ESU = '\x1b[?2026l'

type WriteCallback = (error?: Error | null) => void

/**
 * Terminal allowlist for DEC 2026 synchronized output.
 *
 * This mirrors the practical allowlist used by open-claude-code, with tmux
 * explicitly disabled because it proxies bytes without preserving the
 * synchronized update boundary.
 */
export function isSynchronizedOutputSupported(): boolean {
  if (process.env.TMUX) {
    return false
  }

  const termProgram = process.env.TERM_PROGRAM
  const term = process.env.TERM

  if (
    termProgram === 'iTerm.app' ||
    termProgram === 'WezTerm' ||
    termProgram === 'WarpTerminal' ||
    termProgram === 'ghostty' ||
    termProgram === 'contour' ||
    termProgram === 'vscode' ||
    termProgram === 'alacritty'
  ) {
    return true
  }

  if (term?.includes('kitty') || process.env.KITTY_WINDOW_ID) {
    return true
  }
  if (term === 'xterm-ghostty') {
    return true
  }
  if (term?.startsWith('foot')) {
    return true
  }
  if (term?.includes('alacritty')) {
    return true
  }
  if (process.env.ZED_TERM) {
    return true
  }
  if (process.env.WT_SESSION) {
    return true
  }

  const vteVersion = process.env.VTE_VERSION
  if (vteVersion) {
    const version = Number.parseInt(vteVersion, 10)
    if (!Number.isNaN(version) && version >= 6800) {
      return true
    }
  }

  return false
}

export const SYNC_SUPPORTED = isSynchronizedOutputSupported()

export function wrapStdoutWithAtomicSync(stdout: NodeJS.WriteStream): NodeJS.WriteStream {
  return makeProxy(stdout, SYNC_SUPPORTED)
}

function makeProxy(stdout: NodeJS.WriteStream, syncSupported: boolean): NodeJS.WriteStream {
  let buffer: string | null = null

  const passThroughWrite = (args: unknown[]): boolean => {
    return Reflect.apply(stdout.write, stdout, args) as boolean
  }

  const write = (...args: unknown[]): boolean => {
    const chunk = args[0]
    const callback = callbackFromArgs(args)
    const stringChunk = typeof chunk === 'string' ? chunk : null

    if (syncSupported) {
      if (stringChunk === BSU) {
        buffer = BSU
        callback?.()
        return true
      }

      if (stringChunk === ESU) {
        if (buffer !== null) {
          const fullFrame = buffer + ESU
          buffer = null
          if (callback) {
            return stdout.write(fullFrame, callback)
          }
          return stdout.write(fullFrame)
        }
        return passThroughWrite(args)
      }

      if (buffer !== null) {
        buffer += stringifyChunk(chunk)
        callback?.()
        return true
      }

      return passThroughWrite(args)
    }

    if (stringChunk === BSU || stringChunk === ESU) {
      callback?.()
      return true
    }

    return passThroughWrite(args)
  }

  return new Proxy(stdout, {
    get(target, property, receiver) {
      if (property === 'write') {
        return write
      }
      const value = Reflect.get(target, property, receiver)
      return typeof value === 'function' ? value.bind(target) : value
    }
  }) as NodeJS.WriteStream
}

function callbackFromArgs(args: unknown[]): WriteCallback | undefined {
  const last = args[args.length - 1]
  return typeof last === 'function' ? (last as WriteCallback) : undefined
}

function stringifyChunk(chunk: unknown): string {
  if (typeof chunk === 'string') {
    return chunk
  }
  if (chunk instanceof Uint8Array) {
    return Buffer.from(chunk).toString()
  }
  return String(chunk ?? '')
}
