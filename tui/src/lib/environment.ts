import { existsSync, readFileSync } from 'node:fs'
import { arch, homedir, platform } from 'node:os'
import { join, resolve } from 'node:path'

export interface EnvironmentInfo {
  cwd: string
  /** Repo-relative root (e.g. `/workspace/Aether`) when one can be detected. */
  repoRoot: string | null
  branch: string | null
}

/**
 * Best-effort environment probe used by the banner and the system prompt
 * environment block. All fields fall back to safe defaults; nothing here may
 * throw at boot time.
 */
export function probeEnvironment(): EnvironmentInfo {
  const cwd = resolveWorkspaceCwd()
  const repoRoot = findRepoRoot(cwd)
  const branch = repoRoot ? readGitBranch(repoRoot) : null
  return { cwd: shortenPath(cwd), repoRoot, branch }
}

export function resolveWorkspaceCwd(): string {
  return resolve(process.env.AETHER_WORKSPACE_CWD || process.env.INIT_CWD || safeCwd())
}

export function augmentSystemPrompt(userPrompt: string | null, cwd = resolveWorkspaceCwd()): string {
  const envBlock = buildEnvironmentContext(cwd)
  const trimmed = userPrompt?.trim()
  return trimmed ? `${envBlock}\n\n${trimmed}` : envBlock
}

export function buildEnvironmentContext(cwd = resolveWorkspaceCwd()): string {
  const repoRoot = findRepoRoot(cwd)
  const branch = repoRoot ? readGitBranch(repoRoot) : null
  const lines = [
    '<environment>',
    `working_directory: ${cwd}`,
    `platform: ${platform()} (${arch()})`,
    `is_git_repository: ${repoRoot ? 'yes' : 'no'}`
  ]
  if (repoRoot && repoRoot !== cwd) {
    lines.push(`git_root: ${repoRoot}`)
  }
  if (branch) {
    lines.push(`git_branch: ${branch}`)
  }
  lines.push(`shell: ${process.env.SHELL || 'unknown'}`)
  lines.push(`date: ${formatDate(new Date())}`)
  lines.push('</environment>')
  return lines.join('\n')
}

function safeCwd(): string {
  try {
    return process.cwd()
  } catch {
    return '?'
  }
}

function findRepoRoot(start: string): string | null {
  let current = start
  for (let i = 0; i < 12; i++) {
    if (existsSync(join(current, '.git'))) {
      return current
    }
    const parent = resolve(current, '..')
    if (parent === current) {
      return null
    }
    current = parent
  }
  return null
}

function readGitBranch(repoRoot: string): string | null {
  try {
    const headPath = join(repoRoot, '.git', 'HEAD')
    if (!existsSync(headPath)) {
      return null
    }
    const head = readFileSync(headPath, 'utf8').trim()
    const match = /^ref:\s+refs\/heads\/(.+)$/.exec(head)
    if (match && match[1]) {
      return match[1]
    }
    // Detached HEAD — show short SHA.
    return head.slice(0, 7) || null
  } catch {
    return null
  }
}

function shortenPath(path: string): string {
  const home = homedir()
  if (home && path.startsWith(home)) {
    return `~${path.slice(home.length)}`
  }
  return path
}

function formatDate(date: Date): string {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  return `${year}-${month}-${day}`
}
