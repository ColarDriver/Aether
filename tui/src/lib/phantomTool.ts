/**
 * Phantom-tool heuristics — port of the most user-visible bits of
 * `aether/cli/ui.py:strip_tool_blocks` / `_looks_like_intended_tool_use` /
 * `strip_all_command_fences`.
 *
 * The Aether engine has its own server-side phantom-tool synthesis; this
 * module only cleans what the model accidentally emits as prose so the
 * transcript does not flash half-rendered XML/JSON to the user.
 *
 * All functions are pure string transforms — no UI dependencies.
 */

// Mirrors Python `_TOOL_TAGS` in `aether/cli/ui.py:72`. Every tag a model has
// been observed to emit inline as narration of tool use — including
// thinking / reasoning blocks which are routed to the reasoning channel
// separately.
const TOOL_TAG_NAMES = [
  'tool_call',
  'tool_calls',
  'tool_result',
  'tool_results',
  'tool_use',
  'tool_uses',
  'tool_response',
  'tool_responses',
  'function_call',
  'function_calls',
  'function_result',
  'function_results',
  'function_response',
  'function_responses',
  'invoke',
  'invokes',
  'reasoning_effort',
  'thinking'
] as const

const TAG_GROUP = TOOL_TAG_NAMES.join('|')
// Closed block:  <ns:tool_call …>…</ns:tool_call>  (DOTALL, any namespace)
// JS lacks a named back-reference shared across alternations cheaply, so we
// build one regex per tag and concatenate the matches. Slightly slower but
// it's an inner-loop on streamed text — still O(n).
const TOOL_BLOCK_RES = TOOL_TAG_NAMES.map(
  (tag) =>
    new RegExp(
      `<\\s*(?:[\\w-]+:)?${tag}\\b[^>]*>[\\s\\S]*?<\\s*/\\s*(?:[\\w-]+:)?${tag}\\s*>`,
      'gi'
    )
)
// Open tag with no closer yet — strip everything from the tag onward.
const TOOL_OPEN_RE = new RegExp(
  `<\\s*(?:[\\w-]+:)?(?:${TAG_GROUP})\\b[^>]*>`,
  'i'
)

// `[tool: name]` bracket form (Python `_BRACKET_TOOL_LINE_RE`) plus the
// alternate `[tool_call(name=foo)]` shape some TS callers emit.
const BRACKET_TOOL_LINE_RES: ReadonlyArray<RegExp> = [
  /^[ \t]*\[tool:\s*[^\]\n]+\][ \t]*(?:\n|$)/gim,
  /^\s*\[tool_call\([^)]*\)\]\s*$/gim
]
const BRACKET_TOOL_OPEN_RE = /\[tool:\s*[^\]\n]*$/i

const FUNCTION_EQ_TAG_RE = /<function=[^>\s/]+>[\s\S]*?(?=<function=|<\/function|$)/gi
const FUNCTION_EQ_PARTIAL_RE = /<function=?[^>]*$/i
const FUNCTION_EQ_OPEN_RE = /<function=[^>\s/]+>/i

// Partial inline open tag at end-of-buffer for any known tag name. Python
// keeps a sorted-by-length list to avoid matching `<too` when `<tool_use`
// would also fit. We do the same so streaming half-tokens never leak.
const PARTIAL_INLINE_TAG_NAMES = [...TOOL_TAG_NAMES, 'parameter'].sort(
  (a, b) => b.length - a.length
)

const FENCE_HINT_RE = /```(?:bash|sh|shell|console|zsh|fish)\b/i
const SHELL_PROMPT_RE = /^\s*\$\s+\S/m
const IMPERATIVE_HINT_RE =
  /\b(let me (?:run|check|grep|read|search|look|use)|i(?:\s*'\s*ll|\s+will)\s+(?:run|use|check|read|search|grep))\b/i

/**
 * Conservative scrub of inline tool-call markup. Complete blocks are deleted;
 * a partial open tag (no matching close yet) hides everything from the open
 * tag onward so the user never sees half-rendered JSON before the closer
 * arrives in the stream.
 */
export function stripToolBlocks(text: string): string {
  if (!text) {
    return ''
  }
  let cleaned = text
  for (const re of TOOL_BLOCK_RES) {
    cleaned = cleaned.replace(re, '')
  }

  const openMatch = cleaned.match(TOOL_OPEN_RE)
  if (openMatch && openMatch.index !== undefined) {
    cleaned = cleaned.slice(0, openMatch.index)
  }

  cleaned = stripBracketToolLines(cleaned)
  cleaned = cleaned.replace(FUNCTION_EQ_TAG_RE, '')
  cleaned = cleaned.replace(FUNCTION_EQ_PARTIAL_RE, '')
  cleaned = stripPartialInlineXml(cleaned)
  cleaned = cleaned.replace(/\n{3,}/g, '\n\n')
  return cleaned
}

function stripBracketToolLines(text: string): string {
  let cleaned = text
  for (const re of BRACKET_TOOL_LINE_RES) {
    cleaned = cleaned.replace(re, '')
  }
  const openMatch = cleaned.match(BRACKET_TOOL_OPEN_RE)
  if (openMatch && openMatch.index !== undefined) {
    cleaned = cleaned.slice(0, openMatch.index)
  }
  return cleaned
}

/**
 * Mirror of Python `_strip_partial_inline_xml_tag`: when the text ends with a
 * `<` that has not closed yet, decide whether the partial token looks like
 * one of the known tool tag names. If so, drop it (and the `<`) so the
 * streaming buffer doesn't briefly show `<too` between deltas.
 */
function stripPartialInlineXml(text: string): string {
  const ltIndex = text.lastIndexOf('<')
  if (ltIndex < 0) {
    return text
  }
  const tail = text.slice(ltIndex + 1)
  if (tail.includes('>')) {
    return text
  }
  // Normalise leading namespace prefix (`<ns:tag…`) and a possible `/` for
  // close-tag prefixes (`</tag…`).
  let normalised = tail.trim().toLowerCase().replace(/^\/+/, '')
  const colon = normalised.lastIndexOf(':')
  if (colon >= 0) {
    normalised = normalised.slice(colon + 1)
  }
  // Strip everything after the first whitespace/attribute boundary — only
  // the tag identifier matters here.
  const ident = normalised.split(/[\s>]/)[0] ?? ''
  if (!ident) {
    return text
  }
  for (const name of PARTIAL_INLINE_TAG_NAMES) {
    if (name.startsWith(ident)) {
      return text.slice(0, ltIndex)
    }
  }
  return text
}

/**
 * True when *text* contains positive evidence of attempted tool use —
 * fenced shell block, `$` prompt, imperative phrasing, or a
 * `<function=…>` opener. Used to gate the idle-turn warning so a polite
 * greeting doesn't get flagged.
 */
export function looksLikeIntendedToolUse(text: string): boolean {
  if (!text) {
    return false
  }
  if (FENCE_HINT_RE.test(text)) {
    return true
  }
  if (SHELL_PROMPT_RE.test(text)) {
    return true
  }
  if (IMPERATIVE_HINT_RE.test(text)) {
    return true
  }
  if (FUNCTION_EQ_OPEN_RE.test(text)) {
    return true
  }
  return false
}

const COMMAND_FENCE_RE =
  /```(?:bash|sh|shell|console|zsh|fish)\s*\n([\s\S]*?)```/g

/**
 * Aggressive companion to `stripToolBlocks` — drops every shell-style
 * code-fence block. Caller must gate this behind
 * `noToolsDispatched && looksLikeIntendedToolUse(text)` to avoid stripping
 * legit example fences in long-form responses.
 */
export function stripAllCommandFences(text: string): string {
  if (!text || !text.includes('```')) {
    return text
  }
  let cleaned = text.replace(COMMAND_FENCE_RE, '')
  // Tidy up leading-in punctuation ("Let me run:") that's now sitting on
  // its own line. Conservative pattern — only short trailing colons.
  cleaned = cleaned.replace(/^[^\n]{1,40}[:：]\s*\n+/gm, (segment) =>
    segment.endsWith('\n\n') ? '\n' : '\n'
  )
  return cleaned
}

/**
 * Convenience predicate used by ChatMessage to render a soft hint when the
 * assistant turn ended without a tool call but the prose smells like the
 * model intended to use one.
 */
export function shouldShowPhantomHint(input: {
  text: string
  toolDispatched: boolean
}): boolean {
  if (input.toolDispatched) {
    return false
  }
  return looksLikeIntendedToolUse(input.text)
}
