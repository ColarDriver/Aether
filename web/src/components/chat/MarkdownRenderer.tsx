type Props = {
  text: string
}

export function MarkdownRenderer({ text }: Props) {
  const blocks = text.split(/\n{2,}/)
  return (
    <div className="markdown-renderer">
      {blocks.map((block, index) => renderBlock(block, index))}
    </div>
  )
}

function renderBlock(block: string, index: number) {
  if (block.startsWith('### ')) return <h3 key={index}>{block.slice(4)}</h3>
  if (block.startsWith('## ')) return <h2 key={index}>{block.slice(3)}</h2>
  if (block.startsWith('# ')) return <h1 key={index}>{block.slice(2)}</h1>
  const fence = String.fromCharCode(96, 96, 96)
  if (block.startsWith(fence)) {
    let code = block.slice(fence.length)
    const newline = code.indexOf('\n')
    code = newline >= 0 ? code.slice(newline + 1) : code
    if (code.endsWith(fence)) code = code.slice(0, -fence.length)
    return <pre className="markdown-code" key={index}>{code.trimEnd()}</pre>
  }
  if (block.includes('\n- ') || block.startsWith('- ')) {
    return (
      <ul key={index}>
        {block.split('\n').filter((line) => line.startsWith('- ')).map((line) => (
          <li key={line}>{line.slice(2)}</li>
        ))}
      </ul>
    )
  }
  return <p key={index}>{block}</p>
}
