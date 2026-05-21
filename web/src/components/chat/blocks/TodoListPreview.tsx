import { CheckCircle2, Circle, CircleDashed, Clock3 } from 'lucide-react'

type TodoStatus = 'pending' | 'in_progress' | 'completed' | 'cancelled'

type TodoItem = {
  id: string
  content: string
  status: TodoStatus
}

type Props = {
  todos: TodoItem[]
}

export function TodoListPreview({ todos }: Props) {
  if (todos.length === 0) return null
  const counts = todoCounts(todos)
  return (
    <section className="todo-preview" aria-label="Todo checklist">
      <header>
        <strong>Todo checklist</strong>
        <span>
          {counts.completed}/{todos.length} complete
          {counts.in_progress > 0 ? ' · ' + counts.in_progress + ' active' : ''}
        </span>
      </header>
      <ol>
        {todos.map((todo) => {
          const Icon = iconForStatus(todo.status)
          return (
            <li className={'todo-preview-item todo-preview-' + todo.status} key={todo.id}>
              <Icon aria-hidden="true" size={15} />
              <span>{todo.content}</span>
              <em>{statusLabel(todo.status)}</em>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

export function todosFromToolArguments(args: Record<string, unknown>): TodoItem[] {
  const rawTodos = args.todos
  if (!Array.isArray(rawTodos)) return []
  return rawTodos.flatMap((item, index) => {
    if (!item || typeof item !== 'object' || Array.isArray(item)) return []
    const record = item as Record<string, unknown>
    const content = typeof record.content === 'string' ? record.content.trim() : ''
    const status = normalizeStatus(record.status)
    if (!content || !status) return []
    const id = typeof record.id === 'string' && record.id.trim() ? record.id.trim() : String(index + 1)
    return [{ id, content, status }]
  })
}

function todoCounts(todos: TodoItem[]): Record<TodoStatus, number> {
  return todos.reduce<Record<TodoStatus, number>>((counts, todo) => {
    counts[todo.status] += 1
    return counts
  }, { pending: 0, in_progress: 0, completed: 0, cancelled: 0 })
}

function normalizeStatus(value: unknown): TodoStatus | null {
  return value === 'pending' || value === 'in_progress' || value === 'completed' || value === 'cancelled'
    ? value
    : null
}

function iconForStatus(status: TodoStatus) {
  if (status === 'completed') return CheckCircle2
  if (status === 'in_progress') return Clock3
  if (status === 'cancelled') return CircleDashed
  return Circle
}

function statusLabel(status: TodoStatus): string {
  return status.replace('_', ' ')
}
