import type { TaskNotificationBlock as TaskNotification } from '../../../chat-rendering'
import { InlineTaskSummary } from './InlineTaskSummary'

type Props = {
  block: TaskNotification
  onOpenTask?: (taskId: string) => void
}

export function TaskNotificationBlock({ block, onOpenTask }: Props) {
  return (
    <article className="chat-block task-notification">
      <InlineTaskSummary
        title={'Subagent ' + block.status}
        status={block.status}
        taskId={block.taskId}
        subagentType={block.subagentType}
        durationSeconds={block.durationSeconds}
        summary={block.summary}
        error={block.error}
        outputFile={block.outputFile}
        onOpenTask={onOpenTask}
        ariaLabel="Subagent notification"
      />
    </article>
  )
}
