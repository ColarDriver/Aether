# 05 Nudge 机制

## 配置项

- `memory_nudge_interval`
- `skill_nudge_interval`

## 行为

- memory 计数器按 turn 累积
- skill 计数器按 tool-call 轮次累积
- 触发后仅写 metadata：
  - `should_review_memory`
  - `should_review_skills`

## 原则

Nudge 是建议信号，不直接改写主输出。
