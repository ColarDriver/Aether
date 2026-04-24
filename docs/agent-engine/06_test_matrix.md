# 06 测试矩阵

## 覆盖项

- system_message 首位注入
- session store 续跑复用 system prompt
- stream callback 回调触发与计数
- todo hydration 回填
- memory/skill nudge 元数据触发
- hooks 生命周期事件触发
- 既有工具回路与中断/错误路径回归

## 已执行

- `python -m unittest aether.tests.test_engine -v`
- `python -m unittest aether.tests.test_subagents aether.tests.test_state_machine -v`
