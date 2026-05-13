# PR 9 · Cutover — remove Python TUI

## 摘要

单击切换：`aether` 入口从启动 Python TUI 改成拉起 TS Ink TUI；把 `aether/cli/` 里的纯 UI 文件全部删除；后端逻辑文件（sessions / prefs / providers / commands / main）迁移到非 UI 路径；从 runtime 依赖里剔除 `prompt_toolkit`。本 PR 不保留 `--legacy-tui` 后路（用户在规划阶段已确认 single-shot）。

## Scope

In scope:

- 重写 `aether/cli/main.py` 为一个薄启动器：解析全局参数 → `subprocess.Popen` 拉 TS 入口
- 把 backend 模块从 `aether/cli/` 迁移到合理位置（见下方"迁移目标"）
- 删除 `aether/cli/` 下所有纯 UI 文件
- 修改 `pyproject.toml`：
  - `console_scripts` 入口指向新的 launcher
  - 从运行时依赖中删除 `prompt_toolkit`
  - `rich` 视实际使用决定（仅在日志辅助等非 UI 路径使用则保留；否则删除）
- 更新 README、README.zh-CN（如有）
- 删除 / 调整 `aether/cli/` 相关的测试目录
- CI / lint 配置：`aether-tui/` 进入 lint + type-check 管线

Out of scope:

- WebSocket transport / web UI（留给未来 sprint）
- 任何"两套 TUI 并存"的兼容代码
- 重命名 `aether-tui/` 目录（如果在 PR 6 选定的命名要改，那是另一个 PR）

## 迁移目标

| 原位置 | 新位置 | 备注 |
|---|---|---|
| `aether/cli/sessions.py` | `aether/sessions/store.py` | 新建 `aether/sessions/` 包 |
| `aether/cli/prefs.py` | `aether/sessions/prefs.py` | 与 sessions 共存（同一存储域） |
| `aether/cli/providers.py` | `aether/providers/factory.py` | 已有 `aether/models/provider/*`；新建 `aether/providers/` 持高层工厂 |
| `aether/cli/commands.py` | `aether/gateway/handlers/_commands_catalog.py` | 只保留 catalog（name + description）；删除 `_cmd_*` handler |
| `aether/cli/main.py` | `aether/cli/launcher.py`（同包，但只剩 launcher） | 启动器是一个非 UI 入口 |

要删除的纯 UI 文件：

```
aether/cli/app.py
aether/cli/repl.py
aether/cli/ui.py
aether/cli/ui_middleware.py
aether/cli/activity.py
aether/cli/approval_prompter.py
aether/cli/tool_permission_prompter.py
aether/cli/banner.py
aether/cli/input_box.py
aether/cli/picker.py
aether/cli/theme.py
aether/cli/tool_groups.py
```

每一个都要先 grep 确认没有外部 import 后删。

注意：`aether/cli/approval_prompter.py` 的 `Prompter` 协议要先**搬到** `aether/runtime/prompter.py`（PR 5 已经依赖），再删除原文件。`aether/cli/tool_permission_prompter.py` 在 sprint-7 已经把契约挪进 runtime，本 PR 直接删除胶水代码。

## Launcher 设计

`aether/cli/launcher.py`：

```python
def main() -> int:
    args, passthrough = parse_args()       # provider / model / resume / log-level 等
    cmd = resolve_tui_entry(args)          # 找 aether-tui/dist/entry.js（生产）或 npm exec tsx src/entry.tsx（dev）
    env = build_env(args, passthrough)     # AETHER_PYTHON, AETHER_PYTHON_SRC_ROOT, AETHER_CWD 等
    try:
        proc = subprocess.Popen(cmd, env=env, stdin=None, stdout=None, stderr=None)
        return proc.wait()
    except FileNotFoundError:
        print("aether-tui not installed. Run `(cd aether-tui && npm install)`.", file=sys.stderr)
        return 127
```

- 启动器**不**自己开 gateway。gateway 由 TS 端的 `GatewayClient` 拉。
- 启动器**只**透传环境变量与参数；它不解析任何业务逻辑。
- 当存在 `dist/entry.js` 时优先用 Node 直跑；否则 fallback 用 `npx tsx`，并打印一条 stderr 提示。

`console_scripts`：

```toml
[project.scripts]
aether = "aether.cli.launcher:main"
```

启动参数透传策略：argparse 仍然在 Python 端做（兼容现有用户的 muscle memory），转成 env vars 给 TS 端读取：

| Python flag | env var | TS 侧消费 |
|---|---|---|
| `--provider X` | `AETHER_INITIAL_PROVIDER=X` | sessionStore 初始值 |
| `--model X` | `AETHER_INITIAL_MODEL=X` | sessionStore 初始值 |
| `--resume <id>` | `AETHER_RESUME=<id>` | 启动后自动调 `session.resume` |
| `--system-file PATH` | `AETHER_SYSTEM=$(cat PATH)` | 注入下一次 `session.create` |
| `--log-level X` | `AETHER_LOG_LEVEL=X` | gateway env，影响 Python logging |
| `--no-banner` | `AETHER_NO_BANNER=1` | UI 略过 banner |

## 设计要点

**为什么 launcher 不开 gateway。** 让 TS 端管理 gateway 生命周期更干净：UI 退出 → 子进程退出，没有"我开的我管"的中间层。如果让 launcher 开 gateway，TS 端就要做"已存在 gateway 是连接还是替换"的复杂逻辑。

**为什么单击切换而不是 flag。** 用户在规划阶段确认了 single-shot：减少代码路径、避免长期维护两个 TUI、强制所有测试 / 文档统一到 TS 形态。代价是回滚 = revert PR 9，但本 sprint 的 PR 1~8 都已经独立可 review，回滚不会丢工程。

**为什么把 sessions / prefs 挪到独立包。** PR 3 之后这些模块的消费者是 gateway handlers，不是 CLI。继续放在 `aether/cli/` 名不副实，让未来贡献者以为它们是 UI 层。新结构让 `aether/cli/` 在 cutover 后**仅**包含 launcher，目录意义清晰。

**`commands.py` 大幅缩水。** 现有 565 行里大部分是 `_cmd_*` 实现（ReplState 操作）。这些在 TS 端重写（PR 7），Python 端只留 14 条命令的 name + description。如果未来 server 需要"集中命令注册"，可以在 `aether/gateway/handlers/_commands_catalog.py` 里继续加，不影响 TS。

**ROADMAP / Dependencies cleanup。**

- `pyproject.toml` 删除 `prompt_toolkit`；
- 检查 `rich`：如果没有其它 runtime 使用（grep `^import rich\|^from rich`），也删除；
- `aether/tests/cli/` 大量测试要么删要么改：纯 UI 测试（`test_app.py` 等）直接删；接 backend 的测试（`test_session_resume.py` 等）迁到新位置；
- `aether-tui` 加入 CI：node 20+，跑 `npm test` + `npm run type-check` + `npm run lint`。

**Atomic / monolithic PR.** PR 9 比前 8 个都大。强烈建议在 PR description 列出文件移动清单（git rename detection 把噪音降下来），用至少 2 个 reviewer 分工：一人看 Python 删除 / 迁移正确性，一人看 launcher + CI + 文档。

**Rollback plan.** revert 本 PR → 旧 Python TUI 全部恢复，PR 1~8 的 `aether/gateway/` 与 `aether-tui/` 留下不影响功能。`aether` 入口回到 `aether.cli.main:main`。这要求 PR 1~8 在 main 上没有引入 import 链跨进新代码（gateway 包是孤立的）。

## Files touched

迁移（git mv 或 rename）：

- `aether/cli/sessions.py` → `aether/sessions/store.py`
- `aether/cli/prefs.py` → `aether/sessions/prefs.py`
- `aether/cli/providers.py` → `aether/providers/factory.py`
- `aether/cli/commands.py` → `aether/gateway/handlers/_commands_catalog.py`（瘦身 + 只保留 catalog）
- `aether/cli/approval_prompter.py` 中的 `Prompter` 协议 → `aether/runtime/prompter.py`，文件本身删除

删除：

- `aether/cli/app.py` 等 12 个纯 UI 文件
- 对应的 `aether/tests/cli/test_app.py` 等 UI 测试

新建：

- `aether/cli/launcher.py`
- `aether/sessions/__init__.py`
- `aether/providers/__init__.py`

修改：

- `pyproject.toml`（console_scripts、dependencies、optional-dependencies）
- `README.md`、`README.zh-CN.md`
- `aether/gateway/handlers/*.py`（更新 import 路径）
- CI / lint 配置（仓库根）

## Dependencies

PR 8（TS 端功能闭合）。

## Acceptance criteria

- `pip install -e .` 后 `aether` 命令启动 TS Ink TUI（不再有 prompt_toolkit Application）。
- `prompt_toolkit` 不在 `pyproject.toml` runtime deps 中；`pip show prompt_toolkit` 在新装环境下不存在。
- `python -c "import aether.cli.app"` 报 `ModuleNotFoundError`（已删除）。
- `grep -rn "from aether.cli.app\|aether.cli.repl\|aether.cli.ui\|prompt_toolkit" aether/` 仅命中 launcher 内允许的引用（无）。
- 完整 chat turn（含工具 + approval + permission + resume）通过 manual checklist。
- 全部 backend 测试通过：`pytest aether/tests/agents aether/tests/runtime aether/tests/engine aether/tests/gateway aether/tests/sessions aether/tests/providers aether/tests/tools`。
- TS 端 lint + type-check + vitest 全绿。
- 当 `aether-tui/node_modules` 缺失时，启动 launcher 给出可读引导（错误码 127）。
- `git log --stat` 显示 rename 而非 delete + add（保留 git history）。

## Manual verification

```bash
# 全新环境验证
python -m venv .venv-clean && source .venv-clean/bin/activate
pip install -e .
which aether

# 启动主路径
aether                                  # TS TUI 起来

# 透传参数
aether --provider anthropic --model claude-sonnet-4-6
aether --resume ses_abc                 # 自动 resume

# 反向校验：旧入口已不可用
python -c "from aether.cli import app" 2>&1 | grep "No module"
python -c "import prompt_toolkit" 2>&1 | grep "No module"

# 测试套件
pytest aether/tests -x --tb=short
(cd aether-tui && npm test && npm run type-check && npm run lint)

# CI 干跑
git diff --stat origin/master..HEAD     # 检查 rename 而不是删 + 新
```
