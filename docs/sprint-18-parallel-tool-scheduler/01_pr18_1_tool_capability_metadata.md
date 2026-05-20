# PR 18.1 - Tool Capability Metadata

## Goal

Add explicit tool capability metadata used by the future scheduler.

## Target Files

Modify:

- `aether/tools/base.py`
- built-in tool descriptors as needed

Add:

- `aether/runtime/tools/capabilities.py`

## Capability Fields

Recommended internal capability model:

- `read_only: bool`
- `mutates_files: bool`
- `interactive: bool`
- `requires_permission: bool`
- `path_scoped: bool`
- `parallel_safe: bool`
- `cheap: bool`

Do not expose all of these in public tool schemas unless needed. Internal
metadata is enough for the first version.

## Defaults

Conservative default:

- Unknown tools are not parallel safe.
- Shell is not parallel safe.
- File writes are not parallel safe.
- Read/list/search/fetch style tools may be marked safe.

## Tests

Add:

- `aether/tests/runtime/tools/test_capabilities.py`

Cover:

- Built-in read tools classified safe.
- Mutating tools classified unsafe.
- Unknown external tools classified unsafe.
- Cheap classification remains compatible with iteration refunds.

## Acceptance

- No behavior change yet.
- Tool dispatch still sequential.

## Detailed Implementation Notes

### Files to Add

Create:

- `aether/runtime/tools/capabilities.py`
- `aether/tests/runtime/tools/test_capabilities.py`

### Data Model

Recommended:

```python
@dataclass(frozen=True, slots=True)
class ToolCapabilities:
    read_only: bool = False
    mutates_files: bool = False
    interactive: bool = False
    requires_permission: bool = False
    path_scoped: bool = False
    parallel_safe: bool = False
    cheap: bool = False
```

Add helpers:

- `capabilities_for_tool_name(name: str) -> ToolCapabilities`
- `is_parallel_safe_tool(name: str) -> bool`
- `extract_tool_scope_path(name: str, arguments: Mapping[str, Any], cwd: Path) -> Path | None`

Do not require every `ToolDescriptor` to carry these fields in PR18.1. A central
classifier is easier to add without breaking descriptor schema.

### Initial Built-In Classification

Likely parallel safe:

- `read_file`
- `list_dir`
- `grep`
- `glob`
- local `web_search` only if HTTP client/interrupt behavior is safe enough
- `web_fetch` only if independent and interrupt-aware
- `task_output` as read-only task status

Likely unsafe/sequential:

- `shell`
- `write_file`
- `file_edit`
- `notebook_edit`
- `todo_write`
- `memory_write`
- `memory_forget`
- `enter_plan_mode`
- `exit_plan_mode`
- `ask_user_question`
- `task`
- `send_message`
- `task_stop`
- browser tools with shared browser state
- LSP tools if manager state is not proven thread-safe

Confirm exact tool names by reading built-in descriptors during implementation.

### Cheap Tool Compatibility

`EngineConfig.cheap_tool_names` already exists. Do not replace it. Either:

- keep cheap classification separate, or
- mirror it into `ToolCapabilities.cheap` for scheduler observability only

Cheap-tool budget refund behavior must not change in PR18.1.

### Path Scope Extraction

Support common fields:

- `path`
- `file_path`
- `target`

But only for tools that are explicitly `path_scoped`. Do not infer path scope
for unknown tools.

### Tests

Cover:

- read tools parallel safe
- write tools unsafe
- shell unsafe
- interactive tools unsafe
- unknown tool unsafe
- scope path normalized against cwd
- missing path returns `None`
- cheap classification mirrors config/default where intended

### Review Checklist

- No dispatch behavior changes.
- No new permission behavior.
- Descriptor wire schema unchanged unless backward-compatible metadata is added.
