"""Public-safe runtime log reader service."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path

from aether.services.common import ServiceValidationError
from aether.services.logs.contracts import LogFileSummary, LogReadResult

_DEFAULT_LOG_FILES: dict[str, str] = {
    "gateway": "gateway_crash.log",
    "gateway_crash": "gateway_crash.log",
    "agent": "agent.log",
    "web": "web.log",
    "tui": "tui.log",
}
_LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
    "FATAL": 50,
}


class LogService:
    """Read log files from the Aether logs directory with path-safe filters."""

    def __init__(self, *, log_dir: Path | None = None) -> None:
        self._log_dir = log_dir.expanduser() if log_dir is not None else _aether_home() / "logs"

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def files(self) -> list[LogFileSummary]:
        summaries: dict[str, LogFileSummary] = {}
        for key, name in _DEFAULT_LOG_FILES.items():
            path = self._log_dir / name
            summaries[key] = _summary(key, path)
        if self._log_dir.exists():
            for path in sorted(self._log_dir.glob("*.log")):
                key = path.stem
                summaries.setdefault(key, _summary(key, path))
        return sorted(summaries.values(), key=lambda item: (not item.exists, item.key))

    def read(
        self,
        *,
        file: str = "gateway",
        lines: int = 100,
        level: str | None = None,
        component: str | None = None,
        search: str | None = None,
    ) -> LogReadResult:
        limit = _normalize_limit(lines)
        path = self._resolve_file(file)
        available = self.files()
        if not path.exists():
            return LogReadResult(
                file=_file_key(file),
                path=str(path),
                exists=False,
                lines=[],
                available_files=available,
            )

        candidates = _read_tail(path, max(limit, 5000) if _has_filters(level, component, search) else limit)
        filtered = [
            line for line in candidates
            if _matches_level(line, level)
            and _matches_component(line, component)
            and _matches_search(line, search)
        ][-limit:]
        return LogReadResult(
            file=_file_key(file),
            path=str(path),
            exists=True,
            lines=filtered,
            available_files=available,
        )

    def _resolve_file(self, value: str) -> Path:
        key = _file_key(value)
        name = _DEFAULT_LOG_FILES.get(key, key if key.endswith(".log") else f"{key}.log")
        if Path(name).name != name or not name.endswith(".log"):
            raise ServiceValidationError(
                "invalid log file name",
                details={"file": value},
            )
        return self._log_dir / name


def _aether_home() -> Path:
    return Path(os.getenv("AETHER_HOME", Path.home() / ".aether")).expanduser()


def _file_key(value: str) -> str:
    key = (value or "gateway").strip()
    if not key:
        return "gateway"
    return key[:-4] if key.endswith(".log") else key


def _normalize_limit(value: int) -> int:
    if not isinstance(value, int):
        return 100
    return min(max(value, 1), 1000)


def _summary(key: str, path: Path) -> LogFileSummary:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return LogFileSummary(
        key=key,
        name=path.name,
        path=str(path),
        exists=exists,
        size_bytes=size,
    )


def _read_tail(path: Path, limit: int) -> list[str]:
    out: deque[str] = deque(maxlen=limit)
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            out.append(raw.rstrip("\n"))
    return list(out)


def _has_filters(level: str | None, component: str | None, search: str | None) -> bool:
    return bool(_normalize_level(level) or _normalize_component(component) or (search or "").strip())


def _normalize_level(level: str | None) -> str | None:
    raw = (level or "").strip().upper()
    if not raw or raw == "ALL":
        return None
    if raw == "WARN":
        return "WARNING"
    if raw not in _LEVEL_ORDER:
        raise ServiceValidationError(
            "invalid log level",
            details={"level": level},
        )
    return raw


def _matches_level(line: str, level: str | None) -> bool:
    normalized = _normalize_level(level)
    if normalized is None:
        return True
    line_level = _line_level(line)
    if line_level is None:
        return False
    return _LEVEL_ORDER[line_level] >= _LEVEL_ORDER[normalized]


def _line_level(line: str) -> str | None:
    upper = line.upper()
    for level in ("CRITICAL", "FATAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG"):
        if level in upper:
            return "WARNING" if level == "WARN" else level
    return None


def _normalize_component(component: str | None) -> str | None:
    raw = (component or "").strip().lower()
    return None if not raw or raw == "all" else raw


def _matches_component(line: str, component: str | None) -> bool:
    normalized = _normalize_component(component)
    return True if normalized is None else normalized in line.lower()


def _matches_search(line: str, search: str | None) -> bool:
    needle = (search or "").strip().lower()
    return True if not needle else needle in line.lower()


__all__ = ["LogService"]
