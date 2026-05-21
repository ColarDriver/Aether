from __future__ import annotations

import pytest

from aether.services.common import ServiceValidationError
from aether.services.logs import LogService


def test_log_service_lists_known_and_discovered_logs(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "gateway_crash.log").write_text("ERROR gateway crashed\n", encoding="utf-8")
    (log_dir / "custom.log").write_text("INFO custom ready\n", encoding="utf-8")

    files = LogService(log_dir=log_dir).files()

    keys = {item.key for item in files}
    assert {"gateway", "gateway_crash", "agent", "web", "tui", "custom"} <= keys
    assert next(item for item in files if item.key == "custom").exists is True


def test_log_service_reads_tail_and_filters(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "agent.log").write_text(
        "DEBUG tools noisy\n"
        "INFO agent ready\n"
        "WARNING tools slow\n"
        "ERROR agent failed\n",
        encoding="utf-8",
    )

    result = LogService(log_dir=log_dir).read(
        file="agent",
        lines=10,
        level="WARNING",
        component="agent",
    )

    assert result.exists is True
    assert result.file == "agent"
    assert result.lines == ["ERROR agent failed"]


def test_log_service_searches_and_rejects_path_traversal(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "web.log").write_text("INFO alpha\nINFO beta\n", encoding="utf-8")

    searched = LogService(log_dir=log_dir).read(file="web.log", search="beta")
    assert searched.lines == ["INFO beta"]

    with pytest.raises(ServiceValidationError):
        LogService(log_dir=log_dir).read(file="../secret")
