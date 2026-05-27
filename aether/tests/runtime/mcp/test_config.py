from __future__ import annotations

import json

from aether.runtime.mcp.config import (
    load_mcp_server_configs,
    sanitize_mcp_name_component,
)


def test_loads_explicit_stdio_server_with_env_interpolation() -> None:
    configs = load_mcp_server_configs(
        {
            "local fs": {
                "command": "${NODE_BIN}",
                "args": ["server.js", "--root", "${WORKSPACE_ROOT}"],
                "env": {"TOKEN": "${MCP_TOKEN}"},
                "timeout": 5,
                "connect_timeout": 2,
            }
        },
        environ={
            "NODE_BIN": "node",
            "WORKSPACE_ROOT": "/workspace/Aether",
            "MCP_TOKEN": "secret-token",
        },
    )

    assert len(configs) == 1
    config = configs[0]
    assert config.name == "local_fs"
    assert config.command == "node"
    assert config.args == ("server.js", "--root", "/workspace/Aether")
    assert config.env == {"TOKEN": "secret-token"}
    assert config.transport == "stdio"
    assert config.timeout == 5
    assert config.connect_timeout == 2


def test_loads_remote_server_from_aether_mcp_servers_env() -> None:
    raw = {
        "browser": {
            "url": "https://mcp.example.test/sse",
            "transport": "sse",
            "headers": {"Authorization": "Bearer ${MCP_KEY}"},
        }
    }

    configs = load_mcp_server_configs(
        environ={
            "AETHER_MCP_SERVERS": json.dumps(raw),
            "MCP_KEY": "remote-secret",
        }
    )

    assert len(configs) == 1
    assert configs[0].name == "browser"
    assert configs[0].url == "https://mcp.example.test/sse"
    assert configs[0].transport == "sse"
    assert configs[0].headers == {"Authorization": "Bearer remote-secret"}


def test_loads_servers_from_aether_home_file(tmp_path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "mcp_servers.json").write_text(
        json.dumps({"filesystem": {"command": "python", "args": ["server.py"]}}),
        encoding="utf-8",
    )

    configs = load_mcp_server_configs(environ={"AETHER_HOME": str(home)})

    assert [(config.name, config.command, config.args) for config in configs] == [
        ("filesystem", "python", ("server.py",))
    ]


def test_ignores_disabled_invalid_and_ambiguous_servers() -> None:
    configs = load_mcp_server_configs(
        {
            "disabled": {"command": "node", "enabled": False},
            "missing": {"args": ["server.js"]},
            "ambiguous": {"command": "node", "url": "https://example.test/mcp"},
            "bad-url": {"url": "ftp://example.test/mcp"},
            "ok": {"url": "https://example.test/mcp"},
        }
    )

    assert [config.name for config in configs] == ["ok"]
    assert configs[0].transport == "http"


def test_sanitizes_empty_or_punctuated_names() -> None:
    assert sanitize_mcp_name_component(" filesystem/server ") == "filesystem_server"
    assert sanitize_mcp_name_component("!!!") == "unnamed"
