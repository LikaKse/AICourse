"""Конфигурации трёх MCP-серверов."""

from __future__ import annotations

from manager import ServerSpec
import config


def default_specs(*, catalog_command: str | None = None) -> list[ServerSpec]:
    return [
        ServerSpec(
            alias="catalog",
            command=catalog_command or str(config.PYTHON_BIN),
            args=["-u", str(config.CATALOG_SERVER)],
        ),
        ServerSpec(
            alias="notes",
            command=str(config.PYTHON_BIN),
            args=["-u", str(config.NOTES_SERVER)],
        ),
        ServerSpec(
            alias="filesystem",
            command=str(config.NODE_BIN),
            args=[
                str(config.FS_SERVER_JS),
                "--read-only",
                "--confine-roots",
                str(config.WORKSPACE),
            ],
            verify_allowed_directory=config.WORKSPACE,
        ),
    ]
