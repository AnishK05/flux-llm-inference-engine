"""Run the Flux server: ``python -m flux``."""

from __future__ import annotations

import uvicorn

from flux.config import ServerConfig


def main() -> None:
    server_config = ServerConfig.from_env()
    # Import string keeps uvicorn's reloader/worker model happy and loads the
    # model inside the app lifespan.
    uvicorn.run(
        "flux.server:app",
        host=server_config.host,
        port=server_config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
