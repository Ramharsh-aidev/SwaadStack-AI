"""CLI: Start the API server."""

import uvicorn
from swaadstack.config import api_config


def main():
    uvicorn.run(
        "swaadstack.api.app:app",
        host=api_config.host,
        port=api_config.port,
        reload=api_config.debug,
        workers=1 if api_config.debug else api_config.workers,
    )


if __name__ == "__main__":
    main()
