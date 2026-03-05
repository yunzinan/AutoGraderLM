"""Entry point – run with: python main.py"""

import uvicorn

from autograder.app import create_app
from autograder.config import get_config

app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=cfg.web_server.port,
        reload=True,
    )
