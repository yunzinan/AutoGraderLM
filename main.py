"""Entry point - run with: python main.py -c config.example.yaml"""

import argparse
import uvicorn

from autograder.app import create_app
from autograder.config import get_config, set_config_file


def _parse_config_arg() -> None:
    """在 create_app 之前解析 -c/--config，使后续请求使用指定配置。"""
    parser = argparse.ArgumentParser(description="AutoGraderLM")
    parser.add_argument("-c", "--config", default=None, help="Config file path (e.g. config.example.yaml)")
    args, _ = parser.parse_known_args()
    if args.config:
        set_config_file(args.config)


_parse_config_arg()
app = create_app()

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=cfg.web_server.port,
        reload=True,
    )
