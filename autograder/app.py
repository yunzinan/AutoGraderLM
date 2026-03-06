"""FastAPI application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from autograder.config import get_config, get_assignment_path


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = FastAPI(title="AutoGraderLM", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from autograder.routers.pipeline import router as pipeline_router
    from autograder.routers.questions import router as questions_router
    from autograder.routers.results import router as results_router
    from autograder.routers.review import router as review_router
    from autograder.routers.stats import router as stats_router

    app.include_router(questions_router)
    app.include_router(pipeline_router)
    app.include_router(review_router)
    app.include_router(results_router)
    app.include_router(stats_router)

    @app.get("/api/config")
    def api_config():
        cfg = get_config()
        return cfg.model_dump()

    base = get_assignment_path()
    for dir_name in ("questions", "answers", "results"):
        d = base / dir_name
        d.mkdir(parents=True, exist_ok=True)
        app.mount(f"/files/{dir_name}", StaticFiles(directory=str(d)), name=f"files_{dir_name}")
    # res 使用配置中的 pdf_folder_path（已解析到 assignment_path 下）
    res_dir = Path(get_config().assignment_configuration.pdf_folder_path)
    res_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/files/res", StaticFiles(directory=str(res_dir)), name="files_res")

    static_dir = Path("static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/")
    def index():
        return FileResponse("static/index.html")

    return app
