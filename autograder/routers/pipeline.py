"""Router: pipeline execution (segmentation & grading)."""

from __future__ import annotations

import asyncio
import glob
import logging
from pathlib import Path

from fastapi import APIRouter

from autograder.config import get_config
from autograder.models import PipelineStatus
from autograder.pipeline.grading import run_grading
from autograder.pipeline.report import export_scores, generate_all_reports
from autograder.pipeline.segmentation import run_segmentation
from autograder.routers.questions import load_all_questions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

_status = PipelineStatus()
_answer_map: dict[str, dict[str, list[str]]] = {}
_reports: list = []
_running_task: asyncio.Task | None = None


@router.get("/status")
def get_status() -> PipelineStatus:
    return _status


def _list_pdfs() -> list[str]:
    cfg = get_config()
    folder = cfg.assignment_configuration.pdf_folder_path
    paths: list[str] = []
    for ext in ("*.pdf", "*.PDF"):
        paths.extend(glob.glob(str(Path(folder) / ext)))
    return sorted(paths)


async def _update_status(msg: str, current: int, total: int):
    _status.message = msg
    _status.current = current
    _status.total = total


@router.post("/segment")
async def start_segmentation() -> dict:
    global _running_task, _answer_map
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    _status.stage = "segmentation"
    _status.errors = []
    questions = load_all_questions()
    pdfs = _list_pdfs()
    num_pdfs = len(pdfs)
    _status.total = num_pdfs
    _status.current = 0
    _status.message = f"即将切分 {num_pdfs} 份作业…" if num_pdfs else "未找到 PDF 文件"

    async def _run():
        global _answer_map
        try:
            cfg = get_config()
            _answer_map = await run_segmentation(cfg, questions, pdfs, on_progress=_update_status)
            _status.stage = "done"
            _status.message = f"切分完成，共处理 {len(_answer_map)} 份作业"
        except Exception as e:
            logger.exception("Segmentation pipeline failed")
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True, "num_pdfs": num_pdfs}


@router.post("/grade")
async def start_grading() -> dict:
    global _running_task
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    if not _answer_map:
        return {"error": "请先运行作业切分"}

    _status.stage = "grading"
    _status.errors = []
    questions = load_all_questions()

    async def _run():
        try:
            cfg = get_config()
            await run_grading(cfg, questions, _answer_map, on_progress=_update_status)
            _status.stage = "done"
            _status.message = "评分完成"
        except Exception as e:
            logger.exception("Grading pipeline failed")
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True}


@router.post("/report")
async def start_report_generation() -> dict:
    global _running_task, _reports
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    _status.stage = "report"
    _status.errors = []
    questions = load_all_questions()

    async def _run():
        global _reports
        try:
            cfg = get_config()
            from autograder.pipeline.grading import load_all_results

            results = load_all_results()
            if not results:
                _status.stage = "error"
                _status.errors.append("没有找到评分结果，请先运行评分")
                return

            export_scores(cfg, results)
            _reports = await generate_all_reports(cfg, questions, results, on_progress=_update_status)
            _status.stage = "done"
            _status.message = "报告生成完成"
        except Exception as e:
            logger.exception("Report pipeline failed")
            _status.stage = "error"
            _status.errors.append(str(e))

    _running_task = asyncio.create_task(_run())
    return {"ok": True}


@router.get("/answer_map")
def get_answer_map() -> dict:
    return _answer_map


@router.get("/reports")
def get_reports() -> list:
    return [r.model_dump() for r in _reports]


@router.post("/cancel")
async def cancel_pipeline() -> dict:
    """取消当前正在运行的切分/评分/报告任务。"""
    global _running_task
    if _running_task is None:
        return {"ok": True, "message": "没有正在运行的任务"}
    if _running_task.done():
        _running_task = None
        _status.stage = "idle"
        _status.message = ""
        return {"ok": True, "message": "任务已结束"}
    _running_task.cancel()
    try:
        await _running_task
    except asyncio.CancelledError:
        pass
    finally:
        _running_task = None
        _status.stage = "idle"
        _status.message = "已取消"
        _status.current = 0
        _status.total = 0
        _status.errors = []
    return {"ok": True, "message": "已取消"}


@router.post("/reset")
def reset_status() -> dict:
    _status.stage = "idle"
    _status.message = ""
    _status.current = 0
    _status.total = 0
    _status.errors = []
    return {"ok": True}
