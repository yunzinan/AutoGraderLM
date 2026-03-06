"""Router: pipeline execution (segmentation & grading)."""

from __future__ import annotations

import asyncio
import json
import glob
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import APIRouter, Body

from autograder.config import get_config, get_answers_dir, get_results_dir, resolve_assignment_path
from autograder.models import PipelineStatus, QuestionReport
from autograder.pipeline.grading import load_all_results, run_grading
from autograder.pipeline.report import generate_all_reports, run_report_sync
from autograder.pipeline.segmentation import (
    apply_manual_segments,
    load_segments_for_stem,
    run_segmentation,
)
from autograder.routers.questions import load_all_questions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])

def _reports_json_path() -> Path:
    return get_results_dir() / "question_reports.json"


def _build_answer_map_from_disk() -> dict[str, dict[str, list[str]]]:
    """从 answers 目录恢复 answer_map（服务重启后或未先点「开始切分」时可用）。"""
    result: dict[str, dict[str, list[str]]] = {}
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return result
    for stem_dir in sorted(answers_dir.iterdir()):
        if not stem_dir.is_dir() or stem_dir.name.startswith("_"):
            continue
        stem = stem_dir.name
        by_qid: dict[str, list[str]] = {}
        for f in sorted(stem_dir.glob("*.png")):
            if not f.is_file():
                continue
            # 文件名格式 Q1-0.png, Q2-1.png -> qid 为 Q1, Q2
            parts = f.stem.split("-", 1)
            qid = parts[0] if parts else f.stem
            path_str = str(resolve_assignment_path(f"answers/{stem}/{f.name}"))
            by_qid.setdefault(qid, []).append(path_str)
        if by_qid:
            result[stem] = by_qid
    return result


_status = PipelineStatus()
_status_lock = threading.Lock()
_answer_map: dict[str, dict[str, list[str]]] = {}
_reports: list[QuestionReport] = []
_running_task: asyncio.Task | None = None
_report_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report")


def _load_reports_from_disk() -> list[QuestionReport]:
    """Load question reports from disk so they persist across restarts."""
    path = _reports_json_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        out: list[QuestionReport] = []
        for d in raw:
            dist = d.get("score_distribution") or {}
            d = dict(d)
            d["score_distribution"] = {int(k): v for k, v in dist.items()}
            out.append(QuestionReport.model_validate(d))
        return out
    except Exception as e:
        logger.warning("Failed to load persisted reports from %s: %s", path, e)
        return []


def _save_reports_to_disk(reports: list[QuestionReport]) -> None:
    """Persist question reports to disk."""
    try:
        results_dir = get_results_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        _reports_json_path().write_text(
            json.dumps([r.model_dump() for r in reports], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Failed to save reports to %s: %s", _reports_json_path(), e)


def _update_status_sync(msg: str, current: int, total: int) -> None:
    """线程安全地更新 _status（供报告线程调用）。"""
    with _status_lock:
        _status.message = msg
        _status.current = current
        _status.total = total


@router.get("/status")
def get_status() -> PipelineStatus:
    """返回当前 pipeline 状态（线程安全读）。"""
    with _status_lock:
        return PipelineStatus(
            stage=_status.stage,
            message=_status.message,
            current=_status.current,
            total=_status.total,
            errors=list(_status.errors),
        )


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
    global _running_task, _answer_map
    if _status.stage not in ("idle", "done", "error"):
        return {"error": f"Pipeline already running: {_status.stage}"}

    if not _answer_map:
        _answer_map = _build_answer_map_from_disk()
    if not _answer_map:
        return {"error": "请先运行作业切分，或确认 answers 目录下已有切分结果"}

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
    """在后台线程中生成报告，不阻塞事件循环，评阅/下载等接口可照常响应。"""
    global _running_task, _reports
    with _status_lock:
        if _status.stage not in ("idle", "done", "error"):
            return {"error": f"Pipeline already running: {_status.stage}"}
        _status.stage = "report"
        _status.message = "正在生成报告…"
        _status.current = 0
        _status.total = 0
        _status.errors = []

    questions = load_all_questions()
    cfg = get_config()
    results = load_all_results()
    if not results:
        with _status_lock:
            _status.stage = "error"
            _status.errors.append("没有找到评分结果，请先运行评分")
        return {"error": "没有找到评分结果，请先运行评分"}

    def _run_in_thread() -> tuple[list[QuestionReport] | None, str | None]:
        """在线程中执行，返回 (reports, error_message)。"""
        try:
            reports = run_report_sync(
                cfg, questions, results,
                on_progress=_update_status_sync,
            )
            return (reports, None)
        except Exception as e:
            logger.exception("Report pipeline failed")
            return (None, str(e))

    loop = asyncio.get_event_loop()

    async def _wait_and_finish():
        global _reports
        future = loop.run_in_executor(_report_executor, _run_in_thread)
        reports, err = await future
        with _status_lock:
            if err:
                _status.stage = "error"
                _status.errors.append(err)
            else:
                _reports = reports or []
                _save_reports_to_disk(_reports)
                _status.stage = "done"
                _status.message = "报告生成完成"

    _running_task = asyncio.create_task(_wait_and_finish())
    return {"ok": True}


@router.get("/answer_map")
def get_answer_map() -> dict:
    return _answer_map


# ---------- 人工重新切分（segment 编辑） ----------


def _list_segment_assignments() -> list[dict]:
    """列出已有 _pages/for_llm 的作业（stem + 显示名）。"""
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return []
    out = []
    for sub in sorted(answers_dir.iterdir()):
        if not sub.is_dir():
            continue
        stem = sub.name
        llm_dir = sub / "_pages" / "for_llm"
        if not llm_dir.is_dir():
            continue
        pages = sorted(llm_dir.glob("page_*.png"))
        if not pages:
            continue
        out.append({"stem": stem, "label": stem})
    return out


@router.get("/segment-editor/assignments")
def segment_editor_list_assignments() -> list:
    """获取可编辑 segment 的作业列表。"""
    return _list_segment_assignments()


def _resolve_stem_for_answers(stem: str) -> str | None:
    """解析 stem 对应的 answers 下实际目录名（处理 Unicode 规范化差异）。"""
    import unicodedata
    answers_dir = get_answers_dir()
    if not answers_dir.exists():
        return None
    stem_nfc = unicodedata.normalize("NFC", stem)
    for d in answers_dir.iterdir():
        if d.is_dir() and unicodedata.normalize("NFC", d.name) == stem_nfc:
            return d.name
    return stem if (answers_dir / stem).exists() else None


@router.get("/segment-editor/assignment/{stem}")
def segment_editor_get_assignment(stem: str) -> dict:
    """获取某份作业的 for_llm 页面与当前 segment 数据。"""
    from PIL import Image

    actual_stem = _resolve_stem_for_answers(stem)
    if actual_stem is None:
        return {"error": "未找到该作业的 for_llm 页面"}
    base = get_answers_dir() / actual_stem / "_pages" / "for_llm"
    if not base.exists() or not base.is_dir():
        return {"error": "未找到该作业的 for_llm 页面"}
    page_files = sorted(base.glob("page_*.png"))
    pages = []
    dimensions = []
    for f in page_files:
        try:
            img = Image.open(str(f))
            w, h = img.size
            img.close()
        except Exception:
            w, h = 0, 0
        dimensions.append([w, h])
        # 前端通过 /files/answers/{stem}/_pages/for_llm/page_N.png 访问（用实际目录名）
        rel = f"/files/answers/{actual_stem}/_pages/for_llm/{f.name}"
        pages.append({"url": rel, "width": w, "height": h, "name": f.name})
    loaded = load_segments_for_stem(actual_stem)
    questions = []
    if loaded:
        raw = loaded.get("questions") or loaded.get("question")
        if isinstance(raw, list):
            questions = raw
        if loaded.get("dimensions"):
            dimensions = loaded["dimensions"]
    qids = [q.qid for q in load_all_questions()]
    return {
        "stem": actual_stem,
        "pages": pages,
        "dimensions": dimensions,
        "questions": questions,
        "qids": qids,
    }


@router.post("/segment-editor/assignment/{stem}/save")
def segment_editor_save_assignment(stem: str, body: dict = Body(...)) -> dict:
    """保存人工修改的 segment，并重新生成答案图。"""
    global _answer_map
    dimensions = body.get("dimensions") or []
    questions = body.get("questions") or []
    if not dimensions and not questions:
        return {"error": "缺少 dimensions 或 questions"}
    pdfs = _list_pdfs()
    pdf_path = None
    for p in pdfs:
        if Path(p).stem == stem:
            pdf_path = p
            break
    if not pdf_path:
        return {"error": f"未找到对应 PDF：{stem}"}
    # dimensions 转为 list of tuple 供 apply_manual_segments
    dims_tuples = [tuple(d) if isinstance(d, list) else d for d in dimensions]
    try:
        answer_map = apply_manual_segments(pdf_path, dims_tuples, questions)
        _answer_map[stem] = answer_map
        return {"ok": True, "message": "已更新该作业的 segment 并重新生成答案图"}
    except Exception as e:
        logger.exception("segment-editor save failed for %s", stem)
        return {"error": str(e)}


@router.get("/reports")
def get_reports() -> list:
    global _reports
    if not _reports:
        _reports = _load_reports_from_disk()
    return [r.model_dump() for r in _reports]


@router.post("/cancel")
async def cancel_pipeline() -> dict:
    """取消当前正在运行的切分/评分/报告任务。"""
    global _running_task
    if _running_task is None:
        return {"ok": True, "message": "没有正在运行的任务"}
    if _running_task.done():
        _running_task = None
        with _status_lock:
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
        with _status_lock:
            _status.stage = "idle"
            _status.message = "已取消"
            _status.current = 0
            _status.total = 0
            _status.errors = []
    return {"ok": True, "message": "已取消"}


@router.post("/reset")
def reset_status() -> dict:
    with _status_lock:
        _status.stage = "idle"
        _status.message = ""
        _status.current = 0
        _status.total = 0
        _status.errors = []
    return {"ok": True}
